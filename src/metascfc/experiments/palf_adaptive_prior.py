"""Adaptive Prior-Regularization Solver for Phase 2A.

Replaces binary prior mechanisms (rho=0 or rho=1, tau=0 or tau>0) with
continuous interpolation parameters.

Adaptive objective (FC-only branch):
    min ||y - X*beta||^2 + c*lambda_F * beta^T * D_rho * beta + c*tau_L * beta^T * L_p * beta

where:
    D_rho = (1-rho) * I + rho * D_p  (interpolated diagonal)
    D_p = D(q; gamma=0.5) is the prior-aware diagonal
    tau_L >= 0 is the direct network strength

This module provides:
    1. build_adaptive_prior_cache(): builds shared components for all (rho, tau)
    2. build_cache_for_rho(): builds _MSANCRCache with specific D_rho
    3. solve_and_predict_adaptive(): solves FC-only with specific (rho, tau, lambda_F)
    4. select_adaptive_params(): 3-fold CV with one-SE rule for parameter selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import prediction_metrics
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    compute_diagonal_penalty,
    lift_roi_to_edge,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2  # 6670
C_SCALE = 2 * N_EDGE  # 13340
GAMMA_FIXED = 0.5
DIAGONAL_EPSILON = 0.001
TOP_K = 10


# ---------------------------------------------------------------------------
# Adaptive Prior Cache
# ---------------------------------------------------------------------------

@dataclass
class AdaptivePriorCache:
    """Precomputed shared components for adaptive prior regularization.

    These depend on the ROI prior and topology but NOT on (rho, tau, lambda_F).

    Attributes
    ----------
    edge_prior : (n_edges,) lifted from roi_prior
    D_prior : (n_edges,) the prior-aware diagonal D(q; gamma)
    D_prior_inv_sqrt : (n_edges,) 1/sqrt(D_prior)
    identity_diag : (n_edges,) all ones
    active_indices : indices of edges touching top-k ROIs
    L_A : active Laplacian matrix (n_active, n_active)
    n_edges : total number of edge features
    n_rois : atlas size
    gamma : anisotropy exponent
    top_k : number of top ROIs for Laplacian
    """
    edge_prior: np.ndarray
    D_prior: np.ndarray
    D_prior_inv_sqrt: np.ndarray
    identity_diag: np.ndarray
    active_indices: np.ndarray
    L_A: np.ndarray
    n_edges: int
    n_rois: int
    gamma: float
    top_k: int

    @property
    def n_active(self) -> int:
        return int(len(self.active_indices))


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_adaptive_prior_cache(
    roi_prior: np.ndarray,
    n_rois: int = N_ROI,
    top_k: int = TOP_K,
    gamma: float = GAMMA_FIXED,
    epsilon: float = DIAGONAL_EPSILON,
) -> AdaptivePriorCache:
    """Build the adaptive prior cache with shared components.

    Parameters
    ----------
    roi_prior : (n_rois,) ROI-level prior scores
    n_rois : atlas size
    top_k : number of top ROIs for Laplacian active set
    gamma : anisotropy exponent for D_p
    epsilon : small constant for diagonal penalty

    Returns
    -------
    AdaptivePriorCache with precomputed shared components
    """
    roi_prior = np.asarray(roi_prior, dtype=np.float64).ravel()
    if len(roi_prior) != n_rois:
        raise ValueError(f"roi_prior has {len(roi_prior)} entries; expected {n_rois}")

    # Edge-level prior
    edge_prior = lift_roi_to_edge(roi_prior, n_rois, rule="prod")

    # Prior-aware diagonal
    D_prior = compute_diagonal_penalty(edge_prior, gamma, epsilon, normalize=True)
    D_prior_inv_sqrt = 1.0 / np.sqrt(np.maximum(D_prior, 1e-30))

    # Identity diagonal
    identity_diag = np.ones(n_rois * (n_rois - 1) // 2, dtype=np.float64)

    # Edge Laplacian
    edge_lap = build_edge_laplacian(
        n_rois,
        prior_scores=roi_prior,
        top_k=top_k,
        weighting="binary",
        couple_modalities=False,
        normalize="sym",
    )

    active_indices = edge_lap.active_indices
    L_A = edge_lap.active_laplacian

    return AdaptivePriorCache(
        edge_prior=edge_prior,
        D_prior=D_prior,
        D_prior_inv_sqrt=D_prior_inv_sqrt,
        identity_diag=identity_diag,
        active_indices=active_indices,
        L_A=L_A,
        n_edges=n_rois * (n_rois - 1) // 2,
        n_rois=n_rois,
        gamma=float(gamma),
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Build cache for specific rho
# ---------------------------------------------------------------------------

def build_cache_for_rho(
    adaptive_cache: AdaptivePriorCache,
    rho: float,
) -> _MSANCRCache:
    """Build _MSANCRCache with D_rho = (1-rho)*I + rho*D_prior.

    Parameters
    ----------
    adaptive_cache : shared components from build_adaptive_prior_cache
    rho : interpolation parameter in [0, 1]

    Returns
    -------
    _MSANCRCache compatible with _solve_msancr_kernel
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    # D_rho = (1-rho)*I + rho*D_prior
    D_rho = (1.0 - rho) * adaptive_cache.identity_diag + rho * adaptive_cache.D_prior
    D_rho_inv_sqrt = 1.0 / np.sqrt(np.maximum(D_rho, 1e-30))

    active = adaptive_cache.active_indices
    n_active = len(active)

    # Eigendecomposition of D_A_rho^{-1/2} L_A D_A_rho^{-1/2}
    if n_active > 0:
        d_active_inv_sqrt = D_rho_inv_sqrt[active]
        whitened_laplacian = (
            d_active_inv_sqrt[:, None]
            * adaptive_cache.L_A
            * d_active_inv_sqrt[None, :]
        )
        whitened_laplacian = 0.5 * (whitened_laplacian + whitened_laplacian.T)
        generalized_mu, generalized_u = np.linalg.eigh(whitened_laplacian)
        generalized_mu = np.clip(generalized_mu, 0.0, None)
    else:
        generalized_u = np.empty((0, 0), dtype=np.float64)
        generalized_mu = np.empty(0, dtype=np.float64)

    return _MSANCRCache(
        D=D_rho,
        D_inv_sqrt=D_rho_inv_sqrt,
        active_indices=active,
        D_active=D_rho[active],
        active_laplacian=adaptive_cache.L_A,
        generalized_u=generalized_u,
        generalized_mu=generalized_mu,
        n_edges=adaptive_cache.n_edges,
        n_rois=adaptive_cache.n_rois,
        gamma=adaptive_cache.gamma,
        lifting="prod",
    )


# ---------------------------------------------------------------------------
# Solve and predict
# ---------------------------------------------------------------------------

def solve_and_predict_adaptive(
    adaptive_cache: AdaptivePriorCache,
    X_fc_train: np.ndarray,
    y_train: np.ndarray,
    X_fc_test: np.ndarray,
    rho: float,
    tau: float,
    lambda_F: float,
    lambda_sc: float = 1.0,
) -> Dict:
    """Solve FC-only adaptive prior model and predict on test set.

    Parameters
    ----------
    adaptive_cache : shared components
    X_fc_train : (n_train, n_edges) training FC features
    y_train : (n_train,) training targets
    X_fc_test : (n_test, n_edges) test FC features
    rho : interpolation parameter for D_rho
    tau : network strength (lambda_l in MS-A-NCR notation)
    lambda_F : ridge penalty for FC
    lambda_sc : ridge penalty for SC (ignored for FC-only)

    Returns
    -------
    dict with keys: fp_test_pred, alpha, selected_params
    """
    X_fc_train = np.asarray(X_fc_train, dtype=np.float64)
    X_fc_test = np.asarray(X_fc_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

    # Build cache for this rho
    cache = build_cache_for_rho(adaptive_cache, rho)

    # Standardize features
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc_train)
    X_fc_test_z = scaler_fc.transform(X_fc_test)

    # Standardize target
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-8)
    y_z = (y_train - y_mean) / y_std

    # Solve
    alpha, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z), y_z, cache,
        lambda_F, lambda_sc, tau, fc_only=True,
    )

    # Predict
    pred_z = _predict_msancr(
        X_fc_test_z, np.zeros_like(X_fc_test_z),
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha, cache, lambda_F, lambda_sc, tau, fc_only=True,
    )
    fp_test_pred = pred_z * y_std + y_mean

    return {
        "fp_test_pred": fp_test_pred,
        "alpha": alpha,
        "selected_params": {
            "rho": rho,
            "tau": tau,
            "lambda_F": lambda_F,
        },
    }


# ---------------------------------------------------------------------------
# Parameter selection with one-SE rule
# ---------------------------------------------------------------------------

def _compute_one_se_rule(
    all_results: List[Dict],
) -> Tuple[Dict, Dict]:
    """Apply one-SE rule to select simplest candidate within r_best - SE_best.

    Simplicity order: lowest rho, then lowest tau, then lambda_F closest to 0.1.

    Parameters
    ----------
    all_results : list of dicts with keys: rho, tau, lambda_F, mean_pearson, se_pearson

    Returns
    -------
    (best_max_candidate, best_se_candidate)
    """
    if not all_results:
        return {}, {}

    # Find best by mean Pearson
    best_max = max(all_results, key=lambda r: r["mean_pearson"])
    r_best = best_max["mean_pearson"]
    se_best = best_max["se_pearson"]

    # One-SE threshold
    threshold = r_best - se_best

    # Candidates within threshold
    se_candidates = [r for r in all_results if r["mean_pearson"] >= threshold]

    # Sort by simplicity: lowest rho, then lowest tau, then lambda_F closest to 0.1
    def simplicity_key(r):
        return (r["rho"], r["tau"], abs(r["lambda_F"] - 0.1))

    best_se = min(se_candidates, key=simplicity_key) if se_candidates else best_max

    return best_max, best_se


def select_adaptive_params(
    adaptive_cache: AdaptivePriorCache,
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    rho_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    tau_grid: Sequence[float] = (0.0, 0.03, 0.1, 0.5, 1.0, 2.0, 5.0),
    lambda_F_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    n_inner_folds: int = 3,
    random_state: int = 42,
) -> Dict:
    """Select adaptive prior parameters via 3-fold CV with one-SE rule.

    Parameters
    ----------
    adaptive_cache : shared components
    X_fc : (n_subjects, n_edges) FC features
    y : (n_subjects,) targets
    train_idx : indices of training subjects
    rho_grid : interpolation parameter candidates
    tau_grid : network strength candidates
    lambda_F_grid : ridge penalty candidates
    n_inner_folds : number of inner CV folds
    random_state : random seed for CV splits

    Returns
    -------
    dict with keys: best_rho, best_tau, best_lambda_F, all_results,
                    best_max_candidate, best_se_candidate
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Create inner CV splits
    inner_splitter = KFold(
        n_splits=n_inner_folds,
        shuffle=True,
        random_state=random_state,
    )

    all_results = []

    for rho in rho_grid:
        for tau in tau_grid:
            for lambda_F in lambda_F_grid:
                fold_pearsons = []

                for inner_train_local, inner_val_local in inner_splitter.split(
                    train_idx
                ):
                    inner_train_global = train_idx[inner_train_local]
                    inner_val_global = train_idx[inner_val_local]

                    try:
                        result = solve_and_predict_adaptive(
                            adaptive_cache,
                            X_fc[inner_train_global],
                            y[inner_train_global],
                            X_fc[inner_val_global],
                            rho=rho,
                            tau=tau,
                            lambda_F=lambda_F,
                        )
                        m = prediction_metrics(
                            y[inner_val_global], result["fp_test_pred"]
                        )
                        fold_pearsons.append(m["pearson"])
                    except Exception:
                        fold_pearsons.append(-np.inf)

                mean_pearson = float(np.mean(fold_pearsons))
                se_pearson = (
                    float(np.std(fold_pearsons, ddof=1) / np.sqrt(n_inner_folds))
                    if n_inner_folds > 1
                    else 0.0
                )

                all_results.append({
                    "rho": rho,
                    "tau": tau,
                    "lambda_F": lambda_F,
                    "mean_pearson": mean_pearson,
                    "se_pearson": se_pearson,
                    "fold_pearsons": fold_pearsons,
                })

    # Apply one-SE rule
    best_max, best_se = _compute_one_se_rule(all_results)

    return {
        "best_rho": best_se.get("rho", 0.0),
        "best_tau": best_se.get("tau", 0.0),
        "best_lambda_F": best_se.get("lambda_F", 0.1),
        "all_results": all_results,
        "best_max_candidate": best_max,
        "best_se_candidate": best_se,
    }


# ---------------------------------------------------------------------------
# Convenience for cross-fitted OOF generation
# ---------------------------------------------------------------------------

def generate_crossfit_oof_adaptive(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    adaptive_cache: AdaptivePriorCache,
    seed: int,
    outer_fold: int,
    rho_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    tau_grid: Sequence[float] = (0.0, 0.03, 0.1, 0.5, 1.0, 2.0, 5.0),
    lambda_F_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    n_rois: int = N_ROI,
) -> Dict:
    """Generate cross-fitted OOF predictions for adaptive prior FP branch.

    For each fusion fold (A_k, V_k):
      1. Select (rho, tau, lambda_F) on A_k only (inner CV)
      2. Fit on A_k, predict V_k

    Returns dict with fp_oof, selected_params, selection_info.
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Create fusion folds
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 777)
    perm = rng.permutation(len(train_idx))
    fold_sizes = np.full(n_fusion_folds, len(train_idx) // n_fusion_folds)
    fold_sizes[: len(train_idx) % n_fusion_folds] += 1

    fusion_folds = []
    current = 0
    for k in range(n_fusion_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        fusion_folds.append((train_idx[a_local], train_idx[v_local]))
        current = stop

    n_train = len(train_idx)
    fp_oof = np.full(n_train, np.nan, dtype=np.float64)
    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    fp_params_list = []
    fp_sel_infos = []

    for fold_k, (a_k, v_k) in enumerate(fusion_folds):
        # Select parameters on A_k
        sel_result = select_adaptive_params(
            adaptive_cache, X_fc, y, a_k,
            rho_grid, tau_grid, lambda_F_grid, n_inner,
            random_state=int(seed) * 10000 + int(outer_fold) * 100 + fold_k,
        )

        # Fit on A_k, predict V_k
        result = solve_and_predict_adaptive(
            adaptive_cache, X_fc[a_k], y[a_k], X_fc[v_k],
            rho=sel_result["best_rho"],
            tau=sel_result["best_tau"],
            lambda_F=sel_result["best_lambda_F"],
        )

        # Store OOF predictions
        v_local = np.array([train_local_map[int(idx)] for idx in v_k])
        fp_oof[v_local] = result["fp_test_pred"]

        fp_params_list.append(sel_result["best_se_candidate"])
        fp_sel_infos.append(sel_result)

    # Aggregate selected params
    best_rho = float(np.median([p.get("rho", 0.0) for p in fp_params_list]))
    best_tau = float(np.median([p.get("tau", 0.0) for p in fp_params_list]))
    best_lambda_F = float(np.median([p.get("lambda_F", 0.1) for p in fp_params_list]))

    return {
        "fp_oof": fp_oof,
        "selected_params": {
            "rho": best_rho,
            "tau": best_tau,
            "lambda_F": best_lambda_F,
        },
        "selection_info": fp_sel_infos,
    }
