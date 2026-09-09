"""Multi-Task Residual Network-Constrained Ridge (MT-RNCR).

Phase 2C: Uses individual-task late fusion as backbone, then applies
multi-task residual learning with LLM-derived shared/contrast priors.

Key idea:
- Base late fusion already includes FC + SC
- Residual branch tests incremental semantic FC information
- Shared/contrast decomposition exploits cross-task correlation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from metascfc.models.iclr_backbones.network_constrained_ridge import (
    EdgeLaplacian,
    _LaplacianEig,
    build_edge_laplacian,
    factor_laplacian_eig,
    NetworkConstrainedRidge,
)

N_EDGE = 6670
EPSILON_P = 1e-6


# ---------------------------------------------------------------------------
# Shared/contrast decomposition
# ---------------------------------------------------------------------------

def build_shared_contrast_residuals(
    r_wm: np.ndarray,
    r_fi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct shared/contrast residual targets.

    Parameters
    ----------
    r_wm : (n,) standardized WM residuals
    r_fi : (n,) standardized FI residuals

    Returns
    -------
    z_shared : (n,) = (z_wm + z_fi) / sqrt(2)
    z_contrast : (n,) = (z_wm - z_fi) / sqrt(2)
    """
    z_shared = (r_wm + r_fi) / np.sqrt(2)
    z_contrast = (r_wm - r_fi) / np.sqrt(2)
    return z_shared, z_contrast


def reconstruct_task_residuals(
    z_shared_pred: np.ndarray,
    z_contrast_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct task-specific residual predictions from shared/contrast.

    Returns
    -------
    z_wm_pred : (n,) = (z_shared + z_contrast) / sqrt(2)
    z_fi_pred : (n,) = (z_shared - z_contrast) / sqrt(2)
    """
    z_wm_pred = (z_shared_pred + z_contrast_pred) / np.sqrt(2)
    z_fi_pred = (z_shared_pred - z_contrast_pred) / np.sqrt(2)
    return z_wm_pred, z_fi_pred


# ---------------------------------------------------------------------------
# Multi-task priors
# ---------------------------------------------------------------------------

def build_shared_prior(
    prior_wm: np.ndarray,
    prior_fi: np.ndarray,
    epsilon_p: float = EPSILON_P,
) -> np.ndarray:
    """Shared prior: geometric mean, min-max normalized to [0,1].

    p_shared_i = sqrt((p_WM_i + eps) * (p_FI_i + eps))
    """
    p = np.sqrt((prior_wm + epsilon_p) * (prior_fi + epsilon_p))
    p_min, p_max = p.min(), p.max()
    if p_max > p_min:
        p = (p - p_min) / (p_max - p_min)
    else:
        p = np.ones_like(p) * 0.5
    return p


def build_contrast_prior(
    prior_wm: np.ndarray,
    prior_fi: np.ndarray,
) -> np.ndarray:
    """Contrast prior: absolute difference, min-max normalized to [0,1].

    p_contrast_i = |p_WM_i - p_FI_i|
    """
    p = np.abs(prior_wm - prior_fi)
    p_min, p_max = p.min(), p.max()
    if p_max > p_min:
        p = (p - p_min) / (p_max - p_min)
    else:
        p = np.zeros_like(p)
    return p


# ---------------------------------------------------------------------------
# NCR residual fitting
# ---------------------------------------------------------------------------

def fit_ncr_residual(
    X_fc: np.ndarray,
    y_res: np.ndarray,
    edge_laplacian: Optional[EdgeLaplacian],
    lap_eig: Optional[_LaplacianEig],
    lambda1: float,
    lambda2: float,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit NCR residual model on training fold, predict on validation fold.

    NetworkConstrainedRidge expects 2*N_EDGE features (FC+SC concatenated).
    Since we only use FC features, we pad with zeros for SC.

    Parameters
    ----------
    X_fc : (n, 6670) FC features
    y_res : (n,) full residual target
    edge_laplacian : Laplacian (None for lambda2=0)
    lap_eig : eigendecomposition (None for lambda2=0)
    lambda1 : ridge penalty
    lambda2 : Laplacian penalty
    train_idx : training indices
    val_idx : validation indices

    Returns
    -------
    val_pred : (n_val,) predictions on validation set
    best_alpha : selected alpha (unused, for API compat)
    """
    n_fc = X_fc.shape[1]
    X_fc_sc = np.hstack([X_fc, np.zeros((X_fc.shape[0], n_fc), dtype=X_fc.dtype)])

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_fc_sc[train_idx])
    X_val = scaler_X.transform(X_fc_sc[val_idx])

    y_mean = float(y_res[train_idx].mean())
    y_std = max(float(y_res[train_idx].std()), 1e-8)
    y_tr = (y_res[train_idx] - y_mean) / y_std

    if lambda2 == 0 or edge_laplacian is None:
        # Plain Ridge
        model = Ridge(alpha=lambda1, fit_intercept=False)
        model.fit(X_tr, y_tr)
        pred_z = model.predict(X_val)
    else:
        # NCR with Laplacian
        ncr = NetworkConstrainedRidge(
            alpha1=lambda1, alpha2=lambda2,
            edge_laplacian=edge_laplacian, standardize=False,
        )
        ncr.fit(X_tr, y_tr)
        pred_z = ncr.predict(X_val)

    return pred_z * y_std + y_mean, lambda1


def crossfit_ncr_residual(
    X_fc: np.ndarray,
    y_res: np.ndarray,
    edge_laplacian: Optional[EdgeLaplacian],
    lap_eig: Optional[_LaplacianEig],
    lambda1: float,
    lambda2: float,
    train_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    n_inner_folds: int = 5,
) -> np.ndarray:
    """Cross-fit NCR residual model on outer-training set.

    Parameters
    ----------
    X_fc : (n_full, n_edges) full-dataset FC features
    y_res : (n_train,) training-subset residual target
    train_idx : (n_train,) global indices into full dataset
    ...
    Returns OOF predictions for all outer-training subjects.
    """
    n_train = len(train_idx)
    oof = np.full(n_train, np.nan, dtype=np.float64)

    # Slice X_fc to training subjects once
    X_fc_train = X_fc[train_idx]

    # Deterministic inner folds
    rng = np.random.RandomState(20000 + 100 * seed + outer_fold)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(n_inner_folds, n_train // n_inner_folds)
    fold_sizes[:n_train % n_inner_folds] += 1

    current = 0
    for k in range(n_inner_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])

        val_pred, _ = fit_ncr_residual(
            X_fc_train, y_res, edge_laplacian, lap_eig,
            lambda1, lambda2, a_local, v_local,
        )

        oof[v_local] = val_pred
        current = stop

    return oof


# ---------------------------------------------------------------------------
# Eta selection
# ---------------------------------------------------------------------------

def select_eta(
    base_oof: np.ndarray,
    residual_oof: np.ndarray,
    y: np.ndarray,
    eta_grid: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
) -> Tuple[float, float]:
    """Select eta maximizing Pearson(base_oof + eta * residual_oof, y).

    Returns (best_eta, best_pearson).
    """
    best_eta = 0.0
    best_r = -np.inf

    for eta in eta_grid:
        combined = base_oof + eta * residual_oof
        # Handle NaN
        mask = np.isfinite(combined) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        r, _ = pearsonr(combined[mask], y[mask])
        if r > best_r + 1e-14:
            best_r = r
            best_eta = eta

    return best_eta, best_r


def select_eta_mt(
    base_oof_wm: np.ndarray,
    base_oof_fi: np.ndarray,
    residual_oof_wm: np.ndarray,
    residual_oof_fi: np.ndarray,
    y_wm: np.ndarray,
    y_fi: np.ndarray,
    eta_grid: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
) -> Tuple[float, float, float]:
    """Select task-specific etas for MT model using mean Fisher-z score.

    Returns (best_eta_wm, best_eta_fi, best_score).
    """
    best_eta_wm = 0.0
    best_eta_fi = 0.0
    best_score = -np.inf

    for eta_wm in eta_grid:
        for eta_fi in eta_grid:
            comb_wm = base_oof_wm + eta_wm * residual_oof_wm
            comb_fi = base_oof_fi + eta_fi * residual_oof_fi

            mask_wm = np.isfinite(comb_wm) & np.isfinite(y_wm)
            mask_fi = np.isfinite(comb_fi) & np.isfinite(y_fi)

            if mask_wm.sum() < 10 or mask_fi.sum() < 10:
                continue

            r_wm, _ = pearsonr(comb_wm[mask_wm], y_wm[mask_wm])
            r_fi, _ = pearsonr(comb_fi[mask_fi], y_fi[mask_fi])

            # Fisher-z mean
            r_wm_c = np.clip(r_wm, -0.999999, 0.999999)
            r_fi_c = np.clip(r_fi, -0.999999, 0.999999)
            score = (np.arctanh(r_wm_c) + np.arctanh(r_fi_c)) / 2

            if score > best_score + 1e-14:
                best_score = score
                best_eta_wm = eta_wm
                best_eta_fi = eta_fi

    return best_eta_wm, best_eta_fi, best_score


# ---------------------------------------------------------------------------
# Final refit and predict
# ---------------------------------------------------------------------------

def fit_final_ncr_and_predict(
    X_fc: np.ndarray,
    y_res_train: np.ndarray,
    edge_laplacian: Optional[EdgeLaplacian],
    lambda1: float,
    lambda2: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Fit NCR on full training, predict on test.

    Parameters
    ----------
    X_fc : (n_full, n_edges) full-dataset FC features
    y_res_train : (n_train,) training-subset residual target
    train_idx : (n_train,) global indices for training subjects
    test_idx : (n_test,) global indices for test subjects
    """
    n_fc = X_fc.shape[1]
    X_fc_sc = np.hstack([X_fc, np.zeros((X_fc.shape[0], n_fc), dtype=X_fc.dtype)])

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_fc_sc[train_idx])
    X_te = scaler_X.transform(X_fc_sc[test_idx])

    y_mean = float(y_res_train.mean())
    y_std = max(float(y_res_train.std()), 1e-8)
    y_tr = (y_res_train - y_mean) / y_std

    if lambda2 == 0 or edge_laplacian is None:
        model = Ridge(alpha=lambda1, fit_intercept=False)
        model.fit(X_tr, y_tr)
        pred_z = model.predict(X_te)
    else:
        ncr = NetworkConstrainedRidge(
            alpha1=lambda1, alpha2=lambda2,
            edge_laplacian=edge_laplacian, standardize=False,
        )
        ncr.fit(X_tr, y_tr)
        pred_z = ncr.predict(X_te)

    return pred_z * y_std + y_mean


# ---------------------------------------------------------------------------
# Coefficient conversion
# ---------------------------------------------------------------------------

def convert_coefficients_to_original(
    beta_std: np.ndarray,
    scaler_X: StandardScaler,
    y_std: float,
) -> np.ndarray:
    """Convert standardized-space coefficients to original units.

    β_original = β_std * σ_y / σ_x
    """
    x_scale = scaler_X.scale_
    return beta_std * y_std / x_scale


# ---------------------------------------------------------------------------
# Stability diagnostics
# ---------------------------------------------------------------------------

def compute_coefficient_stability(
    beta_list: List[np.ndarray],
    n_edges: int = N_EDGE,
) -> Dict:
    """Compute pairwise Spearman and top-100 Jaccard for a set of coefficients."""
    from scipy.stats import spearmanr

    n = len(beta_list)
    if n < 2:
        return {"mean_spearman": 0.0, "mean_jaccard_top100": 0.0}

    spearmans = []
    jaccards = []

    for i in range(n):
        for j in range(i + 1, n):
            abs_i = np.abs(beta_list[i])
            abs_j = np.abs(beta_list[j])

            # Spearman on absolute coefficients
            rho, _ = spearmanr(abs_i, abs_j)
            spearmans.append(rho)

            # Top-100 Jaccard
            top_i = set(np.argsort(abs_i)[-100:])
            top_j = set(np.argsort(abs_j)[-100:])
            jacc = len(top_i & top_j) / len(top_i | top_j)
            jaccards.append(jacc)

    return {
        "mean_spearman": float(np.mean(spearmans)),
        "mean_jaccard_top100": float(np.mean(jaccards)),
    }


def aggregate_edge_to_roi(
    beta: np.ndarray,
    n_rois: int = 116,
) -> np.ndarray:
    """Aggregate edge-level saliency to ROI-level by summing |beta| over incident edges."""
    roi_saliency = np.zeros(n_rois, dtype=np.float64)
    iu = np.triu_indices(n_rois, k=1)
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        roi_saliency[i] += abs(beta[idx])
        roi_saliency[j] += abs(beta[idx])
    return roi_saliency
