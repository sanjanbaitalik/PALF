"""PALF Cross-Fitted Ablation Experiment.

Fully cross-fitted component ablation matrix (R0-R3) with correct isolation:
  - OOF predictions exclude held-out subjects from preprocessing, selection, and fitting
  - Final branch parameters reselected on all outer-training subjects
  - Equivalent solver scale across all conditions

Conditions:
  R0: Same-solver no prior (D=I, lambda_L=0)
  R1: Anisotropy only (D(q;0.5), lambda_L=0)
  R2: Network only (D=I, positive lambda_L L_p)
  R3: Full PALF (D(q;0.5), positive lambda_L L_p)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import prediction_metrics
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    compute_diagonal_penalty,
    lift_roi_to_edge,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
    factor_laplacian_eig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2  # 6670
C_SCALE = 2 * N_EDGE  # 13340

RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LAMBDA_L_GRID = [0.0, 0.03, 0.1, 0.5, 1.0, 2.0, 5.0]
LAMBDA_L_GRID_NO_NETWORK = [0.0]
LAMBDA_L_GRID_NETWORK = [0.03, 0.1, 0.5, 1.0, 2.0, 5.0]

WEIGHT_STEP = 0.05
WEIGHT_GRID = [round(i * WEIGHT_STEP, 4) for i in range(0, int(1.0 / WEIGHT_STEP) + 1)]

TOP_K = 10
DIAGONAL_EPSILON = 0.001
GAMMA_FIXED = 0.5
SELECTION_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationCondition:
    """Defines one ablation condition."""
    id: str
    name: str
    use_anisotropy: bool
    use_network: bool
    lambda_l_grid: Tuple[float, ...]

    @property
    def use_fc_only(self) -> bool:
        return True  # All conditions use FC-only for the FP branch


CONDITIONS = {
    "R0": AblationCondition(
        id="R0", name="Same-solver no prior",
        use_anisotropy=False, use_network=False,
        lambda_l_grid=tuple(LAMBDA_L_GRID_NO_NETWORK),
    ),
    "R1": AblationCondition(
        id="R1", name="Anisotropy only",
        use_anisotropy=True, use_network=False,
        lambda_l_grid=tuple(LAMBDA_L_GRID_NO_NETWORK),
    ),
    "R2": AblationCondition(
        id="R2", name="Network only",
        use_anisotropy=False, use_network=True,
        lambda_l_grid=tuple(LAMBDA_L_GRID_NETWORK),
    ),
    "R3": AblationCondition(
        id="R3", name="Full PALF",
        use_anisotropy=True, use_network=True,
        lambda_l_grid=tuple(LAMBDA_L_GRID_NETWORK),
    ),
}


# ---------------------------------------------------------------------------
# Cache builder for different conditions
# ---------------------------------------------------------------------------

def build_condition_cache(
    roi_prior: np.ndarray,
    condition: AblationCondition,
    n_rois: int = N_ROI,
) -> _MSANCRCache:
    """Build _MSANCRCache appropriate for the condition.

    - R0: gamma=0, epsilon=1, D=I (all weights 1.0)
    - R1: gamma=0.5, epsilon=0.001, D=D(q;0.5)
    - R2: gamma=0, epsilon=1, D=I (uses L_p from top-10 ROIs)
    - R3: gamma=0.5, epsilon=0.001, D=D(q;0.5) with L_p
    """
    if condition.use_anisotropy:
        gamma = GAMMA_FIXED
        epsilon = DIAGONAL_EPSILON
    else:
        gamma = 0.0
        epsilon = 1.0  # D = (1 + |q|)^0 = 1.0 (identity)

    # Always build Laplacian with top_k >= 1; lambda_l=0 disables it during solving
    cache = build_msancr_cache(
        roi_prior=roi_prior,
        n_rois=n_rois,
        gamma=gamma,
        lifting="prod",
        top_k=TOP_K,
        epsilon=epsilon,
        weighting="binary",
        couple_modalities=False,
        normalize_laplacian="sym",
        prior_space="node",
    )
    return cache


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def make_outer_splits(
    n_subjects: int,
    seeds: Sequence[int],
    n_outer_folds: int = 5,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Generate outer train/test splits.

    Returns list of (seed, fold, train_idx, test_idx).
    """
    splits = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_subjects)
        fold_sizes = np.full(n_outer_folds, n_subjects // n_outer_folds)
        fold_sizes[: n_subjects % n_outer_folds] += 1
        current = 0
        for fold_i in range(n_outer_folds):
            start, stop = current, current + fold_sizes[fold_i]
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            splits.append((seed, fold_i, train_idx, test_idx))
            current = stop
    return splits


def make_fusion_folds(
    train_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    n_fusion_folds: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create fusion-training folds (A_k, V_k) partitioning train_idx.

    Returns list of (A_k_indices, V_k_indices) as global subject indices.
    """
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 777)
    perm = rng.permutation(len(train_idx))
    fold_sizes = np.full(n_fusion_folds, len(train_idx) // n_fusion_folds)
    fold_sizes[: len(train_idx) % n_fusion_folds] += 1

    folds = []
    current = 0
    for k in range(n_fusion_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        folds.append((train_idx[a_local], train_idx[v_local]))
        current = stop
    return folds


def make_inner_selection_folds(
    a_k: np.ndarray,
    seed: int,
    outer_fold: int,
    fusion_fold: int,
    n_inner: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create inner parameter-selection folds within A_k.

    Returns list of (B_indices, C_indices) as global subject indices.
    """
    rng = np.random.RandomState(
        int(seed) * 10000 + int(outer_fold) * 100 + int(fusion_fold) * 10 + 333
    )
    perm = rng.permutation(len(a_k))
    fold_sizes = np.full(n_inner, len(a_k) // n_inner)
    fold_sizes[: len(a_k) % n_inner] += 1

    folds = []
    current = 0
    for k in range(n_inner):
        start, stop = current, current + fold_sizes[k]
        c_local = perm[start:stop]
        b_local = np.concatenate([perm[:start], perm[stop:]])
        folds.append((a_k[b_local], a_k[c_local]))
        current = stop
    return folds


# ---------------------------------------------------------------------------
# Branch fitting primitives
# ---------------------------------------------------------------------------

def _select_alpha_ridge(
    X: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    ridge_grid: Sequence[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, Dict]:
    """Select Ridge alpha via inner CV on train_global.

    Returns (best_alpha, selection_info).
    """
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X[train_global])

    best_pearson = -np.inf
    best_alpha = ridge_grid[0]
    best_rmse = np.inf
    best_mae = np.inf
    all_scores = {}

    for alpha in ridge_grid:
        fold_ps = []
        fold_rmses = []
        fold_maes = []
        for b_idx, c_idx in inner_folds:
            X_b = scaler.transform(X[b_idx])
            X_c = scaler.transform(X[c_idx])
            y_b = y[b_idx]
            y_c = y[c_idx]

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_b, y_b)
            pred = model.predict(X_c)
            m = prediction_metrics(y_c, pred)
            fold_ps.append(m["pearson"])
            fold_rmses.append(m["rmse"])
            fold_maes.append(m["mae"])

        mp = float(np.mean(fold_ps))
        mr = float(np.mean(fold_rmses))
        mm = float(np.mean(fold_maes))
        all_scores[alpha] = {"pearson": mp, "rmse": mr, "mae": mm}

        if (
            mp > best_pearson + SELECTION_TOLERANCE
            or (abs(mp - best_pearson) < SELECTION_TOLERANCE and mr < best_rmse - SELECTION_TOLERANCE)
            or (abs(mp - best_pearson) < SELECTION_TOLERANCE and abs(mr - best_rmse) < SELECTION_TOLERANCE and mm < best_mae - SELECTION_TOLERANCE)
        ):
            best_pearson = mp
            best_alpha = alpha
            best_rmse = mr
            best_mae = mm

    return best_alpha, {"scores": all_scores, "pearson": best_pearson, "rmse": best_rmse, "mae": best_mae}


def _select_fc_only_params(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    cache: _MSANCRCache,
    condition: AblationCondition,
    ridge_grid: Sequence[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict, Dict]:
    """Select (lambda_fc, lambda_l) for FC-only branch via inner CV on train_global.

    Returns (best_params, selection_info).
    """
    best_pearson = -np.inf
    best_params = {"lambda_fc": ridge_grid[0], "lambda_l": 0.0}
    all_scores = {}

    for lambda_fc in ridge_grid:
        for lambda_l in condition.lambda_l_grid:
            fold_ps = []
            for b_idx, c_idx in inner_folds:
                try:
                    scaler = StandardScaler()
                    X_fc_b = scaler.fit_transform(X_fc[b_idx])
                    X_fc_c = scaler.transform(X_fc[c_idx])

                    y_mean_b = float(y[b_idx].mean())
                    y_std_b = max(float(y[b_idx].std()), 1e-8)
                    y_z = (y[b_idx] - y_mean_b) / y_std_b

                    alpha, _ = _solve_msancr_kernel(
                        X_fc_b, np.zeros_like(X_fc_b), y_z, cache,
                        lambda_fc, 1.0, lambda_l, fc_only=True,
                    )
                    pred_z = _predict_msancr(
                        X_fc_c, np.zeros_like(X_fc_c),
                        X_fc_b, np.zeros_like(X_fc_b),
                        alpha, cache, lambda_fc, 1.0, lambda_l, fc_only=True,
                    )
                    pred = pred_z * y_std_b + y_mean_b
                    m = prediction_metrics(y[c_idx], pred)
                    fold_ps.append(m["pearson"])
                except Exception:
                    fold_ps.append(-np.inf)

            mp = float(np.mean(fold_ps))
            all_scores[(lambda_fc, lambda_l)] = {"pearson": mp}

            if mp > best_pearson + SELECTION_TOLERANCE:
                best_pearson = mp
                best_params = {"lambda_fc": lambda_fc, "lambda_l": lambda_l}

    return best_params, {"scores": all_scores, "pearson": best_pearson}


def _fit_and_predict_ridge_on_subset(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit Ridge on train_idx, predict val_idx. Scaler fit on train only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y[train_idx])
    return model.predict(X_val)


def _fit_and_predict_fc_only_on_subset(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cache: _MSANCRCache,
    params: Dict,
) -> np.ndarray:
    """Fit FC-only MS-A-NCR on train_idx, predict val_idx."""
    scaler = StandardScaler()
    X_fc_train = scaler.fit_transform(X_fc[train_idx])
    X_fc_val = scaler.transform(X_fc[val_idx])

    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_z = (y[train_idx] - y_mean) / y_std

    alpha, _ = _solve_msancr_kernel(
        X_fc_train, np.zeros_like(X_fc_train), y_z, cache,
        params["lambda_fc"], 1.0, params["lambda_l"], fc_only=True,
    )
    pred_z = _predict_msancr(
        X_fc_val, np.zeros_like(X_fc_val),
        X_fc_train, np.zeros_like(X_fc_train),
        alpha, cache, params["lambda_fc"], 1.0, params["lambda_l"], fc_only=True,
    )
    return pred_z * y_std + y_mean


# ---------------------------------------------------------------------------
# Cross-fitted OOF generation
# ---------------------------------------------------------------------------

@dataclass
class CrossFitOOFResult:
    """OOF predictions from cross-fitted branches."""
    fc_oof: np.ndarray  # FC-only (no-prior Ridge) OOF
    sc_oof: np.ndarray  # SC Ridge OOF
    fp_oof: np.ndarray  # FC-only prior-aware OOF
    fc_selection_info: Dict
    sc_selection_info: Dict
    fp_selection_info: Dict
    fc_selected_alpha: float
    sc_selected_alpha: float
    fp_selected_params: Dict


def generate_crossfit_oof(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    condition: AblationCondition,
    roi_prior: np.ndarray,
    seed: int,
    outer_fold: int,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    n_rois: int = N_ROI,
) -> CrossFitOOFResult:
    """Generate cross-fitted OOF predictions for all three branches.

    For each fusion fold (A_k, V_k):
      1. Select parameters on A_k only (inner CV)
      2. Fit on A_k, predict V_k
    """
    cache = build_condition_cache(roi_prior, condition, n_rois)
    fusion_folds = make_fusion_folds(train_idx, seed, outer_fold, n_fusion_folds)

    n_train = len(train_idx)
    fc_oof = np.full(n_train, np.nan, dtype=np.float64)
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)
    fp_oof = np.full(n_train, np.nan, dtype=np.float64)

    fc_sel_infos = []
    sc_sel_infos = []
    fp_sel_infos = []
    fc_alphas = []
    sc_alphas = []
    fp_params_list = []

    # Build global-to-local index mapping for train_idx
    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    for fold_k, (a_k, v_k) in enumerate(fusion_folds):
        # Create inner selection folds within A_k
        inner_folds = make_inner_selection_folds(a_k, seed, outer_fold, fold_k, n_inner)

        # Map global indices to local positions in train_idx for OOF storage
        v_local = np.array([train_local_map[int(idx)] for idx in v_k])

        # --- FC branch (no-prior Ridge) ---
        fc_alpha, fc_info = _select_alpha_ridge(
            X_fc, y, a_k, ridge_grid, inner_folds,
        )
        fc_pred = _fit_and_predict_ridge_on_subset(X_fc, y, a_k, v_k, fc_alpha)
        fc_oof[v_local] = fc_pred
        fc_sel_infos.append(fc_info)
        fc_alphas.append(fc_alpha)

        # --- SC branch (Ridge) ---
        sc_alpha, sc_info = _select_alpha_ridge(
            X_sc, y, a_k, ridge_grid, inner_folds,
        )
        sc_pred = _fit_and_predict_ridge_on_subset(X_sc, y, a_k, v_k, sc_alpha)
        sc_oof[v_local] = sc_pred
        sc_sel_infos.append(sc_info)
        sc_alphas.append(sc_alpha)

        # --- FP branch (FC-only prior-aware) ---
        fp_params, fp_info = _select_fc_only_params(
            X_fc, y, a_k, cache, condition, ridge_grid, inner_folds,
        )
        fp_pred = _fit_and_predict_fc_only_on_subset(X_fc, y, a_k, v_k, cache, fp_params)
        fp_oof[v_local] = fp_pred
        fp_sel_infos.append(fp_info)
        fp_params_list.append(fp_params)

    # Aggregate selection info across fusion folds
    fc_agg = _aggregate_selection_scores(fc_sel_infos, ridge_grid, "alpha")
    sc_agg = _aggregate_selection_scores(sc_sel_infos, ridge_grid, "alpha")
    fp_agg = _aggregate_fp_selection_scores(fp_sel_infos, condition)

    return CrossFitOOFResult(
        fc_oof=fc_oof,
        sc_oof=sc_oof,
        fp_oof=fp_oof,
        fc_selection_info=fc_agg,
        sc_selection_info=sc_agg,
        fp_selection_info=fp_agg,
        fc_selected_alpha=float(np.median(fc_alphas)),
        sc_selected_alpha=float(np.median(sc_alphas)),
        fp_selected_params=_select_most_common_fp_params(fp_params_list),
    )


def _aggregate_selection_scores(
    infos: List[Dict], grid: Sequence[float], key: str,
) -> Dict:
    """Aggregate selection scores across fusion folds."""
    # Count how often each alpha was selected
    from collections import Counter
    alpha_counts = Counter()
    for info in infos:
        if key == "alpha":
            # Find the alpha with best pearson in this fold's scores
            scores = info.get("scores", {})
            best_a = max(scores.keys(), default=grid[0], key=lambda a: scores[a]["pearson"])
            alpha_counts[best_a] += 1
    # Most commonly selected alpha
    if alpha_counts:
        best_alpha = alpha_counts.most_common(1)[0][0]
    else:
        best_alpha = grid[0]
    return {"best_alpha": best_alpha, "fold_alphas": [info.get("scores", {}) for info in infos]}


def _aggregate_fp_selection_scores(
    infos: List[Dict], condition: AblationCondition,
) -> Dict:
    """Aggregate FP selection scores across fusion folds."""
    from collections import Counter
    param_counts = Counter()
    for info in infos:
        scores = info.get("scores", {})
        if scores:
            best_key = max(scores.keys(), key=lambda k: scores[k]["pearson"])
            param_counts[best_key] += 1
    if param_counts:
        best_pair = param_counts.most_common(1)[0][0]
    else:
        best_pair = (RIDGE_GRID[0], 0.0)
    return {
        "best_lambda_fc": best_pair[0],
        "best_lambda_l": best_pair[1],
        "fold_params": [info.get("scores", {}) for info in infos],
    }


def _select_most_common_fp_params(params_list: List[Dict]) -> Dict:
    """Select most common FP parameters across folds."""
    from collections import Counter
    param_keys = []
    for p in params_list:
        key = (p.get("lambda_fc", RIDGE_GRID[0]), p.get("lambda_l", 0.0))
        param_keys.append(key)
    counts = Counter(param_keys)
    if counts:
        best = counts.most_common(1)[0][0]
        return {"lambda_fc": best[0], "lambda_sc": 1.0, "lambda_l": best[1]}
    return {"lambda_fc": RIDGE_GRID[0], "lambda_sc": 1.0, "lambda_l": 0.0}


# ---------------------------------------------------------------------------
# Final reselection on full T
# ---------------------------------------------------------------------------

@dataclass
class FinalBranchFit:
    """Result of final branch fit on all outer-training data."""
    branch_name: str
    selected_params: Dict
    alpha: Optional[np.ndarray] = None  # for FP branch
    scaler: Optional[StandardScaler] = None
    y_mean: float = 0.0
    y_std: float = 1.0
    test_pred: Optional[np.ndarray] = None
    selection_info: Dict = field(default_factory=dict)


def reselect_and_fit_final(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: AblationCondition,
    roi_prior: np.ndarray,
    seed: int,
    outer_fold: int,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_final_cv: int = 3,
    n_rois: int = N_ROI,
) -> Tuple[FinalBranchFit, FinalBranchFit, FinalBranchFit]:
    """Reselect final parameters on all T using 3-fold CV, then fit on all T.

    Returns (fc_result, sc_result, fp_result).
    """
    cache = build_condition_cache(roi_prior, condition, n_rois)
    inner_folds = make_inner_selection_folds(train_idx, seed, outer_fold, 99, n_final_cv)

    # --- FC branch final reselection ---
    fc_alpha, fc_info = _select_alpha_ridge(X_fc, y, train_idx, ridge_grid, inner_folds)

    # Fit on all T, predict test
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_fc_test_z = scaler_fc.transform(X_fc[test_idx])

    fc_model = Ridge(alpha=fc_alpha, fit_intercept=True)
    fc_model.fit(X_fc_train_z, y[train_idx])
    fc_test = fc_model.predict(X_fc_test_z)

    fc_result = FinalBranchFit(
        branch_name="FC",
        selected_params={"alpha": fc_alpha},
        test_pred=fc_test,
        selection_info=fc_info,
    )

    # --- SC branch final reselection ---
    sc_alpha, sc_info = _select_alpha_ridge(X_sc, y, train_idx, ridge_grid, inner_folds)

    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])

    sc_model = Ridge(alpha=sc_alpha, fit_intercept=True)
    sc_model.fit(X_sc_train_z, y[train_idx])
    sc_test = sc_model.predict(X_sc_test_z)

    sc_result = FinalBranchFit(
        branch_name="SC",
        selected_params={"alpha": sc_alpha},
        test_pred=sc_test,
        selection_info=sc_info,
    )

    # --- FP branch final reselection ---
    fp_params, fp_info = _select_fc_only_params(
        X_fc, y, train_idx, cache, condition, ridge_grid, inner_folds,
    )

    # Fit on all T
    scaler_fp = StandardScaler()
    X_fc_fp_train_z = scaler_fp.fit_transform(X_fc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean) / y_std

    alpha_fp, _ = _solve_msancr_kernel(
        X_fc_fp_train_z, np.zeros_like(X_fc_fp_train_z), y_train_z, cache,
        fp_params["lambda_fc"], 1.0, fp_params["lambda_l"], fc_only=True,
    )

    # Predict test
    X_fc_fp_test_z = scaler_fp.transform(X_fc[test_idx])
    fp_test_z = _predict_msancr(
        X_fc_fp_test_z, np.zeros_like(X_fc_fp_test_z),
        X_fc_fp_train_z, np.zeros_like(X_fc_fp_train_z),
        alpha_fp, cache, fp_params["lambda_fc"], 1.0, fp_params["lambda_l"], fc_only=True,
    )
    fp_test = fp_test_z * y_std + y_mean

    fp_result = FinalBranchFit(
        branch_name="FP",
        selected_params=fp_params,
        alpha=alpha_fp,
        scaler=scaler_fp,
        y_mean=y_mean,
        y_std=y_std,
        test_pred=fp_test,
        selection_info=fp_info,
    )

    return fc_result, sc_result, fp_result


# ---------------------------------------------------------------------------
# Fusion weight search
# ---------------------------------------------------------------------------

def search_fusion_weights(
    y_true: np.ndarray,
    oof_preds: Dict[str, np.ndarray],
    branch_names: List[str],
) -> Tuple[Dict[str, float], float]:
    """Search 2-branch convex fusion weights on OOF predictions.

    Selection: Pearson first, RMSE tiebreak, MAE tiebreak, larger SC weight on exact tie.
    """
    assert len(branch_names) == 2
    best_key = None
    best_weights = None

    for w0 in WEIGHT_GRID:
        w1 = round(1.0 - w0, 4)
        combined = w0 * oof_preds[branch_names[0]] + w1 * oof_preds[branch_names[1]]
        m = prediction_metrics(y_true, combined)
        # Sort key: Pearson desc, RMSE asc, MAE asc, SC weight desc
        sc_w = w1 if branch_names[1] == "SC" else w0
        key = (-m["pearson"], m["rmse"], m["mae"], -sc_w)
        if best_key is None or key < best_key:
            best_key = key
            best_weights = {branch_names[0]: w0, branch_names[1]: w1}

    combined = best_weights[branch_names[0]] * oof_preds[branch_names[0]] + best_weights[branch_names[1]] * oof_preds[branch_names[1]]
    selected_pearson = float(pearsonr(y_true, combined).statistic) if len(y_true) > 1 else 0.0
    return best_weights, selected_pearson


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _holm_adjust(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni correction.

    Stable sort raw p-values, multiply by remaining family sizes,
    take forward cumulative maximum, cap at 1, restore original order.
    """
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted_val = min(pval * (n - rank), 1.0)
        running_max = max(running_max, adjusted_val)
        adjusted[orig_idx] = running_max
    return adjusted


# ---------------------------------------------------------------------------
# Main ablation evaluation for one outer split
# ---------------------------------------------------------------------------

@dataclass
class AblationSplitResult:
    """Result for one ablation condition on one outer split."""
    seed: int
    outer_fold: int
    condition_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    # OOF predictions
    fc_oof: np.ndarray
    sc_oof: np.ndarray
    fp_oof: np.ndarray
    # Fusion
    fusion_weights: Dict[str, float]
    # Test predictions
    fc_test_pred: np.ndarray
    sc_test_pred: np.ndarray
    fp_test_pred: np.ndarray
    fused_test_pred: np.ndarray
    equal_weight_pred: np.ndarray
    # Metrics
    fc_metrics: Dict
    sc_metrics: Dict
    fp_metrics: Dict
    fused_metrics: Dict
    equal_weight_metrics: Dict
    # Selected parameters
    fc_selected_alpha: float
    sc_selected_alpha: float
    fp_selected_params: Dict
    # Final branch fits
    fc_final: FinalBranchFit
    sc_final: FinalBranchFit
    fp_final: FinalBranchFit
    # Lineage
    selection_scores: Dict = field(default_factory=dict)


def evaluate_ablation_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: AblationCondition,
    roi_prior: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    n_final_cv: int = 3,
    n_rois: int = N_ROI,
) -> AblationSplitResult:
    """Evaluate one ablation condition on one outer split.

    Steps:
      1. Cross-fitted OOF for stacking
      2. Fusion weight selection on OOF
      3. Final parameter reselection on all T
      4. Predict test
    """
    y_test = y[test_idx]

    # Step 1: Cross-fitted OOF
    oof = generate_crossfit_oof(
        X_fc, X_sc, y, train_idx, condition, roi_prior,
        seed, outer_fold, ridge_grid, n_fusion_folds, n_inner, n_rois,
    )

    # Step 2: Fusion weight selection on OOF
    # R0 fuses FC+SC (no-prior FC); R1/R2/R3 fuse FP+SC (prior-aware FC)
    y_train = y[train_idx]
    if condition.use_anisotropy or condition.use_network:
        # R1/R2/R3: use FP (prior-aware) + SC fusion
        fusion_weights, fusion_pearson = search_fusion_weights(
            y_train, {"FP": oof.fp_oof, "SC": oof.sc_oof}, ["FP", "SC"],
        )
    else:
        # R0: use FC (no-prior) + SC fusion
        fusion_weights, fusion_pearson = search_fusion_weights(
            y_train, {"FC": oof.fc_oof, "SC": oof.sc_oof}, ["FC", "SC"],
        )

    # Equal-weight baseline
    if condition.use_anisotropy or condition.use_network:
        ew_pred = 0.5 * oof.fp_oof + 0.5 * oof.sc_oof
    else:
        ew_pred = 0.5 * oof.fc_oof + 0.5 * oof.sc_oof
    ew_pearson = float(pearsonr(y_train, ew_pred).statistic) if len(y_train) > 1 else 0.0

    # Step 3: Final reselection and fit
    fc_final, sc_final, fp_final = reselect_and_fit_final(
        X_fc, X_sc, y, train_idx, test_idx, condition, roi_prior,
        seed, outer_fold, ridge_grid, n_final_cv, n_rois,
    )

    # Step 4: Fused test prediction
    if condition.use_anisotropy or condition.use_network:
        # R1/R2/R3: FP + SC
        w_fp = fusion_weights.get("FP", 0.5)
        w_sc = fusion_weights.get("SC", 0.5)
        fused_test = w_fp * fp_final.test_pred + w_sc * sc_final.test_pred
    else:
        # R0: FC + SC
        w_fc = fusion_weights.get("FC", 0.5)
        w_sc = fusion_weights.get("SC", 0.5)
        fused_test = w_fc * fc_final.test_pred + w_sc * sc_final.test_pred

    # Equal-weight test prediction
    if condition.use_anisotropy or condition.use_network:
        ew_test = 0.5 * fp_final.test_pred + 0.5 * sc_final.test_pred
    else:
        ew_test = 0.5 * fc_final.test_pred + 0.5 * sc_final.test_pred

    # Compute metrics
    fc_m = prediction_metrics(y_test, fc_final.test_pred)
    sc_m = prediction_metrics(y_test, sc_final.test_pred)
    fp_m = prediction_metrics(y_test, fp_final.test_pred)
    fused_m = prediction_metrics(y_test, fused_test)
    ew_m = prediction_metrics(y_test, ew_test)

    return AblationSplitResult(
        seed=seed,
        outer_fold=outer_fold,
        condition_id=condition.id,
        train_idx=train_idx,
        test_idx=test_idx,
        fc_oof=oof.fc_oof,
        sc_oof=oof.sc_oof,
        fp_oof=oof.fp_oof,
        fusion_weights=fusion_weights,
        fc_test_pred=fc_final.test_pred,
        sc_test_pred=sc_final.test_pred,
        fp_test_pred=fp_final.test_pred,
        fused_test_pred=fused_test,
        equal_weight_pred=ew_test,
        fc_metrics=fc_m,
        sc_metrics=sc_m,
        fp_metrics=fp_m,
        fused_metrics=fused_m,
        equal_weight_metrics=ew_m,
        fc_selected_alpha=fc_final.selected_params["alpha"],
        sc_selected_alpha=sc_final.selected_params["alpha"],
        fp_selected_params=fp_final.selected_params,
        fc_final=fc_final,
        sc_final=sc_final,
        fp_final=fp_final,
        selection_scores={
            "fc": oof.fc_selection_info,
            "sc": oof.sc_selection_info,
            "fp": oof.fp_selection_info,
        },
    )


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

@dataclass
class AblationTaskResult:
    """Complete results for one task across all conditions and splits."""
    task_name: str
    splits: List[AblationSplitResult]
    condition_ids: List[str]
    seeds: List[int]
    n_outer_folds: int


def run_ablation_experiment(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    task_name: str,
    roi_prior: Optional[np.ndarray] = None,
    seeds: Sequence[int] = tuple(range(10)),
    n_outer_folds: int = 5,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    n_final_cv: int = 3,
    n_rois: int = N_ROI,
    output_dir: Optional[str] = None,
    condition_ids: Sequence[str] = ("R0", "R1", "R2", "R3"),
    smoke_mode: bool = False,
) -> AblationTaskResult:
    """Run complete ablation experiment for one task.

    Parameters
    ----------
    roi_prior : np.ndarray, optional
        ROI-level prior vector (length n_rois).  Required for conditions that
        use anisotropy (R1, R3).  If *None*, falls back to a uniform prior
        (which collapses anisotropic conditions to identity D).

    Returns
    -------
    AblationTaskResult with all splits.
    """
    if smoke_mode:
        seeds = seeds[:1]

    outer_splits = make_outer_splits(len(y), seeds, n_outer_folds)
    all_results = []

    total = len(condition_ids) * len(outer_splits)
    done = 0

    for cond_id in condition_ids:
        condition = CONDITIONS[cond_id]
        logger.info(f"Condition {cond_id}: {condition.name}")

        for seed, fold, train_idx, test_idx in outer_splits:
            done += 1
            logger.info(
                f"  [{done}/{total}] seed={seed} fold={fold} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}"
            )

            # Use the provided roi_prior, or fall back to uniform placeholder
            # (R0-only baseline does not use the prior, so placeholder is fine there)
            _prior = roi_prior if roi_prior is not None else np.ones(n_rois) / n_rois

            t0 = time.time()
            result = evaluate_ablation_split(
                X_fc, X_sc, y, seed, fold, train_idx, test_idx,
                condition, _prior,
                ridge_grid, n_fusion_folds, n_inner, n_final_cv, n_rois,
            )
            elapsed = time.time() - t0
            logger.info(
                f"    FC r={result.fc_metrics['pearson']:.4f} "
                f"SC r={result.sc_metrics['pearson']:.4f} "
                f"FP r={result.fp_metrics['pearson']:.4f} "
                f"Fused r={result.fused_metrics['pearson']:.4f} "
                f"({elapsed:.1f}s)"
            )
            all_results.append(result)

    return AblationTaskResult(
        task_name=task_name,
        splits=all_results,
        condition_ids=list(condition_ids),
        seeds=list(seeds),
        n_outer_folds=n_outer_folds,
    )


# ---------------------------------------------------------------------------
# Legacy-grid Ridge reference
# ---------------------------------------------------------------------------

def evaluate_legacy_ridge_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    n_final_cv: int = 3,
) -> AblationSplitResult:
    """Legacy-grid ordinary FC Ridge + SC Ridge reference with corrected CV."""
    # Use R0 condition (no prior, no network)
    condition = CONDITIONS["R0"]
    roi_prior = np.ones(N_ROI) / N_ROI  # Placeholder; R0 ignores prior
    return evaluate_ablation_split(
        X_fc, X_sc, y, seed, outer_fold, train_idx, test_idx,
        condition, roi_prior, ridge_grid, n_fusion_folds, n_inner, n_final_cv,
    )


# ---------------------------------------------------------------------------
# Fixed prior-swap diagnostics
# ---------------------------------------------------------------------------

def evaluate_prior_swap_fixed(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    matched_result: AblationSplitResult,
    control_cache: _MSANCRCache,
    control_prior_type: str,
    n_rois: int = N_ROI,
) -> Dict:
    """Refit FC branch on T with control prior, keep R3's final SC and fusion weights.

    This is a matched-setting fixed prior-swap diagnostic.
    """
    y_test = y[test_idx]

    # Use R3's final SC and fusion weights
    sc_final = matched_result.sc_final
    w_fp = matched_result.fusion_weights["FP"]
    w_sc = matched_result.fusion_weights["SC"]

    # Refit FC branch with control prior
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_fc_test_z = scaler_fc.transform(X_fc[test_idx])

    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean) / y_std

    fp_params = matched_result.fp_selected_params
    alpha_ctrl, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z), y_train_z, control_cache,
        fp_params["lambda_fc"], 1.0, fp_params["lambda_l"], fc_only=True,
    )
    ctrl_test_z = _predict_msancr(
        X_fc_test_z, np.zeros_like(X_fc_test_z),
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha_ctrl, control_cache, fp_params["lambda_fc"], 1.0, fp_params["lambda_l"], fc_only=True,
    )
    ctrl_test = ctrl_test_z * y_std + y_mean

    # SC prediction
    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
    sc_model = Ridge(alpha=sc_final.selected_params["alpha"], fit_intercept=True)
    sc_model.fit(X_sc_train_z, y[train_idx])
    sc_test = sc_model.predict(X_sc_test_z)

    # Fused prediction with matched weights
    fused = w_fp * ctrl_test + w_sc * sc_test

    m = prediction_metrics(y_test, fused)
    fp_only_m = prediction_metrics(y_test, ctrl_test)

    return {
        "prior_type": control_prior_type,
        "fused_pearson": m["pearson"],
        "fused_rmse": m["rmse"],
        "fused_mae": m["mae"],
        "fp_only_pearson": fp_only_m["pearson"],
        "fp_only_rmse": fp_only_m["rmse"],
        "fp_only_mae": fp_only_m["mae"],
        "w_fp": w_fp,
        "w_sc": w_sc,
        "fp_params": fp_params,
    }


# ---------------------------------------------------------------------------
# Statistics and analysis
# ---------------------------------------------------------------------------

def compute_seed_metrics(result: AblationTaskResult) -> pd.DataFrame:
    """Compute seed-level metrics from split results."""
    rows = []
    for cond_id in result.condition_ids:
        cond_splits = [s for s in result.splits if s.condition_id == cond_id]
        for seed in result.seeds:
            seed_splits = [s for s in cond_splits if s.seed == seed]
            if not seed_splits:
                continue
            for metric_name in ["fc", "sc", "fp", "fused", "equal_weight"]:
                key = f"{metric_name}_metrics"
                vals = [getattr(s, key)["pearson"] for s in seed_splits]
                rmse_vals = [getattr(s, key)["rmse"] for s in seed_splits]
                mae_vals = [getattr(s, key)["mae"] for s in seed_splits]
                rows.append({
                    "task": result.task_name,
                    "condition": cond_id,
                    "seed": seed,
                    "model": metric_name,
                    "mean_pearson": float(np.mean(vals)),
                    "mean_rmse": float(np.mean(rmse_vals)),
                    "mean_mae": float(np.mean(mae_vals)),
                    "n_folds": len(seed_splits),
                })
    return pd.DataFrame(rows)


def compute_component_summary(result: AblationTaskResult) -> pd.DataFrame:
    """Compute condition-level summary across seeds."""
    seed_df = compute_seed_metrics(result)
    rows = []
    for cond_id in result.condition_ids:
        for model in ["fc", "sc", "fp", "fused", "equal_weight"]:
            subset = seed_df[(seed_df["condition"] == cond_id) & (seed_df["model"] == model)]
            if subset.empty:
                continue
            rows.append({
                "task": result.task_name,
                "condition": cond_id,
                "model": model,
                "mean_pearson": float(subset["mean_pearson"].mean()),
                "std_pearson": float(subset["mean_pearson"].std(ddof=1)) if len(subset) > 1 else 0.0,
                "mean_rmse": float(subset["mean_rmse"].mean()),
                "mean_mae": float(subset["mean_mae"].mean()),
                "n_seeds": len(subset),
            })
    return pd.DataFrame(rows)
