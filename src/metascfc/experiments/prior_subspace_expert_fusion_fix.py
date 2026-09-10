"""Prior-Selected Subspace NCR Expert Fusion (PS-NCR-EF) — CORRECTED.

Phase 2D: Uses the LLM prior to select a low-dimensional expert subspace,
fits Ridge and NCR experts on that subspace, and fuses the expert with the
untouched strong same-solver no-prior baseline.

Corrections over prior_subspace_expert_fusion.py:
  F3: Inner-CV preprocessing leakage — StandardScaler fit per inner fold.
  F4: Independent FC/SC mask selection — separate mask per modality.
  F5: Broken primal reconstruction validation — expert vs final split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2
IU = np.triu_indices(N_ROI, k=1)

K_EDGE_GRID = [100, 300, 600, 1200]
M_ROI_GRID = [5, 10, 15]
RIDGE_EXPERT_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LAPLACIAN_RATIO_GRID = [0.0, 0.1, 0.3, 1.0]
WEIGHT_STEP = 0.05
WEIGHT_GRID = [round(i * WEIGHT_STEP, 4) for i in range(0, int(1.0 / WEIGHT_STEP) + 1)]


# ---------------------------------------------------------------------------
# Edge mask construction (unchanged)
# ---------------------------------------------------------------------------

def build_edge_product_prior(
    roi_prior: np.ndarray,
) -> np.ndarray:
    """Compute edge-level product prior q_ij = p_i * p_j.

    Parameters
    ----------
    roi_prior : (n_rois,) ROI-level prior scores

    Returns
    -------
    edge_prior : (n_edges,) upper-triangle edge prior scores
    """
    iu = np.triu_indices(len(roi_prior), k=1)
    return (roi_prior[iu[0]] * roi_prior[iu[1]]).astype(np.float64)


def direct_topk_mask(
    edge_prior: np.ndarray,
    k: int,
) -> np.ndarray:
    """Select top-K edges by product prior.

    Returns boolean mask of length n_edges.
    """
    mask = np.zeros(len(edge_prior), dtype=bool)
    top_k_idx = np.argsort(edge_prior)[-k:]
    mask[top_k_idx] = True
    return mask


def roi_incident_mask(
    roi_prior: np.ndarray,
    m: int,
    n_rois: int = N_ROI,
) -> Tuple[np.ndarray, int]:
    """Select edges incident to top-M ROIs.

    Returns (boolean mask, actual_edge_count).
    """
    top_m_rois = np.argsort(roi_prior)[-m:]
    roi_set = set(top_m_rois.tolist())

    iu = np.triu_indices(n_rois, k=1)
    n_edges = len(iu[0])
    mask = np.zeros(n_edges, dtype=bool)

    for idx in range(n_edges):
        i, j = iu[0][idx], iu[1][idx]
        if i in roi_set or j in roi_set:
            mask[idx] = True

    return mask, int(mask.sum())


def get_edge_count_for_roi_mask(
    m: int,
    n_rois: int = N_ROI,
) -> int:
    """Compute the exact edge count for top-M ROI incident mask."""
    _, count = roi_incident_mask(np.zeros(n_rois), m, n_rois)
    return count


# ---------------------------------------------------------------------------
# Prior controls (unchanged)
# ---------------------------------------------------------------------------

def build_control_prior(
    prior_type: str,
    task_prior: np.ndarray,
    other_task_prior: np.ndarray,
    seed: int,
    n_rois: int = N_ROI,
) -> np.ndarray:
    """Build a control prior for ablation studies.

    Parameters
    ----------
    prior_type : one of 'matched', 'cross_task', 'shuffled', 'random'
    task_prior : the matched prior for this task
    other_task_prior : the prior for the other task (for cross_task)
    seed : for reproducibility of shuffled/random
    """
    if prior_type == "matched":
        return task_prior.copy()
    elif prior_type == "cross_task":
        return other_task_prior.copy()
    elif prior_type == "shuffled":
        rng = np.random.RandomState(seed + 7777)
        perm = rng.permutation(n_rois)
        return task_prior[perm].copy()
    elif prior_type == "random":
        rng = np.random.RandomState(seed + 9999)
        return rng.uniform(0, 1, size=n_rois).astype(np.float64)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")


# ---------------------------------------------------------------------------
# Expert fitting
# ---------------------------------------------------------------------------

@dataclass
class ExpertResult:
    """Result of fitting a single expert (FC or SC)."""
    mask: np.ndarray          # boolean mask of selected edges
    n_selected: int
    mask_family: str          # 'direct_topk' or 'roi_incident'
    mask_size: int            # K or M value
    lambda_r: float           # selected Ridge penalty
    laplacian_ratio: float    # selected Laplacian ratio (0 = pure Ridge)
    expert_type: str          # 'ridge' or 'ncr'
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    y_mean: float
    y_std: float
    beta_standardized: np.ndarray   # in standardized feature space
    beta_original: np.ndarray       # in original feature units
    oof_pred: Optional[np.ndarray] = None
    test_pred: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# F3 FIX: Fold-local fitting primitive with leakage-free scaler
# ---------------------------------------------------------------------------

def fit_expert_candidate_on_split(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    lambda_R: float,
    laplacian_ratio: float = 0.0,
    edge_laplacian: Optional[EdgeLaplacian] = None,
) -> np.ndarray:
    """Fit a single Ridge/NCR candidate on inner-train, predict inner-val.

    The StandardScaler is fit on inner_train only (F3 fix).
    No inner CV is performed here — this is a fold-local primitive.

    Parameters
    ----------
    X : (n_subjects, n_edges) full feature matrix
    y : (n_subjects,) target
    mask : (n_edges,) boolean mask
    inner_train_idx : global indices of inner-train subjects
    inner_val_idx : global indices of inner-val subjects
    lambda_R : Ridge penalty
    laplacian_ratio : ratio for NCR (0 = pure Ridge)
    edge_laplacian : EdgeLaplacian for NCR, or None

    Returns
    -------
    val_pred : (len(inner_val_idx),) predictions on inner-val
    """
    X_sub = X[:, mask]
    n_sub = int(mask.sum())

    # F3 FIX: Fit scaler on inner_train ONLY
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(X_sub[inner_train_idx])
    X_val_raw = scaler.transform(X_sub[inner_val_idx])

    y_mean = float(y[inner_train_idx].mean())
    y_std = max(float(y[inner_train_idx].std()), 1e-8)
    y_tr = (y[inner_train_idx] - y_mean) / y_std

    lambda_L = laplacian_ratio * lambda_R

    # Build restricted Laplacian if NCR
    sub_edge_laplacian = None
    if edge_laplacian is not None and lambda_L > 0:
        sub_edge_laplacian = _build_sub_laplacian(mask, edge_laplacian, n_sub)

    if sub_edge_laplacian is None or lambda_L == 0:
        # Pure Ridge path
        X_tr = X_tr_raw
        X_val = X_val_raw
        model = Ridge(alpha=lambda_R, fit_intercept=False)
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val) * y_std + y_mean
    else:
        # NCR path: pad to 2*n_sub
        X_tr = np.hstack([X_tr_raw, np.zeros_like(X_tr_raw)])
        X_val = np.hstack([X_val_raw, np.zeros_like(X_val_raw)])
        ncr = NetworkConstrainedRidge(
            alpha1=lambda_R, alpha2=lambda_L,
            edge_laplacian=sub_edge_laplacian,
            standardize=False,
        )
        ncr.fit(X_tr, y_tr)
        val_pred = ncr.predict(X_val) * y_std + y_mean

    return val_pred


def _build_sub_laplacian(
    mask: np.ndarray,
    edge_laplacian: EdgeLaplacian,
    n_sub: int,
) -> Optional[EdgeLaplacian]:
    """Build restricted Laplacian for a masked subspace."""
    global_active = edge_laplacian.active_indices
    mask_global_idx = np.where(mask)[0]
    local_active_map = {}
    for local_i, global_idx in enumerate(mask_global_idx):
        local_active_map[global_idx] = local_i

    restricted_active = []
    for g_idx in global_active:
        if g_idx in local_active_map:
            restricted_active.append(local_active_map[g_idx])

    if len(restricted_active) == 0:
        return None

    restricted_active = np.array(restricted_active, dtype=np.int64)
    restricted_laplacian = edge_laplacian.active_laplacian[
        :len(global_active), :len(global_active)
    ]
    sub_laplacian = np.zeros((n_sub, n_sub), dtype=np.float64)
    for i_local, i_global in enumerate(global_active):
        if i_global in local_active_map:
            for j_local, j_global in enumerate(global_active):
                if j_global in local_active_map:
                    li = local_active_map[i_global]
                    lj = local_active_map[j_global]
                    sub_laplacian[li, lj] = restricted_laplacian[i_local, j_local]

    active_sub_laplacian = sub_laplacian[np.ix_(restricted_active, restricted_active)]
    return EdgeLaplacian(
        active_indices=restricted_active,
        active_laplacian=active_sub_laplacian,
        n_edges=n_sub,
        n_rois=N_ROI,
        top_k=edge_laplacian.top_k,
        weighting=edge_laplacian.weighting,
        couple_modalities=False,
    )


# ---------------------------------------------------------------------------
# F3+F4 FIX: Inner CV evaluation with per-fold scaler
# ---------------------------------------------------------------------------

def _evaluate_mask_inner_cv_fixed(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    analysis_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    n_inner: int = 3,
    lambda_grid: List[float] = RIDGE_EXPERT_GRID,
) -> float:
    """Evaluate a mask using inner CV Ridge with per-fold scaler (F3 fix).

    Each inner fold fits its own StandardScaler on the inner-train fold only.
    """
    n_analysis = len(analysis_idx)

    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 42)
    perm = rng.permutation(n_analysis)
    fold_sizes = np.full(n_inner, n_analysis // n_inner)
    fold_sizes[:n_analysis % n_inner] += 1

    best_score = -np.inf
    for lam in lambda_grid:
        oof = np.full(n_analysis, np.nan)
        current = 0
        for k in range(n_inner):
            start, stop = current, current + fold_sizes[k]
            v_local = perm[start:stop]
            a_local = np.concatenate([perm[:start], perm[stop:]])

            # F3 FIX: fit scaler on inner-train only
            X_sub = X[analysis_idx, :][:, mask]
            scaler = StandardScaler()
            X_a = scaler.fit_transform(X_sub[a_local])
            X_v = scaler.transform(X_sub[v_local])

            y_mean = float(y[analysis_idx[a_local]].mean())
            y_std = max(float(y[analysis_idx[a_local]].std()), 1e-8)
            y_a = (y[analysis_idx[a_local]] - y_mean) / y_std

            model = Ridge(alpha=lam, fit_intercept=False)
            model.fit(X_a, y_a)
            pred = model.predict(X_v) * y_std + y_mean
            oof[v_local] = pred
            current = stop

        oof_orig = oof
        valid = np.isfinite(oof_orig)
        if valid.sum() < 10:
            continue
        r, _ = pearsonr(oof_orig[valid], y[analysis_idx][valid])
        best_score = max(best_score, r)

    return best_score


# ---------------------------------------------------------------------------
# F4 FIX: Independent mask selection per modality
# ---------------------------------------------------------------------------

def _select_best_mask_for_modality(
    X_modality: np.ndarray,
    y: np.ndarray,
    edge_prior: np.ndarray,
    roi_prior: np.ndarray,
    analysis_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    n_inner: int = 3,
) -> Tuple[np.ndarray, str, int]:
    """Select the best mask family and size for a single modality (F4 fix).

    Evaluates masks independently on X_modality using per-fold scaler inner CV.

    Returns
    -------
    best_mask : boolean array
    best_family : 'direct_topk' or 'roi_incident'
    best_k_or_m : the K or M value
    """
    candidates = []

    for k in K_EDGE_GRID:
        mask = direct_topk_mask(edge_prior, k)
        score = _evaluate_mask_inner_cv_fixed(
            X_modality, y, mask, analysis_idx, seed, outer_fold, n_inner,
        )
        candidates.append((score, mask, "direct_topk", k))

    for m in M_ROI_GRID:
        mask, count = roi_incident_mask(roi_prior, m)
        score = _evaluate_mask_inner_cv_fixed(
            X_modality, y, mask, analysis_idx, seed, outer_fold, n_inner,
        )
        candidates.append((score, mask, "roi_incident", m))

    # Sort: highest Pearson, then larger mask as tiebreak
    candidates.sort(key=lambda x: (-x[0], -x[3]))
    _, best_mask, best_family, best_k_or_m = candidates[0]

    return best_mask, best_family, best_k_or_m


# ---------------------------------------------------------------------------
# F3 FIX: Fixed Ridge expert with leakage-free scaler
# ---------------------------------------------------------------------------

def fit_expert_ridge_fixed(
    X_modality: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lambda_grid: List[float] = RIDGE_EXPERT_GRID,
    seed: int = 0,
    outer_fold: int = 0,
    n_inner: int = 3,
) -> ExpertResult:
    """Fit a restricted-Ridge expert on the selected subspace.

    F3 FIX: StandardScaler is fit per inner fold during hyperparameter
    selection, then a fresh scaler is fit on the complete training scope
    for the final fit.
    """
    X_sub = X_modality[:, mask]

    # Inner CV for lambda selection — each fold gets its own scaler
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 42)
    n_train = len(train_idx)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(n_inner, n_train // n_inner)
    fold_sizes[:n_train % n_inner] += 1

    best_lambda = lambda_grid[0]
    best_score = -np.inf

    for lam in lambda_grid:
        oof = np.full(n_train, np.nan)
        current = 0
        for k in range(n_inner):
            start, stop = current, current + fold_sizes[k]
            v_local = perm[start:stop]
            a_local = np.concatenate([perm[:start], perm[stop:]])

            # F3 FIX: fit scaler on inner-train only
            scaler_inner = StandardScaler()
            X_a = scaler_inner.fit_transform(X_sub[train_idx[a_local]])
            X_v = scaler_inner.transform(X_sub[train_idx[v_local]])

            y_mean_inner = float(y[train_idx[a_local]].mean())
            y_std_inner = max(float(y[train_idx[a_local]].std()), 1e-8)
            y_a = (y[train_idx[a_local]] - y_mean_inner) / y_std_inner

            model = Ridge(alpha=lam, fit_intercept=False)
            model.fit(X_a, y_a)
            oof[v_local] = model.predict(X_v) * y_std_inner + y_mean_inner
            current = stop

        mask_valid = np.isfinite(oof)
        if mask_valid.sum() < 10:
            continue
        r, _ = pearsonr(oof[mask_valid], y[train_idx][mask_valid])
        if r > best_score + 1e-14:
            best_score = r
            best_lambda = lam

    # F3 FIX: Final fit — fit fresh scaler on complete training scope
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_sub[train_idx])
    X_te = scaler.transform(X_sub[test_idx])

    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_tr = (y[train_idx] - y_mean) / y_std

    model = Ridge(alpha=best_lambda, fit_intercept=False)
    model.fit(X_tr, y_tr)

    beta_std = np.zeros(mask.sum(), dtype=np.float64)
    beta_std[:] = model.coef_

    beta_orig = beta_std * y_std / np.maximum(scaler.scale_, 1e-8)

    test_pred = model.predict(X_te) * y_std + y_mean

    return ExpertResult(
        mask=mask.copy(),
        n_selected=int(mask.sum()),
        mask_family="",
        mask_size=0,
        lambda_r=best_lambda,
        laplacian_ratio=0.0,
        expert_type="ridge",
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        y_mean=y_mean,
        y_std=y_std,
        beta_standardized=beta_std,
        beta_original=beta_orig,
        test_pred=test_pred,
    )


# ---------------------------------------------------------------------------
# F3 FIX: Fixed NCR expert with leakage-free scaler
# ---------------------------------------------------------------------------

def fit_expert_ncr_fixed(
    X_modality: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    edge_laplacian: EdgeLaplacian,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lambda_grid: List[float] = RIDGE_EXPERT_GRID,
    ratio_grid: List[float] = LAPLACIAN_RATIO_GRID,
    seed: int = 0,
    outer_fold: int = 0,
    n_inner: int = 3,
) -> ExpertResult:
    """Fit a prior-selected NCR expert on the selected subspace.

    F3 FIX: StandardScaler is fit per inner fold during hyperparameter
    selection, then a fresh scaler is fit on the complete training scope
    for the final fit.
    """
    X_sub = X_modality[:, mask]
    n_sub = int(mask.sum())

    # Build restricted Laplacian once
    sub_edge_laplacian = _build_sub_laplacian(mask, edge_laplacian, n_sub)

    # Inner CV for (lambda_r, ratio) selection — each fold gets its own scaler
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 42)
    n_train = len(train_idx)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(n_inner, n_train // n_inner)
    fold_sizes[:n_train % n_inner] += 1

    best_lambda = lambda_grid[0]
    best_ratio = 0.0
    best_score = -np.inf

    for lam in lambda_grid:
        for ratio in ratio_grid:
            lambda_l = ratio * lam

            oof = np.full(n_train, np.nan)
            current = 0
            for k in range(n_inner):
                start, stop = current, current + fold_sizes[k]
                v_local = perm[start:stop]
                a_local = np.concatenate([perm[:start], perm[stop:]])

                # F3 FIX: fit scaler on inner-train only
                scaler_inner = StandardScaler()
                X_a_raw = scaler_inner.fit_transform(X_sub[train_idx[a_local]])
                X_v_raw = scaler_inner.transform(X_sub[train_idx[v_local]])

                y_mean_inner = float(y[train_idx[a_local]].mean())
                y_std_inner = max(float(y[train_idx[a_local]].std()), 1e-8)
                y_a = (y[train_idx[a_local]] - y_mean_inner) / y_std_inner

                if sub_edge_laplacian is None or lambda_l == 0:
                    model = Ridge(alpha=lam, fit_intercept=False)
                    model.fit(X_a_raw, y_a)
                    oof[v_local] = model.predict(X_v_raw) * y_std_inner + y_mean_inner
                else:
                    X_a = np.hstack([X_a_raw, np.zeros_like(X_a_raw)])
                    X_v = np.hstack([X_v_raw, np.zeros_like(X_v_raw)])
                    ncr = NetworkConstrainedRidge(
                        alpha1=lam, alpha2=lambda_l,
                        edge_laplacian=sub_edge_laplacian,
                        standardize=False,
                    )
                    ncr.fit(X_a, y_a)
                    oof[v_local] = ncr.predict(X_v) * y_std_inner + y_mean_inner
                current = stop

            mask_valid = np.isfinite(oof)
            if mask_valid.sum() < 10:
                continue
            r, _ = pearsonr(oof[mask_valid], y[train_idx][mask_valid])

            if (r > best_score + 1e-14 or
                (abs(r - best_score) < 1e-14 and ratio < best_ratio) or
                (abs(r - best_score) < 1e-14 and ratio == best_ratio and lam < best_lambda)):
                best_score = r
                best_lambda = lam
                best_ratio = ratio

    # F3 FIX: Final fit — fit fresh scaler on complete training scope
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(X_sub[train_idx])
    X_te_raw = scaler.transform(X_sub[test_idx])

    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_tr = (y[train_idx] - y_mean) / y_std

    lambda_l = best_ratio * best_lambda

    if sub_edge_laplacian is None or lambda_l == 0:
        model = Ridge(alpha=best_lambda, fit_intercept=False)
        model.fit(X_tr_raw, y_tr)
        beta_full = model.coef_.copy()
        beta_std = beta_full[:n_sub].copy()
        test_pred = model.predict(X_te_raw) * y_std + y_mean
    else:
        X_tr = np.hstack([X_tr_raw, np.zeros_like(X_tr_raw)])
        X_te = np.hstack([X_te_raw, np.zeros_like(X_te_raw)])
        ncr = NetworkConstrainedRidge(
            alpha1=best_lambda, alpha2=lambda_l,
            edge_laplacian=sub_edge_laplacian,
            standardize=False,
        )
        ncr.fit(X_tr, y_tr)
        beta_std = ncr.beta()[:n_sub].copy()
        test_pred = ncr.predict(X_te) * y_std + y_mean

    beta_orig = beta_std * y_std / np.maximum(scaler.scale_, 1e-8)

    return ExpertResult(
        mask=mask.copy(),
        n_selected=int(mask.sum()),
        mask_family="",
        mask_size=0,
        lambda_r=best_lambda,
        laplacian_ratio=best_ratio,
        expert_type="ncr",
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        y_mean=y_mean,
        y_std=y_std,
        beta_standardized=beta_std,
        beta_original=beta_orig,
        test_pred=test_pred,
    )


# ---------------------------------------------------------------------------
# Cross-fitted expert OOF
# ---------------------------------------------------------------------------

@dataclass
class ExpertOOFResult:
    """OOF predictions for FC and SC experts."""
    fc_oof: np.ndarray       # (n_train,) FC expert OOF predictions
    sc_oof: np.ndarray       # (n_train,) SC expert OOF predictions
    fc_expert: Optional[ExpertResult] = None
    sc_expert: Optional[ExpertResult] = None
    fc_sc_weights: Optional[Dict[str, float]] = None
    expert_fused_oof: Optional[np.ndarray] = None
    selected_mask_family_fc: str = ""
    selected_mask_size_fc: int = 0
    selected_mask_family_sc: str = ""
    selected_mask_size_sc: int = 0


def generate_expert_crossfit_oof_fixed(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    roi_prior: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    expert_type: str = "ncr",
    edge_laplacian: Optional[EdgeLaplacian] = None,
) -> ExpertOOFResult:
    """Generate cross-fitted expert OOF predictions (F3+F4 fix).

    F4 fix: FC mask is selected using FC data only; SC mask is selected
    using SC data only. Masks, mask families, and sizes may differ.
    """
    n_train = len(train_idx)
    fc_oof = np.full(n_train, np.nan, dtype=np.float64)
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)

    # Create fusion folds
    rng_fusion = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 777)
    perm = rng_fusion.permutation(n_train)
    fold_sizes = np.full(n_fusion_folds, n_train // n_fusion_folds)
    fold_sizes[:n_train % n_fusion_folds] += 1

    fusion_folds = []
    current = 0
    for k in range(n_fusion_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        fusion_folds.append((a_local, v_local))
        current = stop

    edge_prior = build_edge_product_prior(roi_prior)

    last_fc_expert = None
    last_sc_expert = None
    last_fc_family = ""
    last_fc_size = 0
    last_sc_family = ""
    last_sc_size = 0

    for a_local, v_local in fusion_folds:
        a_global = train_idx[a_local]
        v_global = train_idx[v_local]

        # F4 FIX: Select best mask independently for each modality
        best_mask_fc, fc_family, fc_size = _select_best_mask_for_modality(
            X_fc, y, edge_prior, roi_prior, a_global, seed, outer_fold, n_inner,
        )
        best_mask_sc, sc_family, sc_size = _select_best_mask_for_modality(
            X_sc, y, edge_prior, roi_prior, a_global, seed, outer_fold, n_inner,
        )

        # Fit FC expert on analysis, predict validation
        if expert_type == "ncr" and edge_laplacian is not None:
            fc_expert = fit_expert_ncr_fixed(
                X_fc, y, best_mask_fc, edge_laplacian,
                a_global, v_global, seed=seed, outer_fold=outer_fold, n_inner=n_inner,
            )
            sc_expert = fit_expert_ncr_fixed(
                X_sc, y, best_mask_sc, edge_laplacian,
                a_global, v_global, seed=seed, outer_fold=outer_fold, n_inner=n_inner,
            )
        else:
            fc_expert = fit_expert_ridge_fixed(
                X_fc, y, best_mask_fc,
                a_global, v_global, seed=seed, outer_fold=outer_fold, n_inner=n_inner,
            )
            sc_expert = fit_expert_ridge_fixed(
                X_sc, y, best_mask_sc,
                a_global, v_global, seed=seed, outer_fold=outer_fold, n_inner=n_inner,
            )

        fc_expert.mask_family = fc_family
        fc_expert.mask_size = fc_size
        sc_expert.mask_family = sc_family
        sc_expert.mask_size = sc_size

        fc_oof[v_local] = fc_expert.test_pred
        sc_oof[v_local] = sc_expert.test_pred

        last_fc_expert = fc_expert
        last_sc_expert = sc_expert
        last_fc_family = fc_family
        last_fc_size = fc_size
        last_sc_family = sc_family
        last_sc_size = sc_size

    # Fuse FC and SC experts
    fc_sc_weights, _ = search_fusion_weights_simple(y[train_idx], fc_oof, sc_oof)
    expert_fused_oof = (fc_sc_weights["fc"] * fc_oof +
                        fc_sc_weights["sc"] * sc_oof)

    return ExpertOOFResult(
        fc_oof=fc_oof,
        sc_oof=sc_oof,
        fc_expert=last_fc_expert,
        sc_expert=last_sc_expert,
        fc_sc_weights=fc_sc_weights,
        expert_fused_oof=expert_fused_oof,
        selected_mask_family_fc=last_fc_family,
        selected_mask_size_fc=last_fc_size,
        selected_mask_family_sc=last_sc_family,
        selected_mask_size_sc=last_sc_size,
    )


# ---------------------------------------------------------------------------
# F5 FIX: Expert-only reconstruction validation
# ---------------------------------------------------------------------------

def validate_expert_reconstruction(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    fc_expert: ExpertResult,
    sc_expert: ExpertResult,
    expert_w_fc: float,
    expert_w_sc: float,
    test_idx: np.ndarray,
    expert_test_pred: np.ndarray,
    tol: float = 1e-8,
) -> float:
    """Validate expert reconstruction: FC expert + SC expert -> expert fused.

    This checks that the primal coefficients of each expert reproduce
    the expert predictions, and that the weighted fusion reproduces
    the expert_test_pred.

    Returns max absolute error.
    """
    # FC expert prediction from primal coefficients
    X_fc_sub = X_fc[test_idx][:, fc_expert.mask]
    X_fc_z = (X_fc_sub - fc_expert.scaler_mean) / np.maximum(fc_expert.scaler_scale, 1e-8)
    fc_pred = X_fc_z @ fc_expert.beta_standardized * fc_expert.y_std + fc_expert.y_mean

    # SC expert prediction from primal coefficients
    X_sc_sub = X_sc[test_idx][:, sc_expert.mask]
    X_sc_z = (X_sc_sub - sc_expert.scaler_mean) / np.maximum(sc_expert.scaler_scale, 1e-8)
    sc_pred = X_sc_z @ sc_expert.beta_standardized * sc_expert.y_std + sc_expert.y_mean

    # Expert fusion: weighted sum of FC and SC expert predictions
    expert_fused = expert_w_fc * fc_pred + expert_w_sc * sc_pred

    # Compare with the saved expert_test_pred
    return float(np.max(np.abs(expert_fused - expert_test_pred)))


def validate_final_reconstruction(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    fc_expert: ExpertResult,
    sc_expert: ExpertResult,
    alpha: float,
    expert_w_fc: float,
    expert_w_sc: float,
    baseline_test: np.ndarray,
    test_idx: np.ndarray,
    final_test_pred: np.ndarray,
    tol: float = 1e-8,
) -> float:
    """Validate final reconstruction: baseline + expert -> final.

    reconstructed_final = (1-alpha) * baseline_test + alpha * reconstructed_expert_fused

    Returns max absolute error.
    """
    # Reconstruct expert predictions from primal coefficients
    X_fc_sub = X_fc[test_idx][:, fc_expert.mask]
    X_fc_z = (X_fc_sub - fc_expert.scaler_mean) / np.maximum(fc_expert.scaler_scale, 1e-8)
    fc_pred = X_fc_z @ fc_expert.beta_standardized * fc_expert.y_std + fc_expert.y_mean

    X_sc_sub = X_sc[test_idx][:, sc_expert.mask]
    X_sc_z = (X_sc_sub - sc_expert.scaler_mean) / np.maximum(sc_expert.scaler_scale, 1e-8)
    sc_pred = X_sc_z @ sc_expert.beta_standardized * sc_expert.y_std + sc_expert.y_mean

    reconstructed_expert_fused = expert_w_fc * fc_pred + expert_w_sc * sc_pred
    reconstructed_final = (1 - alpha) * baseline_test + alpha * reconstructed_expert_fused

    # Compare with the saved final_test_pred
    return float(np.max(np.abs(reconstructed_final - final_test_pred)))


# ---------------------------------------------------------------------------
# Simple fusion weight search (unchanged)
# ---------------------------------------------------------------------------

def search_fusion_weights_simple(
    y_true: np.ndarray,
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    branch_names: Tuple[str, str] = ("fc", "sc"),
) -> Tuple[Dict[str, float], float]:
    """Search convex fusion weights for two branches.

    Returns (weights_dict, best_pearson).
    """
    best_w = 0.5
    best_r = -np.inf

    valid = np.isfinite(oof_a) & np.isfinite(oof_b)
    if valid.sum() < 10:
        return {branch_names[0]: 0.5, branch_names[1]: 0.5}, 0.0

    for w in WEIGHT_GRID:
        combined = w * oof_a + (1 - w) * oof_b
        r, _ = pearsonr(combined[valid], y_true[valid])
        if r > best_r + 1e-14:
            best_r = r
            best_w = w

    return {branch_names[0]: best_w, branch_names[1]: 1.0 - best_w}, best_r


# ---------------------------------------------------------------------------
# Hierarchical fusion (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Result of hierarchical expert-baseline fusion."""
    alpha: float                     # expert weight in final fusion
    expert_w_fc: float               # FC weight within expert
    expert_w_sc: float               # SC weight within expert
    base_oof: np.ndarray
    expert_oof: np.ndarray
    final_oof: np.ndarray
    base_test_pred: Optional[np.ndarray] = None
    expert_test_pred: Optional[np.ndarray] = None
    final_test_pred: Optional[np.ndarray] = None


def hierarchical_fusion(
    y: np.ndarray,
    base_oof: np.ndarray,
    expert_fused_oof: np.ndarray,
    base_test: np.ndarray,
    expert_test: np.ndarray,
    expert_w_fc: float,
    expert_w_sc: float,
) -> FusionResult:
    """Stage B: fuse strong baseline with prior expert."""
    # Search alpha
    best_alpha = 0.0
    best_r = -np.inf

    valid = np.isfinite(base_oof) & np.isfinite(expert_fused_oof)
    if valid.sum() < 10:
        return FusionResult(
            alpha=0.0, expert_w_fc=expert_w_fc, expert_w_sc=expert_w_sc,
            base_oof=base_oof, expert_oof=expert_fused_oof,
            final_oof=base_oof, base_test_pred=base_test,
            expert_test_pred=expert_test, final_test_pred=base_test,
        )

    for alpha in WEIGHT_GRID:
        combined = (1 - alpha) * base_oof + alpha * expert_fused_oof
        r, _ = pearsonr(combined[valid], y[valid])
        if r > best_r + 1e-14:
            best_r = r
            best_alpha = alpha

    final_oof = (1 - best_alpha) * base_oof + best_alpha * expert_fused_oof
    final_test = (1 - best_alpha) * base_test + best_alpha * expert_test

    return FusionResult(
        alpha=best_alpha,
        expert_w_fc=expert_w_fc,
        expert_w_sc=expert_w_sc,
        base_oof=base_oof,
        expert_oof=expert_fused_oof,
        final_oof=final_oof,
        base_test_pred=base_test,
        expert_test_pred=expert_test,
        final_test_pred=final_test,
    )


# ---------------------------------------------------------------------------
# Expert-use diagnostics (unchanged)
# ---------------------------------------------------------------------------

def compute_expert_use_diagnostics(
    alphas: np.ndarray,
    base_oof_error: np.ndarray,
    expert_oof_error: np.ndarray,
    base_oof: np.ndarray,
    expert_oof: np.ndarray,
) -> Dict:
    """Compute conservative expert-use diagnostics."""
    return {
        "fraction_alpha_0": float(np.mean(alphas == 0)),
        "fraction_alpha_0_025": float(np.mean((alphas > 0) & (alphas <= 0.25))),
        "fraction_alpha_gt_025": float(np.mean(alphas > 0.25)),
        "mean_alpha": float(np.mean(alphas)),
        "median_alpha": float(np.median(alphas)),
        "corr_base_expert_pred": float(pearsonr(base_oof, expert_oof)[0])
            if len(base_oof) > 10 else 0.0,
        "corr_base_expert_error": float(pearsonr(base_oof_error, expert_oof_error)[0])
            if len(base_oof_error) > 10 else 0.0,
    }


# ---------------------------------------------------------------------------
# Coefficient export (unchanged)
# ---------------------------------------------------------------------------

def export_expert_coefficients(
    fc_expert: ExpertResult,
    sc_expert: ExpertResult,
    alpha: float,
    expert_w_fc: float,
    expert_w_sc: float,
    n_edges: int = N_EDGE,
) -> Dict:
    """Export primal expert coefficients in full edge space."""
    beta_fc_full = np.zeros(n_edges, dtype=np.float64)
    beta_fc_full[fc_expert.mask] = fc_expert.beta_original

    beta_sc_full = np.zeros(n_edges, dtype=np.float64)
    beta_sc_full[sc_expert.mask] = sc_expert.beta_original

    beta_fc_weighted = alpha * expert_w_fc * beta_fc_full
    beta_sc_weighted = alpha * expert_w_sc * beta_sc_full

    return {
        "beta_fc_expert": beta_fc_full,
        "beta_sc_expert": beta_sc_full,
        "beta_fc_expert_weighted": beta_fc_weighted,
        "beta_sc_expert_weighted": beta_sc_weighted,
        "selected_edge_indices": np.where(fc_expert.mask)[0],
        "fc_scaler_mean": fc_expert.scaler_mean,
        "fc_scaler_scale": fc_expert.scaler_scale,
        "sc_scaler_mean": sc_expert.scaler_mean,
        "sc_scaler_scale": sc_expert.scaler_scale,
        "fc_y_mean": fc_expert.y_mean,
        "fc_y_std": fc_expert.y_std,
        "sc_y_mean": sc_expert.y_mean,
        "sc_y_std": sc_expert.y_std,
    }
