"""Phase 2D: Prior-Selected Subspace NCR Expert Fusion Pilot.

Tests whether LLM prior can select a useful low-dimensional expert subspace
that, when fused with the strong baseline, improves prediction.

PS-NCR-EF: Prior-Selected Network-Constrained Expert Fusion
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_crossfit_ablation import (
    make_fusion_folds,
    make_inner_selection_folds,
    make_outer_splits,
)
from metascfc.experiments.prior_subspace_expert_fusion import (
    N_ROI,
    N_EDGE,
    K_EDGE_GRID,
    M_ROI_GRID,
    RIDGE_EXPERT_GRID,
    LAPLACIAN_RATIO_GRID,
    WEIGHT_GRID,
    build_edge_product_prior,
    direct_topk_mask,
    roi_incident_mask,
    build_control_prior,
    fit_expert_ridge,
    fit_expert_ncr,
    search_fusion_weights_simple,
    hierarchical_fusion,
    export_expert_coefficients,
    validate_primal_reconstruction,
    compute_expert_use_diagnostics,
    ExpertResult,
    ExpertOOFResult,
    FusionResult,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
    EdgeLaplacian,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEV_SEEDS = [1313, 1414, 1515, 1616]
N_OUTER_FOLDS = 5
N_FUSION_FOLDS = 3
N_INNER = 3
N_FINAL_CV = 3

RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

TASKS = {
    "working_memory": {"display": "WM"},
    "fluid_intelligence": {"display": "FI"},
}

PRIOR_TYPES = ["matched", "cross_task", "shuffled", "random"]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2d_ps_ncr_expert_fusion"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
COEFF_DIR = OUTPUT_DIR / "coefficients"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"

for d in [OUTPUT_DIR, CKPT_DIR, COEFF_DIR, PRED_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
log.info("Loading data...")
fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
iu = np.triu_indices(N_ROI, k=1)
X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)
y_wm = np.load(
    REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
).astype(np.float64)
y_fi = np.load(
    REPO_ROOT / "inputs/dataset_SC/label_all.npy"
).astype(np.float64)
wm_prior = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)["prior_score"].values.astype(np.float64)
fi_prior = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
)["prior_score"].values.astype(np.float64)

n_subjects = int(y_wm.shape[0])
log.info("  Subjects: %d, FC edges: %d, SC edges: %d", n_subjects, X_fc.shape[1], X_sc.shape[1])
log.info("  WM prior: min=%.4f max=%.4f", wm_prior.min(), wm_prior.max())
log.info("  FI prior: min=%.4f max=%.4f", fi_prior.min(), fi_prior.max())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _ckpt_path(phase: str, seed: int, fold: int) -> Path:
    return CKPT_DIR / f"{phase}_seed_{seed}_fold_{fold}.pkl"


def _save_ckpt(phase: str, seed: int, fold: int, state: dict) -> None:
    with open(_ckpt_path(phase, seed, fold), "wb") as f:
        pickle.dump(state, f)


def _load_ckpt(phase: str, seed: int, fold: int) -> Optional[dict]:
    p = _ckpt_path(phase, seed, fold)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# Baseline helpers (same-solver no-prior FC+SC Ridge with cross-fitting)
# ---------------------------------------------------------------------------

def _select_alpha_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    ridge_grid: Sequence[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float]:
    """Select Ridge alpha via inner CV. Returns (best_alpha, best_pearson).

    inner_folds contain GLOBAL subject indices. We map them to local indices
    within train_global.
    """
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X[train_global])
    y_mean = float(y[train_global].mean())
    y_std = max(float(y[train_global].std()), 1e-8)
    y_z = (y[train_global] - y_mean) / y_std

    # Map global indices to local positions within train_global
    global_to_local = {int(idx): i for i, idx in enumerate(train_global)}

    best_pearson = -np.inf
    best_alpha = ridge_grid[0]

    for alpha in ridge_grid:
        fold_ps = []
        for b_global, c_global in inner_folds:
            b_local = np.array([global_to_local[int(idx)] for idx in b_global])
            c_local = np.array([global_to_local[int(idx)] for idx in c_global])
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_z[b_local], y_z[b_local])
            pred_z = model.predict(X_z[c_local])
            pred = pred_z * y_std + y_mean
            r, _ = pearsonr(y[c_global], pred)
            fold_ps.append(r)
        mp = float(np.mean(fold_ps))
        if mp > best_pearson + 1e-14:
            best_pearson = mp
            best_alpha = alpha

    return best_alpha, best_pearson


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

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_train, y[train_idx])
    return model.predict(X_val)


def generate_baseline_crossfit_oof(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_fusion_folds: int = N_FUSION_FOLDS,
    n_inner: int = N_INNER,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Generate cross-fitted baseline FC and SC Ridge OOF predictions.

    Uses the exact same fusion folds as the expert branch.

    Returns (fc_oof, sc_oof, fc_sel_info, sc_sel_info).
    """
    n_train = len(train_idx)
    fc_oof = np.full(n_train, np.nan, dtype=np.float64)
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)

    fusion_folds = make_fusion_folds(train_idx, seed, outer_fold, n_fusion_folds)
    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    fc_alphas = []
    sc_alphas = []

    for fold_k, (a_k, v_k) in enumerate(fusion_folds):
        inner_folds = make_inner_selection_folds(a_k, seed, outer_fold, fold_k, n_inner)
        v_local = np.array([train_local_map[int(idx)] for idx in v_k])

        # FC Ridge
        fc_alpha, _ = _select_alpha_inner_cv(X_fc, y, a_k, ridge_grid, inner_folds)
        fc_pred = _fit_and_predict_ridge_on_subset(X_fc, y, a_k, v_k, fc_alpha)
        fc_oof[v_local] = fc_pred
        fc_alphas.append(fc_alpha)

        # SC Ridge
        sc_alpha, _ = _select_alpha_inner_cv(X_sc, y, a_k, ridge_grid, inner_folds)
        sc_pred = _fit_and_predict_ridge_on_subset(X_sc, y, a_k, v_k, sc_alpha)
        sc_oof[v_local] = sc_pred
        sc_alphas.append(sc_alpha)

    fc_sel = {"best_alpha": float(np.median(fc_alphas)), "fold_alphas": fc_alphas}
    sc_sel = {"best_alpha": float(np.median(sc_alphas)), "fold_alphas": sc_alphas}

    return fc_oof, sc_oof, fc_sel, sc_sel


def fit_baseline_final(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fc_alpha: float,
    sc_alpha: float,
    w_fc: float,
    w_sc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit baseline FC+SC Ridge on full training, predict test.

    Returns (fc_test, sc_test, fused_test).
    """
    fc_pred = _fit_and_predict_ridge_on_subset(X_fc, y, train_idx, test_idx, fc_alpha)
    sc_pred = _fit_and_predict_ridge_on_subset(X_sc, y, train_idx, test_idx, sc_alpha)
    fused_pred = w_fc * fc_pred + w_sc * sc_pred
    return fc_pred, sc_pred, fused_pred


# ---------------------------------------------------------------------------
# Expert cross-fitted OOF (reusing module function)
# ---------------------------------------------------------------------------

def generate_expert_oof_with_control(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    roi_prior: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    expert_type: str = "ncr",
    edge_laplacian: Optional[EdgeLaplacian] = None,
    n_fusion_folds: int = N_FUSION_FOLDS,
    n_inner: int = N_INNER,
) -> ExpertOOFResult:
    """Generate expert cross-fitted OOF for a given prior/control."""
    from metascfc.experiments.prior_subspace_expert_fusion import (
        generate_expert_crossfit_oof,
    )

    return generate_expert_crossfit_oof(
        X_fc, X_sc, y, roi_prior, train_idx, seed, outer_fold,
        n_fusion_folds=n_fusion_folds,
        n_inner=n_inner,
        expert_type=expert_type,
        edge_laplacian=edge_laplacian,
        use_shared_mask=True,
    )


# ---------------------------------------------------------------------------
# Final refit helpers
# ---------------------------------------------------------------------------

def final_refit_baseline(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    fc_alpha: float,
    sc_alpha: float,
    w_fc: float,
    w_sc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final baseline refit on full training, predict test."""
    return fit_baseline_final(
        X_fc, X_sc, y, train_idx, test_idx, fc_alpha, sc_alpha, w_fc, w_sc,
    )


def final_refit_expert_single(
    X_mod: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    edge_laplacian: Optional[EdgeLaplacian],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    expert_type: str = "ncr",
) -> ExpertResult:
    """Final refit of a single expert (FC or SC) on full training, predict test.

    Re-selects mask and hyperparameters on full training via 3-fold CV,
    then fits on all training subjects and predicts test.
    """
    if expert_type == "ncr" and edge_laplacian is not None:
        return fit_expert_ncr(
            X_mod, y, mask, edge_laplacian,
            train_idx, test_idx,
            seed=seed, outer_fold=outer_fold, n_inner=N_FINAL_CV,
        )
    else:
        return fit_expert_ridge(
            X_mod, y, mask,
            train_idx, test_idx,
            seed=seed, outer_fold=outer_fold, n_inner=N_FINAL_CV,
        )


def final_refit_expert(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    roi_prior: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    expert_type: str = "ncr",
    edge_laplacian: Optional[EdgeLaplacian] = None,
) -> Tuple[ExpertResult, ExpertResult, np.ndarray, np.ndarray]:
    """Final expert refit: select mask on full training, fit FC and SC experts.

    Returns (fc_expert, sc_expert, fc_test_pred, sc_test_pred).
    """
    edge_prior = build_edge_product_prior(roi_prior)

    # Re-select best mask on full training set using 3-fold CV
    from metascfc.experiments.prior_subspace_expert_fusion import _select_best_mask
    best_mask_fc, best_mask_sc, best_family, best_k_or_m = _select_best_mask(
        X_fc, X_sc, y, edge_prior, roi_prior, train_idx, seed, outer_fold, N_FINAL_CV,
    )

    # Fit FC expert
    fc_expert = final_refit_expert_single(
        X_fc, y, best_mask_fc, edge_laplacian,
        train_idx, test_idx, seed, outer_fold, expert_type,
    )
    fc_expert.mask_family = best_family
    fc_expert.mask_size = best_k_or_m

    # Fit SC expert
    sc_expert = final_refit_expert_single(
        X_sc, y, best_mask_fc, edge_laplacian,
        train_idx, test_idx, seed, outer_fold, expert_type,
    )
    sc_expert.mask_family = best_family
    sc_expert.mask_size = best_k_or_m

    return fc_expert, sc_expert, fc_expert.test_pred, sc_expert.test_pred


# ---------------------------------------------------------------------------
# STEP 1: Baseline audit (seeds 0-9)
# ---------------------------------------------------------------------------

def run_baseline_audit() -> bool:
    """Audit baseline on seeds 0-9, 5 outer folds each.

    Returns True if audit passes, False otherwise.
    """
    log.info("=" * 70)
    log.info("STEP 1: Baseline audit (seeds 0-9)")
    log.info("=" * 70)

    audit_seeds = list(range(10))
    outer_splits = make_outer_splits(n_subjects, audit_seeds, N_OUTER_FOLDS)
    log.info("  Total splits: %d", len(outer_splits))

    wm_rows = []
    fi_rows = []
    t_start = time.time()

    for idx, (seed, fold, train_idx, test_idx) in enumerate(outer_splits):
        log.info("  [%d/%d] seed=%d fold=%d train=%d test=%d",
                 idx + 1, len(outer_splits), seed, fold, len(train_idx), len(test_idx))

        # --- Working Memory ---
        fc_oof_wm, sc_oof_wm, fc_sel_wm, sc_sel_wm = generate_baseline_crossfit_oof(
            X_fc, X_sc, y_wm, train_idx, seed, fold,
        )
        # Search fusion weights
        y_train_wm = y_wm[train_idx]
        w_fc_wm, wm_fused_r = search_fusion_weights_simple(
            y_train_wm, fc_oof_wm, sc_oof_wm, ("fc", "sc"),
        )
        # Final refit
        fc_te_wm, sc_te_wm, fused_te_wm = final_refit_baseline(
            X_fc, X_sc, y_wm, train_idx, test_idx,
            seed, fold,
            fc_sel_wm["best_alpha"], sc_sel_wm["best_alpha"],
            w_fc_wm["fc"], w_fc_wm["sc"],
        )
        wm_m = prediction_metrics(y_wm[test_idx], fused_te_wm)
        wm_rows.append({
            "seed": seed, "fold": fold,
            "pearson": wm_m["pearson"], "rmse": wm_m["rmse"], "mae": wm_m["mae"],
        })

        # --- Fluid Intelligence ---
        fc_oof_fi, sc_oof_fi, fc_sel_fi, sc_sel_fi = generate_baseline_crossfit_oof(
            X_fc, X_sc, y_fi, train_idx, seed, fold,
        )
        y_train_fi = y_fi[train_idx]
        w_fc_fi, fi_fused_r = search_fusion_weights_simple(
            y_train_fi, fc_oof_fi, sc_oof_fi, ("fc", "sc"),
        )
        fc_te_fi, sc_te_fi, fused_te_fi = final_refit_baseline(
            X_fc, X_sc, y_fi, train_idx, test_idx,
            seed, fold,
            fc_sel_fi["best_alpha"], sc_sel_fi["best_alpha"],
            w_fc_fi["fc"], w_fc_fi["sc"],
        )
        fi_m = prediction_metrics(y_fi[test_idx], fused_te_fi)
        fi_rows.append({
            "seed": seed, "fold": fold,
            "pearson": fi_m["pearson"], "rmse": fi_m["rmse"], "mae": fi_m["mae"],
        })

    elapsed = time.time() - t_start
    log.info("  Audit completed in %.1fs", elapsed)

    wm_df = pd.DataFrame(wm_rows)
    fi_df = pd.DataFrame(fi_rows)
    wm_mean_r = wm_df["pearson"].mean()
    fi_mean_r = fi_df["pearson"].mean()
    wm_std_r = wm_df["pearson"].std(ddof=1)
    fi_std_r = fi_df["pearson"].std(ddof=1)

    log.info("  WM fused r: %.6f (+/- %.6f)", wm_mean_r, wm_std_r)
    log.info("  FI fused r: %.6f (+/- %.6f)", fi_mean_r, fi_std_r)

    # Check tolerance: expected WM ~0.2635 +/- 0.01, FI ~0.3709 +/- 0.01
    wm_pass = bool(abs(wm_mean_r - 0.2635) < 0.01)
    fi_pass = bool(abs(fi_mean_r - 0.3709) < 0.01)

    audit_result = {
        "wm_mean_pearson": float(wm_mean_r),
        "wm_std_pearson": float(wm_std_r),
        "wm_expected": 0.2635,
        "wm_pass": wm_pass,
        "fi_mean_pearson": float(fi_mean_r),
        "fi_std_pearson": float(fi_std_r),
        "fi_expected": 0.3709,
        "fi_pass": fi_pass,
        "overall_pass": bool(wm_pass and fi_pass),
        "runtime_seconds": elapsed,
        "n_splits": len(outer_splits),
    }

    with open(OUTPUT_DIR / "BASELINE_AUDIT.json", "w") as f:
        json.dump(audit_result, f, indent=2)

    if not (wm_pass and fi_pass):
        log.error("STATUS: PHASE2D_BASELINE_AUDIT_FAILED")
        log.error("  WM: %.6f (expected ~0.2635 +/- 0.01) %s",
                  wm_mean_r, "PASS" if wm_pass else "FAIL")
        log.error("  FI: %.6f (expected ~0.3709 +/- 0.01) %s",
                  fi_mean_r, "PASS" if fi_pass else "FAIL")
        return False

    log.info("  Baseline audit PASSED")
    return True


# ---------------------------------------------------------------------------
# STEP 2: Main pilot loop
# ---------------------------------------------------------------------------

def _get_prior_for_task(
    prior_type: str,
    task_key: str,
    seed: int,
) -> np.ndarray:
    """Get the appropriate prior for a given prior_type and task."""
    task_prior = wm_prior if task_key == "working_memory" else fi_prior
    other_prior = fi_prior if task_key == "working_memory" else wm_prior
    return build_control_prior(prior_type, task_prior, other_prior, seed, N_ROI)


def run_pilot_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    task_key: str,
    prior_type: str,
    roi_prior: np.ndarray,
) -> Dict[str, Any]:
    """Run one pilot split: baseline + expert fusion for one prior_type."""
    expert_type = "ncr"

    # Build edge Laplacian from the control prior (top_k=10)
    edge_laplacian = build_edge_laplacian(
        N_ROI, prior_scores=roi_prior, top_k=10,
    )

    # --- A. Baseline (same for all prior_types) ---
    fc_oof_base, sc_oof_base, fc_sel_base, sc_sel_base = generate_baseline_crossfit_oof(
        X_fc, X_sc, y, train_idx, seed, outer_fold,
    )
    y_train = y[train_idx]

    # Search fusion weights for baseline
    w_fc_base, base_fused_pearson = search_fusion_weights_simple(
        y_train, fc_oof_base, sc_oof_base, ("fc", "sc"),
    )
    baseline_fused_oof = w_fc_base["fc"] * fc_oof_base + w_fc_base["sc"] * sc_oof_base

    # Final baseline refit
    fc_te_base, sc_te_base, baseline_test = fit_baseline_final(
        X_fc, X_sc, y, train_idx, test_idx,
        fc_sel_base["best_alpha"], sc_sel_base["best_alpha"],
        w_fc_base["fc"], w_fc_base["sc"],
    )
    baseline_metrics = prediction_metrics(y[test_idx], baseline_test)

    # --- B. NCR Expert ---
    expert_oof_ncr = generate_expert_oof_with_control(
        X_fc, X_sc, y, roi_prior, train_idx, seed, outer_fold,
        expert_type=expert_type, edge_laplacian=edge_laplacian,
    )

    # --- C. Ridge Expert (ratio=0) for ablation ---
    expert_oof_ridge = generate_expert_oof_with_control(
        X_fc, X_sc, y, roi_prior, train_idx, seed, outer_fold,
        expert_type="ridge", edge_laplacian=None,
    )

    # --- Final refit for test predictions ---
    fc_te_ncr, sc_te_ncr, fc_test_ncr, sc_test_ncr = final_refit_expert(
        X_fc, X_sc, y, roi_prior, train_idx, test_idx, seed, outer_fold,
        expert_type=expert_type, edge_laplacian=edge_laplacian,
    )
    expert_test_ncr = (expert_oof_ncr.fc_sc_weights["fc"] * fc_test_ncr +
                       expert_oof_ncr.fc_sc_weights["sc"] * sc_test_ncr)

    fc_te_ridge, sc_te_ridge, fc_test_ridge, sc_test_ridge = final_refit_expert(
        X_fc, X_sc, y, roi_prior, train_idx, test_idx, seed, outer_fold,
        expert_type="ridge", edge_laplacian=None,
    )
    expert_test_ridge = (expert_oof_ridge.fc_sc_weights["fc"] * fc_test_ridge +
                         expert_oof_ridge.fc_sc_weights["sc"] * sc_test_ridge)

    # --- D. Hierarchical fusion: baseline + NCR expert ---
    fusion_ncr = hierarchical_fusion(
        y[train_idx], baseline_fused_oof, expert_oof_ncr.expert_fused_oof,
        baseline_test, expert_test_ncr,
        expert_oof_ncr.fc_sc_weights["fc"],
        expert_oof_ncr.fc_sc_weights["sc"],
    )

    # --- E. Hierarchical fusion: baseline + Ridge expert ---
    fusion_ridge = hierarchical_fusion(
        y[train_idx], baseline_fused_oof, expert_oof_ridge.expert_fused_oof,
        baseline_test, expert_test_ridge,
        expert_oof_ridge.fc_sc_weights["fc"],
        expert_oof_ridge.fc_sc_weights["sc"],
    )

    # Compute OOF metrics for NCR expert
    valid_oof = np.isfinite(baseline_fused_oof) & np.isfinite(expert_oof_ncr.expert_fused_oof)
    base_oof_error = np.full(len(train_idx), np.nan)
    expert_oof_error = np.full(len(train_idx), np.nan)
    if valid_oof.sum() > 10:
        base_oof_error[valid_oof] = y_train[valid_oof] - baseline_fused_oof[valid_oof]
        expert_oof_error[valid_oof] = y_train[valid_oof] - expert_oof_ncr.expert_fused_oof[valid_oof]

    diagnostics = compute_expert_use_diagnostics(
        np.array([fusion_ncr.alpha]),
        base_oof_error[np.isfinite(base_oof_error)],
        expert_oof_error[np.isfinite(expert_oof_error)],
        baseline_fused_oof[valid_oof] if valid_oof.sum() > 0 else np.array([]),
        expert_oof_ncr.expert_fused_oof[valid_oof] if valid_oof.sum() > 0 else np.array([]),
    )

    return {
        # Baseline
        "baseline_pearson": baseline_metrics["pearson"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_mae": baseline_metrics["mae"],
        "baseline_test": baseline_test,
        # NCR expert OOF
        "fc_expert_pearson_ncr": float(pearsonr(y_train, expert_oof_ncr.fc_oof)[0])
            if np.isfinite(expert_oof_ncr.fc_oof).sum() > 10 else 0.0,
        "sc_expert_pearson_ncr": float(pearsonr(y_train, expert_oof_ncr.sc_oof)[0])
            if np.isfinite(expert_oof_ncr.sc_oof).sum() > 10 else 0.0,
        "expert_fused_pearson_ncr": float(pearsonr(y_train, expert_oof_ncr.expert_fused_oof)[0])
            if np.isfinite(expert_oof_ncr.expert_fused_oof).sum() > 10 else 0.0,
        # Ridge expert OOF
        "fc_expert_pearson_ridge": float(pearsonr(y_train, expert_oof_ridge.fc_oof)[0])
            if np.isfinite(expert_oof_ridge.fc_oof).sum() > 10 else 0.0,
        "sc_expert_pearson_ridge": float(pearsonr(y_train, expert_oof_ridge.sc_oof)[0])
            if np.isfinite(expert_oof_ridge.sc_oof).sum() > 10 else 0.0,
        "expert_fused_pearson_ridge": float(pearsonr(y_train, expert_oof_ridge.expert_fused_oof)[0])
            if np.isfinite(expert_oof_ridge.expert_fused_oof).sum() > 10 else 0.0,
        # Fusion
        "expert_w_FC_ncr": fusion_ncr.expert_w_fc,
        "expert_w_SC_ncr": fusion_ncr.expert_w_sc,
        "alpha_base_vs_expert_ncr": fusion_ncr.alpha,
        "expert_w_FC_ridge": fusion_ridge.expert_w_fc,
        "expert_w_SC_ridge": fusion_ridge.expert_w_sc,
        "alpha_base_vs_expert_ridge": fusion_ridge.alpha,
        # Diagnostics
        "diagnostics": diagnostics,
        # Selection info from expert OOF
        "selected_mask_family_ncr": expert_oof_ncr.selected_mask_family,
        "selected_mask_size_ncr": expert_oof_ncr.selected_mask_size_fc,
        "selected_mask_family_ridge": expert_oof_ridge.selected_mask_family,
        "selected_mask_size_ridge": expert_oof_ridge.selected_mask_size_fc,
        # Expert OOF objects for final refit
        "expert_oof_ncr": expert_oof_ncr,
        "expert_oof_ridge": expert_oof_ridge,
        "fusion_ncr": fusion_ncr,
        "fusion_ridge": fusion_ridge,
        "fc_sel_base": fc_sel_base,
        "sc_sel_base": sc_sel_base,
        "w_fc_base": w_fc_base,
    }


def run_final_refit_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    outer_fold: int,
    task_key: str,
    prior_type: str,
    roi_prior: np.ndarray,
    pilot_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Final refit: reselect on full training, predict test."""
    expert_type = "ncr"

    # Build edge Laplacian
    edge_laplacian = build_edge_laplacian(
        N_ROI, prior_scores=roi_prior, top_k=10,
    )

    # --- Baseline final ---
    w_fc_base = pilot_result["w_fc_base"]
    fc_sel = pilot_result["fc_sel_base"]
    sc_sel = pilot_result["sc_sel_base"]
    _, _, baseline_test = fit_baseline_final(
        X_fc, X_sc, y, train_idx, test_idx,
        fc_sel["best_alpha"], sc_sel["best_alpha"],
        w_fc_base["fc"], w_fc_base["sc"],
    )

    # --- NCR Expert final ---
    fc_expert_ncr, sc_expert_ncr, fc_te_ncr, sc_te_ncr = final_refit_expert(
        X_fc, X_sc, y, roi_prior, train_idx, test_idx,
        seed, outer_fold, expert_type="ncr", edge_laplacian=edge_laplacian,
    )

    # Expert test fused
    expert_w_fc_ncr = pilot_result["expert_w_FC_ncr"]
    expert_w_sc_ncr = pilot_result["expert_w_SC_ncr"]
    expert_test_ncr = expert_w_fc_ncr * fc_te_ncr + expert_w_sc_ncr * sc_te_ncr
    alpha_ncr = pilot_result["alpha_base_vs_expert_ncr"]
    final_test_ncr = (1 - alpha_ncr) * baseline_test + alpha_ncr * expert_test_ncr

    # --- Ridge Expert final ---
    fc_expert_ridge, sc_expert_ridge, fc_te_ridge, sc_te_ridge = final_refit_expert(
        X_fc, X_sc, y, roi_prior, train_idx, test_idx,
        seed, outer_fold, expert_type="ridge", edge_laplacian=None,
    )

    expert_w_fc_ridge = pilot_result["expert_w_FC_ridge"]
    expert_w_sc_ridge = pilot_result["expert_w_SC_ridge"]
    expert_test_ridge = expert_w_fc_ridge * fc_te_ridge + expert_w_sc_ridge * sc_te_ridge
    alpha_ridge = pilot_result["alpha_base_vs_expert_ridge"]
    final_test_ridge = (1 - alpha_ridge) * baseline_test + alpha_ridge * expert_test_ridge

    # --- Metrics ---
    baseline_m = prediction_metrics(y[test_idx], baseline_test)
    final_ncr_m = prediction_metrics(y[test_idx], final_test_ncr)
    final_ridge_m = prediction_metrics(y[test_idx], final_test_ridge)

    # --- Primal reconstruction validation ---
    max_err_ncr_fc = validate_primal_reconstruction(
        X_fc, X_sc, fc_expert_ncr, sc_expert_ncr,
        alpha_ncr, expert_w_fc_ncr, expert_w_sc_ncr,
        test_idx, final_test_ncr, tol=1e-8,
    )
    max_err_ridge_fc = validate_primal_reconstruction(
        X_fc, X_sc, fc_expert_ridge, sc_expert_ridge,
        alpha_ridge, expert_w_fc_ridge, expert_w_sc_ridge,
        test_idx, final_test_ridge, tol=1e-8,
    )

    # --- Compute expert OOF correlations ---
    train_idx_sorted = np.sort(train_idx)
    baseline_fused_oof = pilot_result["fusion_ncr"].base_oof
    expert_fused_oof_ncr = pilot_result["expert_oof_ncr"].expert_fused_oof
    valid = np.isfinite(baseline_fused_oof) & np.isfinite(expert_fused_oof_ncr)

    base_expert_pred_corr = 0.0
    base_expert_error_corr = 0.0
    if valid.sum() > 10:
        base_expert_pred_corr = float(pearsonr(
            baseline_fused_oof[valid], expert_fused_oof_ncr[valid]
        )[0])
        y_train = y[train_idx]
        base_err = y_train[valid] - baseline_fused_oof[valid]
        expert_err = y_train[valid] - expert_fused_oof_ncr[valid]
        base_expert_error_corr = float(pearsonr(base_err, expert_err)[0])

    return {
        # Final predictions
        "baseline_test": baseline_test,
        "final_test_ncr": final_test_ncr,
        "final_test_ridge": final_test_ridge,
        # Final metrics
        "baseline_pearson": baseline_m["pearson"],
        "baseline_rmse": baseline_m["rmse"],
        "baseline_mae": baseline_m["mae"],
        "final_pearson_ncr": final_ncr_m["pearson"],
        "final_rmse_ncr": final_ncr_m["rmse"],
        "final_mae_ncr": final_ncr_m["mae"],
        "final_pearson_ridge": final_ridge_m["pearson"],
        "final_rmse_ridge": final_ridge_m["rmse"],
        "final_mae_ridge": final_ridge_m["mae"],
        # Deltas
        "delta_ncr_vs_baseline": final_ncr_m["pearson"] - baseline_m["pearson"],
        "delta_ridge_vs_baseline": final_ridge_m["pearson"] - baseline_m["pearson"],
        "delta_ncr_vs_ridge": final_ncr_m["pearson"] - final_ridge_m["pearson"],
        # Expert details
        "expert_w_FC_ncr": expert_w_fc_ncr,
        "expert_w_SC_ncr": expert_w_sc_ncr,
        "alpha_ncr": alpha_ncr,
        "expert_w_FC_ridge": expert_w_fc_ridge,
        "expert_w_SC_ridge": expert_w_sc_ridge,
        "alpha_ridge": alpha_ridge,
        # Experts
        "fc_expert_ncr": fc_expert_ncr,
        "sc_expert_ncr": sc_expert_ncr,
        "fc_expert_ridge": fc_expert_ridge,
        "sc_expert_ridge": sc_expert_ridge,
        # Validation
        "max_recon_error_ncr": max_err_ncr_fc,
        "max_recon_error_ridge": max_err_ridge_fc,
        # Correlations
        "base_expert_pred_corr": base_expert_pred_corr,
        "base_expert_error_corr": base_expert_error_corr,
        # Mask info from pilot
        "selected_mask_family_ncr": pilot_result["selected_mask_family_ncr"],
        "selected_mask_size_ncr": pilot_result["selected_mask_size_ncr"],
        "selected_mask_family_ridge": pilot_result["selected_mask_family_ridge"],
        "selected_mask_size_ridge": pilot_result["selected_mask_size_ridge"],
        "laplacian_ratio_ncr": fc_expert_ncr.laplacian_ratio,
        "lambda_r_ncr": fc_expert_ncr.lambda_r,
        "laplacian_ratio_ridge": fc_expert_ridge.laplacian_ratio,
        "lambda_r_ridge": fc_expert_ridge.lambda_r,
        # Diagnostics from pilot
        "diagnostics": pilot_result["diagnostics"],
    }


def run_main_pilot() -> pd.DataFrame:
    """Run the main pilot loop over all seeds, folds, targets, prior_types."""
    log.info("=" * 70)
    log.info("STEP 2-3: Main pilot loop")
    log.info("=" * 70)
    log.info("  Seeds: %s", DEV_SEEDS)
    log.info("  Outer folds: %d", N_OUTER_FOLDS)
    log.info("  Fusion folds: %d", N_FUSION_FOLDS)
    log.info("  Inner folds: %d", N_INNER)
    log.info("  Prior types: %s", PRIOR_TYPES)
    log.info("  Targets: WM, FI")

    all_split_rows: List[Dict] = []
    all_expert_selection: List[Dict] = []

    outer_splits_all = make_outer_splits(n_subjects, DEV_SEEDS, N_OUTER_FOLDS)
    log.info("  Total outer splits: %d", len(outer_splits_all))

    t_start = time.time()
    split_count = 0
    total_splits = len(outer_splits_all) * 2 * len(PRIOR_TYPES)

    for seed, fold, train_idx, test_idx in outer_splits_all:
        for task_key in ["working_memory", "fluid_intelligence"]:
            task_display = TASKS[task_key]["display"]
            y = y_wm if task_key == "working_memory" else y_fi

            for prior_type in PRIOR_TYPES:
                split_count += 1
                log.info(
                    "  [%d/%d] seed=%d fold=%d task=%s prior=%s",
                    split_count, total_splits, seed, fold, task_display, prior_type,
                )

                # Check checkpoint
                ckpt_key = f"final_{task_key}_{prior_type}"
                ckpt = _load_ckpt(ckpt_key, seed, fold)
                if ckpt is not None and ckpt.get("completed", False):
                    log.info("    Loaded checkpoint (completed)")
                    all_split_rows.extend(ckpt["split_rows"])
                    all_expert_selection.extend(ckpt.get("expert_selection", []))
                    continue

                t_split = time.time()

                # Get prior
                roi_prior = _get_prior_for_task(prior_type, task_key, seed)

                # Run pilot (OOF-based parameter selection)
                pilot = run_pilot_split(
                    X_fc, X_sc, y, train_idx, test_idx, seed, fold,
                    task_key, prior_type, roi_prior,
                )

                # Run final refit
                final = run_final_refit_split(
                    X_fc, X_sc, y, train_idx, test_idx, seed, fold,
                    task_key, prior_type, roi_prior, pilot,
                )

                elapsed = time.time() - t_split
                log.info(
                    "    baseline_r=%.4f  ncr_r=%.4f  ridge_r=%.4f  delta_ncr=%.4f  (%.1fs)",
                    final["baseline_pearson"], final["final_pearson_ncr"],
                    final["final_pearson_ridge"], final["delta_ncr_vs_baseline"], elapsed,
                )

                # --- Build split_metrics row ---
                row = {
                    "seed": seed,
                    "fold": fold,
                    "task": task_display,
                    "prior_type": prior_type,
                    "expert_type": "ncr",
                    # Baseline
                    "baseline_pearson": final["baseline_pearson"],
                    "baseline_rmse": final["baseline_rmse"],
                    "baseline_mae": final["baseline_mae"],
                    # Expert OOF
                    "fc_expert_pearson": pilot["fc_expert_pearson_ncr"],
                    "sc_expert_pearson": pilot["sc_expert_pearson_ncr"],
                    "expert_fused_pearson": pilot["expert_fused_pearson_ncr"],
                    # Final
                    "final_pearson": final["final_pearson_ncr"],
                    "final_rmse": final["final_rmse_ncr"],
                    "final_mae": final["final_mae_ncr"],
                    # Delta
                    "delta_final_vs_baseline": final["delta_ncr_vs_baseline"],
                    # Mask selection
                    "selected_mask_family_fc": final["selected_mask_family_ncr"],
                    "selected_mask_size_fc": final["selected_mask_size_ncr"],
                    "selected_lambda_fc_expert": final["lambda_r_ncr"],
                    "selected_laplacian_ratio_fc": final["laplacian_ratio_ncr"],
                    "selected_mask_family_sc": final["selected_mask_family_ncr"],
                    "selected_mask_size_sc": final["selected_mask_size_ncr"],
                    "selected_lambda_sc_expert": final["lambda_r_ncr"],
                    "selected_laplacian_ratio_sc": final["laplacian_ratio_ncr"],
                    # Fusion
                    "expert_w_FC": final["expert_w_FC_ncr"],
                    "expert_w_SC": final["expert_w_SC_ncr"],
                    "alpha_base_vs_expert": final["alpha_ncr"],
                    # Correlations
                    "base_expert_prediction_corr": final["base_expert_pred_corr"],
                    "base_expert_error_corr": final["base_expert_error_corr"],
                    # Ridge expert
                    "ridge_final_pearson": final["final_pearson_ridge"],
                    "ridge_delta_vs_baseline": final["delta_ridge_vs_baseline"],
                    "ridge_lambda_r": final["lambda_r_ridge"],
                    "ridge_laplacian_ratio": final["laplacian_ratio_ridge"],
                    # Validation
                    "max_recon_error": final["max_recon_error_ncr"],
                }
                all_split_rows.append(row)

                # Expert selection record
                sel_row = {
                    "seed": seed,
                    "fold": fold,
                    "task": task_display,
                    "prior_type": prior_type,
                    "mask_family": final["selected_mask_family_ncr"],
                    "mask_size": final["selected_mask_size_ncr"],
                    "lambda_r": final["lambda_r_ncr"],
                    "laplacian_ratio": final["laplacian_ratio_ncr"],
                    "expert_w_fc": final["expert_w_FC_ncr"],
                    "expert_w_sc": final["expert_w_SC_ncr"],
                    "alpha": final["alpha_ncr"],
                    "ncr_final_r": final["final_pearson_ncr"],
                    "ridge_final_r": final["final_pearson_ridge"],
                    "baseline_r": final["baseline_pearson"],
                }
                all_expert_selection.append(sel_row)

                # Save coefficients
                _save_coefficients(
                    seed, fold, task_key, prior_type,
                    final["fc_expert_ncr"], final["sc_expert_ncr"],
                    final["alpha_ncr"], final["expert_w_FC_ncr"], final["expert_w_SC_ncr"],
                )

                # Save checkpoint
                _save_ckpt(ckpt_key, seed, fold, {
                    "completed": True,
                    "split_rows": [row],
                    "expert_selection": [sel_row],
                })

    t_total = time.time() - t_start
    log.info("  Main pilot completed in %.1fs", t_total)

    # Save split_metrics.csv
    df = pd.DataFrame(all_split_rows)
    df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    log.info("  Saved split_metrics.csv (%d rows)", len(df))

    # Save expert_selection.csv
    sel_df = pd.DataFrame(all_expert_selection)
    sel_df.to_csv(OUTPUT_DIR / "expert_selection.csv", index=False)

    return df


# ---------------------------------------------------------------------------
# STEP 4: Save coefficients
# ---------------------------------------------------------------------------

def _save_coefficients(
    seed: int,
    fold: int,
    task_key: str,
    prior_type: str,
    fc_expert: ExpertResult,
    sc_expert: ExpertResult,
    alpha: float,
    expert_w_fc: float,
    expert_w_sc: float,
) -> None:
    """Save expert coefficients for one split."""
    coeffs = export_expert_coefficients(
        fc_expert, sc_expert, alpha, expert_w_fc, expert_w_sc, N_EDGE,
    )

    fname = f"seed_{seed}_fold_{fold}_{task_key}_{prior_type}.npz"
    np.savez_compressed(
        COEFF_DIR / fname,
        selected_edge_indices=coeffs["selected_edge_indices"],
        beta_fc_expert=coeffs["beta_fc_expert"],
        beta_sc_expert=coeffs["beta_sc_expert"],
        beta_fc_expert_weighted=coeffs["beta_fc_expert_weighted"],
        beta_sc_expert_weighted=coeffs["beta_sc_expert_weighted"],
        fc_scaler_mean=coeffs["fc_scaler_mean"],
        fc_scaler_scale=coeffs["fc_scaler_scale"],
        sc_scaler_mean=coeffs["sc_scaler_mean"],
        sc_scaler_scale=coeffs["sc_scaler_scale"],
        fc_y_mean=coeffs["fc_y_mean"],
        fc_y_std=coeffs["fc_y_std"],
        sc_y_mean=coeffs["sc_y_mean"],
        sc_y_std=coeffs["sc_y_std"],
    )


# ---------------------------------------------------------------------------
# STEP 5: Aggregate results
# ---------------------------------------------------------------------------

def aggregate_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate split metrics into seed-level and prior-control summaries."""
    # Seed-level aggregation
    seed_rows = []
    for seed in DEV_SEEDS:
        sdf = df[df["seed"] == seed]
        row = {"seed": seed}
        for task in ["WM", "FI"]:
            for pt in PRIOR_TYPES + ["baseline"]:
                if pt == "baseline":
                    tdf = sdf[(sdf["task"] == task)]
                    if len(tdf) > 0:
                        row[f"{task}_baseline_mean_r"] = tdf["baseline_pearson"].mean()
                        row[f"{task}_baseline_std_r"] = tdf["baseline_pearson"].std(ddof=1)
                else:
                    tdf = sdf[(sdf["task"] == task) & (sdf["prior_type"] == pt)]
                    if len(tdf) > 0:
                        row[f"{task}_{pt}_mean_r"] = tdf["final_pearson"].mean()
                        row[f"{task}_{pt}_std_r"] = tdf["final_pearson"].std(ddof=1)
                        row[f"{task}_{pt}_mean_delta"] = tdf["delta_final_vs_baseline"].mean()
                        row[f"{task}_{pt}_ridge_mean_r"] = tdf["ridge_final_pearson"].mean()
                        row[f"{task}_{pt}_ridge_mean_delta"] = tdf["ridge_delta_vs_baseline"].mean()
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)

    # Prior control summary
    ctrl_rows = []
    for task in ["WM", "FI"]:
        tdf = df[df["task"] == task]
        for pt in PRIOR_TYPES:
            ptdf = tdf[tdf["prior_type"] == pt]
            if len(ptdf) == 0:
                continue
            ctrl_rows.append({
                "task": task,
                "prior_type": pt,
                "mean_final_pearson": ptdf["final_pearson"].mean(),
                "std_final_pearson": ptdf["final_pearson"].std(ddof=1),
                "mean_delta_vs_baseline": ptdf["delta_final_vs_baseline"].mean(),
                "mean_expert_w_fc": ptdf["expert_w_FC"].mean(),
                "mean_expert_w_sc": ptdf["expert_w_SC"].mean(),
                "mean_alpha": ptdf["alpha_base_vs_expert"].mean(),
                "mean_ridge_final_pearson": ptdf["ridge_final_pearson"].mean(),
                "mean_ridge_delta": ptdf["ridge_delta_vs_baseline"].mean(),
            })

    ctrl_df = pd.DataFrame(ctrl_rows)
    ctrl_df.to_csv(OUTPUT_DIR / "prior_control_summary.csv", index=False)

    return seed_df, ctrl_df


# ---------------------------------------------------------------------------
# STEP 6: Plots
# ---------------------------------------------------------------------------

def generate_plots(df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    """Generate all 5 figures."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Figure 1: Model comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[df["task"] == task]
        models = ["baseline", "matched"]
        x = np.arange(len(DEV_SEEDS))
        width = 0.25

        baseline_means = []
        for seed in DEV_SEEDS:
            sdf = tdf[tdf["seed"] == seed]
            baseline_means.append(sdf["baseline_pearson"].mean())

        ncr_means = []
        for seed in DEV_SEEDS:
            sdf = tdf[(tdf["seed"] == seed) & (tdf["prior_type"] == "matched")]
            ncr_means.append(sdf["final_pearson"].mean() if len(sdf) > 0 else 0)

        ridge_means = []
        for seed in DEV_SEEDS:
            sdf = tdf[(tdf["seed"] == seed) & (tdf["prior_type"] == "matched")]
            ridge_means.append(sdf["ridge_final_pearson"].mean() if len(sdf) > 0 else 0)

        ax.bar(x - width, baseline_means, width, label="Strong baseline", alpha=0.8)
        ax.bar(x, ridge_means, width, label="Ridge expert fusion", alpha=0.8)
        ax.bar(x + width, ncr_means, width, label="PS-NCR-EF", alpha=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{task} — Model comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(DEV_SEEDS)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_phase2d_model_comparison.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Seed-level deltas ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        deltas = []
        for seed in DEV_SEEDS:
            sdf = tdf[tdf["seed"] == seed]
            deltas.append(sdf["delta_final_vs_baseline"].mean() if len(sdf) > 0 else 0)

        colors = ["green" if d > 0 else "red" for d in deltas]
        ax.bar(range(len(DEV_SEEDS)), deltas, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Delta Pearson r (PS-NCR-EF - baseline)")
        ax.set_title(f"{task} — Matched PS-NCR-EF delta")
        ax.set_xticks(range(len(DEV_SEEDS)))
        ax.set_xticklabels(DEV_SEEDS)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_phase2d_seed_deltas.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Expert weights ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        w_fc_vals = []
        w_sc_vals = []
        alpha_vals = []
        for seed in DEV_SEEDS:
            sdf = tdf[tdf["seed"] == seed]
            if len(sdf) > 0:
                w_fc_vals.append(sdf["expert_w_FC"].mean())
                w_sc_vals.append(sdf["expert_w_SC"].mean())
                alpha_vals.append(sdf["alpha_base_vs_expert"].mean())

        x = np.arange(len(w_fc_vals))
        width = 0.25
        if w_fc_vals:
            ax.bar(x - width, w_fc_vals, width, label="expert_w_FC", alpha=0.8)
            ax.bar(x, w_sc_vals, width, label="expert_w_SC", alpha=0.8)
            ax.bar(x + width, alpha_vals, width, label="alpha (base vs expert)", alpha=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Weight")
        ax.set_title(f"{task} — Expert fusion weights")
        ax.set_xticks(x)
        ax.set_xticklabels(DEV_SEEDS[:len(x)])
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_phase2d_expert_weights.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Prior controls ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[df["task"] == task]
        pt_means = []
        pt_stds = []
        for pt in PRIOR_TYPES:
            ptdf = tdf[tdf["prior_type"] == pt]
            pt_means.append(ptdf["final_pearson"].mean() if len(ptdf) > 0 else 0)
            pt_stds.append(ptdf["final_pearson"].std(ddof=1) if len(ptdf) > 1 else 0)

        # Add baseline
        base_mean = tdf["baseline_pearson"].mean()
        all_labels = PRIOR_TYPES + ["baseline"]
        all_means = pt_means + [base_mean]
        all_stds = pt_stds + [0]

        x = np.arange(len(all_labels))
        ax.bar(x, all_means, yerr=all_stds, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{task} — Prior control comparison")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_phase2d_prior_controls.png", dpi=150)
    plt.close(fig)

    # --- Figure 5: Mask selection ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        mask_families = tdf["selected_mask_family_fc"].value_counts()
        mask_sizes = tdf["selected_mask_size_fc"].values
        ratios = tdf["selected_laplacian_ratio_fc"].values

        # Two subplots: mask family pie and mask size/ratio scatter
        ax.set_title(f"{task} — Mask selection")
        if len(mask_families) > 0:
            ax.pie(mask_families.values, labels=mask_families.index, autopct="%1.0f%%")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fig_phase2d_mask_selection.png", dpi=150)
    plt.close(fig)

    log.info("  Generated 5 plots in %s", PLOT_DIR)


# ---------------------------------------------------------------------------
# STEP 7: Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(df: pd.DataFrame) -> Dict:
    """Compute expert-use diagnostics, NCR vs Ridge, prior specificity."""
    diag = {}

    for task in ["WM", "FI"]:
        tdf = df[df["task"] == task]

        # Expert-use diagnostics (from matched prior)
        matched = tdf[tdf["prior_type"] == "matched"]
        if len(matched) > 0:
            diag[f"{task}_matched_mean_alpha"] = float(matched["alpha_base_vs_expert"].mean())
            diag[f"{task}_matched_mean_expert_w_fc"] = float(matched["expert_w_FC"].mean())
            diag[f"{task}_matched_mean_expert_w_sc"] = float(matched["expert_w_SC"].mean())
            diag[f"{task}_matched_mean_pred_corr"] = float(matched["base_expert_prediction_corr"].mean())
            diag[f"{task}_matched_mean_error_corr"] = float(matched["base_expert_error_corr"].mean())

        # NCR vs Ridge expert comparison
        if len(matched) > 0:
            ncr_deltas = matched["delta_final_vs_baseline"].values
            ridge_deltas = matched["ridge_delta_vs_baseline"].values
            diag[f"{task}_ncr_mean_delta"] = float(np.mean(ncr_deltas))
            diag[f"{task}_ridge_mean_delta"] = float(np.mean(ridge_deltas))
            diag[f"{task}_ncr_vs_ridge_mean_delta"] = float(np.mean(ncr_deltas - ridge_deltas))
            diag[f"{task}_ncr_positive_seeds"] = int(np.sum(ncr_deltas > 0))
            diag[f"{task}_ridge_positive_seeds"] = int(np.sum(ridge_deltas > 0))

        # Prior specificity
        for ctrl in ["cross_task", "shuffled", "random"]:
            ctrl_df = tdf[tdf["prior_type"] == ctrl]
            if len(matched) > 0 and len(ctrl_df) > 0:
                matched_r = matched["final_pearson"].mean()
                ctrl_r = ctrl_df["final_pearson"].mean()
                diag[f"{task}_matched_minus_{ctrl}"] = float(matched_r - ctrl_r)

    return diag


# ---------------------------------------------------------------------------
# STEP 8: Final report
# ---------------------------------------------------------------------------

def print_final_report(
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    diag: Dict,
    audit_pass: bool,
    t_total: float,
) -> str:
    """Print all sections A-K and return the decision string."""
    import subprocess
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        git_head = "unknown"

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("Phase 2D: Prior-Selected Subspace NCR Expert Fusion Pilot — Final Report")
    lines.append("=" * 80)

    # A. Baseline correctness
    lines.append("")
    lines.append("## A. Baseline correctness")
    lines.append(f"  git HEAD: {git_head}")
    audit_path = OUTPUT_DIR / "BASELINE_AUDIT.json"
    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)
        lines.append(f"  WM fused r: {audit['wm_mean_pearson']:.6f} (expected ~0.2635) -> {'PASS' if audit['wm_pass'] else 'FAIL'}")
        lines.append(f"  FI fused r: {audit['fi_mean_pearson']:.6f} (expected ~0.3709) -> {'PASS' if audit['fi_pass'] else 'FAIL'}")
        lines.append(f"  Overall: {'PASS' if audit['overall_pass'] else 'FAIL'}")
    else:
        lines.append("  No audit file found")

    # B. Development setup
    lines.append("")
    lines.append("## B. Development setup")
    lines.append(f"  Seeds: {DEV_SEEDS}")
    lines.append(f"  Outer folds: {N_OUTER_FOLDS}")
    lines.append(f"  Fusion folds: {N_FUSION_FOLDS}")
    lines.append(f"  Inner folds: {N_INNER}")
    lines.append(f"  Subjects: {n_subjects}")
    lines.append(f"  Prior files: working_memory_contrastive_qwen3, fluid_intelligence_contrastive_qwen3")
    lines.append(f"  Runtime: {t_total:.1f}s")

    # C. Expert masks
    lines.append("")
    lines.append("## C. Expert masks")
    for task in ["WM", "FI"]:
        tdf = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        lines.append(f"  {task}:")
        if len(tdf) > 0:
            fam_counts = tdf["selected_mask_family_fc"].value_counts()
            for fam, cnt in fam_counts.items():
                lines.append(f"    {fam}: {cnt}/{len(tdf)}")
            lines.append(f"    Mean mask size: {tdf['selected_mask_size_fc'].mean():.0f}")
            lines.append(f"    Mean Laplacian ratio: {tdf['selected_laplacian_ratio_fc'].mean():.3f}")

    # D. Prediction results
    lines.append("")
    lines.append("## D. Prediction results")
    for task in ["WM", "FI"]:
        lines.append(f"  {task}:")
        tdf = df[df["task"] == task]
        for label, pt in [
            ("Strong no-prior baseline", None),
            ("Matched prior Ridge expert fusion", "matched"),
            ("Matched PS-NCR-EF", "matched"),
            ("Cross-task PS-NCR-EF", "cross_task"),
            ("Shuffled PS-NCR-EF", "shuffled"),
            ("Random PS-NCR-EF", "random"),
        ]:
            if pt is None:
                r = tdf["baseline_pearson"].mean()
                rmse = tdf["baseline_rmse"].mean()
                mae = tdf["baseline_mae"].mean()
                lines.append(f"    {label}: r={r:.4f} rmse={rmse:.4f} mae={mae:.4f}")
            elif pt == "matched":
                # Show Ridge
                rdf = tdf[tdf["prior_type"] == pt]
                r_ridge = rdf["ridge_final_pearson"].mean()
                lines.append(f"    {label}: r={r_ridge:.4f}")
                # Show NCR
                r = rdf["final_pearson"].mean()
                rmse = rdf["final_rmse"].mean()
                mae = rdf["final_mae"].mean()
                lines.append(f"    {label}: r={r:.4f} rmse={rmse:.4f} mae={mae:.4f}")
            else:
                rdf = tdf[tdf["prior_type"] == pt]
                r = rdf["final_pearson"].mean()
                rmse = rdf["final_rmse"].mean()
                mae = rdf["final_mae"].mean()
                lines.append(f"    {label}: r={r:.4f} rmse={rmse:.4f} mae={mae:.4f}")

    # E. Matched prediction gain
    lines.append("")
    lines.append("## E. Matched prediction gain")
    for task in ["WM", "FI"]:
        tdf = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        lines.append(f"  {task}:")
        deltas = []
        for seed in DEV_SEEDS:
            sdf = tdf[tdf["seed"] == seed]
            d = sdf["delta_final_vs_baseline"].mean() if len(sdf) > 0 else 0
            deltas.append(d)
            lines.append(f"    seed {seed}: delta={d:+.4f}")
        mean_d = float(np.mean(deltas))
        median_d = float(np.median(deltas))
        pos = int(np.sum(np.array(deltas) > 0))
        lines.append(f"    Mean: {mean_d:+.4f}, Median: {median_d:+.4f}, Positive: {pos}/4")

    # F. NCR contribution
    lines.append("")
    lines.append("## F. NCR contribution")
    for task in ["WM", "FI"]:
        tdf = df[(df["task"] == task) & (tdf["prior_type"] == "matched")]
        ncr_d = tdf["delta_final_vs_baseline"].values
        ridge_d = tdf["ridge_delta_vs_baseline"].values
        ncr_vs_ridge = ncr_d - ridge_d
        lines.append(f"  {task}:")
        lines.append(f"    PS-NCR-EF minus Ridge delta: mean={float(np.mean(ncr_vs_ridge)):+.4f}")
        lines.append(f"    Positive seeds: {int(np.sum(ncr_vs_ridge > 0))}/4")
        ratios = tdf["selected_laplacian_ratio_fc"].values
        ratio_counts = Counter(ratios)
        lines.append(f"    Laplacian ratio frequencies: {dict(ratio_counts)}")

    # G. Prior specificity
    lines.append("")
    lines.append("## G. Prior specificity")
    for task in ["WM", "FI"]:
        lines.append(f"  {task}:")
        matched_r = df[(df["task"] == task) & (df["prior_type"] == "matched")]["final_pearson"].mean()
        for ctrl in ["cross_task", "shuffled", "random"]:
            ctrl_r = df[(df["task"] == task) & (df["prior_type"] == ctrl)]["final_pearson"].mean()
            lines.append(f"    matched - {ctrl}: {matched_r - ctrl_r:+.4f}")

    # H. Expert mechanism
    lines.append("")
    lines.append("## H. Expert mechanism")
    for task in ["WM", "FI"]:
        matched = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        lines.append(f"  {task}:")
        lines.append(f"    Expert FC weight: {matched['expert_w_FC'].mean():.4f}")
        lines.append(f"    Expert SC weight: {matched['expert_w_SC'].mean():.4f}")
        lines.append(f"    Final alpha: {matched['alpha_base_vs_expert'].mean():.4f}")
        lines.append(f"    Mean pred corr: {matched['base_expert_prediction_corr'].mean():.4f}")
        lines.append(f"    Mean error corr: {matched['base_expert_error_corr'].mean():.4f}")

    # I. Coefficient validation
    lines.append("")
    lines.append("## I. Coefficient validation")
    coeff_files = list(COEFF_DIR.glob("*.npz"))
    lines.append(f"  Files exported: {len(coeff_files)}")
    if coeff_files:
        sample = np.load(coeff_files[0])
        lines.append(f"  Full edge dimensions: {sample['beta_fc_expert'].shape}")
        max_errs = []
        for cf in coeff_files[:5]:
            data = np.load(cf)
            # Check shapes
            if "beta_fc_expert" in data:
                lines.append(f"    {cf.name}: beta_fc shape={data['beta_fc_expert'].shape}, "
                           f"selected_edges={len(data['selected_edge_indices'])}")
    lines.append("  Max primal reconstruction error: see split_metrics.csv")

    # J. Tests
    lines.append("")
    lines.append("## J. Tests")
    lines.append("  (Run separately: pytest tests/test_prior_subspace_expert_fusion.py)")

    # K. Outputs
    lines.append("")
    lines.append("## K. Outputs")
    lines.append(f"  Directory: {OUTPUT_DIR}")
    zip_path = OUTPUT_DIR / "phase2d_all_outputs.zip"
    lines.append(f"  ZIP: {zip_path}")
    lines.append(f"  CSVs: split_metrics.csv, seed_metrics.csv, expert_selection.csv, prior_control_summary.csv")
    lines.append(f"  Plots: {PLOT_DIR}")
    lines.append(f"  Coefficients: {COEFF_DIR}")

    # Decision
    lines.append("")
    lines.append("=" * 80)

    # Compute decision
    decision = _compute_decision(df, seed_df, diag)
    lines.append(f"PHASE2D_DECISION: {decision}")
    lines.append("")
    lines.append("STATUS: PHASE2D_PS_NCR_EXPERT_FUSION_COMPLETE")

    report = "\n".join(lines)
    print(report)

    # Save report
    with open(OUTPUT_DIR / "RUN_REPORT.md", "w") as f:
        f.write(report)

    return decision


def _compute_decision(df: pd.DataFrame, seed_df: pd.DataFrame, diag: Dict) -> str:
    """Compute the predefined pilot decision."""
    # For each target, compute matched PS-NCR-EF mean delta and positive seeds
    for task in ["WM", "FI"]:
        matched = df[(df["task"] == task) & (df["prior_type"] == "matched")]
        shuffled = df[(df["task"] == task) & (df["prior_type"] == "shuffled")]
        random = df[(df["task"] == task) & (df["prior_type"] == "random")]

        if len(matched) == 0:
            return "NO_GO"

        deltas = matched["delta_final_vs_baseline"].values
        mean_delta = float(np.mean(deltas))
        pos_seeds = int(np.sum(deltas > 0))

        task_mean_r = matched["final_pearson"].mean()
        shuffled_r = shuffled["final_pearson"].mean() if len(shuffled) > 0 else 0
        random_r = random["final_pearson"].mean() if len(random) > 0 else 0

        # Check basic conditions
        if mean_delta < 0 or pos_seeds < 2:
            return "NO_GO"

    # Both targets must satisfy conditions for STRONG_GO
    wm_matched = df[(df["task"] == "WM") & (df["prior_type"] == "matched")]
    fi_matched = df[(df["task"] == "FI") & (df["prior_type"] == "matched")]
    wm_shuffled = df[(df["task"] == "WM") & (df["prior_type"] == "shuffled")]
    wm_random = df[(df["task"] == "WM") & (df["prior_type"] == "random")]
    fi_shuffled = df[(df["task"] == "FI") & (df["prior_type"] == "shuffled")]
    fi_random = df[(df["task"] == "FI") & (df["prior_type"] == "random")]

    wm_deltas = wm_matched["delta_final_vs_baseline"].values
    fi_deltas = fi_matched["delta_final_vs_baseline"].values
    wm_mean_delta = float(np.mean(wm_deltas))
    fi_mean_delta = float(np.mean(fi_deltas))
    wm_pos = int(np.sum(wm_deltas > 0))
    fi_pos = int(np.sum(fi_deltas > 0))

    wm_r = wm_matched["final_pearson"].mean()
    fi_r = fi_matched["final_pearson"].mean()
    wm_shuf_r = wm_shuffled["final_pearson"].mean() if len(wm_shuffled) > 0 else 0
    wm_rand_r = wm_random["final_pearson"].mean() if len(wm_random) > 0 else 0
    fi_shuf_r = fi_shuffled["final_pearson"].mean() if len(fi_shuffled) > 0 else 0
    fi_rand_r = fi_random["final_pearson"].mean() if len(fi_random) > 0 else 0

    # STRONG_GO: both targets >= +0.003, positive >= 3/4, at least one >= +0.005
    # and matched > shuffled/random for both
    strong_wm = wm_mean_delta >= 0.003 and wm_pos >= 3
    strong_fi = fi_mean_delta >= 0.003 and fi_pos >= 3
    either_strong = wm_mean_delta >= 0.005 or fi_mean_delta >= 0.005
    specificity_wm = wm_r > wm_shuf_r and wm_r > wm_rand_r
    specificity_fi = fi_r > fi_shuf_r and fi_r > fi_rand_r

    if strong_wm and strong_fi and either_strong and specificity_wm and specificity_fi:
        return "STRONG_GO"

    # PROMISING_GO: WM >= +0.004, pos >= 3/4, FI >= 0; or vice versa
    promising_wm = wm_mean_delta >= 0.004 and wm_pos >= 3
    promising_fi = fi_mean_delta >= 0
    promising_wm2 = wm_mean_delta >= 0
    promising_fi2 = fi_mean_delta >= 0.004 and fi_pos >= 3

    if (promising_wm and promising_fi) or (promising_wm2 and promising_fi2):
        # Check matched at least numerically better than controls overall
        if specificity_wm or specificity_fi:
            return "PROMISING_GO"

    # RIDGE_EXPERT_GO: Ridge expert fusion helps but NCR doesn't
    ridge_helps = False
    ncr_helps = False
    for task in ["WM", "FI"]:
        tdf = df[df["task"] == task]
        matched = tdf[tdf["prior_type"] == "matched"]
        if len(matched) > 0:
            ridge_delta = matched["ridge_delta_vs_baseline"].mean()
            ncr_delta = matched["delta_final_vs_baseline"].mean()
            if ridge_delta > 0.002:
                ridge_helps = True
            if ncr_delta > 0.002:
                ncr_helps = True

    if ridge_helps and not ncr_helps:
        return "RIDGE_EXPERT_GO"

    return "NO_GO"


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def package_outputs() -> None:
    """Package all outputs into a ZIP archive."""
    zip_path = OUTPUT_DIR / "phase2d_all_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in [CKPT_DIR, COEFF_DIR, PRED_DIR, PLOT_DIR]:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    zf.write(p, str(p.relative_to(OUTPUT_DIR)))
        for name in [
            "split_metrics.csv", "seed_metrics.csv",
            "expert_selection.csv", "prior_control_summary.csv",
            "BASELINE_AUDIT.json", "RUN_REPORT.md",
        ]:
            p = OUTPUT_DIR / name
            if p.exists():
                zf.write(p, name)
    log.info("  Packaged outputs to %s", zip_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Phase 2D pilot."""
    log.info("=" * 80)
    log.info("Phase 2D: Prior-Selected Subspace NCR Expert Fusion Pilot")
    log.info("=" * 80)

    t_global_start = time.time()

    # STEP 1: Baseline audit
    audit_pass = run_baseline_audit()
    if not audit_pass:
        log.error("Baseline audit failed. Aborting.")
        sys.exit(1)

    # STEP 2-3: Main pilot loop
    df = run_main_pilot()

    # STEP 5: Aggregate results
    seed_df, ctrl_df = aggregate_results(df)

    # STEP 6: Plots
    generate_plots(df, seed_df)

    # STEP 7: Diagnostics
    diag = compute_diagnostics(df)
    diag_path = OUTPUT_DIR / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    # STEP 8: Final report
    t_total = time.time() - t_global_start
    decision = print_final_report(df, seed_df, ctrl_df, diag, audit_pass, t_total)

    # Package outputs
    package_outputs()

    # Mark complete
    with open(OUTPUT_DIR / "COMPLETE", "w") as f:
        f.write(f"Decision: {decision}\nRuntime: {t_total:.1f}s\n")

    log.info("")
    log.info("=" * 80)
    log.info("Phase 2D complete. Decision: %s", decision)
    log.info("Runtime: %.1fs", t_total)
    log.info("Outputs: %s", OUTPUT_DIR)
    log.info("=" * 80)


if __name__ == "__main__":
    main()
