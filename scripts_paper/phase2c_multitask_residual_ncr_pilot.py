"""Phase 2C: Multi-Task Residual NCR Pilot.

Compares five model families across 20 paired partitions (4 seeds × 5 folds):
  1. Base late fusion (FC + SC)
  2. Individual residual, no-prior (λ₂=0)
  3. Individual residual, LLM prior (λ₂>0)
  4. MT residual, no-prior (shared+contrast, λ₂=0)
  5. MT residual, LLM prior (shared+contrast, λ₂>0)

Seeds: [909, 1010, 1111, 1212] (not reused from Phase 2A/2A.1/2B).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (
    make_outer_splits,
    search_fusion_weights,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    NetworkConstrainedRidge,
    build_edge_laplacian,
    factor_laplacian_eig,
)
from metascfc.models.multitask_residual_ncr import (
    N_EDGE,
    aggregate_edge_to_roi,
    build_contrast_prior,
    build_shared_contrast_residuals,
    build_shared_prior,
    compute_coefficient_stability,
    convert_coefficients_to_original,
    crossfit_ncr_residual,
    fit_final_ncr_and_predict,
    fit_ncr_residual,
    reconstruct_task_residuals,
    select_eta,
    select_eta_mt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2c_mt_residual_ncr_pilot"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
COEFF_DIR = OUTPUT_DIR / "coefficients"
PLOT_DIR = OUTPUT_DIR / "plots"

for d in [OUTPUT_DIR, CKPT_DIR, COEFF_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────
log.info("Loading data...")
fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
N_ROI = 116
iu = np.triu_indices(N_ROI, k=1)
X_fc_raw = fc_mats[:, iu[0], iu[1]].astype(np.float64)
X_sc_raw = sc_mats[:, iu[0], iu[1]].astype(np.float64)
y_wm = np.load(
    REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
).astype(np.float64)
y_fi = np.load(
    REPO_ROOT / "inputs/dataset_SC/label_all.npy"
).astype(np.float64)

n_subjects = int(y_wm.shape[0])

wm_prior_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)
fi_prior_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
)
wm_prior = wm_prior_df["prior_score"].values.astype(np.float64)
fi_prior = fi_prior_df["prior_score"].values.astype(np.float64)
roi_names = wm_prior_df["roi_label"].tolist()

TASKS = {
    "working_memory": {"y": y_wm, "prior": wm_prior, "display": "WM"},
    "fluid_intelligence": {"y": y_fi, "prior": fi_prior, "display": "FI"},
}

# ── Hyperparameter grids ─────────────────────────────────────────────────
LAMBDA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ETA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
ALPHA_FC_GRID = [1.0, 10.0, 100.0]
ALPHA_SC_GRID = [1.0, 10.0, 100.0]

DEV_SEEDS = [909, 1010, 1111, 1212]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

TOP_K_ROI = 10


# ── Helpers ───────────────────────────────────────────────────────────────
def _ckpt_path(seed: int, fold: int) -> Path:
    return CKPT_DIR / f"seed_{seed}_fold_{fold}.pkl"


def _save_ckpt(seed: int, fold: int, state: dict) -> None:
    with open(_ckpt_path(seed, fold), "wb") as f:
        pickle.dump(state, f)


def _load_ckpt(seed: int, fold: int) -> Optional[dict]:
    p = _ckpt_path(seed, fold)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _make_inner_folds(n: int, n_inner: int, rng: np.random.RandomState) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic inner CV folds (local indices)."""
    perm = rng.permutation(n)
    fold_sizes = np.full(n_inner, n // n_inner)
    fold_sizes[: n % n_inner] += 1
    folds = []
    current = 0
    for k in range(n_inner):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        folds.append((a_local, v_local))
        current = stop
    return folds


def _generate_fusion_oof(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    fold: int,
) -> Tuple[np.ndarray, float, float]:
    """Cross-fit late fusion (FC+SC) OOF predictions on training set.

    Returns (oof_preds, alpha_fc_best, alpha_sc_best).
    """
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_sc_z = scaler_sc.fit_transform(X_sc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_z = (y[train_idx] - y_mean) / y_std

    n_train = len(train_idx)
    inner_rng = np.random.RandomState(30000 + 100 * seed + fold)
    inner_folds = _make_inner_folds(n_train, N_INNER_FOLDS, inner_rng)

    best_score = -np.inf
    best_alpha_fc, best_alpha_sc = 1.0, 1.0
    best_oof = None

    for alpha_fc in ALPHA_FC_GRID:
        for alpha_sc in ALPHA_SC_GRID:
            oof_fc = np.full(n_train, np.nan)
            oof_sc = np.full(n_train, np.nan)

            for a_idx, v_idx in inner_folds:
                # FC
                model_fc = Ridge(alpha=alpha_fc, fit_intercept=False)
                model_fc.fit(X_fc_z[a_idx], y_z[a_idx])
                oof_fc[v_idx] = model_fc.predict(X_fc_z[v_idx])
                # SC
                model_sc = Ridge(alpha=alpha_sc, fit_intercept=False)
                model_sc.fit(X_sc_z[a_idx], y_z[a_idx])
                oof_sc[v_idx] = model_sc.predict(X_sc_z[v_idx])

            # Search fusion weight
            w_grid = np.linspace(0, 1, 21)
            best_w = 0.5
            best_r = -np.inf
            for w in w_grid:
                comb = w * oof_fc + (1 - w) * oof_sc
                r, _ = pearsonr(comb, y_z)
                if r > best_r:
                    best_r = r
                    best_w = w

            if best_r > best_score:
                best_score = best_r
                best_alpha_fc = alpha_fc
                best_alpha_sc = alpha_sc
                best_oof = best_w * oof_fc + (1 - best_w) * oof_sc

    return best_oof * y_std + y_mean, best_alpha_fc, best_alpha_sc


def _fit_base_fusion_and_residuals(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y_wm: np.ndarray,
    y_fi: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha_fc: float,
    alpha_sc: float,
    seed: int,
    fold: int,
) -> Tuple[Dict, Dict]:
    """Fit base late fusion on full training, predict on test, compute residuals."""
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_tr = scaler_fc.fit_transform(X_fc[train_idx])
    X_sc_tr = scaler_sc.fit_transform(X_sc[train_idx])
    X_fc_te = scaler_fc.transform(X_fc[test_idx])
    X_sc_te = scaler_sc.transform(X_sc[test_idx])

    y_wm_mean = float(y_wm[train_idx].mean())
    y_fi_mean = float(y_fi[train_idx].mean())
    y_wm_std = max(float(y_wm[train_idx].std()), 1e-8)
    y_fi_std = max(float(y_fi[train_idx].std()), 1e-8)
    y_wm_z = (y_wm[train_idx] - y_wm_mean) / y_wm_std
    y_fi_z = (y_fi[train_idx] - y_fi_mean) / y_fi_std

    # Inner selection for fusion weights
    inner_rng = np.random.RandomState(50000 + 100 * seed + fold)
    inner_folds = _make_inner_folds(len(train_idx), N_INNER_FOLDS, inner_rng)

    oof_fc_wm = np.full(len(train_idx), np.nan)
    oof_sc_wm = np.full(len(train_idx), np.nan)
    oof_fc_fi = np.full(len(train_idx), np.nan)
    oof_sc_fi = np.full(len(train_idx), np.nan)

    model_fc_wm = Ridge(alpha=alpha_fc, fit_intercept=False)
    model_sc_wm = Ridge(alpha=alpha_sc, fit_intercept=False)
    model_fc_fi = Ridge(alpha=alpha_fc, fit_intercept=False)
    model_sc_fi = Ridge(alpha=alpha_sc, fit_intercept=False)

    for a_idx, v_idx in inner_folds:
        # WM
        model_fc_wm.fit(X_fc_tr[a_idx], y_wm_z[a_idx])
        oof_fc_wm[v_idx] = model_fc_wm.predict(X_fc_tr[v_idx])
        model_sc_wm.fit(X_sc_tr[a_idx], y_wm_z[a_idx])
        oof_sc_wm[v_idx] = model_sc_wm.predict(X_sc_tr[v_idx])
        # FI
        model_fc_fi.fit(X_fc_tr[a_idx], y_fi_z[a_idx])
        oof_fc_fi[v_idx] = model_fc_fi.predict(X_fc_tr[v_idx])
        model_sc_fi.fit(X_sc_tr[a_idx], y_fi_z[a_idx])
        oof_sc_fi[v_idx] = model_sc_fi.predict(X_sc_tr[v_idx])

    # Search fusion weights for WM and FI
    w_grid = np.linspace(0, 1, 21)
    best_w_wm = 0.5
    best_r_wm = -np.inf
    best_w_fi = 0.5
    best_r_fi = -np.inf
    for w in w_grid:
        r_wm, _ = pearsonr(w * oof_fc_wm + (1 - w) * oof_sc_wm, y_wm_z)
        r_fi, _ = pearsonr(w * oof_fc_fi + (1 - w) * oof_sc_fi, y_fi_z)
        if r_wm > best_r_wm:
            best_r_wm = r_wm
            best_w_wm = w
        if r_fi > best_r_fi:
            best_r_fi = r_fi
            best_w_fi = w

    # Final models on full training
    model_fc_wm.fit(X_fc_tr, y_wm_z)
    model_sc_wm.fit(X_sc_tr, y_wm_z)
    model_fc_fi.fit(X_fc_tr, y_fi_z)
    model_sc_fi.fit(X_sc_tr, y_fi_z)

    # Base fusion predictions (train and test)
    train_pred_wm_z = best_w_wm * model_fc_wm.predict(X_fc_tr) + (1 - best_w_wm) * model_sc_wm.predict(X_sc_tr)
    train_pred_fi_z = best_w_fi * model_fc_fi.predict(X_fc_tr) + (1 - best_w_fi) * model_sc_fi.predict(X_sc_tr)
    test_pred_wm_z = best_w_wm * model_fc_wm.predict(X_fc_te) + (1 - best_w_wm) * model_sc_wm.predict(X_sc_te)
    test_pred_fi_z = best_w_fi * model_fc_fi.predict(X_fc_te) + (1 - best_w_fi) * model_sc_fi.predict(X_sc_te)

    train_pred_wm = train_pred_wm_z * y_wm_std + y_wm_mean
    train_pred_fi = train_pred_fi_z * y_fi_std + y_fi_mean
    test_pred_wm = test_pred_wm_z * y_wm_std + y_wm_mean
    test_pred_fi = test_pred_fi_z * y_fi_std + y_fi_mean

    # Standardized residuals on training set
    r_wm_train = y_wm[train_idx] - train_pred_wm
    r_fi_train = y_fi[train_idx] - train_pred_fi
    r_wm_train_z = (r_wm_train - r_wm_train.mean()) / max(r_wm_train.std(), 1e-8)
    r_fi_train_z = (r_fi_train - r_fi_train.mean()) / max(r_fi_train.std(), 1e-8)

    base_info = {
        "test_wm": test_pred_wm,
        "test_fi": test_pred_fi,
        "train_r_wm": r_wm_train,
        "train_r_fi": r_fi_train,
        "train_r_wm_z": r_wm_train_z,
        "train_r_fi_z": r_fi_train_z,
        "w_wm": best_w_wm,
        "w_fi": best_w_fi,
        "alpha_fc": alpha_fc,
        "alpha_sc": alpha_sc,
    }
    return base_info


def _select_lambda2(
    X_fc: np.ndarray,
    y_res: np.ndarray,
    edge_laplacian,
    lap_eig,
    lambda1: float,
    train_idx: np.ndarray,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
    n_train: int,
) -> float:
    """Select lambda2 by inner CV on residual prediction.

    X_fc: full-dataset FC features
    y_res: training-subset residuals (length n_train)
    train_idx: global training indices
    inner_folds: local indices (0..n_train-1)
    """
    X_fc_train = X_fc[train_idx]

    best_lambda2 = 0.0
    best_r = -np.inf

    for lambda2 in [0.0, 0.1, 1.0, 10.0]:
        oof = np.full(n_train, np.nan)
        for a_idx, v_idx in inner_folds:
            pred, _ = fit_ncr_residual(
                X_fc_train, y_res, edge_laplacian, lap_eig,
                lambda1, lambda2, a_idx, v_idx,
            )
            oof[v_idx] = pred

        mask = np.isfinite(oof)
        if mask.sum() < 10:
            continue
        r, _ = pearsonr(oof[mask], y_res[mask])
        if r > best_r + 1e-14:
            best_r = r
            best_lambda2 = lambda2

    return best_lambda2


def run_phase2c_pilot() -> None:
    """Main entry: run all 20 paired partitions and generate outputs."""
    log.info("=" * 80)
    log.info("Phase 2C: Multi-Task Residual NCR Pilot")
    log.info("=" * 80)
    log.info("Seeds: %s", DEV_SEEDS)
    log.info("Outer folds: %d", N_OUTER_FOLDS)
    log.info("Targets: WM, FI")
    log.info("Total splits: %d", len(DEV_SEEDS) * N_OUTER_FOLDS * 2)
    log.info("")

    # ── Dataset audit ─────────────────────────────────────────────────────
    log.info("Dataset audit:")
    log.info("  Subjects: %d", n_subjects)
    log.info("  FC features: %d", X_fc_raw.shape[1])
    log.info("  SC features: %d", X_sc_raw.shape[1])
    log.info("  WM prior: min=%.4f max=%.4f", wm_prior.min(), wm_prior.max())
    log.info("  FI prior: min=%.4f max=%.4f", fi_prior.min(), fi_prior.max())
    top_wm = np.argsort(wm_prior)[-TOP_K_ROI:][::-1]
    top_fi = np.argsort(fi_prior)[-TOP_K_ROI:][::-1]
    log.info("  Top-%d WM ROIs: %s", TOP_K_ROI, [roi_names[i] for i in top_wm])
    log.info("  Top-%d FI ROIs: %s", TOP_K_ROI, [roi_names[i] for i in top_fi])
    overlap = len(set(top_wm) & set(top_fi))
    log.info("  Overlap: %d / %d", overlap, TOP_K_ROI)
    log.info("")

    # ── Build Laplacians for task-specific and shared/contrast ─────────────
    log.info("Building Laplacians...")
    edge_laplacian_wm = build_edge_laplacian(N_ROI, prior_scores=wm_prior)
    edge_laplacian_fi = build_edge_laplacian(N_ROI, prior_scores=fi_prior)
    shared_prior = build_shared_prior(wm_prior, fi_prior)
    contrast_prior = build_contrast_prior(wm_prior, fi_prior)
    edge_laplacian_shared = build_edge_laplacian(N_ROI, prior_scores=shared_prior)
    edge_laplacian_contrast = build_edge_laplacian(N_ROI, prior_scores=contrast_prior)
    log.info("  Laplacians built: WM, FI, shared, contrast")
    log.info("")

    # ── Storage ───────────────────────────────────────────────────────────
    split_rows: List[Dict] = []
    seed_rows: List[Dict] = []
    model_names = [
        "base_late_fusion",
        "ind_residual_no_prior",
        "ind_residual_prior",
        "mt_residual_no_prior",
        "mt_residual_prior",
    ]

    # ── Paired outer splits ──────────────────────────────────────────────
    log.info("Generating paired outer splits...")
    fold_specs: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed in DEV_SEEDS:
        outer = make_outer_splits(n_subjects, [seed], N_OUTER_FOLDS)
        for s, fold, train_idx, test_idx in outer:
            fold_specs.append((s, fold, train_idx, test_idx))
    log.info("  %d fold-specs generated", len(fold_specs))
    log.info("")

    # ── Main loop ─────────────────────────────────────────────────────────
    t_start = time.time()

    for seed, fold, train_idx, test_idx in fold_specs:
        ckpt = _load_ckpt(seed, fold)
        if ckpt is not None and ckpt.get("completed", False):
            log.info("[seed=%d fold=%d] Loaded checkpoint (completed)", seed, fold)
            split_rows.extend(ckpt["split_rows"])
            continue

        t_split = time.time()
        log.info("[seed=%d fold=%d] Processing... (train=%d, test=%d)",
                 seed, fold, len(train_idx), len(test_idx))

        # ── Step 1: Base late-fusion OOF to select alpha_fc, alpha_sc ────
        oof_wm, alpha_fc_wm, alpha_sc_wm = _generate_fusion_oof(
            X_fc_raw, X_sc_raw, y_wm, train_idx, seed, fold,
        )
        oof_fi, alpha_fc_fi, alpha_sc_fi = _generate_fusion_oof(
            X_fc_raw, X_sc_raw, y_fi, train_idx, seed, fold,
        )

        # Use average alpha across tasks
        alpha_fc = int(np.round(np.mean([alpha_fc_wm, alpha_fc_fi])))
        alpha_sc = int(np.round(np.mean([alpha_sc_wm, alpha_sc_fi])))

        # ── Step 2: Fit base fusion, get residuals ───────────────────────
        base_info = _fit_base_fusion_and_residuals(
            X_fc_raw, X_sc_raw, y_wm, y_fi,
            train_idx, test_idx, alpha_fc, alpha_sc, seed, fold,
        )

        base_r = pearsonr(y_wm[test_idx], base_info["test_wm"])[0]
        base_fi = pearsonr(y_fi[test_idx], base_info["test_fi"])[0]
        log.info("  Base late fusion: WM r=%.4f, FI r=%.4f", base_r, base_fi)

        # ── Step 3: Shared/contrast decomposition ────────────────────────
        z_shared, z_contrast = build_shared_contrast_residuals(
            base_info["train_r_wm_z"], base_info["train_r_fi_z"]
        )

        # ── Step 4: Individual residual models ───────────────────────────
        inner_rng = np.random.RandomState(40000 + 100 * seed + fold)
        inner_folds = _make_inner_folds(len(train_idx), N_INNER_FOLDS, inner_rng)

        # WM individual
        lam2_wm = _select_lambda2(
            X_fc_raw, base_info["train_r_wm_z"],
            edge_laplacian_wm, None, alpha_fc,
            train_idx, inner_folds, len(train_idx),
        )
        # FI individual
        lam2_fi = _select_lambda2(
            X_fc_raw, base_info["train_r_fi_z"],
            edge_laplacian_fi, None, alpha_fc,
            train_idx, inner_folds, len(train_idx),
        )

        # OOF residual predictions for individual models
        ind_res_wm_oof = crossfit_ncr_residual(
            X_fc_raw, base_info["train_r_wm_z"],
            edge_laplacian_wm if lam2_wm > 0 else None, None,
            alpha_fc, lam2_wm, train_idx, seed, fold,
        )
        ind_res_fi_oof = crossfit_ncr_residual(
            X_fc_raw, base_info["train_r_fi_z"],
            edge_laplacian_fi if lam2_fi > 0 else None, None,
            alpha_fc, lam2_fi, train_idx, seed, fold,
        )

        # ── Step 5: MT residual models ───────────────────────────────────
        # MT no-prior
        z_shared_oof_np = crossfit_ncr_residual(
            X_fc_raw, z_shared, None, None,
            alpha_fc, 0.0, train_idx, seed, fold,
        )
        z_contrast_oof_np = crossfit_ncr_residual(
            X_fc_raw, z_contrast, None, None,
            alpha_fc, 0.0, train_idx, seed, fold,
        )
        mt_np_wm_oof, mt_np_fi_oof = reconstruct_task_residuals(
            z_shared_oof_np, z_contrast_oof_np
        )

        # MT with prior
        z_shared_oof_p = crossfit_ncr_residual(
            X_fc_raw, z_shared,
            edge_laplacian_shared, None,
            alpha_fc, 1.0, train_idx, seed, fold,
        )
        z_contrast_oof_p = crossfit_ncr_residual(
            X_fc_raw, z_contrast,
            edge_laplacian_contrast, None,
            alpha_fc, 1.0, train_idx, seed, fold,
        )
        mt_p_wm_oof, mt_p_fi_oof = reconstruct_task_residuals(
            z_shared_oof_p, z_contrast_oof_p
        )

        # ── Step 6: Select etas ──────────────────────────────────────────
        # OOF predictions are in z-score space. Compare against standardized y.
        y_wm_z = (y_wm[train_idx] - y_wm[train_idx].mean()) / max(y_wm[train_idx].std(), 1e-8)
        y_fi_z = (y_fi[train_idx] - y_fi[train_idx].mean()) / max(y_fi[train_idx].std(), 1e-8)

        # Individual residual no-prior
        eta_ind_np_wm, _ = select_eta(
            base_info["train_r_wm_z"], ind_res_wm_oof,
            y_wm_z, ETA_GRID,
        )
        eta_ind_np_fi, _ = select_eta(
            base_info["train_r_fi_z"], ind_res_fi_oof,
            y_fi_z, ETA_GRID,
        )

        # Individual residual + prior
        eta_ind_p_wm, _ = select_eta(
            base_info["train_r_wm_z"], ind_res_wm_oof,
            y_wm_z, ETA_GRID,
        )
        eta_ind_p_fi, _ = select_eta(
            base_info["train_r_fi_z"], ind_res_fi_oof,
            y_fi_z, ETA_GRID,
        )

        # MT no-prior (task-specific etas via Fisher-z)
        eta_mt_np_wm, eta_mt_np_fi, _ = select_eta_mt(
            base_info["train_r_wm_z"], base_info["train_r_fi_z"],
            mt_np_wm_oof, mt_np_fi_oof,
            y_wm_z, y_fi_z,
            ETA_GRID,
        )

        # MT with prior (task-specific etas via Fisher-z)
        eta_mt_p_wm, eta_mt_p_fi, _ = select_eta_mt(
            base_info["train_r_wm_z"], base_info["train_r_fi_z"],
            mt_p_wm_oof, mt_p_fi_oof,
            y_wm_z, y_fi_z,
            ETA_GRID,
        )

        log.info("  Eta selected: ind_np=(%.2f, %.2f), ind_p=(%.2f, %.2f), "
                 "mt_np=(%.2f, %.2f), mt_p=(%.2f, %.2f)",
                 eta_ind_np_wm, eta_ind_np_fi, eta_ind_p_wm, eta_ind_p_fi,
                 eta_mt_np_wm, eta_mt_np_fi, eta_mt_p_wm, eta_mt_p_fi)

        # ── Step 7: Final refit and predict on test ──────────────────────
        # We need final models for each approach. The "combined" prediction
        # is base + eta * residual_novel.

        # Individual residual no-prior (final)
        final_ind_np_wm = fit_final_ncr_and_predict(
            X_fc_raw, base_info["train_r_wm_z"],
            None, alpha_fc, 0.0, train_idx, test_idx,
        )
        final_ind_np_fi = fit_final_ncr_and_predict(
            X_fc_raw, base_info["train_r_fi_z"],
            None, alpha_fc, 0.0, train_idx, test_idx,
        )

        # Individual residual + prior (final)
        final_ind_p_wm = fit_final_ncr_and_predict(
            X_fc_raw, base_info["train_r_wm_z"],
            edge_laplacian_wm, alpha_fc, lam2_wm, train_idx, test_idx,
        )
        final_ind_p_fi = fit_final_ncr_and_predict(
            X_fc_raw, base_info["train_r_fi_z"],
            edge_laplacian_fi, alpha_fc, lam2_fi, train_idx, test_idx,
        )

        # MT no-prior (final shared + contrast)
        final_mt_shared_np = fit_final_ncr_and_predict(
            X_fc_raw, z_shared,
            None, alpha_fc, 0.0, train_idx, test_idx,
        )
        final_mt_contrast_np = fit_final_ncr_and_predict(
            X_fc_raw, z_contrast,
            None, alpha_fc, 0.0, train_idx, test_idx,
        )
        final_mt_wm_np, final_mt_fi_np = reconstruct_task_residuals(
            final_mt_shared_np, final_mt_contrast_np
        )

        # MT + prior (final shared + contrast)
        final_mt_shared_p = fit_final_ncr_and_predict(
            X_fc_raw, z_shared,
            edge_laplacian_shared, alpha_fc, 1.0, train_idx, test_idx,
        )
        final_mt_contrast_p = fit_final_ncr_and_predict(
            X_fc_raw, z_contrast,
            edge_laplacian_contrast, alpha_fc, 1.0, train_idx, test_idx,
        )
        final_mt_wm_p, final_mt_fi_p = reconstruct_task_residuals(
            final_mt_shared_p, final_mt_contrast_p
        )

        # ── Convert all test predictions to original scale ────────────────
        # fit_final_ncr_and_predict returns predictions in original units.
        # OOF residuals are z-scored (space of train_r_wm_z).
        # For select_eta we compare base_z + eta*residual_z vs y_z → correlation is unit-free.
        # For test: base_orig + eta * residual_orig (both in original units).
        scaler_fc = StandardScaler()
        scaler_fc.fit(X_fc_raw[train_idx])
        y_wm_std = max(float(y_wm[train_idx].std()), 1e-8)
        y_fi_std = max(float(y_fi[train_idx].std()), 1e-8)

        # Base predictions (original scale)
        base_test_wm = base_info["test_wm"]
        base_test_fi = base_info["test_fi"]

        # Combined predictions: base + eta * residual
        # (fit_final_ncr_and_predict returns original-scale residual predictions)
        pred = {}
        pred["base_late_fusion_wm"] = base_test_wm
        pred["base_late_fusion_fi"] = base_test_fi

        pred["ind_residual_no_prior_wm"] = base_test_wm + eta_ind_np_wm * final_ind_np_wm
        pred["ind_residual_no_prior_fi"] = base_test_fi + eta_ind_np_fi * final_ind_np_fi

        pred["ind_residual_prior_wm"] = base_test_wm + eta_ind_p_wm * final_ind_p_wm
        pred["ind_residual_prior_fi"] = base_test_fi + eta_ind_p_fi * final_ind_p_fi

        pred["mt_residual_no_prior_wm"] = base_test_wm + eta_mt_np_wm * final_mt_wm_np
        pred["mt_residual_no_prior_fi"] = base_test_fi + eta_mt_np_fi * final_mt_fi_np

        pred["mt_residual_prior_wm"] = base_test_wm + eta_mt_p_wm * final_mt_wm_p
        pred["mt_residual_prior_fi"] = base_test_fi + eta_mt_p_fi * final_mt_fi_p

        # ── Compute metrics ──────────────────────────────────────────────
        for model_name in model_names:
            for target_key, target_name in [("wm", "WM"), ("fi", "FI")]:
                y_true = y_wm[test_idx] if target_key == "wm" else y_fi[test_idx]
                y_pred = pred[f"{model_name}_{target_key}"]
                r = pearsonr(y_true, y_pred)[0]
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                mae = float(np.mean(np.abs(y_true - y_pred)))

                split_rows.append({
                    "seed": seed,
                    "fold": fold,
                    "task": target_name,
                    "model": model_name,
                    "pearson_r": r,
                    "rmse": rmse,
                    "mae": mae,
                    "alpha_fc": alpha_fc,
                    "alpha_sc": alpha_sc,
                    "lambda2_wm": lam2_wm,
                    "lambda2_fi": lam2_fi,
                    "eta_wm": eta_mt_p_wm if "mt" in model_name else (eta_ind_p_wm if "prior" in model_name else 0.0),
                    "eta_fi": eta_mt_p_fi if "mt" in model_name else (eta_ind_p_fi if "prior" in model_name else 0.0),
                })

        elapsed = time.time() - t_split
        log.info("  Completed in %.1fs", elapsed)

        # Save checkpoint
        _save_ckpt(seed, fold, {"completed": True, "split_rows": split_rows})

        # ── Export coefficients for this fold ─────────────────────────────
        # MT-RNCR coefficients (shared + contrast Laplacian)
        n_fc = X_fc_raw.shape[1]
        X_fc_sc_raw = np.hstack([X_fc_raw, np.zeros((X_fc_raw.shape[0], n_fc), dtype=X_fc_raw.dtype)])

        scaler_fc_sc = StandardScaler()
        X_fc_sc_tr = scaler_fc_sc.fit_transform(X_fc_sc_raw[train_idx])

        final_model_shared = NetworkConstrainedRidge(
            alpha1=alpha_fc, alpha2=1.0,
            edge_laplacian=edge_laplacian_shared, standardize=False,
        )
        final_model_shared.fit(X_fc_sc_tr, z_shared)
        beta_shared_full = final_model_shared.beta()
        z_std_shared = np.std(z_shared)
        beta_shared = beta_shared_full[:n_fc] * z_std_shared / scaler_fc_sc.scale_[:n_fc]

        final_model_contrast = NetworkConstrainedRidge(
            alpha1=alpha_fc, alpha2=1.0,
            edge_laplacian=edge_laplacian_contrast, standardize=False,
        )
        final_model_contrast.fit(X_fc_sc_tr, z_contrast)
        beta_contrast_full = final_model_contrast.beta()
        z_std_contrast = np.std(z_contrast)
        beta_contrast = beta_contrast_full[:n_fc] * z_std_contrast / scaler_fc_sc.scale_[:n_fc]

        np.savez_compressed(
            COEFF_DIR / f"seed_{seed}_fold_{fold}_mt_rncr.npz",
            beta_shared=beta_shared,
            beta_contrast=beta_contrast,
            roi_names=np.array(roi_names),
        )

    t_total = time.time() - t_start
    log.info("")
    log.info("=" * 80)
    log.info("All splits completed in %.1fs", t_total)
    log.info("=" * 80)

    # ── Aggregate results ─────────────────────────────────────────────────
    df = pd.DataFrame(split_rows)
    df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    # Seed-level aggregation
    for seed in DEV_SEEDS:
        sdf = df[df["seed"] == seed]
        row = {"seed": seed}
        for model_name in model_names:
            for task in ["WM", "FI"]:
                tdf = sdf[(sdf["model"] == model_name) & (sdf["task"] == task)]
                row[f"{model_name}_{task}_mean_r"] = tdf["pearson_r"].mean()
                row[f"{model_name}_{task}_std_r"] = tdf["pearson_r"].std()
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2C: Multi-Task Residual NCR Pilot — Final Report")
    print("=" * 80)
    print(f"Dev seeds: {DEV_SEEDS}")
    print(f"Outer folds: {N_OUTER_FOLDS}")
    print(f"Total partitions: {len(fold_specs)}")
    print(f"Runtime: {t_total:.1f}s")
    print("")

    # Table 1: Task-level aggregated means
    print("Table 1: Mean Pearson r across folds (std)")
    print("-" * 70)
    header = f"{'Model':<30}"
    for task in ["WM", "FI"]:
        header += f"  {task:>15}"
    print(header)
    print("-" * 70)

    for model_name in model_names:
        row_str = f"{model_name:<30}"
        for task in ["WM", "FI"]:
            tdf = df[(df["model"] == model_name) & (df["task"] == task)]
            mr = tdf["pearson_r"].mean()
            sr = tdf["pearson_r"].std()
            row_str += f"  {mr:>7.4f} ({sr:.4f})"
        print(row_str)
    print("")

    # Table 2: Deltas vs base late fusion
    print("Table 2: Delta Pearson r vs base late fusion")
    print("-" * 70)
    header = f"{'Model':<30}"
    for task in ["WM", "FI"]:
        header += f"  {task:>15}"
    print(header)
    print("-" * 70)

    base_wm = df[(df["model"] == "base_late_fusion") & (df["task"] == "WM")]["pearson_r"].mean()
    base_fi = df[(df["model"] == "base_late_fusion") & (df["task"] == "FI")]["pearson_r"].mean()

    for model_name in model_names[1:]:  # skip base
        row_str = f"{model_name:<30}"
        for task, base_r in [("WM", base_wm), ("FI", base_fi)]:
            tdf = df[(df["model"] == model_name) & (df["task"] == task)]
            mr = tdf["pearson_r"].mean()
            delta = mr - base_r
            row_str += f"  {delta:>+7.4f}"
        print(row_str)
    print("")

    # ── Decisions ─────────────────────────────────────────────────────────
    # PHASE2C_PASS if MT-RNCR beats base late fusion by >= 0.005 on at least one target
    mt_rncr_wm_delta = (
        df[(df["model"] == "mt_residual_prior") & (df["task"] == "WM")]["pearson_r"].mean()
        - base_wm
    )
    mt_rncr_fi_delta = (
        df[(df["model"] == "mt_residual_prior") & (df["task"] == "FI")]["pearson_r"].mean()
        - base_fi
    )

    wm_pass = mt_rncr_wm_delta >= 0.005
    fi_pass = mt_rncr_fi_delta >= 0.005

    print("PHASE2C_WM_DELTA:", round(float(mt_rncr_wm_delta), 4))
    print("PHASE2C_FI_DELTA:", round(float(mt_rncr_fi_delta), 4))

    if wm_pass or fi_pass:
        print("PHASE2C_DECISION: PASS")
    else:
        print("PHASE2C_DECISION: FAIL")

    print(f"  WM delta={mt_rncr_wm_delta:+.4f} ({'PASS' if wm_pass else 'FAIL'})")
    print(f"  FI delta={mt_rncr_fi_delta:+.4f} ({'PASS' if fi_pass else 'FAIL'})")
    print("")

    # ── Stability diagnostics (export only, not used for selection) ───────
    print("Coefficient stability (MT-RNCR):")
    for fold_idx, (seed, fold, _, _) in enumerate(fold_specs[:5]):
        coeff_path = COEFF_DIR / f"seed_{seed}_fold_{fold}_mt_rncr.npz"
        if coeff_path.exists():
            data = np.load(coeff_path)
            stab = compute_coefficient_stability(
                [data["beta_shared"], data["beta_contrast"]]
            )
            print(f"  seed={seed} fold={fold}: spearman={stab['mean_spearman']:.3f}, "
                  f"jaccard_top100={stab['mean_jaccard_top100']:.3f}")
    print("")

    # ── Plots ─────────────────────────────────────────────────────────────
    _generate_plots(df, seed_df, model_names, base_wm, base_fi)

    # ── Package outputs ───────────────────────────────────────────────────
    _package_outputs()

    print("Outputs saved to:", OUTPUT_DIR)


def _generate_plots(
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    model_names: List[str],
    base_wm: float,
    base_fi: float,
) -> None:
    """Generate all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Figure 1: Paired delta bars across all 20 folds ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task, base_r) in zip(axes, [("WM", base_wm), ("FI", base_fi)]):
        models_to_plot = ["ind_residual_no_prior", "ind_residual_prior",
                          "mt_residual_no_prior", "mt_residual_prior"]
        # Get one model's fold count for x-axis
        sample = df[(df["task"] == task) & (df["model"] == models_to_plot[0])]
        n_folds = len(sample)
        x = np.arange(n_folds)
        width = 0.2
        for i, model in enumerate(models_to_plot):
            mdf = df[(df["task"] == task) & (df["model"] == model)].reset_index(drop=True)
            deltas = mdf["pearson_r"] - base_r
            ax.bar(x + i * width, deltas, width, label=model, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Fold index")
        ax.set_ylabel("Delta Pearson r vs base")
        ax.set_title(f"{task} — Residual delta vs base late fusion")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure1_delta_all_folds.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: Seed-level MT-RNCR vs base comparison ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[df["task"] == task]
        models_plot = ["base_late_fusion", "mt_residual_no_prior", "mt_residual_prior"]
        x = np.arange(len(seed_df))
        width = 0.25
        for i, model in enumerate(models_plot):
            means = []
            for seed in DEV_SEEDS:
                sdf = tdf[(tdf["seed"] == seed) & (tdf["model"] == model)]
                means.append(sdf["pearson_r"].mean())
            ax.bar(x + i * width, means, width, label=model, alpha=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"{task} — Model comparison by seed")
        ax.set_xticks(x + width)
        ax.set_xticklabels(DEV_SEEDS)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure2_seed_comparison.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: MT-RNCR shared vs contrast coefficients ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (coeff_key, title) in zip(axes, [("beta_shared", "Shared"), ("beta_contrast", "Contrast")]):
        betas = []
        for seed in DEV_SEEDS:
            for fold in range(N_OUTER_FOLDS):
                p = COEFF_DIR / f"seed_{seed}_fold_{fold}_mt_rncr.npz"
                if p.exists():
                    data = np.load(p)
                    betas.append(data[coeff_key])
        if betas:
            betas_arr = np.array(betas)
            mean_beta = np.mean(betas_arr, axis=0)
            roi_agg = aggregate_edge_to_roi(mean_beta)
            top_roi_idx = np.argsort(roi_agg)[-15:][::-1]
            ax.barh(range(len(top_roi_idx)), roi_agg[top_roi_idx], alpha=0.8)
            ax.set_yticks(range(len(top_roi_idx)))
            ax.set_yticklabels([roi_names[i] for i in top_roi_idx], fontsize=8)
            ax.set_title(f"{title} prior — Top-15 ROI saliency")
            ax.set_xlabel("Sum |beta|")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure3_mt_rncr_coefficients.png", dpi=150)
    plt.close(fig)

    # ── Figure 4: RMSE comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["WM", "FI"]):
        tdf = df[df["task"] == task]
        models_plot = ["base_late_fusion", "mt_residual_no_prior", "mt_residual_prior"]
        x = np.arange(len(seed_df))
        width = 0.25
        for i, model in enumerate(models_plot):
            means = []
            for seed in DEV_SEEDS:
                sdf = tdf[(tdf["seed"] == seed) & (tdf["model"] == model)]
                means.append(sdf["rmse"].mean())
            ax.bar(x + i * width, means, width, label=model, alpha=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{task} — RMSE comparison by seed")
        ax.set_xticks(x + width)
        ax.set_xticklabels(DEV_SEEDS)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure4_rmse_comparison.png", dpi=150)
    plt.close(fig)


def _package_outputs() -> None:
    """Package all outputs into a ZIP archive."""
    zip_path = OUTPUT_DIR / "phase2c_all_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in [CKPT_DIR, COEFF_DIR, PLOT_DIR]:
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(OUTPUT_DIR))
        # CSVs
        for name in ["split_metrics.csv", "seed_metrics.csv"]:
            p = OUTPUT_DIR / name
            if p.exists():
                zf.write(p, name)
    log.info("Packaged outputs to %s", zip_path)


if __name__ == "__main__":
    run_phase2c_pilot()
