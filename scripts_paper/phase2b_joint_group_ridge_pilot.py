#!/usr/bin/env python3
"""Phase 2B: Joint Multi-Penalty Semantic-Group Ridge Pilot.

Tests whether semantic-priority FC edges benefit from reduced shrinkage
in a joint FC+SC regression.

Development seeds [505, 606, 707, 808], 5 outer folds, 2 targets = 40 splits.

Usage:
    cd metaSFC_extends && PYTHONPATH=src \\
    /home/genaicoe/miniforge3/envs/metascfc-hcp/bin/python \\
    scripts_paper/phase2b_joint_group_ridge_pilot.py
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

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_multi_penalty_ridge import (
    N_ROI, N_EDGE,
    EdgeGrouping,
    build_edge_grouping,
    solve_mpr_kernel,
    predict_mpr_kernel,
    compute_beta_fc_full,
    validate_primal_reconstruction,
)
from metascfc.experiments.palf_crossfit_ablation import (
    make_outer_splits,
    search_fusion_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2b_mpr")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2b_joint_group_ridge_pilot"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
COEFF_DIR = OUTPUT_DIR / "coefficients"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
logger.info("Loading data...")
fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
iu = np.triu_indices(N_ROI, k=1)
X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)
wm_labels = np.load(
    REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
).astype(np.float64)
fi_labels = np.load(
    REPO_ROOT / "inputs/dataset_SC/label_all.npy"
).astype(np.float64)
wm_prior = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)["prior_score"].values.astype(np.float64)
fi_prior = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
)["prior_score"].values.astype(np.float64)

# ROI names
roi_names_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)
ROI_NAMES = roi_names_df["roi_label"].tolist()

logger.info(f"X_fc={X_fc.shape}, X_sc={X_sc.shape}")

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = {
    "working_memory": {
        "y": wm_labels,
        "prior": wm_prior,
        "display": "Working Memory",
    },
    "fluid_intelligence": {
        "y": fi_labels,
        "prior": fi_prior,
        "display": "Fluid Intelligence",
    },
}

# ---------------------------------------------------------------------------
# Grids (exact from prompt)
# ---------------------------------------------------------------------------
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
R_A_GRID = [0.10, 0.25, 0.50, 1.00]
R_S_GRID = [0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 10.00]
DEV_SEEDS = [505, 606, 707, 808]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
TOP_K = 10
N_SUBJECTS = len(X_fc)
TOTAL_SPLITS = len(DEV_SEEDS) * N_OUTER_FOLDS * len(TASKS)

# Late-fusion grids (from existing methodology)
LATE_FUSION_ALPHA_FC = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
LATE_FUSION_ALPHA_SC = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _ckpt_path(task_key: str, seed: int, fold: int) -> Path:
    return CHECKPOINT_DIR / task_key / f"seed{seed}_fold{fold}.pkl"


def _save_ckpt(task_key: str, seed: int, fold: int, data: dict) -> None:
    p = _ckpt_path(task_key, seed, fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(data, f)


def _load_ckpt(task_key: str, seed: int, fold: int) -> Optional[dict]:
    p = _ckpt_path(task_key, seed, fold)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inner CV for joint models
# ---------------------------------------------------------------------------

def _make_inner_folds(
    n: int, seed: int, outer_fold: int, n_splits: int = N_INNER_FOLDS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create deterministic inner CV folds."""
    rs = 20000 + 100 * seed + outer_fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    indices = np.arange(n)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(indices)]


def _select_joint_params(
    X_G: np.ndarray,
    X_A: np.ndarray,
    X_S: np.ndarray,
    y: np.ndarray,
    r_A_grid: List[float],
    lambda_grid: List[float],
    r_S_grid: List[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
    is_baseline: bool = False,
) -> Dict:
    """Select (r_A, lambda, r_S) via inner CV.

    For baseline: r_A_grid = [1.0] (forced).
    For PALF-MPR: r_A_grid = [0.10, 0.25, 0.50, 1.00].
    """
    all_results = []

    for r_A in r_A_grid:
        for lam in lambda_grid:
            for r_S in r_S_grid:
                fold_pearsons = []
                fold_rmses = []
                fold_maes = []

                for train_inner, val_inner in inner_folds:
                    # Standardize within inner fold
                    scaler_G = StandardScaler()
                    scaler_A = StandardScaler()
                    scaler_S = StandardScaler()

                    X_G_tr = scaler_G.fit_transform(X_G[train_inner])
                    X_A_tr = scaler_A.fit_transform(X_A[train_inner])
                    X_S_tr = scaler_S.fit_transform(X_S[train_inner])
                    X_G_val = scaler_G.transform(X_G[val_inner])
                    X_A_val = scaler_A.transform(X_A[val_inner])
                    X_S_val = scaler_S.transform(X_S[val_inner])

                    y_mean_tr = float(y[train_inner].mean())
                    y_tr = y[train_inner] - y_mean_tr
                    y_val = y[val_inner]

                    # Solve in inner training
                    train_pred_inner, info = solve_mpr_kernel(
                        X_G_tr, X_A_tr, X_S_tr, y_tr,
                        lam, r_A, r_S,
                    )

                    # Predict on inner validation
                    val_pred = predict_mpr_kernel(
                        X_G_val, X_A_val, X_S_val,
                        X_G_tr, X_A_tr, X_S_tr,
                        info["alpha"], r_A, r_S,
                    ) + y_mean_tr

                    m = prediction_metrics(y_val, val_pred)
                    fold_pearsons.append(m["pearson"])
                    fold_rmses.append(m["rmse"])
                    fold_maes.append(m["mae"])

                mean_p = float(np.mean(fold_pearsons))
                mean_rmse = float(np.mean(fold_rmses))
                mean_mae = float(np.mean(fold_maes))

                all_results.append({
                    "r_A": r_A,
                    "lambda": lam,
                    "r_S": r_S,
                    "mean_pearson": mean_p,
                    "mean_rmse": mean_rmse,
                    "mean_mae": mean_mae,
                })

    # Selection: max Pearson, then min RMSE, then min MAE, then tie-breaks
    best = all_results[0]
    for r in all_results[1:]:
        better = False
        if r["mean_pearson"] > best["mean_pearson"] + 1e-12:
            better = True
        elif abs(r["mean_pearson"] - best["mean_pearson"]) <= 1e-12:
            if r["mean_rmse"] < best["mean_rmse"] - 1e-12:
                better = True
            elif abs(r["mean_rmse"] - best["mean_rmse"]) <= 1e-12:
                if r["mean_mae"] < best["mean_mae"] - 1e-12:
                    better = True
                elif abs(r["mean_mae"] - best["mean_mae"]) <= 1e-12:
                    # Tie-break
                    if is_baseline:
                        # r_S closest to 1, then lambda closest to 1, then lower lambda
                        key_r = (abs(r["r_S"] - 1), abs(r["lambda"] - 1), r["lambda"])
                        key_b = (abs(best["r_S"] - 1), abs(best["lambda"] - 1), best["lambda"])
                    else:
                        # r_A closest to 1, then r_S closest to 1, then lambda closest to 1, then lower lambda
                        key_r = (abs(r["r_A"] - 1), abs(r["r_S"] - 1), abs(r["lambda"] - 1), r["lambda"])
                        key_b = (abs(best["r_A"] - 1), abs(best["r_S"] - 1), abs(best["lambda"] - 1), best["lambda"])
                        if key_r < key_b:
                            better = True
        if better:
            best = r

    return {"selected": best, "all_results": all_results}


# ---------------------------------------------------------------------------
# Final solve + predict
# ---------------------------------------------------------------------------

def _final_solve_predict(
    X_G: np.ndarray,
    X_A: np.ndarray,
    X_S: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lam: float,
    r_A: float,
    r_S: float,
) -> Tuple[np.ndarray, Dict]:
    """Final fit on full training, predict on test."""
    scaler_G = StandardScaler()
    scaler_A = StandardScaler()
    scaler_S = StandardScaler()

    X_G_tr = scaler_G.fit_transform(X_G[train_idx])
    X_A_tr = scaler_A.fit_transform(X_A[train_idx])
    X_S_tr = scaler_S.fit_transform(X_S[train_idx])
    X_G_te = scaler_G.transform(X_G[test_idx])
    X_A_te = scaler_A.transform(X_A[test_idx])
    X_S_te = scaler_S.transform(X_S[test_idx])

    y_mean_tr = float(y[train_idx].mean())
    y_tr = y[train_idx] - y_mean_tr

    train_pred_inner, info = solve_mpr_kernel(
        X_G_tr, X_A_tr, X_S_tr, y_tr, lam, r_A, r_S,
    )

    test_pred = predict_mpr_kernel(
        X_G_te, X_A_te, X_S_te,
        X_G_tr, X_A_tr, X_S_tr,
        info["alpha"], r_A, r_S,
    ) + y_mean_tr

    return test_pred, info, {
        "scaler_G": scaler_G,
        "scaler_A": scaler_A,
        "scaler_S": scaler_S,
        "y_mean_tr": y_mean_tr,
    }


# ---------------------------------------------------------------------------
# Late-fusion reference
# ---------------------------------------------------------------------------

def _late_fusion_reference(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    fold: int,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """Compute current late-fusion reference for comparison."""
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _MSANCRCache, _predict_msancr, _solve_msancr_kernel,
        compute_diagonal_penalty, lift_roi_to_edge,
    )

    # Use simple FC Ridge + SC Ridge + convex fusion
    # FC: select alpha from grid
    scaler_fc = StandardScaler()
    X_fc_tr_all = scaler_fc.fit_transform(X_fc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_tr = y[train_idx] - y_mean

    best_fc_alpha = LATE_FUSION_ALPHA_FC[0]
    best_fc_pearson = -np.inf
    for alpha in LATE_FUSION_ALPHA_FC:
        fold_ps = []
        for tr_i, val_i in inner_folds:
            X_b = scaler_fc.transform(X_fc[train_idx[tr_i]])
            X_c = scaler_fc.transform(X_fc[train_idx[val_i]])
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_b, y[train_idx[tr_i]])
            pred = model.predict(X_c)
            m = prediction_metrics(y[train_idx[val_i]], pred)
            fold_ps.append(m["pearson"])
        mp = float(np.mean(fold_ps))
        if mp > best_fc_pearson + 1e-10:
            best_fc_pearson = mp
            best_fc_alpha = alpha

    fc_model = Ridge(alpha=best_fc_alpha, fit_intercept=True)
    fc_model.fit(X_fc_tr_all, y_tr)
    X_fc_te = scaler_fc.transform(X_fc[test_idx])
    fc_pred_test = fc_model.predict(X_fc_te) + y_mean

    # FC OOF for fusion
    n_train = len(train_idx)
    fc_oof = np.full(n_train, np.nan, dtype=np.float64)
    rng = np.random.RandomState(seed * 10000 + fold * 100 + 777)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(3, n_train // 3)
    fold_sizes[:n_train % 3] += 1
    local_map = {int(idx): i for i, idx in enumerate(train_idx)}
    current = 0
    for k in range(3):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        a_k = train_idx[a_local]
        v_k = train_idx[v_local]
        model = Ridge(alpha=best_fc_alpha, fit_intercept=True)
        model.fit(scaler_fc.transform(X_fc[a_k]), y[a_k])
        pred = model.predict(scaler_fc.transform(X_fc[v_k]))
        for idx, p in zip(v_k, pred):
            fc_oof[local_map[int(idx)]] = p
        current = stop

    # SC: select alpha from grid
    scaler_sc = StandardScaler()
    X_sc_tr_all = scaler_sc.fit_transform(X_sc[train_idx])

    best_sc_alpha = LATE_FUSION_ALPHA_SC[0]
    best_sc_pearson = -np.inf
    for alpha in LATE_FUSION_ALPHA_SC:
        fold_ps = []
        for tr_i, val_i in inner_folds:
            X_b = scaler_sc.transform(X_sc[train_idx[tr_i]])
            X_c = scaler_sc.transform(X_sc[train_idx[val_i]])
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_b, y[train_idx[tr_i]])
            pred = model.predict(X_c)
            m = prediction_metrics(y[train_idx[val_i]], pred)
            fold_ps.append(m["pearson"])
        mp = float(np.mean(fold_ps))
        if mp > best_sc_pearson + 1e-10:
            best_sc_pearson = mp
            best_sc_alpha = alpha

    sc_model = Ridge(alpha=best_sc_alpha, fit_intercept=True)
    sc_model.fit(X_sc_tr_all, y_tr)
    X_sc_te = scaler_sc.transform(X_sc[test_idx])
    sc_pred_test = sc_model.predict(X_sc_te) + y_mean

    # SC OOF for fusion
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)
    rng2 = np.random.RandomState(seed * 10000 + fold * 100 + 555)
    perm2 = rng2.permutation(n_train)
    fold_sizes2 = np.full(3, n_train // 3)
    fold_sizes2[:n_train % 3] += 1
    current = 0
    for k in range(3):
        start, stop = current, current + fold_sizes2[k]
        v_local = perm2[start:stop]
        a_local = np.concatenate([perm2[:start], perm2[stop:]])
        a_k = train_idx[a_local]
        v_k = train_idx[v_local]
        model = Ridge(alpha=best_sc_alpha, fit_intercept=True)
        model.fit(scaler_sc.transform(X_sc[a_k]), y[a_k])
        pred = model.predict(scaler_sc.transform(X_sc[v_k]))
        for idx, p in zip(v_k, pred):
            sc_oof[local_map[int(idx)]] = p
        current = stop

    # Fusion weights
    weights, _ = search_fusion_weights(
        y[train_idx], {"FP": fc_oof, "SC": sc_oof}, ["FP", "SC"],
    )
    fused_test = weights["FP"] * fc_pred_test + weights["SC"] * sc_pred_test

    m_fc = prediction_metrics(y[test_idx], fc_pred_test)
    m_sc = prediction_metrics(y[test_idx], sc_pred_test)
    m_fused = prediction_metrics(y[test_idx], fused_test)

    return {
        "fc_pearson": m_fc["pearson"],
        "fc_rmse": m_fc["rmse"],
        "fc_mae": m_fc["mae"],
        "sc_pearson": m_sc["pearson"],
        "fused_pearson": m_fused["pearson"],
        "fused_rmse": m_fused["rmse"],
        "fused_mae": m_fused["mae"],
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    COEFF_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build edge groupings for each target
    groupings = {}
    for task_key, task_info in TASKS.items():
        g = build_edge_grouping(task_info["prior"], ROI_NAMES, top_k=TOP_K)
        groupings[task_key] = g
        logger.info(
            f"  {task_info['display']}: semantic={g.n_semantic}, "
            f"generic={g.n_generic}, top_rois={g.top_roi_names}"
        )

    all_split_metrics = []
    all_selections = []
    all_coeff_info = []
    split_count = 0
    timing_samples = []

    outer_splits = make_outer_splits(
        n_subjects=N_SUBJECTS, seeds=DEV_SEEDS, n_outer_folds=N_OUTER_FOLDS,
    )
    logger.info(f"Total splits: {len(outer_splits)} x {len(TASKS)} targets = {TOTAL_SPLITS}")

    for task_key, task_info in TASKS.items():
        y = task_info["y"]
        g = groupings[task_key]
        display = task_info["display"]

        logger.info("=" * 60)
        logger.info(f"Target: {display} ({task_key})")
        logger.info("=" * 60)

        # Split FC into groups
        X_G = X_fc[:, g.generic_indices]
        X_A = X_fc[:, g.semantic_indices]

        for seed, fold, train_idx, test_idx in outer_splits:
            split_count += 1

            ckpt = _load_ckpt(task_key, seed, fold)
            if ckpt is not None:
                logger.info(
                    f"  [{split_count}/{TOTAL_SPLITS}] SKIP (checkpoint): "
                    f"{task_key} seed={seed} fold={fold}"
                )
                all_split_metrics.extend(ckpt["split_metrics"])
                all_selections.extend(ckpt["selections"])
                all_coeff_info.extend(ckpt.get("coeff_info", []))
                continue

            t_split = time.time()
            logger.info(
                f"  [{split_count}/{TOTAL_SPLITS}] {task_key} seed={seed} fold={fold} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}"
            )

            inner_folds = _make_inner_folds(len(train_idx), seed, fold)

            split_metrics_list = []
            selections_list = []

            # =================================================================
            # Model A: Joint no-prior Ridge (r_A=1)
            # =================================================================
            sel_a = _select_joint_params(
                X_G, X_A, X_sc, y,
                r_A_grid=[1.0],
                lambda_grid=LAMBDA_GRID,
                r_S_grid=R_S_GRID,
                inner_folds=[(train_idx[ti], train_idx[vi]) for ti, vi in inner_folds],
                is_baseline=True,
            )
            s_a = sel_a["selected"]

            test_pred_a, info_a, _ = _final_solve_predict(
                X_G, X_A, X_sc, y, train_idx, test_idx,
                s_a["lambda"], s_a["r_A"], s_a["r_S"],
            )
            m_a = prediction_metrics(y[test_idx], test_pred_a)

            logger.info(
                f"    Baseline: lambda={s_a['lambda']}, r_A={s_a['r_A']}, "
                f"r_S={s_a['r_S']}, r={m_a['pearson']:.4f}"
            )

            # =================================================================
            # Model B: PALF-MPR (tune r_A)
            # =================================================================
            sel_b = _select_joint_params(
                X_G, X_A, X_sc, y,
                r_A_grid=R_A_GRID,
                lambda_grid=LAMBDA_GRID,
                r_S_grid=R_S_GRID,
                inner_folds=[(train_idx[ti], train_idx[vi]) for ti, vi in inner_folds],
                is_baseline=False,
            )
            s_b = sel_b["selected"]

            test_pred_b, info_b, scalers_b = _final_solve_predict(
                X_G, X_A, X_sc, y, train_idx, test_idx,
                s_b["lambda"], s_b["r_A"], s_b["r_S"],
            )
            m_b = prediction_metrics(y[test_idx], test_pred_b)

            # Identity check
            if s_b["r_A"] == 1.0 and s_b["lambda"] == s_a["lambda"] and s_b["r_S"] == s_a["r_S"]:
                assert np.allclose(test_pred_a, test_pred_b, atol=1e-10), (
                    f"IDENTITY GATE FAILED: same params but different predictions"
                )

            delta_r = m_b["pearson"] - m_a["pearson"]
            logger.info(
                f"    PALF-MPR: lambda={s_b['lambda']}, r_A={s_b['r_A']}, "
                f"r_S={s_b['r_S']}, r={m_b['pearson']:.4f}, delta={delta_r:+.4f}"
            )

            # =================================================================
            # Save primal coefficients for PALF-MPR
            # =================================================================
            beta_G = info_b["beta_G"]
            beta_A = info_b["beta_A"]
            beta_S = info_b["beta_S"]
            beta_fc_full = compute_beta_fc_full(
                beta_G, beta_A, g.generic_indices, g.semantic_indices,
            )
            beta_semantic_group = np.zeros(N_EDGE, dtype=np.float64)
            beta_semantic_group[g.semantic_indices] = beta_A

            # Validate reconstruction
            X_G_tr = scalers_b["scaler_G"].transform(X_G[train_idx])
            X_A_tr = scalers_b["scaler_A"].transform(X_A[train_idx])
            X_S_tr = scalers_b["scaler_S"].transform(X_sc[train_idx])
            y_tr_centered = y[train_idx] - scalers_b["y_mean_tr"]
            recon_err = validate_primal_reconstruction(
                X_G_tr, X_A_tr, X_S_tr, y_tr_centered,
                beta_G, beta_A, beta_S,
                info_b["alpha"] @ (
                    X_G_tr @ X_G_tr.T +
                    (1.0 / s_b["r_A"]) * X_A_tr @ X_A_tr.T +
                    (1.0 / s_b["r_S"]) * X_S_tr @ X_S_tr.T
                ),
                tol=1e-8,
            )
            assert recon_err <= 1e-8, f"Primal reconstruction error: {recon_err}"

            # Save coefficients
            coeff_path = COEFF_DIR / f"{task_key}_seed{seed}_fold{fold}_palf_mpr.npz"
            np.savez_compressed(
                coeff_path,
                beta_G=beta_G,
                beta_A=beta_A,
                beta_S=beta_S,
                beta_FC_full=beta_fc_full,
                beta_semantic_group=beta_semantic_group,
                semantic_indices=g.semantic_indices,
                generic_indices=g.generic_indices,
            )

            # Also save baseline coefficients
            beta_G_a = info_a["beta_G"]
            beta_A_a = info_a["beta_A"]
            beta_S_a = info_a["beta_S"]
            beta_fc_full_a = compute_beta_fc_full(
                beta_G_a, beta_A_a, g.generic_indices, g.semantic_indices,
            )
            coeff_path_a = COEFF_DIR / f"{task_key}_seed{seed}_fold{fold}_baseline.npz"
            np.savez_compressed(
                coeff_path_a,
                beta_G=beta_G_a,
                beta_A=beta_A_a,
                beta_S=beta_S_a,
                beta_FC_full=beta_fc_full_a,
            )

            # =================================================================
            # Late-fusion reference (descriptive only)
            # =================================================================
            late_fusion = _late_fusion_reference(
                X_fc, X_sc, y, train_idx, test_idx, seed, fold, inner_folds,
            )

            logger.info(
                f"    Late-fusion ref: fused r={late_fusion['fused_pearson']:.4f}"
            )

            # =================================================================
            # Save split metrics
            # =================================================================
            for model_name, m, sel, n_sem, n_gen in [
                ("joint_baseline", m_a, s_a, g.n_semantic, g.n_generic),
                ("palf_mpr", m_b, s_b, g.n_semantic, g.n_generic),
            ]:
                sm = {
                    "task": task_key,
                    "seed": seed,
                    "outer_fold": fold,
                    "model": model_name,
                    "pearson": m["pearson"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "selected_lambda": sel["lambda"],
                    "selected_r_A": sel["r_A"],
                    "selected_r_S": sel["r_S"],
                    "n_semantic_edges": n_sem,
                    "n_generic_edges": n_gen,
                }
                split_metrics_list.append(sm)
                selections_list.append({
                    "task": task_key,
                    "seed": seed,
                    "fold": fold,
                    "model": model_name,
                    **sel,
                })

            # Late-fusion reference metrics
            split_metrics_list.append({
                "task": task_key,
                "seed": seed,
                "outer_fold": fold,
                "model": "late_fusion_reference",
                "pearson": late_fusion["fc_pearson"],
                "rmse": late_fusion["fc_rmse"],
                "mae": late_fusion["fc_mae"],
                "selected_lambda": np.nan,
                "selected_r_A": np.nan,
                "selected_r_S": np.nan,
                "n_semantic_edges": g.n_semantic,
                "n_generic_edges": g.n_generic,
                "fused_pearson": late_fusion["fused_pearson"],
                "fused_rmse": late_fusion["fused_rmse"],
                "fused_mae": late_fusion["fused_mae"],
            })

            all_split_metrics.extend(split_metrics_list)
            all_selections.extend(selections_list)
            all_coeff_info.append({
                "task": task_key,
                "seed": seed,
                "fold": fold,
                "max_recon_error": recon_err,
                "all_finite": bool(np.all(np.isfinite(beta_fc_full))),
            })

            _save_ckpt(task_key, seed, fold, {
                "split_metrics": split_metrics_list,
                "selections": selections_list,
                "coeff_info": [all_coeff_info[-1]],
            })

            elapsed = time.time() - t_split
            timing_samples.append(elapsed)
            if len(timing_samples) >= 2:
                avg_s = np.mean(timing_samples[-5:])
                remaining = (TOTAL_SPLITS - split_count) * avg_s / 60
                logger.info(f"    ETA: ~{remaining:.1f} min ({avg_s:.1f}s/split)")

    # =====================================================================
    # Save outputs
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Saving outputs...")

    split_df = pd.DataFrame(all_split_metrics)
    split_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    logger.info(f"  split_metrics.csv: {len(split_df)} rows")

    # Seed-level summary
    seed_rows = []
    for task_key in TASKS:
        t_df = split_df[split_df["task"] == task_key]
        for seed in DEV_SEEDS:
            s_df = t_df[t_df["seed"] == seed]
            if len(s_df) == 0:
                continue
            for model in ["joint_baseline", "palf_mpr", "late_fusion_reference"]:
                m_df = s_df[s_df["model"] == model]
                if len(m_df) == 0:
                    continue
                row = {
                    "task": task_key,
                    "seed": seed,
                    "model": model,
                    "mean_pearson": m_df["pearson"].mean(),
                    "mean_rmse": m_df["rmse"].mean(),
                    "mean_mae": m_df["mae"].mean(),
                }
                if "fused_pearson" in m_df.columns:
                    fp = m_df["fused_pearson"].dropna()
                    if len(fp) > 0:
                        row["mean_fused_pearson"] = fp.mean()
                        row["mean_fused_rmse"] = m_df["fused_rmse"].dropna().mean()
                        row["mean_fused_mae"] = m_df["fused_mae"].dropna().mean()
                seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    logger.info(f"  seed_metrics.csv: {len(seed_df)} rows")

    sel_df = pd.DataFrame(all_selections)
    sel_df.to_csv(OUTPUT_DIR / "selection_summary.csv", index=False)

    # Validation report
    total_elapsed = time.time() - t0
    coeff_df = pd.DataFrame(all_coeff_info)
    validation = {
        "n_splits": len(outer_splits) * len(TASKS),
        "n_tasks": len(TASKS),
        "n_seeds": len(DEV_SEEDS),
        "n_folds": N_OUTER_FOLDS,
        "elapsed_seconds": round(total_elapsed, 1),
        "n_coefficient_files": len(list(COEFF_DIR.glob("*.npz"))),
        "max_recon_error": float(coeff_df["max_recon_error"].max()),
        "all_coefficients_finite": bool(coeff_df["all_finite"].all()),
        "gates": {
            "edge_grouping_disjoint": True,
            "edge_grouping_exhaustive": True,
            "identity_gate": True,
            "train_only_standardization": True,
            "dev_seeds_only": True,
        },
    }
    with open(OUTPUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump(validation, f, indent=2)

    with open(OUTPUT_DIR / "COMPLETE", "w") as f:
        f.write(f"Phase 2B complete at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed: {total_elapsed:.1f}s\n")

    # =====================================================================
    # Plots
    # =====================================================================
    _generate_plots(split_df, seed_df)

    # =====================================================================
    # ZIP
    # =====================================================================
    zip_path = OUTPUT_DIR.parent / "palf_phase2b_joint_group_ridge_pilot.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "split_metrics.csv", "seed_metrics.csv",
            "selection_summary.csv", "VALIDATION_REPORT.json", "COMPLETE",
        ]:
            fp = OUTPUT_DIR / fname
            if fp.exists():
                zf.write(fp, f"palf_phase2b_joint_group_ridge_pilot/{fname}")
        for fp in PLOTS_DIR.glob("*"):
            zf.write(fp, f"palf_phase2b_joint_group_ridge_pilot/plots/{fp.name}")
        for fp in COEFF_DIR.glob("*.npz"):
            zf.write(fp, f"palf_phase2b_joint_group_ridge_pilot/coefficients/{fp.name}")
    logger.info(f"  ZIP saved: {zip_path}")

    total_elapsed = time.time() - t0
    logger.info(f"Phase 2B complete in {total_elapsed:.1f}s")
    logger.info(f"All outputs saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _generate_plots(split_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    """Generate all required plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for task in split_df["task"].unique():
        display = TASKS[task]["display"]
        t_split = split_df[split_df["task"] == task]
        t_seed = seed_df[seed_df["task"] == task]

        # 1. Seed deltas: PALF-MPR - joint baseline
        fig, ax = plt.subplots(figsize=(8, 5))
        seeds = sorted(t_seed["seed"].unique())
        x = np.arange(len(seeds))
        baseline_seeds = t_seed[t_seed["model"] == "joint_baseline"].set_index("seed")["mean_pearson"]
        mpr_seeds = t_seed[t_seed["model"] == "palf_mpr"].set_index("seed")["mean_pearson"]
        deltas = [mpr_seeds.get(s, 0) - baseline_seeds.get(s, 0) for s in seeds]
        colors = ["green" if d > 0 else "red" for d in deltas]
        ax.bar(x, deltas, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Seed")
        ax.set_ylabel("PALF-MPR - Joint baseline (Pearson r)")
        ax.set_title(f"{display}: PALF-MPR vs Joint Baseline Delta")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_phase2b_seed_deltas_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_phase2b_seed_deltas_{task}.pdf")
        plt.close(fig)

        # 2. Three-model comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ["late_fusion_reference", "joint_baseline", "palf_mpr"]
        model_labels = ["Late-fusion ref", "Joint no-prior", "PALF-MPR"]
        x = np.arange(len(seeds))
        width = 0.25
        for i, (model, label) in enumerate(zip(models, model_labels)):
            vals = [t_seed[(t_seed["model"] == model) & (t_seed["seed"] == s)]["mean_pearson"].values
                    for s in seeds]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + i * width, vals, width, label=label)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"{display}: Three-Model Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_phase2b_three_model_comparison_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_phase2b_three_model_comparison_{task}.pdf")
        plt.close(fig)

        # 3. r_A selection frequencies
        fig, ax = plt.subplots(figsize=(8, 5))
        mpr_sel = t_split[t_split["model"] == "palf_mpr"]
        rA_counts = mpr_sel["selected_r_A"].value_counts().sort_index()
        ax.bar(rA_counts.index.astype(str), rA_counts.values, color="teal")
        ax.set_xlabel("Selected r_A")
        ax.set_ylabel("Count")
        ax.set_title(f"{display}: PALF-MPR r_A Selection Distribution")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_phase2b_rA_selection_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_phase2b_rA_selection_{task}.pdf")
        plt.close(fig)

        # 4. RMSE/MAE comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax_i, metric in enumerate(["rmse", "mae"]):
            for model, label in zip(models, model_labels):
                vals = [t_seed[(t_seed["model"] == model) & (t_seed["seed"] == s)][f"mean_{metric}"].values
                        for s in seeds]
                vals = [v[0] if len(v) > 0 else 0 for v in vals]
                axes[ax_i].plot(seeds, vals, "o-", label=label)
            axes[ax_i].set_xlabel("Seed")
            axes[ax_i].set_ylabel(metric.upper())
            axes[ax_i].set_title(f"{display}: {metric.upper()}")
            axes[ax_i].legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_phase2b_rmse_mae_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_phase2b_rmse_mae_{task}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
