#!/usr/bin/env python3
"""Phase 2A: Adaptive Prior Pilot Experiment.

Compares Model A (baseline, rho=0, tau=0) against Model B (adaptive prior
with continuous rho/tau/lambda_F selection via 3-fold CV + one-SE rule).

Development seeds [101, 202, 303, 404], 5 outer folds each, 2 targets = 40 splits.

Usage:
    cd metaSFC_extends && PYTHONPATH=src \\
    /home/genaicoe/miniforge3/envs/metascfc-hcp/bin/python \\
    scripts_paper/phase2a_adaptive_prior_pilot.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_adaptive_prior import (
    AdaptivePriorCache,
    build_adaptive_prior_cache,
    build_cache_for_rho,
)
from metascfc.experiments.palf_crossfit_ablation import (
    C_SCALE,
    DIAGONAL_EPSILON,
    GAMMA_FIXED,
    N_EDGE,
    N_ROI,
    RIDGE_GRID,
    TOP_K,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    compute_diagonal_penalty,
    lift_roi_to_edge,
    make_inner_selection_folds,
    make_outer_splits,
    search_fusion_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2a_adaptive_prior_pilot")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2a_adaptive_prior_pilot"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
logger.info("Loading data...")
fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
iu = np.triu_indices(N_ROI, k=1)
X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)  # (412, 6670)
X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)  # (412, 6670)
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
logger.info(
    f"X_fc={X_fc.shape}, X_sc={X_sc.shape}, "
    f"wm_labels={wm_labels.shape}, fi_labels={fi_labels.shape}"
)

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
# Grids
# ---------------------------------------------------------------------------
RHO_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
TAU_GRID = [0.0, 0.003, 0.01, 0.03, 0.10, 0.50, 1.0, 2.0, 5.0]
LAMBDA_F_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ALPHA_SC_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]
DEV_SEEDS = [101, 202, 303, 404]
N_OUTER_FOLDS = 5
N_FUSION_FOLDS = 3
N_INNER_FOLDS = 3

N_SUBJECTS = len(X_fc)
TOTAL_SPLITS = len(DEV_SEEDS) * N_OUTER_FOLDS * len(TASKS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(task_key: str, seed: int, fold: int) -> Path:
    return CHECKPOINT_DIR / task_key / f"seed{seed}_fold{fold}.pkl"


def _save_checkpoint(task_key: str, seed: int, fold: int, data: dict) -> None:
    path = _checkpoint_path(task_key, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_checkpoint(task_key: str, seed: int, fold: int) -> Optional[dict]:
    path = _checkpoint_path(task_key, seed, fold)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _select_lambda_fc_only(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    cache,
    lambda_F_grid: List[float],
    tau: float,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, Dict]:
    """Select lambda_F for FC-only baseline (rho=0, tau=0) via 3-fold inner CV."""
    best_pearson = -np.inf
    best_lambda = lambda_F_grid[0]
    all_scores = {}

    for lambda_F in lambda_F_grid:
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
                    lambda_F, 1.0, tau, fc_only=True,
                )
                pred_z = _predict_msancr(
                    X_fc_c, np.zeros_like(X_fc_c),
                    X_fc_b, np.zeros_like(X_fc_b),
                    alpha, cache, lambda_F, 1.0, tau, fc_only=True,
                )
                pred = pred_z * y_std_b + y_mean_b
                m = prediction_metrics(y[c_idx], pred)
                fold_ps.append(m["pearson"])
            except Exception:
                fold_ps.append(-np.inf)

        mp = float(np.mean(fold_ps))
        all_scores[lambda_F] = {"pearson": mp}

        if mp > best_pearson + 1e-10:
            best_pearson = mp
            best_lambda = lambda_F

    return best_lambda, {"scores": all_scores, "pearson": best_pearson}


def _select_adaptive_params_inner(
    adaptive_cache: AdaptivePriorCache,
    X_fc: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    rho_grid: List[float],
    tau_grid: List[float],
    lambda_F_grid: List[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """Select (rho, tau, lambda_F) via 3-fold inner CV with one-SE rule."""
    all_results = []

    for rho in rho_grid:
        for tau in tau_grid:
            for lambda_F in lambda_F_grid:
                fold_ps = []
                for b_idx, c_idx in inner_folds:
                    try:
                        cache = build_cache_for_rho(adaptive_cache, rho)
                        scaler = StandardScaler()
                        X_fc_b = scaler.fit_transform(X_fc[b_idx])
                        X_fc_c = scaler.transform(X_fc[c_idx])

                        y_mean_b = float(y[b_idx].mean())
                        y_std_b = max(float(y[b_idx].std()), 1e-8)
                        y_z = (y[b_idx] - y_mean_b) / y_std_b

                        alpha, _ = _solve_msancr_kernel(
                            X_fc_b, np.zeros_like(X_fc_b), y_z, cache,
                            lambda_F, 1.0, tau, fc_only=True,
                        )
                        pred_z = _predict_msancr(
                            X_fc_c, np.zeros_like(X_fc_c),
                            X_fc_b, np.zeros_like(X_fc_b),
                            alpha, cache, lambda_F, 1.0, tau, fc_only=True,
                        )
                        pred = pred_z * y_std_b + y_mean_b
                        m = prediction_metrics(y[c_idx], pred)
                        fold_ps.append(m["pearson"])
                    except Exception:
                        fold_ps.append(-np.inf)

                mean_p = float(np.mean(fold_ps))
                se_p = (
                    float(np.std(fold_ps, ddof=1) / np.sqrt(N_INNER_FOLDS))
                    if N_INNER_FOLDS > 1
                    else 0.0
                )
                all_results.append({
                    "rho": rho,
                    "tau": tau,
                    "lambda_F": lambda_F,
                    "mean_pearson": mean_p,
                    "se_pearson": se_p,
                    "fold_pearsons": fold_ps,
                })

    # One-SE rule
    best_max = max(all_results, key=lambda r: r["mean_pearson"])
    r_best = best_max["mean_pearson"]
    se_best = best_max["se_pearson"]
    threshold = r_best - se_best

    se_candidates = [r for r in all_results if r["mean_pearson"] >= threshold]

    def simplicity_key(r):
        return (r["rho"], r["tau"], abs(r["lambda_F"] - 0.1))

    best_se = min(se_candidates, key=simplicity_key) if se_candidates else best_max

    return {
        "best_rho": best_se["rho"],
        "best_tau": best_se["tau"],
        "best_lambda_F": best_se["lambda_F"],
        "all_results": all_results,
        "best_max_candidate": best_max,
        "best_se_candidate": best_se,
    }


def _select_alpha_sc(
    X_sc: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    alpha_sc_grid: List[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, Dict]:
    """Select alpha_SC for SC Ridge via 3-fold inner CV."""
    scaler = StandardScaler()
    X_sc_train_z = scaler.fit_transform(X_sc[train_global])

    best_pearson = -np.inf
    best_alpha = alpha_sc_grid[0]
    all_scores = {}

    for alpha in alpha_sc_grid:
        fold_ps = []
        for b_idx, c_idx in inner_folds:
            X_b = scaler.transform(X_sc[b_idx])
            X_c = scaler.transform(X_sc[c_idx])
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_b, y[b_idx])
            pred = model.predict(X_c)
            m = prediction_metrics(y[c_idx], pred)
            fold_ps.append(m["pearson"])

        mp = float(np.mean(fold_ps))
        all_scores[alpha] = {"pearson": mp}

        if mp > best_pearson + 1e-10:
            best_pearson = mp
            best_alpha = alpha

    return best_alpha, {"scores": all_scores, "pearson": best_pearson}


def _solve_fc_and_predict(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cache,
    lambda_F: float,
    tau: float,
) -> np.ndarray:
    """Solve FC-only and return test predictions in original scale."""
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_fc[train_idx])
    X_test_z = scaler.transform(X_fc[test_idx])

    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean) / y_std

    alpha, _ = _solve_msancr_kernel(
        X_train_z, np.zeros_like(X_train_z), y_train_z,
        cache, lambda_F, 1.0, tau, fc_only=True,
    )

    fp_test_z = _predict_msancr(
        X_test_z, np.zeros_like(X_test_z),
        X_train_z, np.zeros_like(X_train_z),
        alpha, cache, lambda_F, 1.0, tau, fc_only=True,
    )
    return fp_test_z * y_std + y_mean


def _solve_sc_and_predict(
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha_SC: float,
) -> np.ndarray:
    """Solve SC Ridge and return test predictions."""
    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
    sc_model = Ridge(alpha=alpha_SC, fit_intercept=True)
    sc_model.fit(X_sc_train_z, y[train_idx])
    return sc_model.predict(X_sc_test_z)


def _generate_fusion_oof(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    fp_cache,
    fp_lambda_F: float,
    fp_tau: float,
    sc_alpha: float,
    seed: int,
    outer_fold: int,
    n_fusion_folds: int = N_FUSION_FOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cross-fitted OOF predictions for FP and SC branches."""
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 777)
    perm = rng.permutation(len(train_idx))
    fold_sizes = np.full(n_fusion_folds, len(train_idx) // n_fusion_folds)
    fold_sizes[: len(train_idx) % n_fusion_folds] += 1

    n_train = len(train_idx)
    fp_oof = np.full(n_train, np.nan, dtype=np.float64)
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)
    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    current = 0
    for k in range(n_fusion_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        a_k = train_idx[a_local]
        v_k = train_idx[v_local]

        # FP prediction on V_k
        fp_pred = _solve_fc_and_predict(
            X_fc, y, a_k, v_k, fp_cache, fp_lambda_F, fp_tau,
        )
        # SC prediction on V_k
        sc_pred = _solve_sc_and_predict(
            X_sc, y, a_k, v_k, sc_alpha,
        )

        v_local_idx = np.array([train_local_map[int(idx)] for idx in v_k])
        fp_oof[v_local_idx] = fp_pred
        sc_oof[v_local_idx] = sc_pred

        current = stop

    return fp_oof, sc_oof


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    all_split_metrics = []
    all_adaptive_selections = []
    split_count = 0
    timing_samples = []

    outer_splits = make_outer_splits(
        n_subjects=N_SUBJECTS, seeds=DEV_SEEDS, n_outer_folds=N_OUTER_FOLDS,
    )
    logger.info(f"Total splits: {len(outer_splits)} x {len(TASKS)} targets = {TOTAL_SPLITS}")

    for task_key, task_info in TASKS.items():
        y = task_info["y"]
        roi_prior = task_info["prior"]
        display = task_info["display"]

        logger.info("=" * 60)
        logger.info(f"Target: {display} ({task_key})")
        logger.info("=" * 60)

        # Build adaptive_prior_cache ONCE for this target
        adaptive_cache = build_adaptive_prior_cache(
            roi_prior, n_rois=N_ROI, top_k=TOP_K, gamma=GAMMA_FIXED,
            epsilon=DIAGONAL_EPSILON,
        )
        logger.info(f"  Adaptive prior cache built: n_active={adaptive_cache.n_active}")

        for seed, fold, train_idx, test_idx in outer_splits:
            split_count += 1

            # Check checkpoint
            ckpt = _load_checkpoint(task_key, seed, fold)
            if ckpt is not None:
                logger.info(
                    f"  [{split_count}/{TOTAL_SPLITS}] SKIP (checkpoint exists): "
                    f"{task_key} seed={seed} fold={fold}"
                )
                all_split_metrics.append(ckpt["split_metric"])
                all_adaptive_selections.append(ckpt["adaptive_selection"])
                continue

            t_split = time.time()
            logger.info(
                f"  [{split_count}/{TOTAL_SPLITS}] {task_key} seed={seed} fold={fold} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}"
            )

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Inner selection folds for Model A and Model B
            inner_folds_a = make_inner_selection_folds(
                train_idx, seed, fold, 0, N_INNER_FOLDS,
            )
            inner_folds_b = make_inner_selection_folds(
                train_idx, seed, fold, 1, N_INNER_FOLDS,
            )

            # =================================================================
            # Model A: baseline (rho=0, tau=0)
            # =================================================================
            cache_a = build_cache_for_rho(adaptive_cache, rho=0.0)

            # Select lambda_F
            lambda_F_a, sel_info_a = _select_lambda_fc_only(
                X_fc, y, train_idx, cache_a, LAMBDA_F_GRID, tau=0.0,
                inner_folds=inner_folds_a,
            )

            # Select alpha_SC
            alpha_SC_a, sc_sel_a = _select_alpha_sc(
                X_sc, y, train_idx, ALPHA_SC_GRID, inner_folds_a,
            )

            # Solve FC on full training
            fp_test_a = _solve_fc_and_predict(
                X_fc, y, train_idx, test_idx, cache_a, lambda_F_a, tau=0.0,
            )

            # Solve SC on full training
            sc_test_a = _solve_sc_and_predict(
                X_sc, y, train_idx, test_idx, alpha_SC_a,
            )

            # Fusion OOF
            fp_oof_a, sc_oof_a = _generate_fusion_oof(
                X_fc, X_sc, y, train_idx, cache_a, lambda_F_a, 0.0,
                alpha_SC_a, seed, fold,
            )

            # Fusion weights
            weights_a, _ = search_fusion_weights(
                y_train, {"FP": fp_oof_a, "SC": sc_oof_a}, ["FP", "SC"],
            )
            fused_test_a = weights_a["FP"] * fp_test_a + weights_a["SC"] * sc_test_a

            # Metrics
            m_fp_a = prediction_metrics(y_test, fp_test_a)
            m_sc_a = prediction_metrics(y_test, sc_test_a)
            m_fused_a = prediction_metrics(y_test, fused_test_a)

            # =================================================================
            # Model B: adaptive (rho, tau, lambda_F selection with one-SE)
            # =================================================================
            sel_b = _select_adaptive_params_inner(
                adaptive_cache, X_fc, y, train_idx,
                RHO_GRID, TAU_GRID, LAMBDA_F_GRID, inner_folds_b,
            )
            rho_b = sel_b["best_rho"]
            tau_b = sel_b["best_tau"]
            lambda_F_b = sel_b["best_lambda_F"]

            # Select alpha_SC (same grid, inner folds)
            alpha_SC_b, sc_sel_b = _select_alpha_sc(
                X_sc, y, train_idx, ALPHA_SC_GRID, inner_folds_b,
            )

            # Solve FC with selected params
            cache_b = build_cache_for_rho(adaptive_cache, rho_b)
            fp_test_b = _solve_fc_and_predict(
                X_fc, y, train_idx, test_idx, cache_b, lambda_F_b, tau_b,
            )

            # Solve SC
            sc_test_b = _solve_sc_and_predict(
                X_sc, y, train_idx, test_idx, alpha_SC_b,
            )

            # Fusion OOF
            fp_oof_b, sc_oof_b = _generate_fusion_oof(
                X_fc, X_sc, y, train_idx, cache_b, lambda_F_b, tau_b,
                alpha_SC_b, seed, fold,
            )

            # Fusion weights
            weights_b, _ = search_fusion_weights(
                y_train, {"FP": fp_oof_b, "SC": sc_oof_b}, ["FP", "SC"],
            )
            fused_test_b = weights_b["FP"] * fp_test_b + weights_b["SC"] * sc_test_b

            # Metrics
            m_fp_b = prediction_metrics(y_test, fp_test_b)
            m_sc_b = prediction_metrics(y_test, sc_test_b)
            m_fused_b = prediction_metrics(y_test, fused_test_b)

            # =================================================================
            # Save split metrics
            # =================================================================
            split_metric = {
                "task": task_key,
                "seed": seed,
                "outer_fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                # Model A
                "model_a_lambda_F": lambda_F_a,
                "model_a_alpha_SC": alpha_SC_a,
                "model_a_w_fp": weights_a["FP"],
                "model_a_w_sc": weights_a["SC"],
                "model_a_fp_pearson": m_fp_a["pearson"],
                "model_a_fp_rmse": m_fp_a["rmse"],
                "model_a_fp_mae": m_fp_a["mae"],
                "model_a_sc_pearson": m_sc_a["pearson"],
                "model_a_sc_rmse": m_sc_a["rmse"],
                "model_a_sc_mae": m_sc_a["mae"],
                "model_a_fused_pearson": m_fused_a["pearson"],
                "model_a_fused_rmse": m_fused_a["rmse"],
                "model_a_fused_mae": m_fused_a["mae"],
                # Model B
                "model_b_rho": rho_b,
                "model_b_tau": tau_b,
                "model_b_lambda_F": lambda_F_b,
                "model_b_alpha_SC": alpha_SC_b,
                "model_b_w_fp": weights_b["FP"],
                "model_b_w_sc": weights_b["SC"],
                "model_b_fp_pearson": m_fp_b["pearson"],
                "model_b_fp_rmse": m_fp_b["rmse"],
                "model_b_fp_mae": m_fp_b["mae"],
                "model_b_sc_pearson": m_sc_b["pearson"],
                "model_b_sc_rmse": m_sc_b["rmse"],
                "model_b_sc_mae": m_sc_b["mae"],
                "model_b_fused_pearson": m_fused_b["pearson"],
                "model_b_fused_rmse": m_fused_b["rmse"],
                "model_b_fused_mae": m_fused_b["mae"],
                # Delta
                "delta_fused_pearson": m_fused_b["pearson"] - m_fused_a["pearson"],
                "delta_fused_rmse": m_fused_b["rmse"] - m_fused_a["rmse"],
                "delta_fused_mae": m_fused_b["mae"] - m_fused_a["mae"],
            }
            all_split_metrics.append(split_metric)

            adaptive_selection = {
                "task": task_key,
                "seed": seed,
                "outer_fold": fold,
                "selected_rho": rho_b,
                "selected_tau": tau_b,
                "selected_lambda_F": lambda_F_b,
                "best_max_rho": sel_b["best_max_candidate"]["rho"],
                "best_max_tau": sel_b["best_max_candidate"]["tau"],
                "best_max_lambda_F": sel_b["best_max_candidate"]["lambda_F"],
                "best_max_pearson": sel_b["best_max_candidate"]["mean_pearson"],
                "best_se_pearson": sel_b["best_se_candidate"]["mean_pearson"],
            }
            all_adaptive_selections.append(adaptive_selection)

            # Save checkpoint
            _save_checkpoint(task_key, seed, fold, {
                "split_metric": split_metric,
                "adaptive_selection": adaptive_selection,
            })

            elapsed_split = time.time() - t_split
            timing_samples.append(elapsed_split)

            logger.info(
                f"    Model A: lambda_F={lambda_F_a}, alpha_SC={alpha_SC_a:.1f}, "
                f"fused_r={m_fused_a['pearson']:.4f}"
            )
            logger.info(
                f"    Model B: rho={rho_b}, tau={tau_b}, lambda_F={lambda_F_b}, "
                f"alpha_SC={alpha_SC_b:.1f}, fused_r={m_fused_b['pearson']:.4f}"
            )
            logger.info(f"    delta_r={split_metric['delta_fused_pearson']:+.4f} ({elapsed_split:.1f}s)")

            # Print timing/ETA after first 2 splits per target
            if len(timing_samples) == 2:
                mean_time = np.mean(timing_samples)
                remaining = TOTAL_SPLITS - split_count
                eta_s = mean_time * remaining
                logger.info(
                    f"  ETA: ~{eta_s / 60:.1f} min "
                    f"({mean_time:.1f}s/split, {remaining} splits remaining)"
                )

    # =================================================================
    # Save outputs
    # =================================================================
    logger.info("=" * 60)
    logger.info("Saving outputs...")
    logger.info("=" * 60)

    # split_metrics.csv
    split_df = pd.DataFrame(all_split_metrics)
    split_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    logger.info(f"  split_metrics.csv: {len(split_df)} rows")

    # seed_metrics.csv (4 seeds x 2 models x 2 targets)
    seed_rows = []
    for task_key in TASKS:
        task_splits = split_df[split_df["task"] == task_key]
        for seed in DEV_SEEDS:
            seed_splits = task_splits[task_splits["seed"] == seed]
            if seed_splits.empty:
                continue
            for model_prefix, model_name in [
                ("model_a", "baseline"),
                ("model_b", "adaptive"),
            ]:
                seed_rows.append({
                    "task": task_key,
                    "seed": seed,
                    "model": model_name,
                    "mean_fused_pearson": float(seed_splits[f"{model_prefix}_fused_pearson"].mean()),
                    "mean_fused_rmse": float(seed_splits[f"{model_prefix}_fused_rmse"].mean()),
                    "mean_fused_mae": float(seed_splits[f"{model_prefix}_fused_mae"].mean()),
                    "mean_fp_pearson": float(seed_splits[f"{model_prefix}_fp_pearson"].mean()),
                    "mean_sc_pearson": float(seed_splits[f"{model_prefix}_sc_pearson"].mean()),
                    "n_folds": len(seed_splits),
                })
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    logger.info(f"  seed_metrics.csv: {len(seed_df)} rows")

    # adaptive_selection_summary.csv
    sel_df = pd.DataFrame(all_adaptive_selections)
    sel_df.to_csv(OUTPUT_DIR / "adaptive_selection_summary.csv", index=False)
    logger.info(f"  adaptive_selection_summary.csv: {len(sel_df)} rows")

    # Plots
    _generate_plots(split_df, seed_df)

    # VALIDATION_REPORT.json
    validation = {
        "experiment": "Phase 2A Adaptive Prior Pilot",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_splits_total": len(all_split_metrics),
        "dev_seeds": DEV_SEEDS,
        "n_outer_folds": N_OUTER_FOLDS,
        "n_targets": len(TASKS),
        "grids": {
            "RHO_GRID": RHO_GRID,
            "TAU_GRID": TAU_GRID,
            "LAMBDA_F_GRID": LAMBDA_F_GRID,
            "ALPHA_SC_GRID": ALPHA_SC_GRID,
        },
        "summary": {},
    }

    for task_key in TASKS:
        task_splits = split_df[split_df["task"] == task_key]
        a_fused = task_splits["model_a_fused_pearson"].values
        b_fused = task_splits["model_b_fused_pearson"].values
        delta = b_fused - a_fused
        validation["summary"][task_key] = {
            "display": TASKS[task_key]["display"],
            "n_splits": len(task_splits),
            "baseline_mean_fused_pearson": float(np.mean(a_fused)),
            "adaptive_mean_fused_pearson": float(np.mean(b_fused)),
            "mean_delta_pearson": float(np.mean(delta)),
            "std_delta_pearson": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
            "frac_positive_delta": float(np.mean(delta > 0)),
            "adaptive_selections": {
                "mean_rho": float(sel_df[sel_df["task"] == task_key]["selected_rho"].mean()),
                "mean_tau": float(sel_df[sel_df["task"] == task_key]["selected_tau"].mean()),
                "mean_lambda_F": float(sel_df[sel_df["task"] == task_key]["selected_lambda_F"].mean()),
            },
        }

    with open(OUTPUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)
    logger.info("  VALIDATION_REPORT.json saved")

    # COMPLETE marker
    (OUTPUT_DIR / "COMPLETE").write_text(
        f"Phase 2A adaptive prior pilot completed at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )

    # ZIP
    zip_path = OUTPUT_DIR / "phase2a_adaptive_prior_pilot.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fname in files:
                if fname.endswith(".zip"):
                    continue
                fpath = Path(root) / fname
                zf.write(fpath, fpath.relative_to(OUTPUT_DIR))
    logger.info(f"  ZIP saved: {zip_path}")

    elapsed = time.time() - t0
    logger.info(f"Phase 2A complete in {elapsed:.1f}s")
    logger.info(f"All outputs saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _generate_plots(split_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # 1. Delta fused Pearson r per split
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key, panel in [
        (axes[0], "working_memory", "A"),
        (axes[1], "fluid_intelligence", "B"),
    ]:
        task_df = split_df[split_df["task"] == task_key]
        deltas = task_df["delta_fused_pearson"].values
        x = np.arange(1, len(deltas) + 1)
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_xlabel("Split")
        ax.set_ylabel(r"$\Delta r$ (Adaptive $-$ Baseline)")
        ax.set_title(
            f"({panel}) {TASKS[task_key]['display']}", fontsize=10, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=6)

    fig.tight_layout()
    fig.savefig(plot_dir / "fig_phase2a_delta_fused_pearson.pdf")
    fig.savefig(plot_dir / "fig_phase2a_delta_fused_pearson.png")
    plt.close(fig)

    # 2. Selected rho distribution
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key, panel in [
        (axes2[0], "working_memory", "A"),
        (axes2[1], "fluid_intelligence", "B"),
    ]:
        task_sel = split_df[split_df["task"] == task_key]
        rho_vals = task_sel["model_b_rho"].values
        tau_vals = task_sel["model_b_tau"].values

        ax2_twin = ax.twinx()
        bins_rho = np.arange(-0.125, 1.25, 0.25)
        ax.hist(rho_vals, bins=bins_rho, alpha=0.7, color="#4C72B0",
                edgecolor="black", linewidth=0.3, label=r"$\rho$")
        ax.set_xlabel(r"Selected $\rho$")
        ax.set_ylabel("Count", color="#4C72B0")
        ax.set_title(
            f"({panel}) {TASKS[task_key]['display']}", fontsize=10, fontweight="bold",
        )

        ax2_twin.hist(tau_vals, bins=20, alpha=0.4, color="#C44E52",
                       edgecolor="black", linewidth=0.3, label=r"$\tau$")
        ax2_twin.set_ylabel(r"Count ($\tau$)", color="#C44E52")

    fig2.tight_layout()
    fig2.savefig(plot_dir / "fig_phase2a_param_distribution.pdf")
    fig2.savefig(plot_dir / "fig_phase2a_param_distribution.png")
    plt.close(fig2)

    # 3. Baseline vs Adaptive fused Pearson scatter
    fig3, axes3 = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key, panel in [
        (axes3[0], "working_memory", "A"),
        (axes3[1], "fluid_intelligence", "B"),
    ]:
        task_df = split_df[split_df["task"] == task_key]
        a_r = task_df["model_a_fused_pearson"].values
        b_r = task_df["model_b_fused_pearson"].values
        lims = [min(a_r.min(), b_r.min()) - 0.01, max(a_r.max(), b_r.max()) + 0.01]
        ax.scatter(a_r, b_r, alpha=0.6, s=30, color="#1f77b4",
                   edgecolor="black", linewidth=0.3)
        ax.plot(lims, lims, "--", color="gray", linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)
        ax.set_xlabel("Baseline fused $r$")
        ax.set_ylabel("Adaptive fused $r$")
        ax.set_title(
            f"({panel}) {TASKS[task_key]['display']}", fontsize=10, fontweight="bold",
        )

    fig3.tight_layout()
    fig3.savefig(plot_dir / "fig_phase2a_baseline_vs_adaptive.pdf")
    fig3.savefig(plot_dir / "fig_phase2a_baseline_vs_adaptive.png")
    plt.close(fig3)

    logger.info(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
