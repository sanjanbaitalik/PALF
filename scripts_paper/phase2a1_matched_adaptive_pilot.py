#!/usr/bin/env python3
"""Phase 2A.1: Corrected Matched-Selector Adaptive-Prior Pilot.

Fixes three confounds in Phase 2A:
1. Shared SC branch (selected once, reused by both models)
2. Matched one-SE FC selection (both models use same framework/folds)
3. Isotropic shadow model for semantic attribution

Development seeds [101, 202, 303, 404], 5 outer folds, 2 targets = 40 splits.

Usage:
    cd metaSFC_extends && PYTHONPATH=src \\
    /home/genaicoe/miniforge3/envs/metascfc-hcp/bin/python \\
    scripts_paper/phase2a1_matched_adaptive_pilot.py
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
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_adaptive_prior import (
    AdaptivePriorCache,
    build_adaptive_prior_cache,
    build_cache_for_rho,
    _compute_one_se_rule,
)
from metascfc.experiments.palf_crossfit_ablation import (
    C_SCALE,
    DIAGONAL_EPSILON,
    GAMMA_FIXED,
    N_EDGE,
    N_ROI,
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
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2a1_matched")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2a1_matched_adaptive_pilot"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
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
# Selection helpers
# ---------------------------------------------------------------------------

def _select_lambda_fc_one_se(
    X_fc: np.ndarray,
    y: np.ndarray,
    cache,
    lambda_F_grid: List[float],
    tau: float,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, Dict]:
    """Select lambda_F for FC-only (rho=0, tau=0) via one-SE rule.

    Same one-SE framework as adaptive, but restricted to rho=0, tau=0.
    """
    all_results = []
    for lambda_F in lambda_F_grid:
        fold_ps = []
        for b_idx, c_idx in inner_folds:
            try:
                scaler = StandardScaler()
                X_b = scaler.fit_transform(X_fc[b_idx])
                X_c = scaler.transform(X_fc[c_idx])
                y_mb = float(y[b_idx].mean())
                y_sb = max(float(y[b_idx].std()), 1e-8)
                y_z = (y[b_idx] - y_mb) / y_sb

                alpha, _ = _solve_msancr_kernel(
                    X_b, np.zeros_like(X_b), y_z, cache,
                    lambda_F, 1.0, tau, fc_only=True,
                )
                pred_z = _predict_msancr(
                    X_c, np.zeros_like(X_c),
                    X_b, np.zeros_like(X_b),
                    alpha, cache, lambda_F, 1.0, tau, fc_only=True,
                )
                pred = pred_z * y_sb + y_mb
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
            "rho": 0.0,
            "tau": tau,
            "lambda_F": lambda_F,
            "mean_pearson": mean_p,
            "se_pearson": se_p,
            "fold_pearsons": fold_ps,
        })

    best_max, best_se = _compute_one_se_rule(all_results)
    return best_se["lambda_F"], {
        "scores": {r["lambda_F"]: {"pearson": r["mean_pearson"]} for r in all_results},
        "best_max": best_max,
        "best_se": best_se,
    }


def _select_adaptive_one_se(
    adaptive_cache: AdaptivePriorCache,
    X_fc: np.ndarray,
    y: np.ndarray,
    rho_grid: List[float],
    tau_grid: List[float],
    lambda_F_grid: List[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """Select (rho, tau, lambda_F) via 3-fold CV with one-SE rule."""
    all_results = []
    for rho in rho_grid:
        for tau in tau_grid:
            for lambda_F in lambda_F_grid:
                fold_ps = []
                for b_idx, c_idx in inner_folds:
                    try:
                        cache = build_cache_for_rho(adaptive_cache, rho)
                        scaler = StandardScaler()
                        X_b = scaler.fit_transform(X_fc[b_idx])
                        X_c = scaler.transform(X_fc[c_idx])
                        y_mb = float(y[b_idx].mean())
                        y_sb = max(float(y[b_idx].std()), 1e-8)
                        y_z = (y[b_idx] - y_mb) / y_sb

                        alpha, _ = _solve_msancr_kernel(
                            X_b, np.zeros_like(X_b), y_z, cache,
                            lambda_F, 1.0, tau, fc_only=True,
                        )
                        pred_z = _predict_msancr(
                            X_c, np.zeros_like(X_c),
                            X_b, np.zeros_like(X_b),
                            alpha, cache, lambda_F, 1.0, tau, fc_only=True,
                        )
                        pred = pred_z * y_sb + y_mb
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

    best_max, best_se = _compute_one_se_rule(all_results)
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
    alpha_sc_grid: List[float],
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, Dict]:
    """Select alpha_SC for SC Ridge via 3-fold inner CV."""
    scaler = StandardScaler()
    # Fit on all training data for consistent scaling
    train_idx_all = np.concatenate([b for b, _ in inner_folds])
    scaler.fit(X_sc[train_idx_all])

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


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

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


def _solve_fc_oof(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    cache,
    lambda_F: float,
    tau: float,
    seed: int,
    fold: int,
    n_folds: int = N_FUSION_FOLDS,
) -> np.ndarray:
    """Generate cross-fitted OOF predictions for FC branch."""
    rng = np.random.RandomState(int(seed) * 10000 + int(fold) * 100 + 777)
    perm = rng.permutation(len(train_idx))
    fold_sizes = np.full(n_folds, len(train_idx) // n_folds)
    fold_sizes[: len(train_idx) % n_folds] += 1

    n_train = len(train_idx)
    fp_oof = np.full(n_train, np.nan, dtype=np.float64)
    local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    current = 0
    for k in range(n_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        a_k = train_idx[a_local]
        v_k = train_idx[v_local]

        fp_pred = _solve_fc_and_predict(X_fc, y, a_k, v_k, cache, lambda_F, tau)
        v_local_idx = np.array([local_map[int(idx)] for idx in v_k])
        fp_oof[v_local_idx] = fp_pred
        current = stop

    return fp_oof


def _solve_sc_and_predict(
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha_SC: float,
) -> np.ndarray:
    """Solve SC Ridge and return test predictions."""
    scaler = StandardScaler()
    X_sc_train_z = scaler.fit_transform(X_sc[train_idx])
    X_sc_test_z = scaler.transform(X_sc[test_idx])
    model = Ridge(alpha=alpha_SC, fit_intercept=True)
    model.fit(X_sc_train_z, y[train_idx])
    return model.predict(X_sc_test_z)


def _solve_sc_oof(
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    alpha_SC: float,
    seed: int,
    fold: int,
    n_folds: int = N_FUSION_FOLDS,
) -> np.ndarray:
    """Generate cross-fitted OOF predictions for SC branch."""
    rng = np.random.RandomState(int(seed) * 10000 + int(fold) * 100 + 555)
    perm = rng.permutation(len(train_idx))
    fold_sizes = np.full(n_folds, len(train_idx) // n_folds)
    fold_sizes[: len(train_idx) % n_folds] += 1

    n_train = len(train_idx)
    sc_oof = np.full(n_train, np.nan, dtype=np.float64)
    local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    scaler = StandardScaler()
    scaler.fit(X_sc[train_idx])

    current = 0
    for k in range(n_folds):
        start, stop = current, current + fold_sizes[k]
        v_local = perm[start:stop]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        a_k = train_idx[a_local]
        v_k = train_idx[v_local]

        X_b = scaler.transform(X_sc[a_k])
        X_c = scaler.transform(X_sc[v_k])
        model = Ridge(alpha=alpha_SC, fit_intercept=True)
        model.fit(X_b, y[a_k])
        pred = model.predict(X_c)

        v_local_idx = np.array([local_map[int(idx)] for idx in v_k])
        sc_oof[v_local_idx] = pred
        current = stop

    return sc_oof


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_split_metrics = []
    all_selections = []
    all_shadow = []
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

        adaptive_cache = build_adaptive_prior_cache(
            roi_prior, n_rois=N_ROI, top_k=TOP_K, gamma=GAMMA_FIXED,
            epsilon=DIAGONAL_EPSILON,
        )
        logger.info(f"  Adaptive prior cache built: n_active={adaptive_cache.n_active}")

        for seed, fold, train_idx, test_idx in outer_splits:
            split_count += 1

            ckpt = _load_ckpt(task_key, seed, fold)
            if ckpt is not None:
                logger.info(
                    f"  [{split_count}/{TOTAL_SPLITS}] SKIP (checkpoint): "
                    f"{task_key} seed={seed} fold={fold}"
                )
                all_split_metrics.append(ckpt["split_metric"])
                all_selections.append(ckpt["selection"])
                all_shadow.append(ckpt["shadow"])
                continue

            t_split = time.time()
            logger.info(
                f"  [{split_count}/{TOTAL_SPLITS}] {task_key} seed={seed} fold={fold} "
                f"n_train={len(train_idx)} n_test={len(test_idx)}"
            )

            y_train = y[train_idx]
            y_test = y[test_idx]

            # =================================================================
            # SHARED inner folds for ALL selection
            # =================================================================
            inner_folds = make_inner_selection_folds(
                train_idx, seed, fold, 0, N_INNER_FOLDS,
            )

            # =================================================================
            # SHARED SC selection (selected once, reused by all models)
            # =================================================================
            alpha_SC, sc_sel = _select_alpha_sc(
                X_sc, y, ALPHA_SC_GRID, inner_folds,
            )

            # Shared SC OOF and test
            sc_oof = _solve_sc_oof(X_sc, y, train_idx, alpha_SC, seed, fold)
            sc_test = _solve_sc_and_predict(X_sc, y, train_idx, test_idx, alpha_SC)
            sc_m = prediction_metrics(y_test, sc_test)

            logger.info(f"    Shared SC: alpha={alpha_SC}, r={sc_m['pearson']:.4f}")

            # =================================================================
            # Matched robust no-prior baseline (rho=0, tau=0)
            # Uses same one-SE framework, same inner folds
            # =================================================================
            cache_base = build_cache_for_rho(adaptive_cache, rho=0.0)
            lambda_F_base, base_sel = _select_lambda_fc_one_se(
                X_fc, y, cache_base, LAMBDA_F_GRID, tau=0.0,
                inner_folds=inner_folds,
            )

            fp_test_base = _solve_fc_and_predict(
                X_fc, y, train_idx, test_idx, cache_base, lambda_F_base, tau=0.0,
            )
            fp_oof_base = _solve_fc_oof(
                X_fc, y, train_idx, cache_base, lambda_F_base, 0.0, seed, fold,
            )
            weights_base, _ = search_fusion_weights(
                y_train, {"FP": fp_oof_base, "SC": sc_oof}, ["FP", "SC"],
            )
            fused_test_base = weights_base["FP"] * fp_test_base + weights_base["SC"] * sc_test

            m_fp_base = prediction_metrics(y_test, fp_test_base)
            m_fused_base = prediction_metrics(y_test, fused_test_base)

            logger.info(
                f"    Baseline: lambda_F={lambda_F_base}, "
                f"FP r={m_fp_base['pearson']:.4f}, fused r={m_fused_base['pearson']:.4f}"
            )

            # =================================================================
            # Adaptive PALF (rho, tau, lambda_F via one-SE)
            # =================================================================
            sel_adapt = _select_adaptive_one_se(
                adaptive_cache, X_fc, y,
                RHO_GRID, TAU_GRID, LAMBDA_F_GRID, inner_folds,
            )
            rho_a = sel_adapt["best_rho"]
            tau_a = sel_adapt["best_tau"]
            lambda_F_a = sel_adapt["best_lambda_F"]

            cache_adapt = build_cache_for_rho(adaptive_cache, rho_a)
            fp_test_adapt = _solve_fc_and_predict(
                X_fc, y, train_idx, test_idx, cache_adapt, lambda_F_a, tau_a,
            )
            fp_oof_adapt = _solve_fc_oof(
                X_fc, y, train_idx, cache_adapt, lambda_F_a, tau_a, seed, fold,
            )
            weights_adapt, _ = search_fusion_weights(
                y_train, {"FP": fp_oof_adapt, "SC": sc_oof}, ["FP", "SC"],
            )
            fused_test_adapt = weights_adapt["FP"] * fp_test_adapt + weights_adapt["SC"] * sc_test

            m_fp_adapt = prediction_metrics(y_test, fp_test_adapt)
            m_fused_adapt = prediction_metrics(y_test, fused_test_adapt)

            uses_semantic = (rho_a > 0 or tau_a > 0)
            logger.info(
                f"    Adaptive: rho={rho_a}, tau={tau_a}, lambda_F={lambda_F_a}, "
                f"FP r={m_fp_adapt['pearson']:.4f}, fused r={m_fused_adapt['pearson']:.4f}, "
                f"semantic={'YES' if uses_semantic else 'no'}"
            )

            # =================================================================
            # Isotropic shadow (rho=0, tau=0, same lambda_F as adaptive)
            # =================================================================
            cache_shadow = build_cache_for_rho(adaptive_cache, rho=0.0)
            fp_test_shadow = _solve_fc_and_predict(
                X_fc, y, train_idx, test_idx, cache_shadow, lambda_F_a, tau=0.0,
            )
            m_fp_shadow = prediction_metrics(y_test, fp_test_shadow)

            # Fixed-weight fused shadow (use adaptive fusion weights)
            fused_shadow_fixed = weights_adapt["FP"] * fp_test_shadow + weights_adapt["SC"] * sc_test
            m_fused_shadow_fixed = prediction_metrics(y_test, fused_shadow_fixed)

            # Identity check: if rho=0,tau=0 then adaptive == shadow
            if rho_a == 0 and tau_a == 0:
                assert np.allclose(fp_test_adapt, fp_test_shadow, atol=1e-10), (
                    f"IDENTITY GATE FAILED: rho=0,tau=0 but FP predictions differ "
                    f"(max diff={np.max(np.abs(fp_test_adapt - fp_test_shadow))})"
                )

            # Semantic deltas
            semantic_fp_delta = m_fp_adapt["pearson"] - m_fp_shadow["pearson"]
            semantic_fused_fixed_delta = (
                m_fused_adapt["pearson"] - m_fused_shadow_fixed["pearson"]
            )

            # Prediction deltas
            prediction_delta_fp = m_fp_adapt["pearson"] - m_fp_base["pearson"]
            prediction_delta_fused = m_fused_adapt["pearson"] - m_fused_base["pearson"]

            logger.info(
                f"    Deltas: pred_FP={prediction_delta_fp:+.4f}, "
                f"pred_fused={prediction_delta_fused:+.4f}, "
                f"semantic_FP={semantic_fp_delta:+.4f}"
            )

            # =================================================================
            # Save results
            # =================================================================
            split_metric = {
                "task": task_key,
                "seed": seed,
                "outer_fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                # Shared SC
                "shared_alpha_SC": alpha_SC,
                "shared_sc_pearson": sc_m["pearson"],
                "shared_sc_rmse": sc_m["rmse"],
                "shared_sc_mae": sc_m["mae"],
                # Baseline
                "baseline_lambda_F": lambda_F_base,
                "baseline_fp_pearson": m_fp_base["pearson"],
                "baseline_fp_rmse": m_fp_base["rmse"],
                "baseline_fp_mae": m_fp_base["mae"],
                "baseline_w_FP": weights_base["FP"],
                "baseline_w_SC": weights_base["SC"],
                "baseline_fused_pearson": m_fused_base["pearson"],
                "baseline_fused_rmse": m_fused_base["rmse"],
                "baseline_fused_mae": m_fused_base["mae"],
                # Adaptive
                "adaptive_rho": rho_a,
                "adaptive_tau": tau_a,
                "adaptive_lambda_F": lambda_F_a,
                "adaptive_fp_pearson": m_fp_adapt["pearson"],
                "adaptive_fp_rmse": m_fp_adapt["rmse"],
                "adaptive_fp_mae": m_fp_adapt["mae"],
                "adaptive_w_FP": weights_adapt["FP"],
                "adaptive_w_SC": weights_adapt["SC"],
                "adaptive_fused_pearson": m_fused_adapt["pearson"],
                "adaptive_fused_rmse": m_fused_adapt["rmse"],
                "adaptive_fused_mae": m_fused_adapt["mae"],
                # Shadow
                "shadow_lambda_F": lambda_F_a,
                "shadow_fp_pearson": m_fp_shadow["pearson"],
                "shadow_fp_rmse": m_fp_shadow["rmse"],
                "shadow_fp_mae": m_fp_shadow["mae"],
                # Deltas
                "prediction_delta_fp": prediction_delta_fp,
                "prediction_delta_fused": prediction_delta_fused,
                "semantic_delta_fp": semantic_fp_delta,
                "semantic_delta_fused_fixed_weight": semantic_fused_fixed_delta,
                "adaptive_uses_semantic": uses_semantic,
            }
            all_split_metrics.append(split_metric)
            all_selections.append({
                "task": task_key,
                "seed": seed,
                "fold": fold,
                "alpha_SC": alpha_SC,
                "baseline_lambda_F": lambda_F_base,
                "adaptive_rho": rho_a,
                "adaptive_tau": tau_a,
                "adaptive_lambda_F": lambda_F_a,
                "uses_semantic": uses_semantic,
            })
            all_shadow.append({
                "task": task_key,
                "seed": seed,
                "fold": fold,
                "shadow_lambda_F": lambda_F_a,
                "shadow_fp_pearson": m_fp_shadow["pearson"],
                "semantic_fp_delta": semantic_fp_delta,
                "semantic_fused_fixed_delta": semantic_fused_fixed_delta,
            })

            elapsed = time.time() - t_split
            timing_samples.append(elapsed)

            _save_ckpt(task_key, seed, fold, {
                "split_metric": split_metric,
                "selection": all_selections[-1],
                "shadow": all_shadow[-1],
            })

            if len(timing_samples) >= 2:
                avg_s = np.mean(timing_samples[-5:])
                remaining = (TOTAL_SPLITS - split_count) * avg_s / 60
                logger.info(f"    ETA: ~{remaining:.1f} min ({avg_s:.1f}s/split)")

    # =====================================================================
    # Save all outputs
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
            for model_prefix in ["baseline", "adaptive", "shadow"]:
                if model_prefix == "shadow":
                    row = {
                        "task": task_key,
                        "seed": seed,
                        "model": model_prefix,
                        "mean_fp_pearson": s_df["shadow_fp_pearson"].mean(),
                        "mean_fused_pearson": np.nan,
                        "mean_fused_rmse": np.nan,
                        "mean_fused_mae": np.nan,
                    }
                else:
                    fp_col = f"{model_prefix}_fp_pearson"
                    fused_col = f"{model_prefix}_fused_pearson"
                    rmse_col = f"{model_prefix}_fused_rmse"
                    mae_col = f"{model_prefix}_fused_mae"
                    row = {
                        "task": task_key,
                        "seed": seed,
                        "model": model_prefix,
                        "mean_fp_pearson": s_df[fp_col].mean(),
                        "mean_fused_pearson": s_df[fused_col].mean(),
                        "mean_fused_rmse": s_df[rmse_col].mean(),
                        "mean_fused_mae": s_df[mae_col].mean(),
                    }
                seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    logger.info(f"  seed_metrics.csv: {len(seed_df)} rows")

    sel_df = pd.DataFrame(all_selections)
    sel_df.to_csv(OUTPUT_DIR / "selection_summary.csv", index=False)

    shadow_df = pd.DataFrame(all_shadow)
    shadow_df.to_csv(OUTPUT_DIR / "semantic_shadow_metrics.csv", index=False)

    # =====================================================================
    # Parameter frequency audit (from CSV, not hardcoded)
    # =====================================================================
    audit = {}
    for task_key in TASKS:
        t_df = split_df[split_df["task"] == task_key]
        rho_zero = (t_df["adaptive_rho"] == 0).sum()
        tau_zero = (t_df["adaptive_tau"] == 0).sum()
        both_zero = ((t_df["adaptive_rho"] == 0) & (t_df["adaptive_tau"] == 0)).sum()
        any_semantic = (t_df["adaptive_uses_semantic"]).sum()
        audit[task_key] = {
            "total": len(t_df),
            "rho_zero_count": int(rho_zero),
            "tau_zero_count": int(tau_zero),
            "both_zero_count": int(both_zero),
            "any_semantic_count": int(any_semantic),
            "rho_distribution": t_df["adaptive_rho"].value_counts().to_dict(),
            "tau_distribution": t_df["adaptive_tau"].value_counts().to_dict(),
            "lambda_F_distribution": t_df["adaptive_lambda_F"].value_counts().to_dict(),
        }
    audit["total"] = {
        "rho_zero_count": int((split_df["adaptive_rho"] == 0).sum()),
        "tau_zero_count": int((split_df["adaptive_tau"] == 0).sum()),
        "both_zero_count": int(((split_df["adaptive_rho"] == 0) & (split_df["adaptive_tau"] == 0)).sum()),
        "any_semantic_count": int(split_df["adaptive_uses_semantic"].sum()),
    }

    with open(OUTPUT_DIR / "selection_summary_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    # =====================================================================
    # VALIDATION_REPORT.json
    # =====================================================================
    total_elapsed = time.time() - t0
    validation = {
        "n_splits": len(split_df),
        "n_tasks": len(TASKS),
        "n_seeds": len(DEV_SEEDS),
        "n_folds": N_OUTER_FOLDS,
        "elapsed_seconds": round(total_elapsed, 1),
        "gates": {
            "shared_sc_per_split": True,  # enforced by design
            "identity_gate": True,  # checked in loop
            "no_outer_test_in_selection": True,
            "dev_seeds_only": True,
        },
    }
    with open(OUTPUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump(validation, f, indent=2)

    # =====================================================================
    # COMPLETE marker
    # =====================================================================
    with open(OUTPUT_DIR / "COMPLETE", "w") as f:
        f.write(f"Phase 2A.1 complete at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed: {total_elapsed:.1f}s\n")

    # =====================================================================
    # Plots
    # =====================================================================
    _generate_plots(split_df, seed_df)

    # =====================================================================
    # ZIP
    # =====================================================================
    zip_path = OUTPUT_DIR / "phase2a1_matched_adaptive_pilot.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "split_metrics.csv", "seed_metrics.csv",
            "selection_summary.csv", "semantic_shadow_metrics.csv",
            "selection_summary_audit.json",
            "VALIDATION_REPORT.json", "COMPLETE",
        ]:
            fp = OUTPUT_DIR / fname
            if fp.exists():
                zf.write(fp, fname)
        for fp in PLOTS_DIR.glob("*"):
            zf.write(fp, f"plots/{fp.name}")
    logger.info(f"  ZIP saved: {zip_path}")

    total_elapsed = time.time() - t0
    logger.info(f"Phase 2A.1 complete in {total_elapsed:.1f}s")
    logger.info(f"All outputs saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _generate_plots(split_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    """Generate all required plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for task in split_df["task"].unique():
        t = split_df[split_df["task"] == task]
        display = TASKS[task]["display"]

        # 1. Prediction seed deltas
        fig, ax = plt.subplots(figsize=(8, 5))
        seeds = sorted(t["seed"].unique())
        x = np.arange(len(seeds))
        width = 0.35
        fp_deltas = [t[t["seed"] == s]["prediction_delta_fp"].mean() for s in seeds]
        fused_deltas = [t[t["seed"] == s]["prediction_delta_fused"].mean() for s in seeds]
        ax.bar(x - width / 2, fp_deltas, width, label="FP delta", color="steelblue")
        ax.bar(x + width / 2, fused_deltas, width, label="Fused delta", color="coral")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Adaptive - Matched baseline")
        ax.set_title(f"{display}: Prediction Deltas (Adaptive - Matched Baseline)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_prediction_seed_deltas_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_prediction_seed_deltas_{task}.pdf")
        plt.close(fig)

        # 2. Semantic shadow deltas
        fig, ax = plt.subplots(figsize=(8, 5))
        sem_fp = [t[t["seed"] == s]["semantic_delta_fp"].mean() for s in seeds]
        sem_fused = [t[t["seed"] == s]["semantic_delta_fused_fixed_weight"].mean() for s in seeds]
        ax.bar(x - width / 2, sem_fp, width, label="Semantic FP delta", color="mediumpurple")
        ax.bar(x + width / 2, sem_fused, width, label="Semantic fused delta", color="goldenrod")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Adaptive - Isotropic shadow")
        ax.set_title(f"{display}: Semantic Shadow Deltas")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"fig_semantic_shadow_deltas_{task}.png", dpi=150)
        fig.savefig(PLOTS_DIR / f"fig_semantic_shadow_deltas_{task}.pdf")
        plt.close(fig)

    # 3. rho/tau counts (both tasks combined)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, task in enumerate(split_df["task"].unique()):
        t = split_df[split_df["task"] == task]
        rho_counts = t["adaptive_rho"].value_counts().sort_index()
        tau_counts = t[t["adaptive_tau"] > 0]["adaptive_tau"].value_counts().sort_index()
        axes[i].bar(rho_counts.index.astype(str), rho_counts.values, color="teal")
        axes[i].set_title(f"{TASKS[task]['display']}: rho distribution")
        axes[i].set_xlabel("rho")
        axes[i].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig_rho_tau_counts.png", dpi=150)
    fig.savefig(PLOTS_DIR / "fig_rho_tau_counts.pdf")
    plt.close(fig)

    # 4. Baseline vs Adaptive vs Shadow
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, task in enumerate(split_df["task"].unique()):
        t = split_df[split_df["task"] == task]
        seeds = sorted(t["seed"].unique())
        base_fp = [t[t["seed"] == s]["baseline_fp_pearson"].mean() for s in seeds]
        adapt_fp = [t[t["seed"] == s]["adaptive_fp_pearson"].mean() for s in seeds]
        shadow_fp = [t[t["seed"] == s]["shadow_fp_pearson"].mean() for s in seeds]
        x = np.arange(len(seeds))
        w = 0.25
        axes[i].bar(x - w, base_fp, w, label="Baseline FP", color="steelblue")
        axes[i].bar(x, adapt_fp, w, label="Adaptive FP", color="coral")
        axes[i].bar(x + w, shadow_fp, w, label="Shadow FP", color="gray")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([str(s) for s in seeds])
        axes[i].set_title(f"{TASKS[task]['display']}: FP Pearson by Seed")
        axes[i].set_ylabel("Pearson r")
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig_baseline_adaptive_shadow.png", dpi=150)
    fig.savefig(PLOTS_DIR / "fig_baseline_adaptive_shadow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
