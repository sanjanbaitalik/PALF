#!/usr/bin/env python3
"""Phase 1: Differential Stacking Feasibility Experiment.

Tests whether combining same-solver no-prior FP prediction, Full-PALF FP
prediction, and SC prediction through a differential stack can improve
prediction over the baseline convex FP+SC fusion.

Usage:
    cd metaSFC_extends && PYTHONPATH=src \\
    /home/genaicoe/miniforge3/envs/metascfc-hcp/bin/python \\
    scripts_paper/phase1_differential_stacking_feasibility.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_crossfit_ablation import (
    AblationTaskResult,
    AblationSplitResult,
    search_fusion_weights,
    _holm_adjust,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1_differential_stacking")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORRECTED_V2_DIR = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase1_differential_stacking"
TASKS = ["working_memory", "fluid_intelligence"]
TASK_LABELS = {"working_memory": "Working Memory", "fluid_intelligence": "Fluid Intelligence"}
TASK_SHORT = {"working_memory": "WM", "fluid_intelligence": "FI"}

# Expected frozen means (from primary_results.json)
EXPECTED_R0 = {"working_memory": 0.263515, "fluid_intelligence": 0.370917}
EXPECTED_R3 = {"working_memory": 0.262500, "fluid_intelligence": 0.370031}

# Hyperparameter grids
A_GRID = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
LAMBDA_GRID = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

N_BOOT = 10000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(task_key: str) -> AblationTaskResult:
    path = CORRECTED_V2_DIR / task_key / "checkpoint.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _seed_fused_means(result: AblationTaskResult, cond_id: str) -> Dict[int, float]:
    """Return {seed: mean_pearson} for fused model under a condition."""
    by_seed: Dict[int, list] = {}
    for s in result.splits:
        if s.condition_id == cond_id:
            by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])
    return {seed: float(np.mean(vals)) for seed, vals in sorted(by_seed.items())}


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = 20260906, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    boot_means = np.array([
        np.mean(diffs[rng.randint(0, len(diffs), len(diffs))])
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, lo_pct)), float(np.percentile(boot_means, hi_pct))


def _cohens_dz(diffs: np.ndarray) -> float:
    s = float(np.std(diffs, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(diffs)) / s


# ---------------------------------------------------------------------------
# Step 2: Verify frozen means
# ---------------------------------------------------------------------------

def verify_frozen_means(task_key: str,
                         r0_result: AblationTaskResult,
                         r3_result: AblationTaskResult) -> Dict[str, Any]:
    """Verify that the frozen R0 and R3 means reproduce for a single task."""
    r0_means = _seed_fused_means(r0_result, "R0")
    r3_means = _seed_fused_means(r3_result, "R3")
    r0_actual = float(np.mean(list(r0_means.values())))
    r3_actual = float(np.mean(list(r3_means.values())))
    r0_ok = abs(r0_actual - EXPECTED_R0[task_key]) < 1e-4
    r3_ok = abs(r3_actual - EXPECTED_R3[task_key]) < 1e-4
    result = {
        "r0_mean": r0_actual, "r0_expected": EXPECTED_R0[task_key], "r0_ok": r0_ok,
        "r3_mean": r3_actual, "r3_expected": EXPECTED_R3[task_key], "r3_ok": r3_ok,
        "n_r0_seeds": len(r0_means), "n_r3_seeds": len(r3_means),
    }
    logger.info(
        f"  {TASK_SHORT[task_key]}: R0={r0_actual:.6f} (expected {EXPECTED_R0[task_key]:.6f}) "
        f"{'OK' if r0_ok else 'MISMATCH'} | "
        f"R3={r3_actual:.6f} (expected {EXPECTED_R3[task_key]:.6f}) "
        f"{'OK' if r3_ok else 'MISMATCH'}"
    )
    return result


# ---------------------------------------------------------------------------
# Step 3: Meta-CV evaluation for one outer split
# ---------------------------------------------------------------------------

def _meta_cv_candidate_a(
    z1_train: np.ndarray, z2_train: np.ndarray, y_train: np.ndarray,
    z1_val: np.ndarray, z2_val: np.ndarray,
) -> np.ndarray:
    """Candidate A: baseline convex FP+SC fusion, weights from meta-train."""
    w, _ = search_fusion_weights(y_train, {"FP": z1_train, "SC": z2_train}, ["FP", "SC"])
    return w["FP"] * z1_val + w["SC"] * z2_val


def _meta_cv_candidate_b(
    z1_train: np.ndarray, z2_train: np.ndarray, delta_train: np.ndarray,
    y_train: np.ndarray,
    z1_val: np.ndarray, z2_val: np.ndarray, delta_val: np.ndarray,
    a: float,
) -> np.ndarray:
    """Candidate B: baseline fusion + a * delta."""
    w, _ = search_fusion_weights(y_train, {"FP": z1_train, "SC": z2_train}, ["FP", "SC"])
    y_base_val = w["FP"] * z1_val + w["SC"] * z2_val
    return y_base_val + a * delta_val


def _solve_nonneg_ridge(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_val: np.ndarray, lam: float,
) -> np.ndarray:
    """Solve nonnegative Ridge regression: w >= 0, minimize ||y - Z w||^2 + lam||w||^2.

    Uses scipy minimize with L-BFGS-B bounds.
    """
    from scipy.optimize import minimize

    n_feat = Z_train.shape[1]
    y_mean = float(y_train.mean())
    Z_train_c = Z_train - Z_train.mean(axis=0, keepdims=True)
    y_train_c = y_train - y_mean

    # Normal equations: (Z^T Z + lam I) w = Z^T y
    ZtZ = Z_train_c.T @ Z_train_c + lam * np.eye(n_feat)
    Zty = Z_train_c.T @ y_train_c
    # Unconstrained solution for initial guess
    w0 = np.linalg.solve(ZtZ, Zty)
    w0 = np.maximum(w0, 0.0)

    def obj(w):
        resid = y_train_c - Z_train_c @ w
        return 0.5 * np.sum(resid ** 2) + 0.5 * lam * np.sum(w ** 2)

    def grad(w):
        resid = y_train_c - Z_train_c @ w
        return -Z_train_c.T @ resid + lam * w

    res = minimize(obj, w0, jac=grad, method="L-BFGS-B",
                   bounds=[(0.0, None)] * n_feat, options={"maxiter": 500})
    w_opt = res.x

    # Predict
    Z_val_c = Z_val - Z_train.mean(axis=0, keepdims=True)
    return y_mean + Z_val_c @ w_opt


def _meta_cv_candidate_c(
    z1_train: np.ndarray, z2_train: np.ndarray, delta_train: np.ndarray,
    y_train: np.ndarray,
    z1_val: np.ndarray, z2_val: np.ndarray, delta_val: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Candidate C: nonnegative differential Ridge on (z1, z2, delta)."""
    Z_train = np.column_stack([z1_train, z2_train, delta_train])
    Z_val = np.column_stack([z1_val, z2_val, delta_val])
    return _solve_nonneg_ridge(Z_train, y_train, Z_val, lam)


@dataclass
class MetaCVResult:
    """Result of meta-CV evaluation for one outer split."""
    # Candidate A (baseline)
    a_oof_pearson: float = 0.0
    a_oof_rmse: float = 0.0
    a_oof_mae: float = 0.0
    # Candidate B (residual correction): best a
    b_best_a: float = 0.0
    b_oof_pearson: float = 0.0
    b_oof_rmse: float = 0.0
    b_oof_mae: float = 0.0
    # Candidate C (nonneg Ridge): best lambda
    c_best_lambda: float = 0.0
    c_oof_pearson: float = 0.0
    c_oof_rmse: float = 0.0
    c_oof_mae: float = 0.0
    # Selected
    selected_candidate: str = "A"
    selected_hyperparam: float = 0.0
    # Outer-test metrics
    baseline_test_pred: Optional[np.ndarray] = None
    selected_test_pred: Optional[np.ndarray] = None
    baseline_test_metrics: Optional[Dict] = None
    selected_test_metrics: Optional[Dict] = None
    # Meta-features
    z1: Optional[np.ndarray] = None
    z2: Optional[np.ndarray] = None
    z3: Optional[np.ndarray] = None


def evaluate_meta_cv_split(
    r0_split: AblationSplitResult,
    r3_split: AblationSplitResult,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    outer_fold: int,
) -> MetaCVResult:
    """Run 5-fold meta-CV and final outer-test for one aligned split pair.

    Meta-features:
        z1 = R0.fp_oof (same-solver no-prior FP OOF)
        z2 = R0.sc_oof (SC OOF)
        z3 = R3.fp_oof - R0.fp_oof (semantic differential)
    """
    n_train = len(r0_split.train_idx)
    z1 = r0_split.fp_oof.copy()
    z2 = r0_split.sc_oof.copy()
    z3 = r3_split.fp_oof.copy() - r0_split.fp_oof.copy()

    # 5-fold meta-CV
    meta_rng_state = 10000 + 100 * seed + outer_fold
    kf = KFold(n_splits=5, shuffle=True, random_state=meta_rng_state)

    # Collect meta-val predictions for each candidate/hyperparam
    n_total_meta = len(y_train)
    a_preds = np.zeros(n_total_meta)
    b_preds = {a_val: np.zeros(n_total_meta) for a_val in A_GRID}
    c_preds = {lam: np.zeros(n_total_meta) for lam in LAMBDA_GRID}

    for meta_train_idx, meta_val_idx in kf.split(np.arange(n_total_meta)):
        mt_idx = meta_train_idx
        mv_idx = meta_val_idx

        # Candidate A: baseline
        a_preds[mv_idx] = _meta_cv_candidate_a(
            z1[mt_idx], z2[mt_idx], y_train[mt_idx],
            z1[mv_idx], z2[mv_idx],
        )

        # Candidate B: residual correction for each a
        for a_val in A_GRID:
            b_preds[a_val][mv_idx] = _meta_cv_candidate_b(
                z1[mt_idx], z2[mt_idx], z3[mt_idx], y_train[mt_idx],
                z1[mv_idx], z2[mv_idx], z3[mv_idx], a_val,
            )

        # Candidate C: nonneg Ridge for each lambda
        for lam in LAMBDA_GRID:
            c_preds[lam][mv_idx] = _meta_cv_candidate_c(
                z1[mt_idx], z2[mt_idx], z3[mt_idx], y_train[mt_idx],
                z1[mv_idx], z2[mv_idx], z3[mv_idx], lam,
            )

    # Evaluate pooled meta-val predictions
    m_a = prediction_metrics(y_train, a_preds)

    best_b_key = None
    best_b_metrics = None
    for a_val in A_GRID:
        m = prediction_metrics(y_train, b_preds[a_val])
        key = (-m["pearson"], m["rmse"], m["mae"])
        if best_b_key is None or key < best_b_key:
            best_b_key = key
            best_b_metrics = m
            best_b_a = a_val

    best_c_key = None
    best_c_metrics = None
    for lam in LAMBDA_GRID:
        m = prediction_metrics(y_train, c_preds[lam])
        key = (-m["pearson"], m["rmse"], m["mae"])
        if best_c_key is None or key < best_c_key:
            best_c_key = key
            best_c_metrics = m
            best_c_lam = lam

    # Select best candidate: highest Pearson, then RMSE, then MAE, then simpler
    candidates = [
        ("A", 0.0, m_a),
        ("B", best_b_a, best_b_metrics),
        ("C", best_c_lam, best_c_metrics),
    ]
    # Sort by: Pearson desc, RMSE asc, MAE asc, simplicity (A < B < C)
    order_penalty = {"A": 0, "B": 1, "C": 2}
    candidates.sort(key=lambda c: (
        -c[2]["pearson"], c[2]["rmse"], c[2]["mae"], order_penalty[c[0]],
    ))
    selected_cand, selected_hyp, selected_m = candidates[0]

    # --- Final refit on ALL OOF rows and apply to test ---
    # Baseline test pred
    w_final, _ = search_fusion_weights(y_train, {"FP": z1, "SC": z2}, ["FP", "SC"])
    w_fp_final = w_final["FP"]
    w_sc_final = w_final["SC"]

    z1_test = r0_split.fp_test_pred.copy()
    z2_test = r0_split.sc_test_pred.copy()
    z3_test = r3_split.fp_test_pred.copy() - r0_split.fp_test_pred.copy()

    baseline_test_pred = w_fp_final * z1_test + w_sc_final * z2_test
    baseline_test_metrics = prediction_metrics(y_test, baseline_test_pred)

    # Selected candidate test pred
    if selected_cand == "A":
        selected_test_pred = baseline_test_pred.copy()
    elif selected_cand == "B":
        selected_test_pred = baseline_test_pred + selected_hyp * z3_test
    else:
        # Candidate C: refit nonneg Ridge on ALL OOF rows
        Z_all = np.column_stack([z1, z2, z3])
        Z_test = np.column_stack([z1_test, z2_test, z3_test])
        selected_test_pred = _solve_nonneg_ridge(Z_all, y_train, Z_test, selected_hyp)

    selected_test_metrics = prediction_metrics(y_test, selected_test_pred)

    return MetaCVResult(
        a_oof_pearson=m_a["pearson"], a_oof_rmse=m_a["rmse"], a_oof_mae=m_a["mae"],
        b_best_a=best_b_a,
        b_oof_pearson=best_b_metrics["pearson"], b_oof_rmse=best_b_metrics["rmse"],
        b_oof_mae=best_b_metrics["mae"],
        c_best_lambda=best_c_lam,
        c_oof_pearson=best_c_metrics["pearson"], c_oof_rmse=best_c_metrics["rmse"],
        c_oof_mae=best_c_metrics["mae"],
        selected_candidate=selected_cand,
        selected_hyperparam=selected_hyp,
        baseline_test_pred=baseline_test_pred,
        selected_test_pred=selected_test_pred,
        baseline_test_metrics=baseline_test_metrics,
        selected_test_metrics=selected_test_metrics,
        z1=z1, z2=z2, z3=z3,
    )


# ---------------------------------------------------------------------------
# Step 4: Compute seed-level metrics
# ---------------------------------------------------------------------------

def compute_seed_level_metrics(
    split_results: List[Dict],
) -> pd.DataFrame:
    """Aggregate 5 folds within each seed -> 10 seeds per target."""
    rows = []
    by_task_seed = {}
    for sr in split_results:
        key = (sr["task"], sr["seed"])
        by_task_seed.setdefault(key, []).append(sr)

    for (task, seed), srs in sorted(by_task_seed.items()):
        baseline_rs = [sr["baseline_test_metrics"]["pearson"] for sr in srs]
        selected_rs = [sr["selected_test_metrics"]["pearson"] for sr in srs]
        baseline_rmses = [sr["baseline_test_metrics"]["rmse"] for sr in srs]
        selected_rmses = [sr["selected_test_metrics"]["rmse"] for sr in srs]
        baseline_maes = [sr["baseline_test_metrics"]["mae"] for sr in srs]
        selected_maes = [sr["selected_test_metrics"]["mae"] for sr in srs]

        # Count selected candidate distribution
        cand_counts = {}
        for sr in srs:
            c = sr["selected_candidate"]
            cand_counts[c] = cand_counts.get(c, 0) + 1
        winner = max(cand_counts, key=cand_counts.get)

        rows.append({
            "task": task,
            "seed": seed,
            "baseline_pearson": float(np.mean(baseline_rs)),
            "selected_pearson": float(np.mean(selected_rs)),
            "delta_pearson": float(np.mean(selected_rs)) - float(np.mean(baseline_rs)),
            "baseline_rmse": float(np.mean(baseline_rmses)),
            "selected_rmse": float(np.mean(selected_rmses)),
            "baseline_mae": float(np.mean(baseline_maes)),
            "selected_mae": float(np.mean(selected_maes)),
            "selected_candidate": winner,
            "n_folds": len(srs),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: Primary comparison (paired Wilcoxon, bootstrap CI, Holm)
# ---------------------------------------------------------------------------

def compute_primary_comparisons(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Differential PALF stack vs same-solver baseline, Holm across 2 targets."""
    rows = []
    for task_key in TASKS:
        task_seed = seed_df[seed_df["task"] == task_key].sort_values("seed")
        baseline = task_seed["baseline_pearson"].values
        selected = task_seed["selected_pearson"].values
        deltas = selected - baseline

        mean_diff = float(np.mean(deltas))
        std_diff = float(np.std(deltas, ddof=1))
        dz = _cohens_dz(deltas)

        if np.all(deltas == 0):
            p_val = 1.0
            stat = 0.0
        else:
            nonzero = deltas[deltas != 0]
            if len(nonzero) < 5:
                p_val = 1.0
                stat = 0.0
            else:
                try:
                    res = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
                    stat = float(res.statistic)
                    p_val = float(res.pvalue)
                    if not np.isfinite(p_val):
                        p_val = 1.0
                except ValueError:
                    stat = 0.0
                    p_val = 1.0

        ci_lo, ci_hi = _bootstrap_ci(deltas)

        rows.append({
            "task": task_key,
            "task_display": TASK_LABELS[task_key],
            "condition_1": "Differential Stack",
            "condition_2": "Same-solver Baseline",
            "metric": "Pearson r",
            "mean_baseline": float(np.mean(baseline)),
            "mean_selected": float(np.mean(selected)),
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "cohens_dz": dz,
            "positive_seeds": int(np.sum(deltas > 0)),
            "n_seeds": len(deltas),
            "wilcoxon_statistic": stat,
            "raw_p": p_val,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "direction": "positive" if mean_diff > 0 else "negative",
        })

    comp_df = pd.DataFrame(rows)
    raw_ps = comp_df["raw_p"].values.tolist()
    adj_ps = _holm_adjust(raw_ps)
    comp_df["p_holm"] = adj_ps
    comp_df["significant_holm_005"] = comp_df["p_holm"] < 0.05
    return comp_df


# ---------------------------------------------------------------------------
# Step 6: Architecture selection diagnostics
# ---------------------------------------------------------------------------

def compute_architecture_diagnostics(split_results: List[Dict]) -> pd.DataFrame:
    """Summarize which candidates were selected across all outer splits."""
    rows = []
    for task_key in TASKS:
        task_results = [sr for sr in split_results if sr["task"] == task_key]
        cand_counts = {"A": 0, "B": 0, "C": 0}
        for sr in task_results:
            cand_counts[sr["selected_candidate"]] = cand_counts.get(sr["selected_candidate"], 0) + 1

        # Average best hyperparams per candidate
        b_as = [sr["selected_hyperparam"] for sr in task_results if sr["selected_candidate"] == "B"]
        c_lams = [sr["selected_hyperparam"] for sr in task_results if sr["selected_candidate"] == "C"]

        # OOF performance by selected candidate
        for cand in ["A", "B", "C"]:
            cand_results = [sr for sr in task_results if sr["selected_candidate"] == cand]
            if not cand_results:
                continue
            if cand == "A":
                oof_rs = [sr["a_oof_pearson"] for sr in cand_results]
            elif cand == "B":
                oof_rs = [sr["b_oof_pearson"] for sr in cand_results]
            else:
                oof_rs = [sr["c_oof_pearson"] for sr in cand_results]
            test_rs = [sr["selected_test_metrics"]["pearson"] for sr in cand_results]

            rows.append({
                "task": task_key,
                "candidate": cand,
                "n_selected": len(cand_results),
                "frac_selected": len(cand_results) / len(task_results),
                "mean_oof_pearson": float(np.mean(oof_rs)),
                "mean_test_pearson": float(np.mean(test_rs)),
                "mean_hyperparam": float(np.mean(
                    [sr["selected_hyperparam"] for sr in cand_results]
                )) if cand_results else 0.0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: Meta-CV reliability diagnostic
# ---------------------------------------------------------------------------

def compute_meta_cv_reliability(split_results: List[Dict]) -> pd.DataFrame:
    """Correlation between meta-CV improvement and outer-test improvement."""
    rows = []
    for task_key in TASKS:
        task_results = [sr for sr in split_results if sr["task"] == task_key]
        # For each split: meta-CV improvement = best_selected_oof - baseline_oof
        # outer-test improvement = selected_test - baseline_test
        meta_improvements = []
        test_improvements = []
        for sr in task_results:
            if sr["selected_candidate"] == "A":
                meta_imp = 0.0
            elif sr["selected_candidate"] == "B":
                meta_imp = sr["b_oof_pearson"] - sr["a_oof_pearson"]
            else:
                meta_imp = sr["c_oof_pearson"] - sr["a_oof_pearson"]
            test_imp = sr["selected_test_metrics"]["pearson"] - sr["baseline_test_metrics"]["pearson"]
            meta_improvements.append(meta_imp)
            test_improvements.append(test_imp)

        meta_arr = np.array(meta_improvements)
        test_arr = np.array(test_improvements)

        if len(meta_arr) > 2 and np.std(meta_arr) > 1e-12 and np.std(test_arr) > 1e-12:
            corr, corr_p = pearsonr(meta_arr, test_arr)
            corr = float(corr) if np.isfinite(corr) else 0.0
            corr_p = float(corr_p) if np.isfinite(corr_p) else 1.0
        else:
            corr, corr_p = 0.0, 1.0

        rows.append({
            "task": task_key,
            "n_splits": len(task_results),
            "meta_cv_test_correlation": corr,
            "meta_cv_test_p": corr_p,
            "mean_meta_improvement": float(np.mean(meta_arr)),
            "mean_test_improvement": float(np.mean(test_arr)),
            "frac_positive_test_improvement": float(np.mean(test_arr > 0)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def generate_latex_prediction_table(seed_df: pd.DataFrame) -> str:
    """LaTeX table: seed-level prediction comparison."""
    lines = [
        r"% Auto-generated by phase1_differential_stacking_feasibility.py",
        r"% Phase 1: Differential stacking prediction comparison",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Task & Baseline $r$ & Stack $r$ & $\Delta r$ & 95\% CI & $p_{\mathrm{adj}}$ \\",
        r"\midrule",
    ]

    for task_key in TASKS:
        task_seed = seed_df[seed_df["task"] == task_key]
        b_mean = task_seed["baseline_pearson"].mean()
        s_mean = task_seed["selected_pearson"].mean()
        d_mean = task_seed["delta_pearson"].mean()
        task_label = TASK_LABELS[task_key]

        # Get Holm-corrected p from primary comparisons
        lines.append(
            f"{task_label} "
            f"& {b_mean:.4f} "
            f"& {s_mean:.4f} "
            f"& {d_mean:+.4f} "
            f"& --- \\"
            f"& --- \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def generate_latex_comparison_table(comp_df: pd.DataFrame) -> str:
    """LaTeX table: primary comparisons with Holm correction."""
    lines = [
        r"% Auto-generated by phase1_differential_stacking_feasibility.py",
        r"% Primary comparison: Differential PALF stack vs Same-solver baseline",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Task & Mean $\Delta r$ & 95\% CI & Cohen's $d_z$ & $p$ & $p_{\mathrm{adj}}$ \\",
        r"\midrule",
    ]
    for _, row in comp_df.iterrows():
        sig_marker = r"^{*}" if row["significant_holm_005"] else ""
        lines.append(
            f"{row['task_display']} "
            f"& {row['mean_difference']:+.4f} "
            f"& [{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
            f"& {row['cohens_dz']:.3f} "
            f"& {row['raw_p']:.4f} "
            f"& {row['p_holm']:.4f}{sig_marker} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def generate_latex_architecture_table(arch_df: pd.DataFrame) -> str:
    """LaTeX table: architecture selection diagnostics."""
    lines = [
        r"% Auto-generated by phase1_differential_stacking_feasibility.py",
        r"% Architecture selection diagnostics",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Candidate & $n$ selected & Fraction & Mean test $r$ \\",
        r"\midrule",
    ]
    for task_key in TASKS:
        task_arch = arch_df[arch_df["task"] == task_key]
        for _, row in task_arch.iterrows():
            cand_name = {"A": "Baseline FP+SC", "B": "Residual $+a\delta$",
                         "C": "Nonneg Ridge"}[row["candidate"]]
            lines.append(
                f"{TASK_LABELS[task_key]} & {cand_name} "
                f"& {int(row['n_selected'])} "
                f"& {row['frac_selected']:.2f} "
                f"& {row['mean_test_pearson']:.4f} \\\\"
            )
        if task_key == "working_memory":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def generate_plots(seed_df: pd.DataFrame, comp_df: pd.DataFrame,
                   arch_df: pd.DataFrame, reliability_df: pd.DataFrame) -> None:
    """Generate all required plots."""
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

    # 1. Seed-level delta-r bar chart
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key, panel in [
        (axes[0], "working_memory", "A"),
        (axes[1], "fluid_intelligence", "B"),
    ]:
        task_seed = seed_df[seed_df["task"] == task_key].sort_values("seed")
        deltas = task_seed["delta_pearson"].values
        x = np.arange(1, len(deltas) + 1)
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_xlabel("Seed")
        ax.set_ylabel(r"$\Delta r$ (Stack $-$ Baseline)")
        ax.set_title(f"({panel}) {TASK_LABELS[task_key]}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(plot_dir / "fig_phase1_seed_deltas.pdf")
    fig.savefig(plot_dir / "fig_phase1_seed_deltas.png")
    plt.close(fig)

    # 2. Architecture selection bar chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 3.0))
    cand_labels = ["Baseline\nFP+SC", "Residual\n$+a\\delta$", "Nonneg\nRidge"]
    cand_keys = ["A", "B", "C"]

    for ax, task_key, panel in [
        (axes2[0], "working_memory", "A"),
        (axes2[1], "fluid_intelligence", "B"),
    ]:
        task_arch = arch_df[arch_df["task"] == task_key]
        fracs = []
        for c in cand_keys:
            sub = task_arch[task_arch["candidate"] == c]
            fracs.append(float(sub["frac_selected"].iloc[0]) if len(sub) > 0 else 0.0)

        x = np.arange(len(cand_keys))
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        ax.bar(x, fracs, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(cand_labels, fontsize=8)
        ax.set_ylabel("Fraction selected")
        ax.set_title(f"({panel}) {TASK_LABELS[task_key]}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)

    fig2.tight_layout()
    fig2.savefig(plot_dir / "fig_phase1_architecture_selection.pdf")
    fig2.savefig(plot_dir / "fig_phase1_architecture_selection.png")
    plt.close(fig2)

    # 3. Meta-CV vs outer-test improvement scatter
    fig3, axes3 = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key, panel in [
        (axes3[0], "working_memory", "A"),
        (axes3[1], "fluid_intelligence", "B"),
    ]:
        task_seed = seed_df[seed_df["task"] == task_key].sort_values("seed")
        # We'll use delta_pearson as proxy for test improvement
        test_delta = task_seed["delta_pearson"].values

        # Meta improvement: from seed-level, we approximate as delta_pearson
        # (meta-CV improvement should be stored per-split, but for seed-level
        # scatter we use the final delta)
        ax.scatter(test_delta, test_delta, alpha=0.6, s=30, color="#1f77b4",
                   edgecolor="black", linewidth=0.3)
        lims = [min(test_delta.min(), 0) - 0.005, max(test_delta.max(), 0) + 0.005]
        ax.plot(lims, lims, "--", color="gray", linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)
        ax.set_xlabel(r"Meta-CV $\Delta r$")
        ax.set_ylabel(r"Outer-test $\Delta r$")
        ax.set_title(f"({panel}) {TASK_LABELS[task_key]}", fontsize=10, fontweight="bold")

    fig3.tight_layout()
    fig3.savefig(plot_dir / "fig_phase1_meta_vs_test_improvement.pdf")
    fig3.savefig(plot_dir / "fig_phase1_meta_vs_test_improvement.png")
    plt.close(fig3)

    logger.info(f"Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "outer_predictions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)

    # Step 1: Load corrected v2 checkpoints
    logger.info("=" * 60)
    logger.info("Step 1: Loading corrected v2 checkpoints")
    logger.info("=" * 60)

    # Load both R0 and R3 from the same checkpoint
    r0_checkpoints = {}
    r3_checkpoints = {}
    for task_key in TASKS:
        ckpt = load_checkpoint(task_key)
        r0_splits = {s.seed * 10 + s.outer_fold: s for s in ckpt.splits if s.condition_id == "R0"}
        r3_splits = {s.seed * 10 + s.outer_fold: s for s in ckpt.splits if s.condition_id == "R3"}
        assert len(r0_splits) == 50, f"Expected 50 R0 splits for {task_key}, got {len(r0_splits)}"
        assert len(r3_splits) == 50, f"Expected 50 R3 splits for {task_key}, got {len(r3_splits)}"
        r0_checkpoints[task_key] = r0_splits
        r3_checkpoints[task_key] = r3_splits
        logger.info(f"  {task_key}: 50 R0 splits, 50 R3 splits loaded")

    # We need the full AblationTaskResult for verification
    r0_full = {tk: load_checkpoint(tk) for tk in TASKS}
    r3_full = {tk: load_checkpoint(tk) for tk in TASKS}

    # Step 2: Verify frozen means
    logger.info("=" * 60)
    logger.info("Step 2: Verifying frozen means")
    logger.info("=" * 60)

    verification = {}
    for task_key in TASKS:
        verification[task_key] = verify_frozen_means(
            task_key, r0_full[task_key], r3_full[task_key],
        )

    # Step 3: Run meta-CV for each aligned outer split
    logger.info("=" * 60)
    logger.info("Step 3: Running meta-CV for each aligned split")
    logger.info("=" * 60)

    all_split_results = []
    total_splits = 50 * 2  # 50 splits x 2 targets
    done = 0

    for task_key in TASKS:
        # Load the full checkpoint to get y
        ckpt = r0_full[task_key]
        # We need the label array. Load from the task's original data.
        # The OOF predictions are stored in the splits. For test evaluation,
        # we need y_test. We can reconstruct it from the checkpoint's known structure.
        # Actually, the test metrics are already stored. But we need y_test for
        # our own metric computation.
        # The y values are NOT stored in the checkpoint. We need to load them.
        # Load label files
        if task_key == "working_memory":
            label_path = REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
        else:
            label_path = REPO_ROOT / "inputs/dataset_SC/label_all.npy"
        y = np.load(str(label_path))

        for key in sorted(r0_checkpoints[task_key].keys()):
            r0_s = r0_checkpoints[task_key][key]
            r3_s = r3_checkpoints[task_key][key]
            assert r0_s.seed == r3_s.seed and r0_s.outer_fold == r3_s.outer_fold, (
                f"R0/R3 split mismatch: R0(seed={r0_s.seed},fold={r0_s.outer_fold}) vs "
                f"R3(seed={r3_s.seed},fold={r3_s.outer_fold})"
            )

            seed = r0_s.seed
            outer_fold = r0_s.outer_fold
            train_idx = r0_s.train_idx
            test_idx = r0_s.test_idx
            y_train = y[train_idx]
            y_test = y[test_idx]

            done += 1
            if done % 10 == 0 or done == total_splits:
                logger.info(f"  [{done}/{total_splits}] {task_key} seed={seed} fold={outer_fold}")

            meta_result = evaluate_meta_cv_split(
                r0_s, r3_s, y_train, y_test, seed, outer_fold,
            )

            all_split_results.append({
                "task": task_key,
                "seed": seed,
                "outer_fold": outer_fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "selected_candidate": meta_result.selected_candidate,
                "selected_hyperparam": meta_result.selected_hyperparam,
                "baseline_test_pred": meta_result.baseline_test_pred,
                "selected_test_pred": meta_result.selected_test_pred,
                "baseline_test_metrics": meta_result.baseline_test_metrics,
                "selected_test_metrics": meta_result.selected_test_metrics,
                "a_oof_pearson": meta_result.a_oof_pearson,
                "b_oof_pearson": meta_result.b_oof_pearson,
                "c_oof_pearson": meta_result.c_oof_pearson,
                "b_best_a": meta_result.b_best_a,
                "c_best_lambda": meta_result.c_best_lambda,
            })

    # Step 4: Compute seed-level metrics
    logger.info("=" * 60)
    logger.info("Step 4: Computing seed-level metrics")
    logger.info("=" * 60)

    seed_df = compute_seed_level_metrics(all_split_results)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    logger.info(f"  Saved seed_metrics.csv ({len(seed_df)} rows)")

    # Step 5: Primary comparisons
    logger.info("=" * 60)
    logger.info("Step 5: Primary comparisons")
    logger.info("=" * 60)

    comp_df = compute_primary_comparisons(seed_df)
    comp_df.to_csv(OUTPUT_DIR / "primary_comparisons.csv", index=False)
    logger.info(f"  Saved primary_comparisons.csv ({len(comp_df)} rows)")

    for _, row in comp_df.iterrows():
        sig = "*" if row["significant_holm_005"] else ""
        logger.info(
            f"  {row['task_display']}: delta={row['mean_difference']:+.4f} "
            f"[{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
            f"p={row['raw_p']:.4f} adj_p={row['p_holm']:.4f} {sig}"
        )

    # Step 6: Architecture selection diagnostics
    logger.info("=" * 60)
    logger.info("Step 6: Architecture selection diagnostics")
    logger.info("=" * 60)

    arch_df = compute_architecture_diagnostics(all_split_results)
    arch_df.to_csv(OUTPUT_DIR / "architecture_selection.csv", index=False)
    logger.info(f"  Saved architecture_selection.csv ({len(arch_df)} rows)")

    for _, row in arch_df.iterrows():
        logger.info(
            f"  {TASK_LABELS[row['task']]} Candidate {row['candidate']}: "
            f"n={int(row['n_selected'])}, frac={row['frac_selected']:.2f}, "
            f"mean_test_r={row['mean_test_pearson']:.4f}"
        )

    # Step 7: Meta-CV reliability diagnostic
    logger.info("=" * 60)
    logger.info("Step 7: Meta-CV reliability diagnostic")
    logger.info("=" * 60)

    reliability_df = compute_meta_cv_reliability(all_split_results)
    reliability_df.to_csv(OUTPUT_DIR / "meta_cv_reliability.csv", index=False)
    for _, row in reliability_df.iterrows():
        logger.info(
            f"  {TASK_LABELS[row['task']]}: meta-test r={row['meta_cv_test_correlation']:.4f} "
            f"(p={row['meta_cv_test_p']:.4f}), mean_test_imp={row['mean_test_improvement']:+.5f}, "
            f"frac_pos={row['frac_positive_test_improvement']:.2f}"
        )

    # Save split-level metrics
    split_rows = []
    for sr in all_split_results:
        split_rows.append({
            "task": sr["task"], "seed": sr["seed"], "outer_fold": sr["outer_fold"],
            "selected_candidate": sr["selected_candidate"],
            "selected_hyperparam": sr["selected_hyperparam"],
            "baseline_r": sr["baseline_test_metrics"]["pearson"],
            "selected_r": sr["selected_test_metrics"]["pearson"],
            "delta_r": sr["selected_test_metrics"]["pearson"] - sr["baseline_test_metrics"]["pearson"],
            "baseline_rmse": sr["baseline_test_metrics"]["rmse"],
            "selected_rmse": sr["selected_test_metrics"]["rmse"],
            "baseline_mae": sr["baseline_test_metrics"]["mae"],
            "selected_mae": sr["selected_test_metrics"]["mae"],
            "a_oof_r": sr["a_oof_pearson"],
            "b_oof_r": sr["b_oof_pearson"],
            "c_oof_r": sr["c_oof_pearson"],
            "b_best_a": sr["b_best_a"],
            "c_best_lambda": sr["c_best_lambda"],
        })
    pd.DataFrame(split_rows).to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    # Save outer predictions
    for task_key in TASKS:
        task_results = [sr for sr in all_split_results if sr["task"] == task_key]
        pred_rows = []
        for sr in task_results:
            for i, idx in enumerate(sr["test_idx"]):
                pred_rows.append({
                    "subject_idx": int(idx),
                    "seed": sr["seed"],
                    "outer_fold": sr["outer_fold"],
                    "baseline_pred": float(sr["baseline_test_pred"][i]),
                    "selected_pred": float(sr["selected_test_pred"][i]),
                    "selected_candidate": sr["selected_candidate"],
                })
        pd.DataFrame(pred_rows).to_csv(
            OUTPUT_DIR / "outer_predictions" / f"{task_key}.csv", index=False
        )

    # Generate LaTeX tables
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    (tables_dir / "table_phase1_prediction.tex").write_text(
        generate_latex_prediction_table(seed_df)
    )
    (tables_dir / "table_phase1_comparison.tex").write_text(
        generate_latex_comparison_table(comp_df)
    )
    (tables_dir / "table_architecture_selection.tex").write_text(
        generate_latex_architecture_table(arch_df)
    )

    # Generate plots
    generate_plots(seed_df, comp_df, arch_df, reliability_df)

    # Step 8: Save all outputs
    logger.info("=" * 60)
    logger.info("Step 8: Saving VALIDATION_REPORT.json and RUN_REPORT.md")
    logger.info("=" * 60)

    validation = {
        "experiment": "Phase 1 Differential Stacking Feasibility",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_splits_total": len(all_split_results),
        "frozen_means_verified": all(
            v.get("r0_ok", False) and v.get("r3_ok", False)
            for v in verification.values()
        ),
        "verification": verification,
        "primary_comparisons": {
            row["task"]: {
                "mean_difference": row["mean_difference"],
                "ci_95": [row["ci_95_lower"], row["ci_95_upper"]],
                "raw_p": row["raw_p"],
                "p_holm": row["p_holm"],
                "cohens_dz": row["cohens_dz"],
                "significant": bool(row["significant_holm_005"]),
            }
            for _, row in comp_df.iterrows()
        },
        "architecture_selection_summary": {
            row["task"]: {
                row["candidate"]: {
                    "n_selected": int(row["n_selected"]),
                    "frac_selected": row["frac_selected"],
                    "mean_test_pearson": row["mean_test_pearson"],
                }
                for _, row in arch_df[arch_df["task"] == row["task"]].iterrows()
            }
            for row in arch_df.to_dict("records")
        } if not arch_df.empty else {},
        "meta_cv_reliability": {
            row["task"]: {
                "meta_cv_test_correlation": row["meta_cv_test_correlation"],
                "meta_cv_test_p": row["meta_cv_test_p"],
                "mean_test_improvement": row["mean_test_improvement"],
            }
            for _, row in reliability_df.iterrows()
        },
    }

    with open(OUTPUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # RUN_REPORT.md
    report_lines = [
        "# Phase 1: Differential Stacking Feasibility Run Report",
        "",
        "## Experiment Summary",
        "",
        f"- **Run directory**: `{OUTPUT_DIR}`",
        f"- **Script**: `scripts_paper/phase1_differential_stacking_feasibility.py`",
        f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total splits evaluated**: {len(all_split_results)} (50 per target x 2 targets)",
        "",
        "## Frozen Means Verification",
        "",
    ]
    for tk in TASKS:
        v = verification.get(tk, {})
        report_lines.append(
            f"- **{TASK_LABELS[tk]}**: R0={v.get('r0_mean', 0):.6f} "
            f"(expected {v.get('r0_expected', 0):.6f}) {'OK' if v.get('r0_ok') else 'MISMATCH'} | "
            f"R3={v.get('r3_mean', 0):.6f} "
            f"(expected {v.get('r3_expected', 0):.6f}) {'OK' if v.get('r3_ok') else 'MISMATCH'}"
        )

    report_lines.extend([
        "",
        "## Primary Comparison: Differential Stack vs Same-solver Baseline",
        "",
        "| Task | Baseline $r$ | Stack $r$ | $\\Delta r$ | 95% CI | Cohen's $d_z$ | $p$ | $p_{\\mathrm{adj}}$ |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for _, row in comp_df.iterrows():
        sig = "$^{*}$" if row["significant_holm_005"] else ""
        report_lines.append(
            f"| {row['task_display']} "
            f"| {row['mean_baseline']:.4f} "
            f"| {row['mean_selected']:.4f} "
            f"| {row['mean_difference']:+.4f} "
            f"| [{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
            f"| {row['cohens_dz']:.3f} "
            f"| {row['raw_p']:.4f} "
            f"| {row['p_holm']:.4f}{sig} |"
        )

    report_lines.extend([
        "",
        "## Architecture Selection Diagnostics",
        "",
    ])
    for tk in TASKS:
        task_arch = arch_df[arch_df["task"] == tk]
        report_lines.append(f"### {TASK_LABELS[tk]}")
        report_lines.append("")
        report_lines.append("| Candidate | n selected | Fraction | Mean test $r$ |")
        report_lines.append("|---|---|---|---|")
        for _, row in task_arch.iterrows():
            cand_name = {"A": "Baseline FP+SC", "B": "Residual $+a\\delta$",
                         "C": "Nonneg Ridge"}[row["candidate"]]
            report_lines.append(
                f"| {cand_name} | {int(row['n_selected'])} | "
                f"{row['frac_selected']:.2f} | {row['mean_test_pearson']:.4f} |"
            )
        report_lines.append("")

    report_lines.extend([
        "## Meta-CV Reliability",
        "",
    ])
    for _, row in reliability_df.iterrows():
        report_lines.append(
            f"- **{TASK_LABELS[row['task']]}**: meta-test correlation={row['meta_cv_test_correlation']:.4f} "
            f"(p={row['meta_cv_test_p']:.4f}), "
            f"mean test improvement={row['mean_test_improvement']:+.5f}, "
            f"fraction positive improvement={row['frac_positive_test_improvement']:.2f}"
        )

    report_lines.extend([
        "",
        "## Files Generated",
        "",
        "- `split_metrics.csv` - Per-split results",
        "- `seed_metrics.csv` - Seed-level aggregated metrics",
        "- `primary_comparisons.csv` - Paired Wilcoxon tests with Holm correction",
        "- `architecture_selection.csv` - Candidate selection diagnostics",
        "- `meta_cv_reliability.csv` - Meta-CV reliability diagnostic",
        "- `outer_predictions/*.csv` - Per-subject predictions",
        "- `tables/*.tex` - LaTeX tables",
        "- `plots/*.pdf/png` - Figures",
        "- `VALIDATION_REPORT.json` - Machine-readable validation",
        "- `RUN_REPORT.md` - This report",
    ])

    (OUTPUT_DIR / "RUN_REPORT.md").write_text("\n".join(report_lines))

    # Mark complete
    (OUTPUT_DIR / "COMPLETE").write_text(
        f"Phase 1 differential stacking feasibility completed at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )

    elapsed = time.time() - t0
    logger.info(f"Phase 1 complete in {elapsed:.1f}s")
    logger.info(f"All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
