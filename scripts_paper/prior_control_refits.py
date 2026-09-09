#!/usr/bin/env python3
"""Prior control fixed-swap refits (Section 7.2).

For each R3 outer split, refit the FC branch on T with each frozen
cross-task, shuffled, and random prior. Rebuild D and L_p for that
control prior, retain gamma 0.5 and product lifting, and use R3's
final selected lambda_F/lambda_L. Keep R3's final SC model and
fusion weights fixed. Evaluate each control on E_test.

Expected: 300 FC refits per task (100 R3 splits × 3 control priors).
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    build_condition_cache,
    GAMMA_FIXED,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _solve_msancr_kernel,
    _predict_msancr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"

PRIOR_DIR = REPO_ROOT / "outputs/priors/llm"
RANDOM_PRIOR_DIR = REPO_ROOT / "outputs/priors/random_prior/aal116"

CONTROL_PRIORS = {
    "cross_task": {
        "working_memory": PRIOR_DIR / "fluid_intelligence_contrastive_qwen3" / "roi_prior.csv",
        "fluid_intelligence": PRIOR_DIR / "working_memory_contrastive_qwen3" / "roi_prior.csv",
        "display": "Cross-task",
    },
    "shuffled": {
        "working_memory": PRIOR_DIR / "working_memory_contrastive_qwen3_shuffled" / "roi_prior.csv",
        "fluid_intelligence": PRIOR_DIR / "fluid_intelligence_contrastive_qwen3_shuffled" / "roi_prior.csv",
        "display": "Shuffled",
    },
    "random": {
        "working_memory": RANDOM_PRIOR_DIR / "roi_prior.csv",
        "fluid_intelligence": RANDOM_PRIOR_DIR / "roi_prior.csv",
        "display": "Random",
    },
}


def load_prior(path: Path) -> np.ndarray:
    """Load ROI prior score column from CSV."""
    df = pd.read_csv(path)
    return df["prior_score"].values.astype(np.float64)


def load_checkpoint(task: str):
    path = OUTPUT_BASE / task / "checkpoint.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def refit_control_prior(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    control_prior: np.ndarray,
    lambda_fc: float,
    lambda_l: float,
    sc_test_pred: np.ndarray,
    fusion_weights: dict,
) -> dict:
    """Refit FC branch with a control prior and evaluate on test set.

    Uses R3's final selected lambda_F/lambda_L, refits FC with the
    control prior's D and L_p. Keeps SC branch fixed.
    """
    from metascfc.experiments.palf_crossfit_ablation import (
        AblationCondition,
        TOP_K,
        DIAGONAL_EPSILON,
    )

    # R3 condition (anisotropy + network)
    condition = CONDITIONS["R3"]

    # Build cache with control prior
    cache = build_condition_cache(control_prior, condition)

    # Standardize features
    scaler = StandardScaler()
    X_fc_train_z = scaler.fit_transform(X_fc[train_idx])
    X_fc_test_z = scaler.transform(X_fc[test_idx])

    # Target transform
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean) / y_std

    # Solve FC branch with control prior
    alpha, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z), y_train_z, cache,
        lambda_fc, 1.0, lambda_l, fc_only=True,
    )

    # Predict test
    fp_test_z = _predict_msancr(
        X_fc_test_z, np.zeros_like(X_fc_test_z),
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha, cache, lambda_fc, 1.0, lambda_l, fc_only=True,
    )
    fp_test = fp_test_z * y_std + y_mean

    # Fuse with fixed SC
    w_fp = fusion_weights["FP"]
    w_sc = fusion_weights["SC"]
    fused_test = w_fp * fp_test + w_sc * sc_test_pred

    y_test = y[test_idx]
    fp_metrics = prediction_metrics(y_test, fp_test)
    fused_metrics = prediction_metrics(y_test, fused_test)

    return {
        "fp_test_pred": fp_test,
        "fused_test_pred": fused_test,
        "fp_metrics": fp_metrics,
        "fused_metrics": fused_metrics,
        "alpha": alpha,
    }


def main():
    # Load data
    log.info("Loading data...")
    fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
    sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
    iu = np.triu_indices(116, k=1)
    X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)

    targets = {
        "working_memory": np.load(REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64),
        "fluid_intelligence": np.load(REPO_ROOT / "inputs/dataset_SC/label_all.npy").astype(np.float64),
    }

    all_results = []

    for task_key, y in targets.items():
        log.info(f"\n{'='*60}")
        log.info(f"Processing {task_key}...")
        log.info(f"{'='*60}")

        result = load_checkpoint(task_key)
        r3_splits = [s for s in result.splits if s.condition_id == "R3"]
        log.info(f"  Found {len(r3_splits)} R3 splits")

        for pri_key, pri_info in CONTROL_PRIORS.items():
            prior_path = pri_info[task_key]
            control_prior = load_prior(prior_path)
            log.info(f"  Control: {pri_info['display']} ({pri_key})")

            t0 = time.time()
            refit_count = 0

            for s in r3_splits:
                # Get R3's final selected parameters
                fp_params = s.fp_final.selected_params
                lambda_fc = fp_params["lambda_fc"]
                lambda_l = fp_params["lambda_l"]

                # Refit with control prior
                refit = refit_control_prior(
                    X_fc, X_sc, y,
                    s.train_idx, s.test_idx,
                    control_prior,
                    lambda_fc, lambda_l,
                    s.sc_test_pred,
                    s.fusion_weights,
                )

                # Compare with R3 matched prior
                r3_fp_r = s.fp_metrics["pearson"]
                r3_fused_r = s.fused_metrics["pearson"]
                ctrl_fp_r = refit["fp_metrics"]["pearson"]
                ctrl_fused_r = refit["fused_metrics"]["pearson"]

                all_results.append({
                    "task": task_key,
                    "seed": s.seed,
                    "outer_fold": s.outer_fold,
                    "control_prior": pri_key,
                    "control_display": pri_info["display"],
                    "lambda_fc": lambda_fc,
                    "lambda_l": lambda_l,
                    # FP branch (unfused)
                    "fp_r_matched": r3_fp_r,
                    "fp_r_control": ctrl_fp_r,
                    "fp_delta_r": ctrl_fp_r - r3_fp_r,
                    "fp_rmse_matched": s.fp_metrics["rmse"],
                    "fp_rmse_control": refit["fp_metrics"]["rmse"],
                    "fp_mae_matched": s.fp_metrics["mae"],
                    "fp_mae_control": refit["fp_metrics"]["mae"],
                    # Fused prediction
                    "fused_r_matched": r3_fused_r,
                    "fused_r_control": ctrl_fused_r,
                    "fused_delta_r": ctrl_fused_r - r3_fused_r,
                    "fused_rmse_matched": s.fused_metrics["rmse"],
                    "fused_rmse_control": refit["fused_metrics"]["rmse"],
                    "fused_mae_matched": s.fused_metrics["mae"],
                    "fused_mae_control": refit["fused_metrics"]["mae"],
                    # Fusion weights (fixed from R3)
                    "w_fp": s.fusion_weights["FP"],
                    "w_sc": s.fusion_weights["SC"],
                })
                refit_count += 1

            elapsed = time.time() - t0
            log.info(f"    {refit_count} refits in {elapsed:.1f}s ({elapsed/max(refit_count,1):.2f}s each)")

    # Save raw results
    df = pd.DataFrame(all_results)
    out_path = OUTPUT_BASE / "prior_control_metrics.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nSaved {len(df)} rows to {out_path}")

    # Compute paired comparisons per task per control
    log.info("\nComputing paired comparisons...")
    comparison_rows = []
    for task in ["working_memory", "fluid_intelligence"]:
        task_df = df[df["task"] == task]
        for ctrl in ["cross_task", "shuffled", "random"]:
            ctrl_df = task_df[task_df["control_prior"] == ctrl]
            if ctrl_df.empty:
                continue

            # FP branch comparison: matched vs control
            delta_fp = ctrl_df["fp_delta_r"].values
            if np.all(delta_fp == 0):
                p_fp = 1.0
                stat_fp = 0.0
            else:
                try:
                    res = wilcoxon(delta_fp)
                    stat_fp = res.statistic
                    p_fp = res.pvalue
                except ValueError:
                    stat_fp = 0.0
                    p_fp = 1.0

            # Fused comparison: matched vs control
            delta_fused = ctrl_df["fused_delta_r"].values
            if np.all(delta_fused == 0):
                p_fused = 1.0
                stat_fused = 0.0
            else:
                try:
                    res = wilcoxon(delta_fused)
                    stat_fused = res.statistic
                    p_fused = res.pvalue
                except ValueError:
                    stat_fused = 0.0
                    p_fused = 1.0

            # Bootstrap CI for fused delta
            rng = np.random.RandomState(20260906)
            n_boot = 10000
            boot_means = np.array([
                np.mean(delta_fused[rng.randint(0, len(delta_fused), len(delta_fused))])
                for _ in range(n_boot)
            ])
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            ctrl_display = CONTROL_PRIORS[ctrl]["display"]
            comparison_rows.append({
                "task": task,
                "control_prior": ctrl,
                "control_display": ctrl_display,
                "metric": "Pearson r (fused)",
                "mean_matched": ctrl_df["fused_r_matched"].mean(),
                "mean_control": ctrl_df["fused_r_control"].mean(),
                "mean_delta": ctrl_df["fused_delta_r"].mean(),
                "median_delta": ctrl_df["fused_delta_r"].median(),
                "positive_seeds": int(np.sum(delta_fused > 0)),
                "n_seeds": len(delta_fused),
                "wilcoxon_statistic": stat_fused,
                "raw_p": p_fused,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "direction": "positive" if ctrl_df["fused_delta_r"].mean() > 0 else "negative",
            })
            # Also FP branch
            comparison_rows.append({
                "task": task,
                "control_prior": ctrl,
                "control_display": ctrl_display,
                "metric": "Pearson r (FP unfused)",
                "mean_matched": ctrl_df["fp_r_matched"].mean(),
                "mean_control": ctrl_df["fp_r_control"].mean(),
                "mean_delta": ctrl_df["fp_delta_r"].mean(),
                "median_delta": ctrl_df["fp_delta_r"].median(),
                "positive_seeds": int(np.sum(delta_fp > 0)),
                "n_seeds": len(delta_fp),
                "wilcoxon_statistic": stat_fp,
                "raw_p": p_fp,
                "ci_95_lower": np.percentile([
                    np.mean(delta_fp[rng.randint(0, len(delta_fp), len(delta_fp))])
                    for _ in range(n_boot)
                ], 2.5),
                "ci_95_upper": np.percentile([
                    np.mean(delta_fp[rng.randint(0, len(delta_fp), len(delta_fp))])
                    for _ in range(n_boot)
                ], 97.5),
                "direction": "positive" if ctrl_df["fp_delta_r"].mean() > 0 else "negative",
            })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = OUTPUT_BASE / "prior_control_comparisons.csv"
    comp_df.to_csv(comp_path, index=False)
    log.info(f"Saved {len(comp_df)} comparison rows to {comp_path}")

    # Generate summary table
    log.info("\n=== PRIOR CONTROL SUMMARY ===")
    for task in ["working_memory", "fluid_intelligence"]:
        log.info(f"\n--- {task.upper()} ---")
        task_comp = comp_df[(comp_df["task"] == task) & (comp_df["metric"] == "Pearson r (fused)")]
        for _, row in task_comp.iterrows():
            log.info(
                f"  {row['control_display']:12s}: matched={row['mean_matched']:.4f}, "
                f"control={row['mean_control']:.4f}, "
                f"delta={row['mean_delta']:+.4f} "
                f"(95% CI [{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}], "
                f"p={row['raw_p']:.4f})"
            )

    # Generate LaTeX table
    tex_lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Task & Control & Matched $r$ & Control $r$ & $\\Delta r$ & 95\\% CI & $p$ \\\\",
        "\\midrule",
    ]
    for task_label, task_key in [("WM", "working_memory"), ("FI", "fluid_intelligence")]:
        task_comp = comp_df[
            (comp_df["task"] == task_key) & (comp_df["metric"] == "Pearson r (fused)")
        ]
        for _, row in task_comp.iterrows():
            tex_lines.append(
                f"{task_label} & {row['control_display']} & {row['mean_matched']:.4f} "
                f"& {row['mean_control']:.4f} & {row['mean_delta']:+.4f} "
                f"& [{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] & {row['raw_p']:.3f} \\\\"
            )
        if task_label == "WM":
            tex_lines.append("\\midrule")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    tables_dir = OUTPUT_BASE / "tables"
    tables_dir.mkdir(exist_ok=True)
    (tables_dir / "table_prior_controls.tex").write_text("\n".join(tex_lines))
    log.info("LaTeX table saved to tables/table_prior_controls.tex")

    log.info("\nSTATUS: PRIOR_CONTROL_COMPLETE")


if __name__ == "__main__":
    main()
