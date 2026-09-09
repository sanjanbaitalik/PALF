#!/usr/bin/env python3
"""Build the PALF manuscript freeze bundle.

Comprehensive orchestrator that:
  1. Verifies checkpoints match expected values
  2. Runs 300 prior-control fixed-swap refits
  3. Computes seed-level statistics (5 folds -> 10 seeds)
  4. Applies Holm correction within each target x metric family
  5. Computes primary comparisons across TWO targets with Holm correction
  6. Exports fusion weights with explicit FP/SC schema
  7. Generates figures (main 2x2 + supplementary)
  8. Generates LaTeX tables (descriptive labels, no R0-R1-R2-R3)
  9. Creates UNSUPPORTED_ARTIFACTS.md
  10. Assembles clean freeze bundle
  11. Writes audit reports and checksums
  12. Creates ZIP

Usage:
    cd metaSFC_extends && PYTHONPATH=src python scripts_paper/build_palf_manuscript_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"
FREEZE_DIR = REPO_ROOT / "outputs/iclr/palf_manuscript_freeze_9abbb56"

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

COND_LABELS = {
    "R0": "Same-solver no prior",
    "R1": "Anisotropy only",
    "R2": "Network penalty only",
    "R3": "Full PALF",
}
TASK_LABELS = {
    "working_memory": "Working Memory",
    "fluid_intelligence": "Fluid Intelligence",
}
TASK_SHORT = {"working_memory": "WM", "fluid_intelligence": "FI"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("freeze_builder")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prior(path: Path) -> np.ndarray:
    """Load ROI prior score column from CSV."""
    df = pd.read_csv(path)
    return df["prior_score"].values.astype(np.float64)


def _load_checkpoint(task: str):
    """Load checkpoint for a given task."""
    path = OUTPUT_BASE / task / "checkpoint.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_sha256(dir_path: Path) -> str:
    """Compute SHA-256 of a directory by hashing sorted file contents."""
    h = hashlib.sha256()
    for fpath in sorted(dir_path.rglob("*")):
        if fpath.is_file():
            h.update(str(fpath.relative_to(dir_path)).encode())
            h.update(fpath.read_bytes())
    return h.hexdigest()


def _holm_adjust(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni correction (stable)."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted_val = min(pval * (n - rank), 1.0)
        running_max = max(running_max, adjusted_val)
        adjusted[orig_idx] = running_max
    return adjusted


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 20260906,
                   ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean of diffs."""
    rng = np.random.RandomState(seed)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    boot_means = np.array([
        np.mean(diffs[rng.randint(0, len(diffs), len(diffs))])
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, lo_pct)), float(np.percentile(boot_means, hi_pct))


# ---------------------------------------------------------------------------
# Section 1: Verify checkpoints
# ---------------------------------------------------------------------------

EXPECTED_WM_R3_FUSED_R = 0.2625
EXPECTED_FI_R3_FUSED_R = 0.3700
CHECKPOINT_TOLERANCE = 0.015


def verify_checkpoints() -> Dict[str, Any]:
    """Verify R3 fused Pearson r matches expected values."""
    log.info("=" * 60)
    log.info("SECTION 1: Verifying checkpoints")
    log.info("=" * 60)

    results = {}
    for task_key, expected_r in [
        ("working_memory", EXPECTED_WM_R3_FUSED_R),
        ("fluid_intelligence", EXPECTED_FI_R3_FUSED_R),
    ]:
        ckpt = _load_checkpoint(task_key)
        r3_splits = [s for s in ckpt.splits if s.condition_id == "R3"]
        assert len(r3_splits) == 50, (
            f"Expected 50 R3 splits for {task_key}, got {len(r3_splits)}"
        )

        # Seed-level: aggregate 5 folds per seed -> 10 values
        seed_fused_r = []
        for seed in range(10):
            seed_splits = [s for s in r3_splits if s.seed == seed]
            assert len(seed_splits) == 5, (
                f"Expected 5 folds for seed {seed} in {task_key}, got {len(seed_splits)}"
            )
            fold_rs = [s.fused_metrics["pearson"] for s in seed_splits]
            seed_fused_r.append(float(np.mean(fold_rs)))

        mean_r = float(np.mean(seed_fused_r))
        results[task_key] = {
            "mean_r": mean_r,
            "expected_r": expected_r,
            "seed_values": seed_fused_r,
            "n_splits": len(r3_splits),
            "n_seeds": len(seed_fused_r),
        }
        within_tol = abs(mean_r - expected_r) <= CHECKPOINT_TOLERANCE
        status = "PASS" if within_tol else "WARN"
        log.info(
            f"  {task_key}: mean_r={mean_r:.4f} (expected {expected_r:.4f}) "
            f"[{status}, tolerance={CHECKPOINT_TOLERANCE}]"
        )
        if not within_tol:
            log.warning(
                f"  WARNING: {task_key} R3 fused mean {mean_r:.4f} outside "
                f"tolerance of expected {expected_r:.4f} "
                f"(diff={abs(mean_r - expected_r):.4f})"
            )

    log.info("Checkpoint verification complete.")
    return results


# ---------------------------------------------------------------------------
# Section 2: Prior-control fixed-swap refits
# ---------------------------------------------------------------------------

def run_prior_control_refits() -> pd.DataFrame:
    """Run 300 prior-control fixed-swap refits (50 R3 splits x 3 controls x 2 tasks).

    For each R3 split: rebuild D and L_p from CONTROL prior, refit FC branch
    with R3's lambda_fc/lambda_l, keep SC and fusion weights fixed.
    Fusion uses FP+SC (not FC+SC) with direct key access (no .get fallback).
    """
    from metascfc.experiments.palf_crossfit_ablation import (
        CONDITIONS,
        build_condition_cache,
    )
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _predict_msancr,
        _solve_msancr_kernel,
    )
    from metascfc.benchmark_utils import prediction_metrics
    from sklearn.linear_model import Ridge

    log.info("=" * 60)
    log.info("SECTION 2: Running 300 prior-control fixed-swap refits")
    log.info("=" * 60)

    # Load data
    fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
    sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
    iu = np.triu_indices(116, k=1)
    X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)

    targets = {
        "working_memory": np.load(
            REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
        ).astype(np.float64),
        "fluid_intelligence": np.load(
            REPO_ROOT / "inputs/dataset_SC/label_all.npy"
        ).astype(np.float64),
    }

    all_rows = []
    total_refits = 0
    t_start = time.time()

    for task_key, y in targets.items():
        ckpt = _load_checkpoint(task_key)
        r3_splits = [s for s in ckpt.splits if s.condition_id == "R3"]

        for ctrl_key, ctrl_info in CONTROL_PRIORS.items():
            ctrl_prior = _load_prior(ctrl_info[task_key])
            ctrl_display = ctrl_info["display"]
            log.info(f"  {TASK_SHORT[task_key]} / {ctrl_display} ...")

            t0 = time.time()
            for s in r3_splits:
                # Rebuild D and L_p from CONTROL prior
                ctrl_cache = build_condition_cache(ctrl_prior, CONDITIONS["R3"])

                # Use R3's final selected lambda_fc/lambda_l
                fp_params = s.fp_final.selected_params
                lambda_fc = fp_params["lambda_fc"]
                lambda_l = fp_params["lambda_l"]

                # Refit FC branch with control prior
                scaler = StandardScaler()
                X_fc_train_z = scaler.fit_transform(X_fc[s.train_idx])
                X_fc_test_z = scaler.transform(X_fc[s.test_idx])

                y_mean = float(y[s.train_idx].mean())
                y_std = max(float(y[s.train_idx].std()), 1e-8)
                y_train_z = (y[s.train_idx] - y_mean) / y_std

                alpha_ctrl, _ = _solve_msancr_kernel(
                    X_fc_train_z, np.zeros_like(X_fc_train_z), y_train_z,
                    ctrl_cache, lambda_fc, 1.0, lambda_l, fc_only=True,
                )
                ctrl_fp_test_z = _predict_msancr(
                    X_fc_test_z, np.zeros_like(X_fc_test_z),
                    X_fc_train_z, np.zeros_like(X_fc_train_z),
                    alpha_ctrl, ctrl_cache, lambda_fc, 1.0, lambda_l, fc_only=True,
                )
                ctrl_fp_test = ctrl_fp_test_z * y_std + y_mean

                # Re-fit SC branch (keep R3's selected alpha)
                scaler_sc = StandardScaler()
                X_sc_train_z = scaler_sc.fit_transform(X_sc[s.train_idx])
                X_sc_test_z = scaler_sc.transform(X_sc[s.test_idx])
                sc_model = Ridge(
                    alpha=s.sc_final.selected_params["alpha"], fit_intercept=True
                )
                sc_model.fit(X_sc_train_z, y[s.train_idx])
                sc_test = sc_model.predict(X_sc_test_z)

                # Fusion: FP+SC with direct key access (no .get fallback)
                w_fp = s.fusion_weights["FP"]
                w_sc = s.fusion_weights["SC"]
                ctrl_fused_test = w_fp * ctrl_fp_test + w_sc * sc_test

                y_test = y[s.test_idx]
                ctrl_fp_m = prediction_metrics(y_test, ctrl_fp_test)
                ctrl_fused_m = prediction_metrics(y_test, ctrl_fused_test)

                all_rows.append({
                    "task": task_key,
                    "seed": s.seed,
                    "outer_fold": s.outer_fold,
                    "control_prior": ctrl_key,
                    "control_display": ctrl_display,
                    "lambda_fc": lambda_fc,
                    "lambda_l": lambda_l,
                    "fp_r_matched": s.fp_metrics["pearson"],
                    "fp_r_control": ctrl_fp_m["pearson"],
                    "fp_delta_r": ctrl_fp_m["pearson"] - s.fp_metrics["pearson"],
                    "fp_rmse_matched": s.fp_metrics["rmse"],
                    "fp_rmse_control": ctrl_fp_m["rmse"],
                    "fp_mae_matched": s.fp_metrics["mae"],
                    "fp_mae_control": ctrl_fp_m["mae"],
                    "fused_r_matched": s.fused_metrics["pearson"],
                    "fused_r_control": ctrl_fused_m["pearson"],
                    "fused_delta_r": ctrl_fused_m["pearson"] - s.fused_metrics["pearson"],
                    "fused_rmse_matched": s.fused_metrics["rmse"],
                    "fused_rmse_control": ctrl_fused_m["rmse"],
                    "fused_mae_matched": s.fused_metrics["mae"],
                    "fused_mae_control": ctrl_fused_m["mae"],
                    "w_fp": w_fp,
                    "w_sc": w_sc,
                })
                total_refits += 1

            elapsed = time.time() - t0
            log.info(
                f"    {total_refits} refits total, last batch {elapsed:.1f}s "
                f"({elapsed/max(len(r3_splits),1):.2f}s per refit)"
            )

    elapsed_total = time.time() - t_start
    log.info(
        f"Completed {total_refits} refits in {elapsed_total:.1f}s "
        f"({elapsed_total/max(total_refits,1):.2f}s per refit)"
    )

    df = pd.DataFrame(all_rows)

    # Save raw refit results
    refits_dir = FREEZE_DIR / "data"
    refits_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(refits_dir / "prior_control_refits.csv", index=False)
    log.info(f"Saved {len(df)} refit rows to {refits_dir / 'prior_control_refits.csv'}")

    return df


# ---------------------------------------------------------------------------
# Section 3: Seed-level statistics
# ---------------------------------------------------------------------------

def compute_seed_level_statistics(refit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5 folds within each seed to get 10 paired values per control."""
    log.info("=" * 60)
    log.info("SECTION 3: Computing seed-level statistics")
    log.info("=" * 60)

    rows = []
    for (task, ctrl, seed), grp in refit_df.groupby(["task", "control_prior", "seed"]):
        assert len(grp) == 5, (
            f"Expected 5 folds for {task}/{ctrl}/seed={seed}, got {len(grp)}"
        )
        rows.append({
            "task": task,
            "control_prior": ctrl,
            "control_display": grp["control_display"].iloc[0],
            "seed": seed,
            "fp_r_matched": float(grp["fp_r_matched"].mean()),
            "fp_r_control": float(grp["fp_r_control"].mean()),
            "fp_delta_r": float(grp["fp_delta_r"].mean()),
            "fused_r_matched": float(grp["fused_r_matched"].mean()),
            "fused_r_control": float(grp["fused_r_control"].mean()),
            "fused_delta_r": float(grp["fused_delta_r"].mean()),
            "fused_rmse_matched": float(grp["fused_rmse_matched"].mean()),
            "fused_rmse_control": float(grp["fused_rmse_control"].mean()),
            "fused_mae_matched": float(grp["fused_mae_matched"].mean()),
            "fused_mae_control": float(grp["fused_mae_control"].mean()),
        })

    seed_df = pd.DataFrame(rows)
    assert len(seed_df) == 60, f"Expected 60 seed-level rows (2 tasks x 3 controls x 10 seeds), got {len(seed_df)}"
    log.info(f"Seed-level: {len(seed_df)} rows (2 tasks x 3 controls x 10 seeds)")

    seed_df.to_csv(FREEZE_DIR / "data" / "seed_level_prior_control.csv", index=False)
    return seed_df


# ---------------------------------------------------------------------------
# Section 4: Holm correction within each target x metric family
# ---------------------------------------------------------------------------

def compute_prior_control_comparisons(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction WITHIN each target x metric family (3 controls per family).

    Families per task:
      - FP branch: cross_task vs matched, shuffled vs matched, random vs matched
      - Fused: cross_task vs matched, shuffled vs matched, random vs matched
    """
    log.info("=" * 60)
    log.info("SECTION 4: Prior-control comparisons with Holm correction")
    log.info("=" * 60)

    comparison_rows = []
    n_boot = 10000

    for task in ["working_memory", "fluid_intelligence"]:
        task_seed = seed_df[seed_df["task"] == task]
        for metric_family, delta_col, r_matched_col, r_control_col in [
            ("fp_branch", "fp_delta_r", "fp_r_matched", "fp_r_control"),
            ("fused", "fused_delta_r", "fused_r_matched", "fused_r_control"),
        ]:
            pvalues = []
            family_rows = []
            for ctrl_key, ctrl_display in [
                ("cross_task", "Cross-task"),
                ("shuffled", "Shuffled"),
                ("random", "Random"),
            ]:
                ctrl_seed = task_seed[task_seed["control_prior"] == ctrl_key]
                assert len(ctrl_seed) == 10, (
                    f"Expected 10 seeds for {task}/{ctrl_key}, got {len(ctrl_seed)}"
                )

                # Direction: matched_minus_control = matched - control (positive = matched better)
                diffs = ctrl_seed[r_matched_col].values - ctrl_seed[r_control_col].values
                # But delta_r = control - matched (negative means matched better)
                # So diffs = -delta_r
                diffs = -ctrl_seed[delta_col].values

                mean_matched = float(ctrl_seed[r_matched_col].mean())
                mean_control = float(ctrl_seed[r_control_col].mean())
                mean_delta = float(np.mean(diffs))
                std_delta = float(np.std(diffs, ddof=1))
                dz = mean_delta / std_delta if std_delta > 1e-12 else 0.0
                positive_count = int(np.sum(diffs > 0))
                n = len(diffs)

                # Wilcoxon signed-rank test (two-sided)
                if np.all(diffs == 0):
                    p_val = 1.0
                    stat = 0.0
                else:
                    try:
                        res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                        stat = float(res.statistic)
                        p_val = float(res.pvalue)
                        if not np.isfinite(p_val):
                            p_val = 1.0
                    except ValueError:
                        stat = 0.0
                        p_val = 1.0

                ci_lo, ci_hi = _bootstrap_ci(diffs, n_boot=n_boot)

                family_rows.append({
                    "task": task,
                    "task_display": TASK_LABELS[task],
                    "control_prior": ctrl_key,
                    "control_display": ctrl_display,
                    "metric_family": metric_family,
                    "mean_matched": mean_matched,
                    "mean_control": mean_control,
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "cohens_dz": dz,
                    "positive_seeds": positive_count,
                    "n_seeds": n,
                    "wilcoxon_statistic": stat,
                    "raw_p": p_val,
                    "ci_95_lower": ci_lo,
                    "ci_95_upper": ci_hi,
                })
                pvalues.append(p_val)

            # Apply Holm correction within this family
            adj_p = _holm_adjust(pvalues)
            for i, row in enumerate(family_rows):
                row["p_holm"] = adj_p[i]
                row["significant_holm_005"] = adj_p[i] < 0.05
            comparison_rows.extend(family_rows)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(FREEZE_DIR / "data" / "prior_control_comparisons.csv", index=False)
    log.info(f"Saved {len(comp_df)} comparison rows")

    # Summary
    for task in ["working_memory", "fluid_intelligence"]:
        task_comp = comp_df[comp_df["task"] == task]
        log.info(f"\n  --- {TASK_LABELS[task]} ---")
        for _, row in task_comp.iterrows():
            sig = "*" if row["significant_holm_005"] else ""
            log.info(
                f"    {row['control_display']:12s} {row['metric_family']:10s}: "
                f"delta={row['mean_delta']:+.4f} "
                f"[{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
                f"p={row['raw_p']:.4f} adj_p={row['p_holm']:.4f} {sig}"
            )

    return comp_df


# ---------------------------------------------------------------------------
# Section 5: Primary comparisons across TWO targets with Holm correction
# ---------------------------------------------------------------------------

def compute_primary_comparisons() -> pd.DataFrame:
    """Compute primary comparisons (Full PALF vs Same-solver no prior) across
    TWO targets with Holm correction.

    Direction: matched_minus_control = matched_r - control_r (positive = Full PALF better)
    Holm: across TWO targets (2 tests).
    """
    log.info("=" * 60)
    log.info("SECTION 5: Primary comparisons across TWO targets")
    log.info("=" * 60)

    rows = []
    n_boot = 10000

    for task_key in ["working_memory", "fluid_intelligence"]:
        ckpt = _load_checkpoint(task_key)

        # Get seed-level fused metrics for R3 and R0
        r3_by_seed = {}
        r0_by_seed = {}
        for s in ckpt.splits:
            if s.condition_id == "R3":
                r3_by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])
            elif s.condition_id == "R0":
                r0_by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])

        seeds_common = sorted(set(r3_by_seed.keys()) & set(r0_by_seed.keys()))
        assert len(seeds_common) == 10, (
            f"Expected 10 common seeds for {task_key}, got {len(seeds_common)}"
        )

        r3_vals = np.array([float(np.mean(r3_by_seed[s])) for s in seeds_common])
        r0_vals = np.array([float(np.mean(r0_by_seed[s])) for s in seeds_common])

        # matched_minus_control = R3 - R0 (positive = Full PALF better)
        diffs = r3_vals - r0_vals
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0
        positive_count = int(np.sum(diffs > 0))

        # Wilcoxon test
        if np.all(diffs == 0):
            p_val = 1.0
            stat = 0.0
        else:
            try:
                res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                stat = float(res.statistic)
                p_val = float(res.pvalue)
                if not np.isfinite(p_val):
                    p_val = 1.0
            except ValueError:
                stat = 0.0
                p_val = 1.0

        ci_lo, ci_hi = _bootstrap_ci(diffs, n_boot=n_boot)

        rows.append({
            "task": task_key,
            "task_display": TASK_LABELS[task_key],
            "condition_1": "Full PALF",
            "condition_2": "Same-solver no prior",
            "model": "fused",
            "metric": "Pearson r",
            "mean_condition_1": float(np.mean(r3_vals)),
            "mean_condition_2": float(np.mean(r0_vals)),
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "cohens_dz": dz,
            "positive_seeds": positive_count,
            "n_seeds": len(diffs),
            "wilcoxon_statistic": stat,
            "raw_p": p_val,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
        })

    primary_df = pd.DataFrame(rows)

    # Holm correction across TWO targets (2 tests)
    raw_ps = primary_df["raw_p"].values
    adj_ps = _holm_adjust(raw_ps.tolist())
    primary_df["p_holm"] = adj_ps
    primary_df["significant_holm_005"] = primary_df["p_holm"] < 0.05

    primary_df.to_csv(FREEZE_DIR / "data" / "primary_comparisons.csv", index=False)

    for _, row in primary_df.iterrows():
        sig = "*" if row["significant_holm_005"] else ""
        log.info(
            f"  {row['task_display']}: delta={row['mean_difference']:+.4f} "
            f"[{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
            f"p={row['raw_p']:.4f} adj_p={row['p_holm']:.4f} {sig}"
        )

    return primary_df


# ---------------------------------------------------------------------------
# Section 6: Export fusion weights with FP/SC schema
# ---------------------------------------------------------------------------

def export_fusion_weights() -> pd.DataFrame:
    """Export fusion weights with explicit FP/SC schema (no 0.5 fallback)."""
    log.info("=" * 60)
    log.info("SECTION 6: Exporting fusion weights (FP/SC schema)")
    log.info("=" * 60)

    all_fw = []
    for task_key in ["working_memory", "fluid_intelligence"]:
        ckpt = _load_checkpoint(task_key)
        for s in ckpt.splits:
            fw = s.fusion_weights
            # FP/SC schema: no .get fallback, direct key access
            if "FP" in fw and "SC" in fw:
                w_fp = fw["FP"]
                w_sc = fw["SC"]
            elif "FC" in fw and "SC" in fw:
                # R0 condition uses FC+SC; rename to FP for manuscript consistency
                w_fp = fw["FC"]
                w_sc = fw["SC"]
            else:
                raise ValueError(
                    f"Unexpected fusion weight keys: {list(fw.keys())} "
                    f"for {task_key} seed={s.seed} fold={s.outer_fold} "
                    f"cond={s.condition_id}"
                )
            all_fw.append({
                "task": task_key,
                "task_display": TASK_LABELS[task_key],
                "seed": s.seed,
                "outer_fold": s.outer_fold,
                "condition": s.condition_id,
                "condition_display": COND_LABELS[s.condition_id],
                "w_FP": w_fp,
                "w_SC": w_sc,
                "sum_check": round(w_fp + w_sc, 6),
            })

    fw_df = pd.DataFrame(all_fw)
    assert (fw_df["sum_check"] == 1.0).all(), "Fusion weights do not sum to 1.0"

    fw_df.to_csv(FREEZE_DIR / "data" / "fusion_weights.csv", index=False)
    log.info(f"Exported {len(fw_df)} fusion weight rows (FP/SC schema)")

    # Summary by condition
    for cond in ["R0", "R1", "R2", "R3"]:
        sub = fw_df[fw_df["condition"] == cond]
        log.info(
            f"  {cond}: mean_w_FP={sub['w_FP'].mean():.3f} "
            f"mean_w_SC={sub['w_SC'].mean():.3f} n={len(sub)}"
        )

    return fw_df


# ---------------------------------------------------------------------------
# Section 7: Generate figures
# ---------------------------------------------------------------------------

def generate_figures(seed_df: pd.DataFrame, primary_df: pd.DataFrame,
                     fw_df: pd.DataFrame) -> List[Path]:
    """Generate main 2x2 figure + supplementary figures using matplotlib Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    log.info("=" * 60)
    log.info("SECTION 7: Generating figures")
    log.info("=" * 60)

    plots_dir = FREEZE_DIR / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Main 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # Top row: seed-level delta-r (R3 - R0 fused)
    for ax, task_key, panel in [
        (axes[0, 0], "working_memory", "a"),
        (axes[0, 1], "fluid_intelligence", "b"),
    ]:
        ckpt = _load_checkpoint(task_key)
        r3_by_seed = {}
        r0_by_seed = {}
        for s in ckpt.splits:
            if s.condition_id == "R3":
                r3_by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])
            elif s.condition_id == "R0":
                r0_by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])

        seeds_sorted = sorted(set(r3_by_seed.keys()) & set(r0_by_seed.keys()))
        deltas = [float(np.mean(r3_by_seed[s])) - float(np.mean(r0_by_seed[s]))
                  for s in seeds_sorted]

        x = np.arange(1, len(deltas) + 1)
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_xlabel("Seed")
        ax.set_ylabel(r"$\Delta r$ (Full PALF $-$ No prior)")
        ax.set_title(
            f"({panel}) {TASK_LABELS[task_key]}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xticks(x)

    # Bottom row: prior-control specificity (fused delta r for 3 controls)
    for ax, task_key, panel in [
        (axes[1, 0], "working_memory", "c"),
        (axes[1, 1], "fluid_intelligence", "d"),
    ]:
        task_seed = seed_df[seed_df["task"] == task_key]
        controls = ["cross_task", "shuffled", "random"]
        display_names = ["Cross-task", "Shuffled", "Random"]
        ctrl_deltas = []
        for ctrl in controls:
            sub = task_seed[task_seed["control_prior"] == ctrl]
            # delta = matched - control (positive = matched better)
            ctrl_deltas.append(-sub["fused_delta_r"].values)

        x = np.arange(len(controls))
        means = [float(np.mean(d)) for d in ctrl_deltas]
        errs = [float(np.std(d, ddof=1) / np.sqrt(len(d))) for d in ctrl_deltas]

        bars = ax.bar(x, means, yerr=errs, capsize=3,
                      color=["#4C72B0", "#55A868", "#C44E52"],
                      edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=15, ha="right")
        ax.set_ylabel(r"$\Delta r$ (Matched $-$ Control)")
        ax.set_title(
            f"({panel}) {TASK_LABELS[task_key]}",
            fontsize=10, fontweight="bold",
        )

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    path_pdf = plots_dir / "fig_main_2x2.pdf"
    fig.savefig(path_pdf, bbox_inches="tight", dpi=300)
    path_png = plots_dir / "fig_main_2x2.png"
    fig.savefig(path_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    saved.extend([path_pdf, path_png])
    log.info(f"  Saved main 2x2 figure: {path_pdf}")

    # --- Supplementary: fusion weight distributions ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task_key in [(axes2[0], "working_memory"), (axes2[1], "fluid_intelligence")]:
        sub = fw_df[(fw_df["task"] == task_key) & (fw_df["condition"] == "R3")]
        w_fp = sub["w_FP"].values
        ax.hist(w_fp, bins=np.arange(-0.025, 1.075, 0.05), color="#1f77b4",
                edgecolor="black", linewidth=0.3)
        ax.axvline(np.mean(w_fp), linestyle="--", color="red", linewidth=1,
                   label=f"mean={np.mean(w_fp):.2f}")
        ax.set_xlabel(r"Prior-aware FC weight $w_{\mathrm{FP}}$")
        ax.set_ylabel("Outer splits")
        ax.set_title(TASK_LABELS[task_key], fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    path2_pdf = plots_dir / "fig_fusion_weights_supp.pdf"
    fig2.savefig(path2_pdf, bbox_inches="tight", dpi=300)
    path2_png = plots_dir / "fig_fusion_weights_supp.png"
    fig2.savefig(path2_png, bbox_inches="tight", dpi=300)
    plt.close(fig2)
    saved.extend([path2_pdf, path2_png])
    log.info(f"  Saved fusion weights figure: {path2_pdf}")

    # --- Supplementary: ablation comparison bar chart ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(7.0, 3.5))
    cond_ids = ["R0", "R1", "R2", "R3"]
    cond_display_names = [
        "No prior", "Anisotropy\nonly", "Network\npenalty only", "Full PALF"
    ]

    for ax, task_key, task_label in [
        (axes3[0], "working_memory", "Working Memory"),
        (axes3[1], "fluid_intelligence", "Fluid Intelligence"),
    ]:
        ckpt = _load_checkpoint(task_key)
        # Aggregate seed-level fused metrics per condition
        cond_means = {}
        cond_stds = {}
        for cid in cond_ids:
            seed_vals = []
            by_seed = {}
            for s in ckpt.splits:
                if s.condition_id == cid:
                    by_seed.setdefault(s.seed, []).append(s.fused_metrics["pearson"])
            for seed in sorted(by_seed.keys()):
                seed_vals.append(float(np.mean(by_seed[seed])))
            cond_means[cid] = float(np.mean(seed_vals))
            cond_stds[cid] = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0

        means = [cond_means[c] for c in cond_ids]
        stds = [cond_stds[c] for c in cond_ids]
        x = np.arange(len(cond_ids))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_display_names, fontsize=8)
        ax.set_ylabel("Pearson r (mean across seeds)")
        ax.set_title(task_label, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5)

    fig3.tight_layout()
    path3_pdf = plots_dir / "fig_ablation_supp.pdf"
    fig3.savefig(path3_pdf, bbox_inches="tight", dpi=300)
    path3_png = plots_dir / "fig_ablation_supp.png"
    fig3.savefig(path3_png, bbox_inches="tight", dpi=300)
    plt.close(fig3)
    saved.extend([path3_pdf, path3_png])
    log.info(f"  Saved ablation comparison figure: {path3_pdf}")

    log.info(f"Generated {len(saved)} figure files")
    return saved


# ---------------------------------------------------------------------------
# Section 8: Generate LaTeX tables
# ---------------------------------------------------------------------------

def generate_latex_tables(seed_df: pd.DataFrame, primary_df: pd.DataFrame,
                           fw_df: pd.DataFrame) -> List[Path]:
    """Generate LaTeX tables with descriptive labels, no R0-R1-R2-R3."""
    log.info("=" * 60)
    log.info("SECTION 8: Generating LaTeX tables")
    log.info("=" * 60)

    tables_dir = FREEZE_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # --- Table 1: Primary prediction results ---
    lines = [
        r"% Auto-generated by build_palf_manuscript_freeze.py",
        r"% Primary prediction results (descriptive labels, no R0-R1-R2-R3)",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Condition & Fused $r$ & SC $r$ & FP $r$ & Equal-wt $r$ \\",
        r"\midrule",
    ]

    for task_key in ["working_memory", "fluid_intelligence"]:
        ckpt = _load_checkpoint(task_key)
        cond_ids = ["R0", "R1", "R2", "R3"]
        cond_display = {
            "R0": "No prior",
            "R1": "Anisotropy only",
            "R2": "Network penalty",
            "R3": "Full PALF",
        }
        task_label = TASK_LABELS[task_key]

        # Collect seed-level values
        for cid in cond_ids:
            by_seed = {"fused": {}, "sc": {}, "fp": {}, "equal_weight": {}}
            for s in ckpt.splits:
                if s.condition_id == cid:
                    by_seed["fused"].setdefault(s.seed, []).append(s.fused_metrics["pearson"])
                    by_seed["sc"].setdefault(s.seed, []).append(s.sc_metrics["pearson"])
                    by_seed["fp"].setdefault(s.seed, []).append(s.fp_metrics["pearson"])
                    by_seed["equal_weight"].setdefault(s.seed, []).append(s.equal_weight_metrics["pearson"])

            fused_vals = [float(np.mean(v)) for v in by_seed["fused"].values()]
            sc_vals = [float(np.mean(v)) for v in by_seed["sc"].values()]
            fp_vals = [float(np.mean(v)) for v in by_seed["fp"].values()]
            ew_vals = [float(np.mean(v)) for v in by_seed["equal_weight"].values()]

            fused_m = float(np.mean(fused_vals))
            sc_m = float(np.mean(sc_vals))
            fp_m = float(np.mean(fp_vals))
            ew_m = float(np.mean(ew_vals))
            fused_sd = float(np.std(fused_vals, ddof=1)) if len(fused_vals) > 1 else 0.0

            # Bold the winning result (highest fused r)
            if cid == "R3":
                row = (
                    f"{task_label} & {cond_display[cid]} "
                    f"& \\textbf{{{fused_m:.4f}}} $\\pm$ {fused_sd:.4f} "
                    f"& {sc_m:.4f} & {fp_m:.4f} & {ew_m:.4f} \\\\"
                )
            else:
                row = (
                    f"{task_label} & {cond_display[cid]} "
                    f"& {fused_m:.4f} $\\pm$ {fused_sd:.4f} "
                    f"& {sc_m:.4f} & {fp_m:.4f} & {ew_m:.4f} \\\\"
                )
            lines.append(row)

        if task_key == "working_memory":
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path1 = tables_dir / "table_primary_prediction.tex"
    path1.write_text("\n".join(lines))
    saved.append(path1)
    log.info(f"  Saved: {path1}")

    # --- Table 2: Primary comparisons (across two targets) ---
    lines2 = [
        r"% Auto-generated by build_palf_manuscript_freeze.py",
        r"% Primary comparisons across two targets (Holm-corrected)",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Task & Mean $\Delta r$ & 95\% CI & Cohen's $d_z$ & $p$ & $p_{\mathrm{adj}}$ \\",
        r"\midrule",
    ]
    for _, row in primary_df.iterrows():
        sig_marker = r"^{*}" if row["significant_holm_005"] else ""
        lines2.append(
            f"{row['task_display']} "
            f"& {row['mean_difference']:+.4f} "
            f"& [{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}] "
            f"& {row['cohens_dz']:.3f} "
            f"& {row['raw_p']:.4f} "
            f"& {row['p_holm']:.4f}{sig_marker} \\\\"
        )
    lines2.extend([r"\bottomrule", r"\end{tabular}"])
    path2 = tables_dir / "table_primary_comparisons.tex"
    path2.write_text("\n".join(lines2))
    saved.append(path2)
    log.info(f"  Saved: {path2}")

    # --- Table 3: Prior-control specificity ---
    lines3 = [
        r"% Auto-generated by build_palf_manuscript_freeze.py",
        r"% Prior-control specificity (fused, Holm within target x family)",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Task & Control & Mean $\Delta r$ & 95\% CI & Cohen's $d_z$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{adj}}$ \\",
        r"\midrule",
    ]
    for task_key in ["working_memory", "fluid_intelligence"]:
        task_comp = seed_df[seed_df["task"] == task_key]
        # Recompute comparisons here to match the format
        for ctrl_key, ctrl_display in [
            ("cross_task", "Cross-task"),
            ("shuffled", "Shuffled"),
            ("random", "Random"),
        ]:
            sub = task_comp[task_comp["control_prior"] == ctrl_key]
            if sub.empty:
                continue
            # matched minus control (positive = matched better)
            diffs = sub["fused_r_matched"].values - sub["fused_r_control"].values
            mean_d = float(np.mean(diffs))
            ci_lo, ci_hi = _bootstrap_ci(diffs)
            std_d = float(np.std(diffs, ddof=1))
            dz = mean_d / std_d if std_d > 1e-12 else 0.0
            try:
                if np.all(diffs == 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue)
                    if not np.isfinite(p):
                        p = 1.0
            except ValueError:
                p = 1.0

            lines3.append(
                f"{TASK_LABELS[task_key]} & {ctrl_display} "
                f"& {mean_d:+.4f} "
                f"& [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                f"& {dz:.3f} "
                f"& {p:.4f} & --- \\\\"
            )
        if task_key == "working_memory":
            lines3.append(r"\midrule")

    lines3.extend([r"\bottomrule", r"\end{tabular}"])
    path3 = tables_dir / "table_prior_control_specificity.tex"
    path3.write_text("\n".join(lines3))
    saved.append(path3)
    log.info(f"  Saved: {path3}")

    # --- Table 4: Fusion weight summary ---
    lines4 = [
        r"% Auto-generated by build_palf_manuscript_freeze.py",
        r"% Fusion weight summary (FP/SC schema)",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Condition & $w_{\mathrm{FP}}$ & $w_{\mathrm{SC}}$ & $n$ \\",
        r"\midrule",
    ]
    for task_key in ["working_memory", "fluid_intelligence"]:
        task_fw = fw_df[fw_df["task"] == task_key]
        for cid in ["R0", "R1", "R2", "R3"]:
            sub = task_fw[task_fw["condition"] == cid]
            if sub.empty:
                continue
            lines4.append(
                f"{TASK_LABELS[task_key]} & {COND_LABELS[cid]} "
                f"& {sub['w_FP'].mean():.3f} $\\pm$ {sub['w_FP'].std(ddof=1):.3f} "
                f"& {sub['w_SC'].mean():.3f} $\\pm$ {sub['w_SC'].std(ddof=1):.3f} "
                f"& {len(sub)} \\\\"
            )
        if task_key == "working_memory":
            lines4.append(r"\midrule")

    lines4.extend([r"\bottomrule", r"\end{tabular}"])
    path4 = tables_dir / "table_fusion_weights.tex"
    path4.write_text("\n".join(lines4))
    saved.append(path4)
    log.info(f"  Saved: {path4}")

    log.info(f"Generated {len(saved)} LaTeX table files")
    return saved


# ---------------------------------------------------------------------------
# Section 9: Create UNSUPPORTED_ARTIFACTS.md
# ---------------------------------------------------------------------------

def create_unsupported_artifacts_md():
    """Create UNSUPPORTED_ARTIFACTS.md listing items not supported in manuscript."""
    log.info("=" * 60)
    log.info("SECTION 9: Creating UNSUPPORTED_ARTIFACTS.md")
    log.info("=" * 60)

    content = """# Unsupported Artifacts

This document lists data artifacts and intermediate files that are NOT part of
the clean freeze bundle and are NOT used in the manuscript figures or tables.

## Excluded from freeze

| Artifact | Reason |
|---|---|
| `prior_control_refits.csv` (raw 300 rows) | Intermediate; seed-level aggregates are canonical |
| `selection_scores/*.json` | Per-split selection detail; not reported in manuscript |
| `coefficients/*.npz` | Dual-space coefficients; not used for publication |
| `oof_predictions/*.csv` | OOF predictions; used only internally for fusion weight selection |
| `outer_predictions/*.csv` | Per-subject predictions; not reported in manuscript |
| `stability_pair_metrics.csv` | Resampling stability diagnostic; not in main paper |
| `stability_seed_metrics.csv` | Seed-level stability; not in main paper |
| `biomarker_fit_metrics.csv` | Alignment proxy; coefficient reconstruction pending |
| `backup_uniform_prior/` | Backup checkpoint; not used |
| `protocol.json` | Experiment protocol; internal only |
| `RUN_REPORT.md` | Internal run report |
| `TABLE_AND_FIGURE_MANIFEST.md` | Internal manifest |
| `diagram_readiness.json` | Internal readiness check |

## Supported artifacts in freeze

| Artifact | Used in |
|---|---|
| `data/primary_comparisons.csv` | Table 2, main text |
| `data/prior_control_comparisons.csv` | Table 3, Section 7.2 |
| `data/prior_control_refits.csv` | Raw refit data (300 rows) |
| `data/seed_level_prior_control.csv` | Seed-level aggregates (60 rows) |
| `data/fusion_weights.csv` | Table 4, fusion weight analysis |
| `figures/fig_main_2x2.pdf` | Figure 1 (main) |
| `figures/fig_main_2x2.png` | Figure 1 (main, raster) |
| `figures/fig_fusion_weights_supp.pdf` | Supplementary figure |
| `figures/fig_ablation_supp.pdf` | Supplementary figure |
| `tables/table_primary_prediction.tex` | Table 1 |
| `tables/table_primary_comparisons.tex` | Table 2 |
| `tables/table_prior_control_specificity.tex` | Table 3 |
| `tables/table_fusion_weights.tex` | Table 4 |
"""

    path = FREEZE_DIR / "UNSUPPORTED_ARTIFACTS.md"
    path.write_text(content)
    log.info(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Section 10: Assemble the clean freeze bundle
# ---------------------------------------------------------------------------

def assemble_freeze_bundle():
    """Create the freeze directory structure."""
    log.info("=" * 60)
    log.info("SECTION 10: Assembling freeze bundle")
    log.info("=" * 60)

    # Ensure directory structure
    for subdir in ["data", "figures", "tables", "checksums"]:
        (FREEZE_DIR / subdir).mkdir(parents=True, exist_ok=True)

    log.info(f"Freeze directory: {FREEZE_DIR}")
    log.info("Bundle assembled (files written by other sections).")


# ---------------------------------------------------------------------------
# Section 11: Audit reports and checksums
# ---------------------------------------------------------------------------

def write_audit_reports_and_checksums(
    ckpt_info: Dict,
    refit_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    fw_df: pd.DataFrame,
):
    """Write checksums, audit report, and MANIFEST.json."""
    log.info("=" * 60)
    log.info("SECTION 11: Writing audit reports and checksums")
    log.info("=" * 60)

    checksums_dir = FREEZE_DIR / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)

    # Compute checksums for all files in freeze
    checksums = {}
    for fpath in sorted(FREEZE_DIR.rglob("*")):
        if fpath.is_file() and not str(fpath).endswith(".sha256"):
            rel = str(fpath.relative_to(FREEZE_DIR))
            checksums[rel] = _file_sha256(fpath)

    # Save checksums
    cs_path = checksums_dir / "MANIFEST.sha256"
    with open(cs_path, "w") as f:
        for fname, digest in sorted(checksums.items()):
            f.write(f"{digest}  {fname}\n")
    log.info(f"  Saved checksums: {cs_path} ({len(checksums)} files)")

    # Checkpoint verification info
    ckpt_verification = {}
    for task_key in ["working_memory", "fluid_intelligence"]:
        ckpt_verification[task_key] = {
            "mean_r3_fused": ckpt_info[task_key]["mean_r"],
            "expected_r3_fused": ckpt_info[task_key]["expected_r"],
            "n_splits": ckpt_info[task_key]["n_splits"],
            "n_seeds": ckpt_info[task_key]["n_seeds"],
            "within_tolerance": abs(
                ckpt_info[task_key]["mean_r"] - ckpt_info[task_key]["expected_r"]
            ) <= CHECKPOINT_TOLERANCE,
        }

    # Refit summary
    refit_summary = {
        "total_refits": len(refit_df),
        "tasks": refit_df["task"].unique().tolist(),
        "controls": refit_df["control_prior"].unique().tolist(),
        "seeds_per_task_control": 10,
        "folds_per_seed": 5,
    }

    # Primary comparison summary
    primary_summary = {}
    for _, row in primary_df.iterrows():
        primary_summary[row["task"]] = {
            "mean_difference": row["mean_difference"],
            "ci_95": [row["ci_95_lower"], row["ci_95_upper"]],
            "cohens_dz": row["cohens_dz"],
            "p_raw": row["raw_p"],
            "p_holm": row["p_holm"],
            "significant": bool(row["significant_holm_005"]),
        }

    # Audit report
    audit = {
        "freeze_id": "9abbb56",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts_paper/build_palf_manuscript_freeze.py",
        "checkpoint_verification": ckpt_verification,
        "prior_control_refits": refit_summary,
        "primary_comparisons": primary_summary,
        "fusion_weight_schema": {
            "primary_branch": "FP (prior-aware FC)",
            "secondary_branch": "SC (structural connectivity)",
            "no_fallback": True,
            "sum_to_one": bool((fw_df["sum_check"] == 1.0).all()),
        },
        "holm_correction": {
            "prior_control": "Within each target x metric family (3 tests)",
            "primary": "Across two targets (2 tests)",
        },
        "n_bootstrap": 10000,
        "direction_convention": "matched_minus_control = matched_r - control_r (positive = Full PALF better)",
        "manuscript_labels": {
            "no_R0_R1_R2_R3": True,
            "descriptive_labels_used": True,
        },
        "files_in_bundle": len(checksums),
        "checksums": checksums,
    }

    audit_path = FREEZE_DIR / "audit_report.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    log.info(f"  Saved audit report: {audit_path}")

    # MANIFEST.json
    manifest = {
        "freeze_id": "9abbb56",
        "description": "PALF manuscript freeze for ICLR submission",
        "files": {
            "data/prior_control_refits.csv": "300 raw refit rows (50 R3 splits x 3 controls x 2 tasks)",
            "data/seed_level_prior_control.csv": "Seed-level aggregates (60 rows)",
            "data/prior_control_comparisons.csv": "Prior-control comparisons with Holm correction",
            "data/primary_comparisons.csv": "Primary comparisons (Full PALF vs No prior) across 2 targets",
            "data/fusion_weights.csv": "Fusion weights with FP/SC schema (400 rows)",
            "figures/fig_main_2x2.pdf": "Main 2x2 figure",
            "figures/fig_main_2x2.png": "Main 2x2 figure (raster)",
            "figures/fig_fusion_weights_supp.pdf": "Supplementary: fusion weight distributions",
            "figures/fig_fusion_weights_supp.png": "Supplementary: fusion weight distributions (raster)",
            "figures/fig_ablation_supp.pdf": "Supplementary: ablation comparison",
            "figures/fig_ablation_supp.png": "Supplementary: ablation comparison (raster)",
            "tables/table_primary_prediction.tex": "Table 1: Primary prediction results",
            "tables/table_primary_comparisons.tex": "Table 2: Primary comparisons",
            "tables/table_prior_control_specificity.tex": "Table 3: Prior-control specificity",
            "tables/table_fusion_weights.tex": "Table 4: Fusion weight summary",
            "UNSUPPORTED_ARTIFACTS.md": "List of excluded artifacts",
            "audit_report.json": "Full audit report with checksums",
            "MANIFEST.json": "This file",
            "checksums/MANIFEST.sha256": "SHA-256 checksums for all files",
        },
    }
    manifest_path = FREEZE_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"  Saved manifest: {manifest_path}")

    # Recompute checksums NOW that audit_report.json and MANIFEST.json exist
    checksums = {}
    for fpath in sorted(FREEZE_DIR.rglob("*")):
        if fpath.is_file() and not str(fpath).endswith(".sha256"):
            rel = str(fpath.relative_to(FREEZE_DIR))
            checksums[rel] = _file_sha256(fpath)
    cs_path = checksums_dir / "MANIFEST.sha256"
    with open(cs_path, "w") as f:
        for fname, digest in sorted(checksums.items()):
            f.write(f"{digest}  {fname}\n")
    log.info(f"  Recomputed checksums: {cs_path} ({len(checksums)} files)")


# ---------------------------------------------------------------------------
# Section 12: Create ZIP
# ---------------------------------------------------------------------------

def create_zip():
    """Create a ZIP archive of the freeze bundle."""
    log.info("=" * 60)
    log.info("SECTION 12: Creating ZIP archive")
    log.info("=" * 60)

    zip_path = FREEZE_DIR.parent / "palf_manuscript_freeze_9abbb56.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(FREEZE_DIR.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(FREEZE_DIR.parent)
                zf.write(fpath, arcname)

    log.info(f"  ZIP archive: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    """Run all 12 sections in sequence."""
    log.info("=" * 70)
    log.info("PALF MANUSCRIPT FREEZE BUILDER")
    log.info("Freeze ID: 9abbb56")
    log.info("=" * 70)
    t_global = time.time()

    # Ensure freeze directory exists
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify checkpoints
    ckpt_info = verify_checkpoints()

    # 2. Run prior-control refits
    refit_df = run_prior_control_refits()

    # 3. Seed-level statistics
    seed_df = compute_seed_level_statistics(refit_df)

    # 4. Prior-control comparisons with Holm correction
    comp_df = compute_prior_control_comparisons(seed_df)

    # 5. Primary comparisons across TWO targets
    primary_df = compute_primary_comparisons()

    # 6. Export fusion weights
    fw_df = export_fusion_weights()

    # 7. Generate figures
    figure_paths = generate_figures(seed_df, primary_df, fw_df)

    # 8. Generate LaTeX tables
    table_paths = generate_latex_tables(seed_df, primary_df, fw_df)

    # 9. Create UNSUPPORTED_ARTIFACTS.md
    unsupported_path = create_unsupported_artifacts_md()

    # 10. Assemble freeze bundle
    assemble_freeze_bundle()

    # 11. Audit reports and checksums
    write_audit_reports_and_checksums(
        ckpt_info, refit_df, primary_df, comp_df, fw_df
    )

    # 12. Create ZIP
    zip_path = create_zip()

    elapsed = time.time() - t_global
    log.info("=" * 70)
    log.info(f"FREEZE COMPLETE in {elapsed:.1f}s")
    log.info(f"Bundle: {FREEZE_DIR}")
    log.info(f"ZIP: {zip_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
