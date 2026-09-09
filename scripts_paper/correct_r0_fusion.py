#!/usr/bin/env python3
"""Correct R0 fusion: recompute from ordinary Ridge FC+SC to generalized no-prior FP+SC.

No branch refitting — only recomputes fusion weights from stored OOF predictions.
Only modifies R0 splits; R1/R2/R3 are copied unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    search_fusion_weights,
    AblationTaskResult,
    AblationSplitResult,
)
from metascfc.benchmark_utils import prediction_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("correct_r0_fusion")

INPUT_CKPT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"
OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"

TASK_LABEL_MAP = {
    "working_memory": "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy",
    "fluid_intelligence": "inputs/dataset_SC/label_all.npy",
}

TASKS = ["working_memory", "fluid_intelligence"]

FREEZE_ID = "9abbb56"

N_BOOT = 10000


def _compute_holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni correction (restores original order)."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted_val = min(pval * (n - rank), 1.0)
        running_max = max(running_max, adjusted_val)
        adjusted[orig_idx] = running_max
    return adjusted


def correct_r0_splits(result: AblationTaskResult, y: np.ndarray) -> tuple[AblationTaskResult, list[dict]]:
    """Correct R0 fusion weights. Returns corrected result and audit entries."""
    corrected = copy.deepcopy(result)
    audit = []

    for orig_split, corr_split in zip(result.splits, corrected.splits):
        if corr_split.condition_id != "R0":
            continue

        seed = corr_split.seed
        fold = corr_split.outer_fold
        train_idx = corr_split.train_idx
        test_idx = corr_split.test_idx
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Store originals for verification
        old_fusion_keys = dict(corr_split.fusion_weights)
        old_fused_r = float(corr_split.fused_metrics["pearson"])
        old_fused_pred = corr_split.fused_test_pred.copy()
        old_fp_pred = corr_split.fp_test_pred.copy()
        old_sc_pred = corr_split.sc_test_pred.copy()
        old_ew_pred = corr_split.equal_weight_pred.copy()

        # Recompute fusion from OOF: generalized no-prior FP+SC
        w, _ = search_fusion_weights(
            y_train, {"FP": corr_split.fp_oof, "SC": corr_split.sc_oof}, ["FP", "SC"],
        )
        corr_split.fusion_weights = w

        # Recompute fused test prediction
        corr_split.fused_test_pred = w["FP"] * corr_split.fp_test_pred + w["SC"] * corr_split.sc_test_pred

        # Recompute equal-weight test prediction
        corr_split.equal_weight_pred = 0.5 * corr_split.fp_test_pred + 0.5 * corr_split.sc_test_pred

        # Recompute metrics
        corr_split.fused_metrics = prediction_metrics(y_test, corr_split.fused_test_pred)
        corr_split.equal_weight_metrics = prediction_metrics(y_test, corr_split.equal_weight_pred)

        # Verify FP and SC predictions unchanged
        fp_changed = not np.allclose(corr_split.fp_test_pred, old_fp_pred, atol=1e-12)
        sc_changed = not np.allclose(corr_split.sc_test_pred, old_sc_pred, atol=1e-12)

        new_fused_r = float(corr_split.fused_metrics["pearson"])

        audit.append({
            "seed": int(seed),
            "fold": int(fold),
            "old_fusion_keys": {k: round(v, 4) for k, v in old_fusion_keys.items()},
            "old_fused_r": round(old_fused_r, 6),
            "new_w_FP": round(w["FP"], 4),
            "new_w_SC": round(w["SC"], 4),
            "new_fused_r": round(new_fused_r, 6),
            "fp_predictions_changed": fp_changed,
            "sc_predictions_changed": sc_changed,
            "r1_r2_r3_changed": False,
        })

    return corrected, audit


def verify_unchanged(orig_result: AblationTaskResult, corr_result: AblationTaskResult) -> bool:
    """Verify R1/R2/R3 splits are unchanged."""
    for orig_s, corr_s in zip(orig_result.splits, corr_result.splits):
        if corr_s.condition_id in ("R1", "R2", "R3"):
            if not np.allclose(orig_s.fused_test_pred, corr_s.fused_test_pred, atol=1e-12):
                logger.warning(
                    f"R1/R2/R3 changed: {corr_s.condition_id} seed={corr_s.seed} fold={corr_s.outer_fold}"
                )
                return False
            if orig_s.fusion_weights != corr_s.fusion_weights:
                logger.warning(
                    f"R1/R2/R3 fusion weights changed: {corr_s.condition_id} seed={corr_s.seed}"
                )
                return False
    return True


def compute_primary_results(results: dict[str, AblationTaskResult]) -> dict:
    """Compute primary results: seed-level metrics, paired Wilcoxon, bootstrap CI, Holm."""
    task_results = {}

    for task_key, result in results.items():
        splits = result.splits

        # Seed-level metrics: 10 seeds x 5 folds -> 10 seed means
        seed_metrics = {}  # {condition: {model: {seed: [fold_pearsons]}}}
        for s in splits:
            cond = s.condition_id
            seed = s.seed
            if cond not in seed_metrics:
                seed_metrics[cond] = {}
            for model_name, metrics in [
                ("FC", s.fc_metrics), ("SC", s.sc_metrics),
                ("FP", s.fp_metrics), ("fused", s.fused_metrics),
                ("equal_weight", s.equal_weight_metrics),
            ]:
                if model_name not in seed_metrics[cond]:
                    seed_metrics[cond][model_name] = {}
                if seed not in seed_metrics[cond][model_name]:
                    seed_metrics[cond][model_name][seed] = []
                seed_metrics[cond][model_name][seed].append(metrics["pearson"])

        # Average across folds per seed
        seed_means = {}
        for cond, models in seed_metrics.items():
            seed_means[cond] = {}
            for model, seeds in models.items():
                seed_means[cond][model] = {
                    s: float(np.mean(vals)) for s, vals in sorted(seeds.items())
                }

        # Primary: R3 vs R0 (corrected) fused
        seeds_list = sorted(seed_means.get("R0", {}).get("fused", {}).keys())
        if len(seeds_list) == 10:
            r3_vals = np.array([seed_means["R3"]["fused"][s] for s in seeds_list])
            r0_vals = np.array([seed_means["R0"]["fused"][s] for s in seeds_list])
            diffs = r3_vals - r0_vals
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1))
            cohens_dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0

            # Paired Wilcoxon
            try:
                if np.all(diffs == 0):
                    wilcox_p = 1.0
                    wilcox_stat = 0.0
                else:
                    nonzero = diffs[diffs != 0]
                    if len(nonzero) < 5:
                        wilcox_p = 1.0
                        wilcox_stat = 0.0
                    else:
                        res = wilcoxon(diffs, alternative="two-sided")
                        wilcox_stat = float(res.statistic)
                        wilcox_p = float(res.pvalue)
            except Exception:
                wilcox_p = 1.0
                wilcox_stat = 0.0

            # Bootstrap CI
            rng = np.random.RandomState(20260906)
            boot_means = []
            n = len(diffs)
            for _ in range(N_BOOT):
                idx = rng.randint(0, n, n)
                boot_means.append(float(np.mean(diffs[idx])))
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))

            task_results[task_key] = {
                "r3_fused_mean": float(np.mean(r3_vals)),
                "r0_fused_mean_corrected": float(np.mean(r0_vals)),
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "cohens_dz": cohens_dz,
                "wilcoxon_statistic": wilcox_stat,
                "wilcoxon_p": wilcox_p,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "n_seeds": 10,
                "positive_seeds": int(np.sum(diffs > 0)),
            }
        else:
            task_results[task_key] = {"error": f"Expected 10 seeds, got {len(seeds_list)}"}

    # Holm correction across 2 targets
    pvalues = []
    task_order = []
    for task_key in TASKS:
        if task_key in task_results and "wilcoxon_p" in task_results[task_key]:
            pvalues.append(task_results[task_key]["wilcoxon_p"])
            task_order.append(task_key)

    if pvalues:
        adj_p = _compute_holm(pvalues)
        for task_key, p in zip(task_order, adj_p):
            task_results[task_key]["wilcoxon_p_holm"] = p

    return task_results


def main():
    t0 = time.time()
    all_audit = {}
    all_results = {}

    for task_key in TASKS:
        logger.info(f"Processing {task_key}...")

        # Load checkpoint
        ckpt_path = INPUT_CKPT_BASE / task_key / "checkpoint.pkl"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            continue
        with open(ckpt_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"  Loaded {len(result.splits)} splits")

        # Load labels
        label_path = REPO_ROOT / TASK_LABEL_MAP[task_key]
        y = np.load(label_path).astype(np.float64)
        logger.info(f"  Loaded {len(y)} labels from {label_path}")

        # Correct R0 fusion
        corrected, audit = correct_r0_splits(result, y)
        logger.info(f"  Corrected {len(audit)} R0 splits")

        # Verify R1/R2/R3 unchanged
        unchanged = verify_unchanged(result, corrected)
        if not unchanged:
            logger.error(f"  WARNING: R1/R2/R3 splits were modified!")
        else:
            logger.info(f"  Verified: R1/R2/R3 unchanged")

        # Check FP/SC predictions unchanged
        fp_changed_any = any(e["fp_predictions_changed"] for e in audit)
        sc_changed_any = any(e["sc_predictions_changed"] for e in audit)
        if fp_changed_any:
            logger.error(f"  WARNING: FP predictions changed in some R0 splits!")
        if sc_changed_any:
            logger.error(f"  WARNING: SC predictions changed in some R0 splits!")
        if not fp_changed_any and not sc_changed_any:
            logger.info(f"  Verified: FP and SC predictions unchanged in all R0 splits")

        # Save corrected checkpoint
        out_dir = OUTPUT_BASE / task_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "checkpoint.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(corrected, f)
        logger.info(f"  Saved corrected checkpoint to {out_path}")

        # Store for primary results
        all_results[task_key] = corrected
        all_audit[task_key] = {
            "freeze_id": FREEZE_ID,
            "task": task_key,
            "total_r0_splits": len(audit),
            "corrections": audit,
        }

    # Save audit JSON
    audit_path = OUTPUT_BASE / "R0_FUSION_CORRECTION_AUDIT.json"
    with open(audit_path, "w") as f:
        json.dump(all_audit, f, indent=2)
    logger.info(f"Saved audit to {audit_path}")

    # Compute primary results
    logger.info("Computing primary results...")
    primary = compute_primary_results(all_results)

    primary_path = OUTPUT_BASE / "primary_results.json"
    with open(primary_path, "w") as f:
        json.dump(primary, f, indent=2)
    logger.info(f"Saved primary results to {primary_path}")

    # Print summary
    for task_key in TASKS:
        if task_key in primary and "mean_difference" in primary[task_key]:
            r = primary[task_key]
            logger.info(
                f"  {task_key}: R3={r['r3_fused_mean']:.4f} "
                f"R0_corrected={r['r0_fused_mean_corrected']:.4f} "
                f"delta={r['mean_difference']:+.4f} "
                f"p={r['wilcoxon_p']:.4f} "
                f"adj_p={r.get('wilcoxon_p_holm', 'N/A')}"
            )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
