#!/usr/bin/env python3
"""Fix fusion weights: recompute from branch OOF predictions in checkpoints."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    search_fusion_weights,
    compute_seed_metrics,
    compute_component_summary,
)
from metascfc.benchmark_utils import prediction_metrics


OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"


def recompute_fusion(result):
    """Recompute fusion weights and fused predictions from branch OOF/test predictions."""
    for split in result.splits:
        condition = CONDITIONS[split.condition_id]
        y_train = split.fc_oof  # Dummy; we need the actual y for train_idx

        # We don't have y_train directly in the split, but we have fc_oof and sc_oof
        # which were computed against y_train. We need y_train to recompute fusion.
        # Fortunately, we can recover it from the branch OOF predictions + their metrics.
        # But actually we need y_true for the OOF subjects.

        # The simplest approach: recompute fusion from OOF predictions
        # We don't have y_train stored, but we can get it from the inner metrics.
        # Actually, let's just store y_train in the checkpoint.

        # For now, let's use the fact that fc_oof and fp_oof are OOF predictions
        # and we need y[train_idx] to select fusion weights.
        # We'll load the data separately.

        pass  # Placeholder - need y_train

    return result


def recompute_fusion_with_data(result, y):
    """Recompute fusion from OOF predictions and full y vector."""
    for split in result.splits:
        condition = CONDITIONS[split.condition_id]
        train_idx = split.train_idx
        test_idx = split.test_idx
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Recompute fusion weights
        if condition.use_anisotropy or condition.use_network:
            fusion_weights, _ = search_fusion_weights(
                y_train, {"FP": split.fp_oof, "SC": split.sc_oof}, ["FP", "SC"],
            )
            w_fp = fusion_weights.get("FP", 0.5)
            w_sc = fusion_weights.get("SC", 0.5)
            fused_test = w_fp * split.fp_test_pred + w_sc * split.sc_test_pred
            ew_test = 0.5 * split.fp_test_pred + 0.5 * split.sc_test_pred
        else:
            fusion_weights, _ = search_fusion_weights(
                y_train, {"FC": split.fc_oof, "SC": split.sc_oof}, ["FC", "SC"],
            )
            w_fc = fusion_weights.get("FC", 0.5)
            w_sc = fusion_weights.get("SC", 0.5)
            fused_test = w_fc * split.fc_test_pred + w_sc * split.sc_test_pred
            ew_test = 0.5 * split.fc_test_pred + 0.5 * split.sc_test_pred

        # Update split
        split.fusion_weights = fusion_weights
        split.fused_test_pred = fused_test
        split.equal_weight_pred = ew_test
        split.fused_metrics = prediction_metrics(y_test, fused_test)
        split.equal_weight_metrics = prediction_metrics(y_test, ew_test)

    return result


def main():
    # Load data
    fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
    sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
    iu = np.triu_indices(116, k=1)
    X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)

    targets = {
        "working_memory": np.load(REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64),
        "fluid_intelligence": np.load(REPO_ROOT / "inputs/dataset_SC/label_all.npy").astype(np.float64),
    }

    for task_key, y in targets.items():
        ckpt_path = OUTPUT_BASE / task_key / "checkpoint.pkl"
        if not ckpt_path.exists():
            print(f"No checkpoint: {ckpt_path}")
            continue

        print(f"Loading {task_key}...")
        with open(ckpt_path, "rb") as f:
            result = pickle.load(f)

        print(f"  {len(result.splits)} splits loaded, recomputing fusion...")
        result = recompute_fusion_with_data(result, y)

        # Save updated checkpoint
        with open(ckpt_path, "wb") as f:
            pickle.dump(result, f)

        # Verify
        seed_df = compute_seed_metrics(result)
        fused = seed_df[seed_df["model"] == "fused"]
        for cond in ["R0", "R1", "R2", "R3"]:
            sub = fused[fused["condition"] == cond]
            print(f"  {cond}: mean_fused_r={sub['mean_pearson'].mean():.4f} (+/-{sub['mean_pearson'].std(ddof=1):.4f})")

    print("\nDone! Run postprocess_palf_ablation.py to regenerate all outputs.")


if __name__ == "__main__":
    main()
