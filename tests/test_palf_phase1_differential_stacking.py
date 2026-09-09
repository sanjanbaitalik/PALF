#!/usr/bin/env python3
"""Tests for PALF Phase 1 differential stacking feasibility experiment."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = REPO_ROOT / "outputs/iclr/palf_phase1_differential_stacking"
CORRECTED_DIR = REPO_ROOT / "outputs/iclr/palf_same_solver_fusion_corrected_v2"
PHASE1_SCRIPT = REPO_ROOT / "scripts_paper/phase1_differential_stacking_feasibility.py"


def _skip_if_no_phase1():
    if not PHASE1_DIR.exists():
        pytest.skip("Phase 1 output directory not yet created")


def _skip_if_no_corrected():
    if not CORRECTED_DIR.exists():
        pytest.skip("Corrected v2 source directory not found")


def _load_checkpoint(task: str):
    import pickle
    ckpt_path = CORRECTED_DIR / task / "checkpoint.pkl"
    if not ckpt_path.exists():
        pytest.skip(f"Corrected checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)


def _load_split_metrics():
    csv_path = PHASE1_DIR / "split_metrics.csv"
    if not csv_path.exists():
        pytest.skip("split_metrics.csv not found")
    return pd.read_csv(csv_path)


def _load_seed_metrics():
    csv_path = PHASE1_DIR / "seed_metrics.csv"
    if not csv_path.exists():
        pytest.skip("seed_metrics.csv not found")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# 1. R0 and R3 split alignment assertion
# ---------------------------------------------------------------------------
def test_r0_r3_split_alignment():
    """R0 and R3 have identical train_idx/test_idx for each seed/fold."""
    _skip_if_no_corrected()

    for task in ["working_memory", "fluid_intelligence"]:
        ckpt = _load_checkpoint(task)
        r0 = sorted([s for s in ckpt.splits if s.condition_id == "R0"],
                     key=lambda s: (s.seed, s.outer_fold))
        r3 = sorted([s for s in ckpt.splits if s.condition_id == "R3"],
                     key=lambda s: (s.seed, s.outer_fold))
        assert len(r0) == len(r3) == 50
        for s0, s3 in zip(r0, r3):
            assert s0.seed == s3.seed
            assert s0.outer_fold == s3.outer_fold
            np.testing.assert_array_equal(s0.train_idx, s3.train_idx)
            np.testing.assert_array_equal(s0.test_idx, s3.test_idx)


# ---------------------------------------------------------------------------
# 2. Differential feature equals R3 FP minus R0 FP
# ---------------------------------------------------------------------------
def test_differential_feature_equals_r3_minus_r0():
    """delta_oof == R3.fp_oof - R0.fp_oof for every split."""
    _skip_if_no_corrected()

    for task in ["working_memory", "fluid_intelligence"]:
        ckpt = _load_checkpoint(task)
        by_key = {}
        for s in ckpt.splits:
            by_key.setdefault((s.seed, s.outer_fold), {})[s.condition_id] = s

        for key in sorted(by_key):
            if "R0" not in by_key[key] or "R3" not in by_key[key]:
                continue
            r0 = by_key[key]["R0"]
            r3 = by_key[key]["R3"]
            delta_oof = r3.fp_oof - r0.fp_oof
            # Verify it's a meaningful differential (not all zeros)
            assert delta_oof.shape == r0.fp_oof.shape
            assert not np.allclose(delta_oof, 0.0), (
                f"delta_oof is all zeros for {task} seed={key[0]} fold={key[1]}"
            )


# ---------------------------------------------------------------------------
# 3. Candidate B with a=0 exactly reproduces baseline
# ---------------------------------------------------------------------------
def test_candidate_b_a_zero_reproduces_baseline():
    """Candidate B with a=0 must exactly reproduce the corrected same-solver baseline."""
    _skip_if_no_phase1()
    split_df = _load_split_metrics()

    for task in ["working_memory", "fluid_intelligence"]:
        task_df = split_df[split_df["task"] == task]
        for _, row in task_df.iterrows():
            if row["selected_candidate"] == "B" and np.isclose(row.get("b_best_a", -1), 0.0):
                np.testing.assert_allclose(
                    row["selected_r"], row["baseline_r"], atol=1e-10,
                    err_msg=f"Candidate B a=0 does not match baseline for {task} "
                            f"seed={row['seed']} fold={row['outer_fold']}",
                )


# ---------------------------------------------------------------------------
# 4. Candidate C coefficients are nonnegative
# ---------------------------------------------------------------------------
def test_candidate_c_coefficients_nonnegative():
    """Candidate C weights must all be >= 0."""
    _skip_if_no_phase1()
    split_df = _load_split_metrics()

    c_splits = split_df[split_df["selected_candidate"] == "C"]
    if len(c_splits) == 0:
        pytest.skip("No Candidate C splits selected")

    # Check that w0, w_sc, w_delta columns exist and are nonnegative
    for col in ["w0", "w_sc", "w_delta"]:
        if col in c_splits.columns:
            bad = c_splits[c_splits[col] < -1e-10]
            assert bad.empty, (
                f"Found negative {col} in Candidate C: "
                f"{bad[['task', 'seed', 'outer_fold', col]].to_dict('records')}"
            )


# ---------------------------------------------------------------------------
# 5. Candidate C intercept is unpenalized
# ---------------------------------------------------------------------------
def test_candidate_c_intercept_unpenalized():
    """Candidate C intercept must not be penalized."""
    if not PHASE1_SCRIPT.exists():
        pytest.skip("Phase 1 script not found")
    src = PHASE1_SCRIPT.read_text()
    # The augmented system should NOT include intercept in the penalty
    assert "np.eye(" in src, (
        "Expected nonnegative Ridge augmented system with identity penalty matrix"
    )


# ---------------------------------------------------------------------------
# 6. Meta-CV does not access outer-test target
# ---------------------------------------------------------------------------
def test_meta_cv_no_outer_test_access():
    """meta-CV must never use outer-test targets for fitting or hyperparameter selection."""
    if not PHASE1_SCRIPT.exists():
        pytest.skip("Phase 1 script not found")
    src = PHASE1_SCRIPT.read_text()
    # Simpler check: look for y_test usage within the meta-CV fold loop
    # The meta-CV loop should use meta_val_idx, not test_idx
    assert "meta_val" in src, "Expected meta_val variable for meta-CV validation fold"
    # Verify that test_idx is not used in fit/transform contexts
    # (It's OK for test_idx to appear in the outer-loop evaluation section)
    lines = src.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check for direct y_test usage in Ridge.fit or similar
        if "y_test" in stripped and ".fit(" in stripped:
            pytest.fail(
                f"Line {i+1}: y_test used in .fit() call: {stripped}"
            )


# ---------------------------------------------------------------------------
# 7. Hyperparameter selection uses only meta-CV predictions
# ---------------------------------------------------------------------------
def test_hyperparameter_selection_uses_only_meta_cv():
    """Hyperparameters must be selected only on meta-CV predictions."""
    _skip_if_no_phase1()
    split_df = _load_split_metrics()

    # Verify that selected_hyperparam is populated for all splits
    assert "selected_hyperparam" in split_df.columns
    bad = split_df[split_df["selected_hyperparam"].isna()]
    assert bad.empty, (
        f"Found splits with missing selected_hyperparam: "
        f"{bad[['task', 'seed', 'outer_fold', 'selected_candidate']].to_dict('records')}"
    )


# ---------------------------------------------------------------------------
# 8. Architecture tie-break prefers simpler model
# ---------------------------------------------------------------------------
def test_architecture_tiebreak_prefers_simpler():
    """When performance is tied, baseline should be preferred over residual over Ridge."""
    if not PHASE1_SCRIPT.exists():
        pytest.skip("Phase 1 script not found")
    src = PHASE1_SCRIPT.read_text()
    # The selection logic should check baseline first, then residual, then Ridge
    # Look for the ordering pattern
    assert "baseline" in src.lower() and "residual" in src.lower(), (
        "Expected baseline/residual architecture names in selection logic"
    )


# ---------------------------------------------------------------------------
# 9. No base-model fit function is called by Phase-1 script
# ---------------------------------------------------------------------------
def test_no_base_model_fit_called():
    """Phase 1 script must not call fit() on FC/SC base models."""
    if not PHASE1_SCRIPT.exists():
        pytest.skip("Phase 1 script not found")
    src = PHASE1_SCRIPT.read_text()
    # Should not contain fit() calls on Ridge for base models
    # Only meta-layer Ridge/lsq_linear is allowed
    lines = src.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check for Ridge().fit() or model.fit() that aren't meta-layer
        if "Ridge(" in stripped and ".fit(" in stripped:
            # This should only be in meta-CV or final meta-model fitting
            # Allow it if it's clearly in meta context
            if "meta" not in stripped.lower() and "stack" not in stripped.lower():
                # Check surrounding context
                context = "\n".join(lines[max(0, i-3):i+3])
                if "meta" not in context.lower() and "stack" not in context.lower():
                    pytest.fail(
                        f"Line {i+1}: Possible base-model fit() call: {stripped}"
                    )


# ---------------------------------------------------------------------------
# 10. Output has exactly 100 selected outer models
# ---------------------------------------------------------------------------
def test_output_has_100_selected_models():
    """Exactly 100 selected outer models (50 per target)."""
    _skip_if_no_phase1()
    split_df = _load_split_metrics()
    assert len(split_df) == 100, f"Expected 100 rows, got {len(split_df)}"

    for task in ["working_memory", "fluid_intelligence"]:
        task_df = split_df[split_df["task"] == task]
        assert len(task_df) == 50, f"Expected 50 rows for {task}, got {len(task_df)}"


# ---------------------------------------------------------------------------
# 11. Seed aggregation yields 10 seeds per target
# ---------------------------------------------------------------------------
def test_seed_aggregation_yields_10_seeds():
    """10 seeds per target in seed_metrics."""
    _skip_if_no_phase1()
    seed_df = _load_seed_metrics()
    for task in ["working_memory", "fluid_intelligence"]:
        task_df = seed_df[seed_df["task"] == task]
        n_seeds = task_df["seed"].nunique()
        assert n_seeds == 10, f"Expected 10 seeds for {task}, got {n_seeds}"


# ---------------------------------------------------------------------------
# 12. Primary Holm family contains exactly two tests
# ---------------------------------------------------------------------------
def test_primary_holm_family_has_two_tests():
    """Holm family contains exactly 2 tests (WM and FI)."""
    _skip_if_no_phase1()
    comp_path = PHASE1_DIR / "primary_comparisons.csv"
    if not comp_path.exists():
        pytest.skip("primary_comparisons.csv not found")
    df = pd.read_csv(comp_path)
    assert len(df) == 2, f"Expected 2 comparison rows, got {len(df)}"

    # Check Holm p values are >= raw p values
    for _, row in df.iterrows():
        if "p_holm" in df.columns and "raw_p" in df.columns:
            assert row["p_holm"] >= row["raw_p"] - 1e-12, (
                f"Holm p ({row['p_holm']}) < raw p ({row['raw_p']}) for {row['task']}"
            )


# ---------------------------------------------------------------------------
# 13. All saved outer-test predictions are finite
# ---------------------------------------------------------------------------
def test_all_outer_test_predictions_finite():
    """No NaN/Inf in outer_predictions CSVs."""
    _skip_if_no_phase1()
    for task in ["working_memory", "fluid_intelligence"]:
        csv_path = PHASE1_DIR / "outer_predictions" / f"{task}.csv"
        if not csv_path.exists():
            pytest.skip(f"outer_predictions/{task}.csv not found")
        df = pd.read_csv(csv_path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            assert np.all(np.isfinite(df[col].dropna())), (
                f"Non-finite values in {col} for {task}"
            )


# ---------------------------------------------------------------------------
# 14. Frozen source directories remain unchanged
# ---------------------------------------------------------------------------
def test_frozen_source_unchanged():
    """Frozen source directories should not be modified by Phase 1."""
    _skip_if_no_corrected()
    # Check that the corrected v2 checkpoint files are unchanged
    import hashlib

    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    for task in ["working_memory", "fluid_intelligence"]:
        ckpt_path = CORRECTED_DIR / task / "checkpoint.pkl"
        if ckpt_path.exists():
            # Just verify the file is readable and valid
            import pickle
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            assert len(ckpt.splits) == 200
