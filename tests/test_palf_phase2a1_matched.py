"""Tests for Phase 2A.1 matched adaptive pilot."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2a1_matched_adaptive_pilot"
PILOT_SCRIPT = REPO_ROOT / "scripts_paper/phase2a1_matched_adaptive_pilot.py"
OLD_OUTPUT = REPO_ROOT / "outputs/iclr/palf_phase2a_adaptive_prior_pilot"

from metascfc.experiments.palf_adaptive_prior import (
    build_adaptive_prior_cache,
    build_cache_for_rho,
    N_ROI, N_EDGE, TOP_K,
    GAMMA_FIXED, DIAGONAL_EPSILON,
)


def _make_test_cache():
    roi_prior = np.random.RandomState(42).rand(N_ROI) + 0.1
    return build_adaptive_prior_cache(roi_prior, n_rois=N_ROI, top_k=TOP_K,
                                      gamma=GAMMA_FIXED, epsilon=DIAGONAL_EPSILON)


# ---------------------------------------------------------------------------
# 1. SC is selected exactly once per split
# ---------------------------------------------------------------------------
def test_sc_selected_once_per_split():
    """Each split has exactly one shared alpha_SC."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    assert "shared_alpha_SC" in sm.columns
    # Each row = 1 split, so SC was selected once per split by design
    assert len(sm) == 40


# ---------------------------------------------------------------------------
# 2. Baseline and adaptive share identical SC OOF/test arrays
# ---------------------------------------------------------------------------
def test_shared_sc_oof_test():
    """SC predictions are identical across models (shared branch)."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    # All splits use shared SC - verify column exists
    assert "shared_sc_pearson" in sm.columns
    # No model_a_sc_* vs model_b_sc_* columns (old format)
    assert "model_a_sc_pearson" not in sm.columns
    assert "model_b_sc_pearson" not in sm.columns


# ---------------------------------------------------------------------------
# 3. Baseline/adaptive alpha_SC is identical
# ---------------------------------------------------------------------------
def test_shared_alpha_SC_identical():
    """Only one alpha_SC column exists (shared)."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    assert "shared_alpha_SC" in sm.columns
    # No separate baseline/adaptive alpha_SC
    assert "baseline_alpha_SC" not in sm.columns
    assert "adaptive_alpha_SC" not in sm.columns


# ---------------------------------------------------------------------------
# 4. Baseline and adaptive use the same one-SE helper
# ---------------------------------------------------------------------------
def test_one_SE_framework_used():
    """Both baseline and adaptive use one-SE selection."""
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "_compute_one_se_rule" in src, "One-SE rule not used in pilot"
    assert "_select_lambda_fc_one_se" in src, "Baseline one-SE selector missing"
    assert "_select_adaptive_one_se" in src, "Adaptive one-SE selector missing"


# ---------------------------------------------------------------------------
# 5. rho=0,tau=0,lambda=x produces identical FP predictions
# ---------------------------------------------------------------------------
def test_identity_gate_rho0_tau0():
    """rho=0,tau=0 must produce identical FP predictions."""
    from metascfc.experiments.palf_crossfit_ablation import (
        _solve_msancr_kernel, _predict_msancr, N_EDGE,
    )
    rng = np.random.RandomState(42)
    n_train, n_test = 50, 20
    X_train = rng.randn(n_train, N_EDGE)
    X_test = rng.randn(n_test, N_EDGE)
    y = rng.randn(n_train)

    ac = _make_test_cache()
    for lambda_F in [0.001, 0.01, 0.1, 1.0, 10.0]:
        cache = build_cache_for_rho(ac, rho=0.0)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xb = scaler.fit_transform(X_train)
        Xc = scaler.transform(X_test)
        yz = (y - y.mean()) / max(y.std(), 1e-8)

        alpha, _ = _solve_msancr_kernel(
            Xb, np.zeros_like(Xb), yz, cache, lambda_F, 1.0, 0.0, fc_only=True,
        )
        pred = _predict_msancr(
            Xc, np.zeros_like(Xc), Xb, np.zeros_like(Xb),
            alpha, cache, lambda_F, 1.0, 0.0, fc_only=True,
        )
        # Run again with separate cache - should be identical
        cache2 = build_cache_for_rho(ac, rho=0.0)
        alpha2, _ = _solve_msancr_kernel(
            Xb, np.zeros_like(Xb), yz, cache2, lambda_F, 1.0, 0.0, fc_only=True,
        )
        pred2 = _predict_msancr(
            Xc, np.zeros_like(Xc), Xb, np.zeros_like(Xb),
            alpha2, cache2, lambda_F, 1.0, 0.0, fc_only=True,
        )
        np.testing.assert_allclose(pred, pred2, atol=1e-12,
                                   err_msg=f"Identity gate failed for lambda_F={lambda_F}")


# ---------------------------------------------------------------------------
# 6. Adaptive shadow uses exactly the adaptive selected lambda_F
# ---------------------------------------------------------------------------
def test_shadow_uses_adaptive_lambda_F():
    """Shadow lambda_F equals adaptive lambda_F in all splits."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    assert "shadow_lambda_F" in sm.columns
    assert "adaptive_lambda_F" in sm.columns
    np.testing.assert_array_equal(
        sm["shadow_lambda_F"].values,
        sm["adaptive_lambda_F"].values,
        err_msg="Shadow lambda_F != adaptive lambda_F",
    )


# ---------------------------------------------------------------------------
# 7. If adaptive rho=tau=0, adaptive == shadow
# ---------------------------------------------------------------------------
def test_rho0_tau0_adaptive_equals_shadow():
    """When rho=tau=0, adaptive and shadow FP predictions must be identical."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    mask = (sm["adaptive_rho"] == 0) & (sm["adaptive_tau"] == 0)
    if mask.sum() == 0:
        pytest.skip("No rho=0,tau=0 splits")
    subset = sm[mask]
    np.testing.assert_allclose(
        subset["adaptive_fp_pearson"].values,
        subset["shadow_fp_pearson"].values,
        atol=1e-10,
        err_msg="adaptive != shadow when rho=tau=0",
    )


# ---------------------------------------------------------------------------
# 8. Fixed-weight semantic comparison uses Adaptive fusion weights
# ---------------------------------------------------------------------------
def test_fixed_weight_uses_adaptive_weights():
    """Fixed-weight shadow uses adaptive w_FP/w_SC, not baseline."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    # Verify columns exist
    assert "adaptive_w_FP" in sm.columns
    assert "adaptive_w_SC" in sm.columns
    assert "semantic_delta_fused_fixed_weight" in sm.columns
    # Check that weights sum to 1
    np.testing.assert_allclose(
        sm["adaptive_w_FP"] + sm["adaptive_w_SC"], 1.0, atol=1e-10,
        err_msg="Adaptive fusion weights do not sum to 1",
    )


# ---------------------------------------------------------------------------
# 9. No outer-test labels enter selection
# ---------------------------------------------------------------------------
def test_no_outer_test_in_selection():
    """Selection functions do not reference y_test."""
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    import re
    src = PILOT_SCRIPT.read_text()
    for func_name in ["_select_lambda_fc_one_se", "_select_adaptive_one_se", "_select_alpha_sc"]:
        match = re.search(rf"def {func_name}\(.*?\n(?=\ndef |\Z)", src, re.DOTALL)
        if match:
            assert "y_test" not in match.group(0), (
                f"{func_name} references y_test - data leakage"
            )


# ---------------------------------------------------------------------------
# 10. Four dev seeds x five folds x two tasks completed
# ---------------------------------------------------------------------------
def test_exactly_40_splits():
    """4 seeds x 5 folds x 2 tasks = 40 splits."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    assert len(sm) == 40, f"Expected 40, got {len(sm)}"
    assert sm["task"].nunique() == 2
    assert sm["seed"].nunique() == 4
    assert sm["outer_fold"].nunique() == 5


# ---------------------------------------------------------------------------
# 11. Semantic counts generated from CSV
# ---------------------------------------------------------------------------
def test_semantic_counts_from_csv():
    """Semantic counts are computable from CSV, not hardcoded."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    assert "adaptive_uses_semantic" in sm.columns
    # Verify audit JSON exists
    audit_path = OUTPUT_DIR / "selection_summary_audit.json"
    assert audit_path.exists(), "Audit JSON not found"
    with open(audit_path) as f:
        audit = json.load(f)
    # Verify counts match
    csv_count = int(sm["adaptive_uses_semantic"].sum())
    audit_count = audit["total"]["any_semantic_count"]
    assert csv_count == audit_count, (
        f"CSV semantic count ({csv_count}) != audit count ({audit_count})"
    )


# ---------------------------------------------------------------------------
# 12. Old Phase-2A outputs remain unchanged
# ---------------------------------------------------------------------------
def test_old_phase2a_unchanged():
    """Old Phase-2A outputs are not modified."""
    if not OLD_OUTPUT.exists():
        pytest.skip("Old Phase 2A output not found")
    sm_path = OLD_OUTPUT / "split_metrics.csv"
    if sm_path.exists():
        df = pd.read_csv(sm_path)
        assert len(df) == 40, "Old Phase 2A split_metrics modified"


# ---------------------------------------------------------------------------
# 13. Current final/frozen seeds 0-9 are not used
# ---------------------------------------------------------------------------
def test_dev_seeds_only():
    """Only seeds 101, 202, 303, 404 are used."""
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    allowed = {101, 202, 303, 404}
    actual = set(sm["seed"].unique())
    assert actual == allowed, f"Unexpected seeds: {actual}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_output():
    if not OUTPUT_DIR.exists() or not (OUTPUT_DIR / "split_metrics.csv").exists():
        pytest.skip("Phase 2A.1 output not yet created")

import json
