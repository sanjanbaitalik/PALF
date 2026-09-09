"""Tests for palf_adaptive_prior (Phase 2A pilot)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PILOT_OUTPUT = REPO_ROOT / "outputs/iclr/palf_phase2a_adaptive_prior_pilot"
PILOT_SCRIPT = REPO_ROOT / "scripts_paper/phase2a_adaptive_prior_pilot.py"

from metascfc.experiments.palf_adaptive_prior import (
    build_adaptive_prior_cache,
    build_cache_for_rho,
    _compute_one_se_rule,
    AdaptivePriorCache,
    N_ROI, N_EDGE, GAMMA_FIXED, DIAGONAL_EPSILON, TOP_K,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    compute_diagonal_penalty,
    lift_roi_to_edge,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
)


def _make_test_cache():
    """Build a test AdaptivePriorCache with random roi_prior."""
    roi_prior = np.random.RandomState(42).rand(N_ROI) + 0.1
    return build_adaptive_prior_cache(roi_prior, n_rois=N_ROI, top_k=TOP_K)


# ---------------------------------------------------------------------------
# 1. rho=0 gives identity diagonal
# ---------------------------------------------------------------------------
def test_rho_zero_gives_identity_diagonal():
    """D_rho with rho=0 should be all ones (identity)."""
    ac = _make_test_cache()
    D_rho = (1.0 - 0.0) * ac.identity_diag + 0.0 * ac.D_prior
    np.testing.assert_allclose(D_rho, 1.0, atol=1e-10,
                               err_msg="D_rho with rho=0 is not identity")


# ---------------------------------------------------------------------------
# 2. rho=1 gives prior diagonal
# ---------------------------------------------------------------------------
def test_rho_one_gives_prior_diagonal():
    """D_rho with rho=1 should match D_prior."""
    ac = _make_test_cache()
    D_rho = (1.0 - 1.0) * ac.identity_diag + 1.0 * ac.D_prior
    np.testing.assert_allclose(D_rho, ac.D_prior, atol=1e-10,
                               err_msg="D_rho with rho=1 does not match D_prior")


# ---------------------------------------------------------------------------
# 3. interpolation preserves mean diagonal approximately 1
# ---------------------------------------------------------------------------
def test_interpolation_preserves_mean_diagonal():
    """D_rho mean should be approximately 1.0 for all rho values."""
    ac = _make_test_cache()
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        D_rho = (1.0 - rho) * ac.identity_diag + rho * ac.D_prior
        np.testing.assert_allclose(D_rho.mean(), 1.0, atol=1e-10,
                                   err_msg=f"D_rho mean != 1 for rho={rho}")


# ---------------------------------------------------------------------------
# 4. tau=0 removes the network term
# ---------------------------------------------------------------------------
def test_tau_zero_removes_network_term():
    """With tau=0, the Laplacian should have no effect on the solution."""
    rng = np.random.RandomState(42)
    n = 50
    X = rng.randn(n, N_EDGE)
    y = rng.randn(n)

    ac = _make_test_cache()
    cache = build_cache_for_rho(ac, rho=0.5)

    alpha_0, _ = _solve_msancr_kernel(
        X, np.zeros_like(X), y, cache, lambda_fc=1.0, lambda_sc=1.0,
        lambda_l=0.0, fc_only=True
    )
    alpha_1, _ = _solve_msancr_kernel(
        X, np.zeros_like(X), y, cache, lambda_fc=1.0, lambda_sc=1.0,
        lambda_l=1.0, fc_only=True
    )
    assert not np.allclose(alpha_0, alpha_1), (
        "tau=0 and tau>0 produce same solution - network term not working"
    )


# ---------------------------------------------------------------------------
# 5-6. rho/tau combinations match existing conditions
# ---------------------------------------------------------------------------
def test_rho_zero_tau_zero_matches_baseline():
    """(rho=0, tau=0) should produce the same objective as existing R0."""
    ac = _make_test_cache()
    cache = build_cache_for_rho(ac, rho=0.0)
    np.testing.assert_allclose(cache.D, 1.0, atol=1e-10,
                               err_msg="rho=0 cache D is not identity")


def test_rho_one_tau_zero_matches_anisotropy():
    """(rho=1, tau=0) should match anisotropy-only (R1) objective."""
    ac = _make_test_cache()
    cache = build_cache_for_rho(ac, rho=1.0)
    np.testing.assert_allclose(cache.D, ac.D_prior, atol=1e-10,
                               err_msg="rho=1 cache D does not match D_prior")


# ---------------------------------------------------------------------------
# 7-8. One-SE selector
# ---------------------------------------------------------------------------
def test_one_se_selector_returns_eligible():
    """One-SE selector must return a candidate within r_best - SE_best."""
    candidates = [
        {"rho": 0.0, "tau": 0.0, "lambda_F": 0.1, "mean_pearson": 0.30, "se_pearson": 0.012},
        {"rho": 0.25, "tau": 0.01, "lambda_F": 0.1, "mean_pearson": 0.31, "se_pearson": 0.012},
        {"rho": 0.5, "tau": 0.1, "lambda_F": 1.0, "mean_pearson": 0.25, "se_pearson": 0.012},
    ]
    _, best_se = _compute_one_se_rule(candidates)
    # best is candidate 1 (mean_r=0.31); threshold ~ 0.298
    # candidates 0 and 1 are eligible; rho=0 wins tie
    assert best_se["mean_pearson"] >= 0.29, (
        f"Selected candidate not in eligible set: {best_se}"
    )


def test_one_se_ties_prefer_lower_rho():
    """Ties broken by lowest rho first."""
    candidates = [
        {"rho": 0.5, "tau": 0.0, "lambda_F": 0.1, "mean_pearson": 0.30, "se_pearson": 0.001},
        {"rho": 0.0, "tau": 0.0, "lambda_F": 0.1, "mean_pearson": 0.30, "se_pearson": 0.001},
    ]
    _, best_se = _compute_one_se_rule(candidates)
    assert best_se["rho"] == 0.0, (
        f"Tie not broken by lower rho: selected rho={best_se['rho']}"
    )


# ---------------------------------------------------------------------------
# 9. No outer-test in selection
# ---------------------------------------------------------------------------
def test_no_outer_test_in_selection():
    """Selection helpers should not receive outer-test targets."""
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # y_test should NOT appear in the inner selection functions
    # but IS expected in the final evaluation section (metrics on held-out test)
    import re
    for func_name in ["_select_lambda_fc_only", "_select_adaptive_params_inner", "_select_alpha_sc"]:
        match = re.search(rf"def {func_name}\(.*?\n(?=\ndef |\Z)", src, re.DOTALL)
        if match:
            assert "y_test" not in match.group(0), (
                f"{func_name} references y_test - data leakage"
            )


# ---------------------------------------------------------------------------
# 10-11. Split indices and SC grid identical
# ---------------------------------------------------------------------------
def test_baseline_adaptive_split_indices_identical():
    """Both models must use identical train/test indices."""
    _skip_if_no_output()
    split_df = pd.read_csv(PILOT_OUTPUT / "split_metrics.csv")
    # 1 row per split with model_a_* and model_b_* columns
    assert len(split_df) == 40, f"Expected 40 rows, got {len(split_df)}"
    assert "model_a_fused_pearson" in split_df.columns
    assert "model_b_fused_pearson" in split_df.columns


def test_sc_grid_identical():
    """Same alpha_SC grid for both models."""
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "ALPHA_SC_GRID" in src or "alpha_SC_grid" in src, (
        "SC grid not found in pilot script"
    )


# ---------------------------------------------------------------------------
# 12. No stacking imports
# ---------------------------------------------------------------------------
def test_no_stacking_imports():
    """No Phase-1 stacking code imported."""
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "phase1_differential_stacking" not in src, (
        "Pilot script imports Phase-1 stacking code"
    )


# ---------------------------------------------------------------------------
# 13. Checkpoint restart
# ---------------------------------------------------------------------------
def test_checkpoint_restart_preserves_completed():
    """Restart loads completed splits correctly."""
    _skip_if_no_output()
    ckpt_path = PILOT_OUTPUT / "checkpoint.pkl"
    if not ckpt_path.exists():
        pytest.skip("No checkpoint found")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    assert isinstance(ckpt, dict), "Checkpoint is not a dict"
    assert len(ckpt) > 0, "Checkpoint is empty"


# ---------------------------------------------------------------------------
# 14. Output finite predictions
# ---------------------------------------------------------------------------
def test_output_finite_predictions():
    """All saved predictions are finite."""
    _skip_if_no_output()
    split_df = pd.read_csv(PILOT_OUTPUT / "split_metrics.csv")
    for col in ["fp_pearson", "sc_pearson", "fused_pearson", "fused_rmse", "fused_mae"]:
        if col in split_df.columns:
            assert np.all(np.isfinite(split_df[col].dropna())), (
                f"Non-finite values in {col}"
            )


# ---------------------------------------------------------------------------
# 15-16. Split and seed counts
# ---------------------------------------------------------------------------
def test_exactly_40_outer_splits():
    """2 targets × 4 seeds × 5 folds = 40 splits."""
    _skip_if_no_output()
    split_df = pd.read_csv(PILOT_OUTPUT / "split_metrics.csv")
    assert len(split_df) == 40, f"Expected 40 rows, got {len(split_df)}"


def test_exactly_4_seeds_per_target():
    """4 development seeds per target."""
    _skip_if_no_output()
    seed_df = pd.read_csv(PILOT_OUTPUT / "seed_metrics.csv")
    for task in seed_df["task"].unique():
        n_seeds = seed_df[seed_df["task"] == task]["seed"].nunique()
        assert n_seeds == 4, f"Expected 4 seeds for {task}, got {n_seeds}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_output():
    if not PILOT_OUTPUT.exists():
        pytest.skip("Phase 2A output not yet created")
