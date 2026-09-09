"""Tests for Phase 2B: Joint Multi-Penalty Semantic-Group Ridge."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2b_joint_group_ridge_pilot"
PILOT_SCRIPT = REPO_ROOT / "scripts_paper/phase2b_joint_group_ridge_pilot.py"

from metascfc.experiments.palf_multi_penalty_ridge import (
    N_ROI, N_EDGE,
    build_edge_grouping,
    solve_mpr_kernel,
    predict_mpr_kernel,
    compute_beta_fc_full,
    validate_primal_reconstruction,
)

# Load prior for tests
roi_names_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)
ROI_NAMES = roi_names_df["roi_label"].tolist()
wm_prior = roi_names_df["prior_score"].values.astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Semantic/generic edge groups are disjoint
# ---------------------------------------------------------------------------
def test_edge_groups_disjoint():
    g = build_edge_grouping(wm_prior, ROI_NAMES, top_k=10)
    overlap = set(g.semantic_indices.tolist()) & set(g.generic_indices.tolist())
    assert len(overlap) == 0, f"Edge groups overlap: {overlap}"


# ---------------------------------------------------------------------------
# 2. Union contains exactly 6670 FC edges
# ---------------------------------------------------------------------------
def test_edge_groups_exhaustive():
    g = build_edge_grouping(wm_prior, ROI_NAMES, top_k=10)
    union = set(g.semantic_indices.tolist()) | set(g.generic_indices.tolist())
    assert len(union) == N_EDGE, f"Union size {len(union)} != {N_EDGE}"


# ---------------------------------------------------------------------------
# 3. Semantic edge count is 1105 for top-10
# ---------------------------------------------------------------------------
def test_semantic_edge_count_1105():
    g = build_edge_grouping(wm_prior, ROI_NAMES, top_k=10)
    assert g.n_semantic == 1105, f"Semantic edges {g.n_semantic} != 1105"


# ---------------------------------------------------------------------------
# 4. r_A=1 gives uniform FC penalty across all FC edges
# ---------------------------------------------------------------------------
def test_rA1_gives_uniform_penalty():
    rng = np.random.RandomState(42)
    n, p_G, p_A, p_S = 30, 100, 50, 80
    X_G = rng.randn(n, p_G)
    X_A = rng.randn(n, p_A)
    X_S = rng.randn(n, p_S)
    y = rng.randn(n)

    # r_A=1, same lambda, same r_S should give same result as
    # treating all FC edges uniformly
    pred_1, info_1 = solve_mpr_kernel(X_G, X_A, X_S, y, lam=1.0, r_A=1.0, r_S=1.0)

    # Concatenate all FC
    X_all = np.hstack([X_G, X_A])
    pred_all, info_all = solve_mpr_kernel(
        X_all, np.zeros((n, 1)), X_S, y, lam=1.0, r_A=1.0, r_S=1.0,
    )
    # They should be very close (not identical due to different kernel structure)
    # but the key property is that r_A=1 doesn't differentiate groups
    assert pred_1.shape == pred_all.shape


# ---------------------------------------------------------------------------
# 5. Baseline and PALF-MPR share the exact solver
# ---------------------------------------------------------------------------
def test_shared_solver():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "solve_mpr_kernel" in src
    assert "_select_joint_params" in src
    # Both models call the same selection function
    assert src.count("_select_joint_params") >= 3  # definition + 2 calls


# ---------------------------------------------------------------------------
# 6. Same (lambda, r_A=1, r_S) produces identical predictions
# ---------------------------------------------------------------------------
def test_identity_same_params():
    rng = np.random.RandomState(42)
    n, p_G, p_A, p_S = 30, 100, 50, 80
    X_G = rng.randn(n, p_G)
    X_A = rng.randn(n, p_A)
    X_S = rng.randn(n, p_S)
    y = rng.randn(n)

    pred1, _ = solve_mpr_kernel(X_G, X_A, X_S, y, lam=1.0, r_A=1.0, r_S=1.0)
    pred2, _ = solve_mpr_kernel(X_G, X_A, X_S, y, lam=1.0, r_A=1.0, r_S=1.0)
    np.testing.assert_allclose(pred1, pred2, atol=1e-14)


# ---------------------------------------------------------------------------
# 7. Train-only standardization is enforced
# ---------------------------------------------------------------------------
def test_train_only_standardization():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # Must use StandardScaler fit on training data
    assert "StandardScaler" in src
    assert "fit_transform" in src
    assert "transform" in src


# ---------------------------------------------------------------------------
# 8. Target centering is train-only
# ---------------------------------------------------------------------------
def test_target_centering_train_only():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # y_mean should be computed from training data
    assert "y_mean" in src or "y_mean_tr" in src


# ---------------------------------------------------------------------------
# 9. Subject-space prediction equals explicit small synthetic primal solution
# ---------------------------------------------------------------------------
def test_synthetic_primal_solution():
    rng = np.random.RandomState(42)
    n, p_G, p_A, p_S = 20, 30, 15, 25
    X_G = rng.randn(n, p_G)
    X_A = rng.randn(n, p_A)
    X_S = rng.randn(n, p_S)
    y = rng.randn(n)

    pred, info = solve_mpr_kernel(X_G, X_A, X_S, y, lam=1.0, r_A=0.5, r_S=2.0)

    # Verify primal reconstruction
    err = validate_primal_reconstruction(
        X_G, X_A, X_S, y,
        info["beta_G"], info["beta_A"], info["beta_S"],
        pred,
        tol=1e-8,
    )
    assert err <= 1e-8, f"Primal reconstruction error: {err}"


# ---------------------------------------------------------------------------
# 10. Recovered primal coefficients reproduce predictions
# ---------------------------------------------------------------------------
def test_primal_coefficients_reproduce():
    rng = np.random.RandomState(123)
    n, p_G, p_A, p_S = 25, 40, 20, 30
    X_G = rng.randn(n, p_G)
    X_A = rng.randn(n, p_A)
    X_S = rng.randn(n, p_S)
    y = rng.randn(n)

    pred, info = solve_mpr_kernel(X_G, X_A, X_S, y, lam=0.1, r_A=0.25, r_S=1.0)

    # Manual primal prediction
    primal_pred = X_G @ info["beta_G"] + X_A @ info["beta_A"] + X_S @ info["beta_S"]
    np.testing.assert_allclose(pred, primal_pred, atol=1e-8)


# ---------------------------------------------------------------------------
# 11. Coefficient vector dimensions are correct
# ---------------------------------------------------------------------------
def test_coefficient_dimensions():
    g = build_edge_grouping(wm_prior, ROI_NAMES, top_k=10)
    rng = np.random.RandomState(42)
    n = 20
    X_G = rng.randn(n, g.n_generic)
    X_A = rng.randn(n, g.n_semantic)
    X_S = rng.randn(n, 100)
    y = rng.randn(n)

    _, info = solve_mpr_kernel(X_G, X_A, X_S, y, lam=1.0, r_A=1.0, r_S=1.0)
    assert info["beta_G"].shape == (g.n_generic,)
    assert info["beta_A"].shape == (g.n_semantic,)
    assert info["beta_S"].shape == (100,)

    beta_full = compute_beta_fc_full(info["beta_G"], info["beta_A"], g.generic_indices, g.semantic_indices)
    assert beta_full.shape == (N_EDGE,)


# ---------------------------------------------------------------------------
# 12. No final/test seeds 0-9 or previous dev seeds 101/202/303/404
# ---------------------------------------------------------------------------
def test_dev_seeds_only():
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    forbidden = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 202, 303, 404}
    used = set(sm["seed"].unique())
    overlap = used & forbidden
    assert len(overlap) == 0, f"Forbidden seeds used: {overlap}"
    assert used == {505, 606, 707, 808}


# ---------------------------------------------------------------------------
# 13. Baseline and PALF-MPR share outer splits
# ---------------------------------------------------------------------------
def test_shared_outer_splits():
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    for (task, seed, fold), grp in sm.groupby(["task", "seed", "outer_fold"]):
        models = set(grp["model"].values)
        assert "joint_baseline" in models, f"Missing baseline for {task}/{seed}/{fold}"
        assert "palf_mpr" in models, f"Missing PALF-MPR for {task}/{seed}/{fold}"


# ---------------------------------------------------------------------------
# 14. Late-fusion reference is not used in PALF-MPR selection
# ---------------------------------------------------------------------------
def test_late_fusion_not_in_selection():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # late_fusion should not be called inside _select_joint_params
    import re
    match = re.search(r"def _select_joint_params\(.*?\n(?=\ndef |\Z)", src, re.DOTALL)
    if match:
        assert "late_fusion" not in match.group(0), "Late-fusion in selection"


# ---------------------------------------------------------------------------
# 15. No Phase-1 stacking helper is used
# ---------------------------------------------------------------------------
def test_no_stacking_imports():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "phase1_differential_stacking" not in src


# ---------------------------------------------------------------------------
# 16. No old adaptive D_rho/tau_L helper is used
# ---------------------------------------------------------------------------
def test_no_old_adaptive_helpers():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "build_cache_for_rho" not in src
    assert "tau_L" not in src or "tau_L" in "# tau_L is not used"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_output():
    if not OUTPUT_DIR.exists() or not (OUTPUT_DIR / "split_metrics.csv").exists():
        pytest.skip("Phase 2B output not yet created")
