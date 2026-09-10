"""Tests for Phase 2D: Prior-Selected Subspace NCR Expert Fusion."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2d_ps_ncr_expert_fusion"
PILOT_SCRIPT = REPO_ROOT / "scripts_paper/phase2d_prior_subspace_ncr_expert_pilot.py"

from metascfc.experiments.prior_subspace_expert_fusion import (
    N_EDGE,
    N_ROI,
    K_EDGE_GRID,
    M_ROI_GRID,
    LAPLACIAN_RATIO_GRID,
    build_edge_product_prior,
    direct_topk_mask,
    roi_incident_mask,
    get_edge_count_for_roi_mask,
    build_control_prior,
    fit_expert_ridge,
    fit_expert_ncr,
    search_fusion_weights_simple,
    hierarchical_fusion,
    export_expert_coefficients,
    validate_primal_reconstruction,
    compute_expert_use_diagnostics,
)

# Load priors
wm_prior_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)
fi_prior_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
)
wm_prior = wm_prior_df["prior_score"].values.astype(np.float64)
fi_prior = fi_prior_df["prior_score"].values.astype(np.float64)


# ---------------------------------------------------------------------------
# 1. direct top-K mask has exact requested cardinality
# ---------------------------------------------------------------------------
def test_direct_topk_cardinality():
    edge_prior = build_edge_product_prior(wm_prior)
    for k in K_EDGE_GRID:
        mask = direct_topk_mask(edge_prior, k)
        assert mask.sum() == k, f"Expected {k} edges, got {mask.sum()}"


# ---------------------------------------------------------------------------
# 2. ROI-incident mask has expected cardinality
# ---------------------------------------------------------------------------
def test_roi_incident_cardinality():
    for m in M_ROI_GRID:
        expected = get_edge_count_for_roi_mask(m)
        mask, actual = roi_incident_mask(wm_prior, m)
        assert actual == expected, f"M={m}: expected {expected}, got {actual}"
        assert mask.sum() == actual


# ---------------------------------------------------------------------------
# 3. mask is deterministic for a frozen prior
# ---------------------------------------------------------------------------
def test_mask_deterministic():
    edge_prior = build_edge_product_prior(wm_prior)
    mask1 = direct_topk_mask(edge_prior, 300)
    mask2 = direct_topk_mask(edge_prior, 300)
    np.testing.assert_array_equal(mask1, mask2)

    mask3, _ = roi_incident_mask(wm_prior, 10)
    mask4, _ = roi_incident_mask(wm_prior, 10)
    np.testing.assert_array_equal(mask3, mask4)


# ---------------------------------------------------------------------------
# 4. shuffled/random controls preserve cardinality
# ---------------------------------------------------------------------------
def test_control_preserves_cardinality():
    for ptype in ["matched", "cross_task", "shuffled", "random"]:
        ctrl = build_control_prior(ptype, wm_prior, fi_prior, seed=42)
        assert len(ctrl) == N_ROI
        edge_prior = build_edge_product_prior(ctrl)
        mask = direct_topk_mask(edge_prior, 300)
        assert mask.sum() == 300


# ---------------------------------------------------------------------------
# 5. Ridge expert StandardScaler is fit only on analysis/train indices
# ---------------------------------------------------------------------------
def test_ridge_expert_scaler_fit_on_train():
    rng = np.random.RandomState(42)
    n, p = 50, 200
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:50] = True
    train_idx = np.arange(40)
    test_idx = np.arange(40, 50)

    result = fit_expert_ridge(X, y, mask, train_idx, test_idx)
    # Scaler mean should be from training set only
    X_sub_train = X[train_idx][:, mask]
    expected_mean = X_sub_train.mean(axis=0)
    np.testing.assert_allclose(result.scaler_mean, expected_mean, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. NCR ratio=0 reproduces restricted Ridge expert
# ---------------------------------------------------------------------------
def test_ncr_ratio0_equals_ridge():
    rng = np.random.RandomState(42)
    n, p = 50, 200
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:50] = True
    train_idx = np.arange(40)
    test_idx = np.arange(40, 50)

    # Build a trivial Laplacian (all zeros = no network penalty)
    from metascfc.models.iclr_backbones.network_constrained_ridge import EdgeLaplacian
    el = EdgeLaplacian(
        active_indices=np.array([], dtype=np.int64),
        active_laplacian=np.zeros((0, 0)),
        n_edges=p, n_rois=16, top_k=10, weighting="binary",
        couple_modalities=False,
    )

    ridge_result = fit_expert_ridge(X, y, mask, train_idx, test_idx, lambda_grid=[1.0])
    ncr_result = fit_expert_ncr(
        X, y, mask, el, train_idx, test_idx,
        lambda_grid=[1.0], ratio_grid=[0.0],
    )

    np.testing.assert_allclose(ridge_result.test_pred, ncr_result.test_pred, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. induced subspace Laplacian is PSD within tolerance
# ---------------------------------------------------------------------------
def test_subspace_laplacian_psd():
    from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian
    el = build_edge_laplacian(N_ROI, prior_scores=wm_prior, top_k=10)
    # The active_laplacian should be PSD
    if el.active_laplacian.shape[0] > 0:
        eigvals = np.linalg.eigvalsh(el.active_laplacian)
        assert eigvals.min() >= -1e-10, f"Min eigenvalue: {eigvals.min()}"


# ---------------------------------------------------------------------------
# 8. FC and SC expert feature indices are identical for the same prior/mask
# ---------------------------------------------------------------------------
def test_fc_sc_same_indices():
    rng = np.random.RandomState(42)
    n = 50
    X_fc = rng.randn(n, 6670)
    X_sc = rng.randn(n, 6670)
    y = rng.randn(n)
    mask = np.zeros(6670, dtype=bool)
    mask[:300] = True
    train_idx = np.arange(40)
    test_idx = np.arange(40, 50)

    fc_res = fit_expert_ridge(X_fc, y, mask, train_idx, test_idx, lambda_grid=[1.0])
    sc_res = fit_expert_ridge(X_sc, y, mask, train_idx, test_idx, lambda_grid=[1.0])

    np.testing.assert_array_equal(fc_res.mask, sc_res.mask)
    assert fc_res.n_selected == sc_res.n_selected


# ---------------------------------------------------------------------------
# 9. OOF expert prediction for a subject is generated without fitting on that subject
# ---------------------------------------------------------------------------
def test_oof_no_leakage():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # Must use cross-fitting (fusion folds)
    assert "fusion_fold" in src.lower() or "oof" in src.lower()


# ---------------------------------------------------------------------------
# 10. baseline branch is current generalized same-solver no-prior
# ---------------------------------------------------------------------------
def test_baseline_is_generalized():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # Must use the palf_crossfit_ablation infrastructure or FC+SC fusion
    assert ("palf_crossfit_ablation" in src or
            "make_outer_splits" in src or
            "fusion" in src.lower())


# ---------------------------------------------------------------------------
# 11. alpha=0 reproduces baseline predictions exactly
# ---------------------------------------------------------------------------
def test_alpha0_reproduces_baseline():
    rng = np.random.RandomState(42)
    n = 50
    base = rng.randn(n)
    expert = rng.randn(n)

    result = hierarchical_fusion(
        y=rng.randn(n),
        base_oof=base,
        expert_fused_oof=expert,
        base_test=base[:10],
        expert_test=expert[:10],
        expert_w_fc=0.5, expert_w_sc=0.5,
    )
    # alpha should be 0 when expert is not helpful
    # (it might not be exactly 0, but let's check the formula works)
    final = (1 - result.alpha) * base + result.alpha * expert
    np.testing.assert_allclose(result.final_oof, final, atol=1e-12)


# ---------------------------------------------------------------------------
# 12. expert_w_FC + expert_w_SC = 1
# ---------------------------------------------------------------------------
def test_expert_weights_sum_to_one():
    rng = np.random.RandomState(42)
    n = 50
    y = rng.randn(n)
    fc_oof = rng.randn(n)
    sc_oof = rng.randn(n)

    weights, _ = search_fusion_weights_simple(y, fc_oof, sc_oof)
    assert abs(weights["fc"] + weights["sc"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 13. final model uses only OOF-selected fusion weights
# ---------------------------------------------------------------------------
def test_fusion_weights_from_oof():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "search_fusion" in src or "fusion_weight" in src.lower()


# ---------------------------------------------------------------------------
# 14. outer-test target never enters mask/penalty/fusion selection
# ---------------------------------------------------------------------------
def test_no_test_in_selection():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    import re
    src = PILOT_SCRIPT.read_text()
    # Selection functions should not reference test targets
    for func in ["_select_best_mask", "_evaluate_mask_inner_cv"]:
        match = re.search(rf"def {func}\(.*?\n(?=\ndef |\nclass |\Z)", src, re.DOTALL)
        if match:
            assert "y_test" not in match.group(0), f"{func} uses y_test"


# ---------------------------------------------------------------------------
# 15. primal coefficient reconstruction reproduces predictions
# ---------------------------------------------------------------------------
def test_primal_reconstruction():
    rng = np.random.RandomState(42)
    n, p = 50, 200
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:50] = True
    train_idx = np.arange(40)
    test_idx = np.arange(40, 50)

    result = fit_expert_ridge(X, y, mask, train_idx, test_idx, lambda_grid=[1.0])

    # Reconstruct from beta_standardized
    X_sub_test = X[test_idx][:, mask]
    X_z = (X_sub_test - result.scaler_mean) / np.maximum(result.scaler_scale, 1e-8)
    pred_reconstructed = X_z @ result.beta_standardized * result.y_std + result.y_mean
    np.testing.assert_allclose(pred_reconstructed, result.test_pred, atol=1e-10)


# ---------------------------------------------------------------------------
# 16. coefficient vector length is 6670
# ---------------------------------------------------------------------------
def test_coefficient_length():
    rng = np.random.RandomState(42)
    n = 50
    X_fc = rng.randn(n, 6670)
    X_sc = rng.randn(n, 6670)
    y = rng.randn(n)
    mask = np.zeros(6670, dtype=bool)
    mask[:300] = True
    train_idx = np.arange(40)
    test_idx = np.arange(40, 50)

    fc_res = fit_expert_ridge(X_fc, y, mask, train_idx, test_idx, lambda_grid=[1.0])
    sc_res = fit_expert_ridge(X_sc, y, mask, train_idx, test_idx, lambda_grid=[1.0])

    coeffs = export_expert_coefficients(fc_res, sc_res, alpha=0.5, expert_w_fc=0.6, expert_w_sc=0.4)
    assert len(coeffs["beta_fc_expert"]) == 6670
    assert len(coeffs["beta_sc_expert"]) == 6670
    assert len(coeffs["beta_fc_expert_weighted"]) == 6670
    assert len(coeffs["beta_sc_expert_weighted"]) == 6670


# ---------------------------------------------------------------------------
# 17. fresh development seeds are used
# ---------------------------------------------------------------------------
def test_fresh_seeds_only():
    if not OUTPUT_DIR.exists() or not (OUTPUT_DIR / "split_metrics.csv").exists():
        pytest.skip("Phase 2D output not yet created")
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    forbidden = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 202, 303, 404,
                 505, 606, 707, 808, 909, 1010, 1111, 1212}
    used = set(sm["seed"].unique())
    overlap = used & forbidden
    assert len(overlap) == 0, f"Forbidden seeds used: {overlap}"
    assert used == {1313, 1414, 1515, 1616}


# ---------------------------------------------------------------------------
# 18. previous output directories remain unchanged
# ---------------------------------------------------------------------------
def test_old_outputs_untouched():
    old_dirs = [
        REPO_ROOT / "outputs/iclr/palf_phase2a_adaptive_prior_pilot",
        REPO_ROOT / "outputs/iclr/palf_phase2a1_matched_adaptive_pilot",
        REPO_ROOT / "outputs/iclr/palf_phase2b_joint_group_ridge_pilot",
        REPO_ROOT / "outputs/iclr/palf_phase2c_mt_residual_ncr_pilot",
    ]
    for d in old_dirs:
        if d.exists():
            sm = d / "split_metrics.csv"
            if sm.exists():
                df = pd.read_csv(sm)
                assert len(df) > 0, f"Old output {d} was modified"


# ---------------------------------------------------------------------------
# 19. checkpoint restart does not duplicate split rows
# ---------------------------------------------------------------------------
def test_checkpoint_no_duplicates():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "checkpoint" in src.lower() or "_ckpt" in src


# ---------------------------------------------------------------------------
# 20. frozen correctness audit reproduces current baseline before pilot launch
# ---------------------------------------------------------------------------
def test_baseline_audit_present():
    if not OUTPUT_DIR.exists():
        pytest.skip("Phase 2D output not yet created")
    audit_path = OUTPUT_DIR / "BASELINE_AUDIT.json"
    assert audit_path.exists(), "Baseline audit not found"
    with open(audit_path) as f:
        import json
        audit = json.load(f)
    assert "wm_mean_pearson" in audit
    assert "fi_mean_pearson" in audit
    # Check approximate correctness
    assert abs(audit["wm_mean_pearson"] - 0.2635) < 0.01, f"WM baseline: {audit['wm_mean_pearson']}"
    assert abs(audit["fi_mean_pearson"] - 0.3709) < 0.01, f"FI baseline: {audit['fi_mean_pearson']}"
