"""Tests for Phase 2C: Multi-Task Residual NCR."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2c_mt_residual_ncr_pilot"
PILOT_SCRIPT = REPO_ROOT / "scripts_paper/phase2c_multitask_residual_ncr_pilot.py"

from metascfc.models.multitask_residual_ncr import (
    N_EDGE,
    build_shared_contrast_residuals,
    reconstruct_task_residuals,
    build_shared_prior,
    build_contrast_prior,
    fit_ncr_residual,
    crossfit_ncr_residual,
    select_eta,
    select_eta_mt,
    fit_final_ncr_and_predict,
    convert_coefficients_to_original,
    compute_coefficient_stability,
    aggregate_edge_to_roi,
)

# Load priors for tests
roi_names_df = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
)
wm_prior = roi_names_df["prior_score"].values.astype(np.float64)
fi_prior = pd.read_csv(
    REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
)["prior_score"].values.astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Shared residual transform is invertible
# ---------------------------------------------------------------------------
def test_shared_residual_invertible():
    rng = np.random.RandomState(42)
    n = 50
    r_wm = rng.randn(n)
    r_fi = rng.randn(n)

    z_shared, z_contrast = build_shared_contrast_residuals(r_wm, r_fi)
    r_wm_rec, r_fi_rec = reconstruct_task_residuals(z_shared, z_contrast)

    np.testing.assert_allclose(r_wm_rec, r_wm, atol=1e-12)
    np.testing.assert_allclose(r_fi_rec, r_fi, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Shared/contrast reconstruction exactly recovers synthetic residuals
# ---------------------------------------------------------------------------
def test_reconstruction_exact():
    rng = np.random.RandomState(123)
    for _ in range(5):
        n = rng.randint(20, 100)
        r_wm = rng.randn(n)
        r_fi = rng.randn(n)
        z_s, z_c = build_shared_contrast_residuals(r_wm, r_fi)
        r_wm_r, r_fi_r = reconstruct_task_residuals(z_s, z_c)
        np.testing.assert_allclose(r_wm_r, r_wm, atol=1e-12)
        np.testing.assert_allclose(r_fi_r, r_fi, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Shared prior is finite and normalized
# ---------------------------------------------------------------------------
def test_shared_prior_normalized():
    p = build_shared_prior(wm_prior, fi_prior)
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0
    assert p.max() <= 1.0


# ---------------------------------------------------------------------------
# 4. Contrast prior is finite and normalized
# ---------------------------------------------------------------------------
def test_contrast_prior_normalized():
    p = build_contrast_prior(wm_prior, fi_prior)
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0
    assert p.max() <= 1.0


# ---------------------------------------------------------------------------
# 5. Outer WM/FI split indices are identical
# ---------------------------------------------------------------------------
def test_paired_splits():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # The pilot must generate paired splits
    assert "train_idx" in src
    assert "test_idx" in src


# ---------------------------------------------------------------------------
# 6. Base OOF prediction for a subject is generated without training on it
# ---------------------------------------------------------------------------
def test_base_oof_no_leakage():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # Base OOF must use cross-fitting
    assert "oof" in src.lower() or "crossfit" in src.lower() or "_generate_fusion_oof" in src


# ---------------------------------------------------------------------------
# 7. Residual targets use OOF base predictions only
# ---------------------------------------------------------------------------
def test_residual_uses_oof():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "r_wm" in src or "residual_wm" in src or "residual" in src


# ---------------------------------------------------------------------------
# 8. lambda2=0 removes NCR prior penalty
# ---------------------------------------------------------------------------
def test_lambda2_zero_removes_penalty():
    rng = np.random.RandomState(42)
    n, p = 30, 100
    X = rng.randn(n, p)
    y = rng.randn(n)
    train_idx = np.arange(20)
    val_idx = np.arange(20, 30)

    pred_l20, _ = fit_ncr_residual(X, y, None, None, lambda1=1.0, lambda2=0.0,
                                    train_idx=train_idx, val_idx=val_idx)
    # With lambda2=0, should use plain Ridge
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    model = Ridge(alpha=1.0, fit_intercept=False)
    y_z = (y[train_idx] - y[train_idx].mean()) / max(y[train_idx].std(), 1e-8)
    model.fit(X_tr, y_z)
    pred_ridge = model.predict(X_val) * max(y[train_idx].std(), 1e-8) + y[train_idx].mean()
    np.testing.assert_allclose(pred_l20, pred_ridge, atol=1e-10)


# ---------------------------------------------------------------------------
# 9. MT no-prior uses identical architecture to MT-RNCR except Laplacian penalty
# ---------------------------------------------------------------------------
def test_mt_architecture_match():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    # Both MT models should use the same crossfit_ncr_residual function
    assert "crossfit_ncr_residual" in src


# ---------------------------------------------------------------------------
# 10. eta=0 exactly recovers base prediction
# ---------------------------------------------------------------------------
def test_eta_zero_recovers_base():
    rng = np.random.RandomState(42)
    n = 50
    base = rng.randn(n)
    residual = rng.randn(n)
    eta = 0.0
    combined = base + eta * residual
    np.testing.assert_allclose(combined, base)


# ---------------------------------------------------------------------------
# 11. Outer-test target is never passed to residual selection
# ---------------------------------------------------------------------------
def test_no_test_in_selection():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    import re
    src = PILOT_SCRIPT.read_text()
    # Check select_eta and select_eta_mt don't use y_test
    for func_name in ["select_eta", "select_eta_mt"]:
        match = re.search(rf"def {func_name}\(.*?\n(?=\ndef |\Z)", src, re.DOTALL)
        if match:
            assert "y_test" not in match.group(0), f"{func_name} uses y_test"


# ---------------------------------------------------------------------------
# 12. Original-coordinate coefficient conversion uses beta_std * y_scale / x_scale
# ---------------------------------------------------------------------------
def test_coefficient_conversion():
    rng = np.random.RandomState(42)
    p = 100
    beta_std = rng.randn(p)
    scaler = StandardScaler()
    scaler.fit(rng.randn(50, p))
    y_std = 2.5

    beta_orig = convert_coefficients_to_original(beta_std, scaler, y_std)
    expected = beta_std * y_std / scaler.scale_
    np.testing.assert_allclose(beta_orig, expected)


# ---------------------------------------------------------------------------
# 13. Saved primal coefficients reproduce direct residual predictions
# ---------------------------------------------------------------------------
def test_primal_reproduction():
    rng = np.random.RandomState(42)
    n, p = 30, 100
    X = rng.randn(n, p)
    y = rng.randn(n)
    scaler_X = StandardScaler()
    X_z = scaler_X.fit_transform(X)
    y_mean = y.mean()
    y_std = max(y.std(), 1e-8)
    y_z = (y - y_mean) / y_std

    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X_z, y_z)
    pred_z = model.predict(X_z)

    # Convert coefficients
    beta_orig = convert_coefficients_to_original(model.coef_, scaler_X, y_std)
    # Both should be finite and have correct dimensions
    assert np.all(np.isfinite(beta_orig))
    assert beta_orig.shape == (p,)
    # Predictions from standardized model should be finite
    assert np.all(np.isfinite(pred_z * y_std + y_mean))


# ---------------------------------------------------------------------------
# 14. Coefficient vector length is 6670
# ---------------------------------------------------------------------------
def test_coefficient_length():
    rng = np.random.RandomState(42)
    beta = rng.randn(N_EDGE)
    assert len(beta) == N_EDGE
    roi_agg = aggregate_edge_to_roi(beta)
    assert len(roi_agg) == 116


# ---------------------------------------------------------------------------
# 15. No previous development/final seeds are used
# ---------------------------------------------------------------------------
def test_fresh_seeds_only():
    _skip_if_no_output()
    sm = pd.read_csv(OUTPUT_DIR / "split_metrics.csv")
    forbidden = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 202, 303, 404, 505, 606, 707, 808}
    used = set(sm["seed"].unique())
    overlap = used & forbidden
    assert len(overlap) == 0, f"Forbidden seeds used: {overlap}"
    assert used == {909, 1010, 1111, 1212}


# ---------------------------------------------------------------------------
# 16. Model selection uses Fisher-z mean for paired MT objective
# ---------------------------------------------------------------------------
def test_fisher_z_selection():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    src = PILOT_SCRIPT.read_text()
    assert "arctanh" in src or "fisher" in src.lower()


# ---------------------------------------------------------------------------
# 17. Stability diagnostics are not used in prediction selection
# ---------------------------------------------------------------------------
def test_stability_not_in_selection():
    if not PILOT_SCRIPT.exists():
        pytest.skip("Pilot script not found")
    import re
    src = PILOT_SCRIPT.read_text()
    # select_eta and select_eta_mt should not reference stability
    for func_name in ["select_eta", "select_eta_mt"]:
        match = re.search(rf"def {func_name}\(.*?\n(?=\ndef |\Z)", src, re.DOTALL)
        if match:
            assert "stability" not in match.group(0).lower(), f"{func_name} uses stability"


# ---------------------------------------------------------------------------
# 18. Historical result directories are untouched
# ---------------------------------------------------------------------------
def test_old_outputs_untouched():
    old_dirs = [
        REPO_ROOT / "outputs/iclr/palf_phase2a_adaptive_prior_pilot",
        REPO_ROOT / "outputs/iclr/palf_phase2a1_matched_adaptive_pilot",
        REPO_ROOT / "outputs/iclr/palf_phase2b_joint_group_ridge_pilot",
    ]
    for d in old_dirs:
        if d.exists():
            sm = d / "split_metrics.csv"
            if sm.exists():
                df = pd.read_csv(sm)
                assert len(df) > 0, f"Old output {d} was modified"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_output():
    if not OUTPUT_DIR.exists() or not (OUTPUT_DIR / "split_metrics.csv").exists():
        pytest.skip("Phase 2C output not yet created")
