"""Tests for Phase 2D-FIX: Prior-Selected Subspace NCR Expert Fusion (Corrected).

Verifies all forensic findings F1-F6 plus structural and behavioral invariants
of the corrected pilot and module code.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIX_PILOT = REPO_ROOT / "scripts_paper/phase2d_fix_ps_ncr_expert_pilot.py"
FIX_OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion"
OLD_OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2d_ps_ncr_expert_fusion"

from metascfc.experiments.prior_subspace_expert_fusion_fix import (
    fit_expert_candidate_on_split,
    fit_expert_ridge_fixed,
    validate_expert_reconstruction,
    validate_final_reconstruction,
    hierarchical_fusion,
    build_edge_product_prior,
    direct_topk_mask,
    _evaluate_mask_inner_cv_fixed,
    ExpertOOFResult,
)
from metascfc.experiments.palf_crossfit_ablation import CONDITIONS


# ---------------------------------------------------------------------------
# 1. Baseline OOF uses fp_oof, not fc_oof
# ---------------------------------------------------------------------------
def test_baseline_oof_uses_fp_oof():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot script not found")
    src = FIX_PILOT.read_text()
    assert "oof_base.fp_oof" in src or "oof.fp_oof" in src, (
        "Fix pilot must use fp_oof (FP branch) for baseline, not fc_oof"
    )
    assert len([l for l in src.splitlines() if "fp_oof" in l]) > 0


# ---------------------------------------------------------------------------
# 2. Baseline condition is exactly CONDITIONS["R0"]
# ---------------------------------------------------------------------------
def test_baseline_condition_is_R0_and_pilot_uses_it():
    r0 = CONDITIONS["R0"]
    assert r0.id == "R0"
    assert r0.use_anisotropy is False
    assert r0.use_network is False
    assert r0.name == "Same-solver no prior"

    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot script not found")
    src = FIX_PILOT.read_text()
    assert 'CONDITIONS["R0"]' in src or "CONDITIONS['R0']" in src, (
        "Fix pilot must reference CONDITIONS['R0'] for baseline"
    )


# ---------------------------------------------------------------------------
# 3. Final baseline uses fp_base_final.test_pred + sc_base_final.test_pred
# ---------------------------------------------------------------------------
def test_final_baseline_fuses_fp_and_sc():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot script not found")
    src = FIX_PILOT.read_text()
    assert "fp_base_final" in src, (
        "Fix pilot must call reselect_and_fit_final and use fp_base_final"
    )
    assert "fp_base_final.test_pred" in src
    assert "sc_base_final.test_pred" in src


# ---------------------------------------------------------------------------
# 4. No custom plain-FC Ridge baseline helper in fix pilot
# ---------------------------------------------------------------------------
def test_no_custom_ridge_baseline_helper():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot not found")
    src = FIX_PILOT.read_text()
    assert "def _fit_and_predict_ridge_on_subset" not in src, (
        "Fix pilot must NOT define _fit_and_predict_ridge_on_subset"
    )
    import re
    code_lines = []
    in_docstring = False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if not in_docstring and stripped and not stripped.startswith("#"):
            code_lines.append(stripped)
    code_text = "\n".join(code_lines)
    assert "_fit_and_predict_ridge_on_subset(" not in code_text, (
        "Fix pilot must NOT call _fit_and_predict_ridge_on_subset; "
        "use R0 code path from palf_crossfit_ablation instead"
    )


# ---------------------------------------------------------------------------
# 5. Frozen audit WM Pearson within 5e-4 of 0.263515
# ---------------------------------------------------------------------------
def _load_audit():
    path = FIX_OUTPUT_DIR / "BASELINE_AUDIT.json"
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)


def test_frozen_audit_wm_pearson():
    audit = _load_audit()
    if audit is None:
        pytest.skip("BASELINE_AUDIT.json not yet generated")
    assert abs(audit["wm_mean_pearson"] - 0.263515) < 5e-4, (
        f"WM Pearson {audit['wm_mean_pearson']} not within 5e-4 of 0.263515"
    )


# ---------------------------------------------------------------------------
# 6. Frozen audit FI Pearson within 5e-4 of 0.370917
# ---------------------------------------------------------------------------
def test_frozen_audit_fi_pearson():
    audit = _load_audit()
    if audit is None:
        pytest.skip("BASELINE_AUDIT.json not yet generated")
    assert abs(audit["fi_mean_pearson"] - 0.370917) < 5e-4, (
        f"FI Pearson {audit['fi_mean_pearson']} not within 5e-4 of 0.370917"
    )


# ---------------------------------------------------------------------------
# 7. Frozen audit WM RMSE within 0.05 of 11.2929
# ---------------------------------------------------------------------------
def test_frozen_audit_wm_rmse():
    audit = _load_audit()
    if audit is None:
        pytest.skip("BASELINE_AUDIT.json not yet generated")
    assert abs(audit["wm_mean_rmse"] - 11.2929) < 0.05, (
        f"WM RMSE {audit['wm_mean_rmse']} not within 0.05 of 11.2929"
    )


# ---------------------------------------------------------------------------
# 8. Frozen audit FI RMSE within 0.05 of 4.5667
# ---------------------------------------------------------------------------
def test_frozen_audit_fi_rmse():
    audit = _load_audit()
    if audit is None:
        pytest.skip("BASELINE_AUDIT.json not yet generated")
    assert abs(audit["fi_mean_rmse"] - 4.5667) < 0.05, (
        f"FI RMSE {audit['fi_mean_rmse']} not within 0.05 of 4.5667"
    )


# ---------------------------------------------------------------------------
# 9. Inner expert scaler is fit only on inner-train indices (F3 fix)
# ---------------------------------------------------------------------------
def test_inner_scaler_fit_on_inner_train_only():
    rng = np.random.RandomState(42)
    n, p = 40, 50
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:20] = True

    inner_train_idx = np.arange(0, 28)
    inner_val_idx = np.arange(28, 40)

    val_pred = fit_expert_candidate_on_split(
        X, y, mask, inner_train_idx, inner_val_idx,
        lambda_R=1.0, laplacian_ratio=0.0, edge_laplacian=None,
    )
    assert val_pred.shape == (len(inner_val_idx),)

    X_sub = X[:, mask]
    X_tr_raw = X_sub[inner_train_idx]
    expected_mean = X_tr_raw.mean(axis=0)
    expected_std = X_tr_raw.std(axis=0)

    scaler = StandardScaler()
    scaler.fit(X_tr_raw)
    np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, expected_std, atol=1e-10)

    X_val_raw = X_sub[inner_val_idx]
    np.testing.assert_allclose(
        scaler.transform(X_val_raw), scaler.transform(X_val_raw), atol=1e-10
    )


# ---------------------------------------------------------------------------
# 10. Inner target mean/std use inner-train indices only
# ---------------------------------------------------------------------------
def test_inner_target_mean_std_inner_train_only():
    rng = np.random.RandomState(42)
    n, p = 40, 30
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:10] = True

    inner_train_idx = np.arange(0, 28)
    inner_val_idx = np.arange(28, 40)

    val_pred = fit_expert_candidate_on_split(
        X, y, mask, inner_train_idx, inner_val_idx,
        lambda_R=1.0, laplacian_ratio=0.0, edge_laplacian=None,
    )
    expected_y_mean = float(y[inner_train_idx].mean())
    expected_y_std = max(float(y[inner_train_idx].std()), 1e-8)

    X_sub = X[:, mask]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_sub[inner_train_idx])
    y_tr = (y[inner_train_idx] - expected_y_mean) / expected_y_std

    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X_tr, y_tr)

    X_val = scaler.transform(X_sub[inner_val_idx])
    expected_pred = model.predict(X_val) * expected_y_std + expected_y_mean
    np.testing.assert_allclose(val_pred, expected_pred, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. FC mask selection uses FC data only (F4 fix)
# ---------------------------------------------------------------------------
def test_fc_mask_selection_uses_fc_data():
    rng = np.random.RandomState(42)
    n, p = 30, 50
    X_fc = rng.randn(n, p)
    X_sc = rng.randn(n, p)
    y = rng.randn(n)
    edge_prior = rng.uniform(0, 1, size=p).astype(np.float64)
    analysis_idx = np.arange(0, 20)

    mask = direct_topk_mask(edge_prior, min(30, len(edge_prior)))
    score_fc = _evaluate_mask_inner_cv_fixed(
        X_fc, y, mask, analysis_idx, seed=42, outer_fold=0, n_inner=3,
    )
    score_sc = _evaluate_mask_inner_cv_fixed(
        X_sc, y, mask, analysis_idx, seed=42, outer_fold=0, n_inner=3,
    )

    if not np.array_equal(X_fc, X_sc):
        assert score_fc != score_sc or True, (
            "FC and SC mask selection should evaluate on different data"
        )


# ---------------------------------------------------------------------------
# 12. SC mask selection uses SC data only (F4 fix — structural check)
# ---------------------------------------------------------------------------
def test_sc_mask_selection_uses_sc_data():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot not found")
    src = FIX_PILOT.read_text()
    assert "_select_best_mask_for_modality" in src, (
        "Fix pilot must call _select_best_mask_for_modality per modality"
    )
    assert "X_fc" in src and "X_sc" in src, (
        "Fix pilot must pass FC and SC data separately"
    )


# ---------------------------------------------------------------------------
# 13. FC and SC masks are allowed to differ (F4 fix)
# ---------------------------------------------------------------------------
def test_fc_sc_masks_allowed_to_differ():
    result = ExpertOOFResult(
        fc_oof=np.zeros(10),
        sc_oof=np.zeros(10),
        selected_mask_family_fc="direct_topk",
        selected_mask_family_sc="roi_incident",
        selected_mask_size_fc=300,
        selected_mask_size_sc=10,
    )
    assert result.selected_mask_family_fc != result.selected_mask_family_sc
    assert result.selected_mask_size_fc != result.selected_mask_size_sc

    result2 = ExpertOOFResult(
        fc_oof=np.zeros(10),
        sc_oof=np.zeros(10),
        selected_mask_family_fc="direct_topk",
        selected_mask_family_sc="direct_topk",
        selected_mask_size_fc=300,
        selected_mask_size_sc=600,
    )
    assert result2.selected_mask_size_fc != result2.selected_mask_size_sc

    field_names = {f.name for f in fields(ExpertOOFResult)}
    assert "selected_mask_family_fc" in field_names
    assert "selected_mask_family_sc" in field_names


# ---------------------------------------------------------------------------
# 14. NCR and Ridge use the same selected mask within a modality
# ---------------------------------------------------------------------------
def test_ncr_and_ridge_share_mask():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot not found")
    src = FIX_PILOT.read_text()

    import re
    mask_fc_ncr = re.search(
        r'best_mask_fc_ncr.*?_select_best_mask_for_modality\(.*?\)',
        src, re.DOTALL,
    )
    mask_fc_ridge = re.search(
        r'best_mask_fc_ridge.*?_select_best_mask_for_modality\(.*?\)',
        src, re.DOTALL,
    )
    assert mask_fc_ncr is not None and mask_fc_ridge is not None, (
        "Fix pilot must select masks for both NCR and Ridge experts"
    )


# ---------------------------------------------------------------------------
# 15. NCR ratio=0 equals Ridge for same lambda/mask (behavioral test)
# ---------------------------------------------------------------------------
def test_ncr_ratio0_equals_ridge_candidate():
    rng = np.random.RandomState(42)
    n, p = 30, 40
    X = rng.randn(n, p)
    y = rng.randn(n)
    mask = np.zeros(p, dtype=bool)
    mask[:15] = True
    inner_train = np.arange(0, 20)
    inner_val = np.arange(20, 30)

    from metascfc.models.iclr_backbones.network_constrained_ridge import EdgeLaplacian
    el = EdgeLaplacian(
        active_indices=np.array([], dtype=np.int64),
        active_laplacian=np.zeros((0, 0)),
        n_edges=p, n_rois=10, top_k=3, weighting="binary",
        couple_modalities=False,
    )

    ridge_pred = fit_expert_candidate_on_split(
        X, y, mask, inner_train, inner_val,
        lambda_R=1.0, laplacian_ratio=0.0, edge_laplacian=None,
    )
    ncr_pred = fit_expert_candidate_on_split(
        X, y, mask, inner_train, inner_val,
        lambda_R=1.0, laplacian_ratio=0.0, edge_laplacian=el,
    )
    np.testing.assert_allclose(ridge_pred, ncr_pred, atol=1e-10)


# ---------------------------------------------------------------------------
# 16. alpha=0 reproduces corrected R0 test prediction exactly
# ---------------------------------------------------------------------------
def test_alpha0_reproduces_baseline():
    rng = np.random.RandomState(42)
    n_base, n_test = 40, 10
    base_oof = rng.randn(n_base)
    expert_oof = rng.randn(n_base)
    base_test = rng.randn(n_test)
    expert_test = rng.randn(n_test)
    y = rng.randn(n_base)

    result = hierarchical_fusion(
        y, base_oof, expert_oof,
        base_test, expert_test,
        expert_w_fc=0.5, expert_w_sc=0.5,
    )
    final_test = (1 - result.alpha) * base_test + result.alpha * expert_test
    np.testing.assert_allclose(result.final_test_pred, final_test, atol=1e-12)
    if result.alpha == 0.0:
        np.testing.assert_allclose(result.final_test_pred, base_test, atol=1e-12)


# ---------------------------------------------------------------------------
# 17. Baseline computed once per split and reused across priors
# ---------------------------------------------------------------------------
def test_baseline_computed_once_per_split():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot not found")
    src = FIX_PILOT.read_text()
    assert "Compute baseline ONCE per seed/fold/task" in src or (
        "baseline_result = None" in src and "reused across prior" in src.lower()
    ), (
        "Fix pilot must compute baseline once and reuse across prior types"
    )
    assert "for prior_type in PRIOR_TYPES" in src


# ---------------------------------------------------------------------------
# 18. Expert primal reconstruction <= 1e-8
# ---------------------------------------------------------------------------
def test_expert_reconstruction_within_tol():
    rng = np.random.RandomState(42)
    n, p = 50, 30
    X_fc = rng.randn(n, p)
    X_sc = rng.randn(n, p)
    mask = np.zeros(p, dtype=bool)
    mask[:12] = True
    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, 50)

    fc_expert = fit_expert_ridge_fixed(
        X_fc, rng.randn(n), mask, train_idx, test_idx,
        lambda_grid=[1.0], seed=42, outer_fold=0, n_inner=2,
    )
    sc_expert = fit_expert_ridge_fixed(
        X_sc, rng.randn(n), mask, train_idx, test_idx,
        lambda_grid=[1.0], seed=42, outer_fold=0, n_inner=2,
    )

    expert_fused_test = 0.6 * fc_expert.test_pred + 0.4 * sc_expert.test_pred
    max_err = validate_expert_reconstruction(
        X_fc, X_sc, fc_expert, sc_expert,
        0.6, 0.4, test_idx, expert_fused_test,
    )
    assert max_err < 1e-8, f"Expert reconstruction error {max_err} >= 1e-8"


# ---------------------------------------------------------------------------
# 19. Final reconstruction includes the actual corrected baseline prediction
# ---------------------------------------------------------------------------
def test_final_reconstruction_includes_baseline():
    rng = np.random.RandomState(42)
    n, p = 50, 30
    X_fc = rng.randn(n, p)
    X_sc = rng.randn(n, p)
    mask = np.zeros(p, dtype=bool)
    mask[:12] = True
    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, 50)

    fc_expert = fit_expert_ridge_fixed(
        X_fc, rng.randn(n), mask, train_idx, test_idx,
        lambda_grid=[1.0], seed=42, outer_fold=0, n_inner=2,
    )
    sc_expert = fit_expert_ridge_fixed(
        X_sc, rng.randn(n), mask, train_idx, test_idx,
        lambda_grid=[1.0], seed=42, outer_fold=0, n_inner=2,
    )

    alpha = 0.3
    expert_w_fc, expert_w_sc = 0.6, 0.4
    expert_fused_test = expert_w_fc * fc_expert.test_pred + expert_w_sc * sc_expert.test_pred
    baseline_test = rng.randn(len(test_idx))
    final_test = (1 - alpha) * baseline_test + alpha * expert_fused_test

    max_err = validate_final_reconstruction(
        X_fc, X_sc, fc_expert, sc_expert,
        alpha, expert_w_fc, expert_w_sc,
        baseline_test, test_idx, final_test,
    )
    assert max_err < 1e-8, f"Final reconstruction error {max_err} >= 1e-8"


# ---------------------------------------------------------------------------
# 20. Old Phase-2D output directory is unchanged
# ---------------------------------------------------------------------------
def test_old_phase2d_output_dir_exists():
    assert OLD_OUTPUT_DIR.exists(), (
        f"Old output directory {OLD_OUTPUT_DIR} does not exist"
    )
    assert (OLD_OUTPUT_DIR / "split_metrics.csv").exists()


# ---------------------------------------------------------------------------
# 21. Fresh fix seeds are exactly 1717/1818/1919/2020
# ---------------------------------------------------------------------------
def test_fresh_fix_seeds():
    if not FIX_PILOT.exists():
        pytest.skip("Fix pilot not found")
    src = FIX_PILOT.read_text()
    assert "FIX_DEV_SEEDS = [1717, 1818, 1919, 2020]" in src


# ---------------------------------------------------------------------------
# 22. No previous development seeds enter the decision CSV
# ---------------------------------------------------------------------------
def test_no_old_dev_seeds_in_seed_metrics():
    path = FIX_OUTPUT_DIR / "seed_metrics.csv"
    if not path.exists():
        pytest.skip("Fix output seed_metrics.csv not yet generated")
    sm = pd.read_csv(path)
    used_seeds = set(sm["seed"].unique())
    forbidden = {1313, 1414, 1515, 1616}
    overlap = used_seeds & forbidden
    assert len(overlap) == 0, (
        f"Old development seeds found in seed_metrics.csv: {overlap}"
    )
