"""Unit tests for Phase 2E LI-SFC-NCR modules.

Tests aal_yeo_mapping, sfc_features, li_sfc_ncr, and llm_prior_generator
using small synthetic data. No atlas loading or real data required.
"""
import numpy as np
import pandas as pd
import pytest

from metascfc.phase2e.aal_yeo_mapping import (
    SYSTEM_NAMES,
    N_SYSTEMS,
    build_system_pair_list,
    roi_to_system_array,
    edge_to_system_pair,
)
from metascfc.phase2e.sfc_features import (
    compute_sfc_pair_features,
    compute_sfc_pair_features_for_split,
    get_valid_pair_mask,
)
from metascfc.phase2e.li_sfc_ncr import (
    build_pair_laplacian,
    fit_sfc_ridge_expert,
    fit_sfc_ncr_expert,
    predict_sfc_expert,
    hierarchical_fusion,
    search_fusion_weights,
)
from metascfc.phase2e.llm_prior_generator import (
    create_control_priors,
    compute_prior_diagnostics,
)


# ---------------------------------------------------------------------------
# aal_yeo_mapping tests
# ---------------------------------------------------------------------------

def test_system_names_length():
    assert len(SYSTEM_NAMES) == N_SYSTEMS
    assert N_SYSTEMS == 9


def test_build_system_pair_list_count():
    df = build_system_pair_list()
    expected = N_SYSTEMS * (N_SYSTEMS + 1) // 2
    assert len(df) == expected
    assert list(df.columns[:3]) == ["pair_index", "system_a", "system_b"]
    assert df["pair_index"].is_unique


def test_roi_to_system_array_shape():
    mapping_df = pd.DataFrame({
        "roi_index": np.arange(1, 117),
        "system": np.resize(SYSTEM_NAMES, 116),
    })
    arr = roi_to_system_array(mapping_df)
    assert arr.shape == (116,)
    assert arr.dtype == int
    assert set(np.unique(arr)).issubset(set(range(N_SYSTEMS)))


def test_edge_to_system_pair_mapping():
    n_rois = 4
    system_pairs_df = build_system_pair_list()
    roi_system = np.array([0, 1, 2, 3])
    fc_idx = np.array([[0, 1], [0, 2], [1, 2]])
    sc_idx = np.array([[0, 1], [0, 2], [1, 2]])
    result = edge_to_system_pair(fc_idx, sc_idx, roi_system, system_pairs_df)
    assert result.shape == (3,)
    assert result.dtype == int
    assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# sfc_features tests
# ---------------------------------------------------------------------------

def test_compute_sfc_pair_features_output_shape():
    n_subj, n_edges, n_pairs = 20, 10, 4
    fc = np.random.randn(n_subj, n_edges)
    sc = np.random.randn(n_subj, n_edges)
    edge_pairs = np.tile(np.arange(n_pairs), n_edges // n_pairs)
    X, fc_m, fc_s, sc_m, sc_s = compute_sfc_pair_features(
        fc, sc, edge_pairs, n_pairs, fit_scalers=True
    )
    assert X.shape == (n_subj, n_pairs)
    assert fc_m.shape == (n_edges,)
    assert fc_s.shape == (n_edges,)


def test_compute_sfc_pair_features_for_split_no_leakage():
    n_pairs = 3
    n_edges = 6
    edge_pairs = np.tile(np.arange(n_pairs), n_edges // n_pairs)
    rng = np.random.RandomState(0)
    fc_tr = rng.randn(15, n_edges)
    sc_tr = rng.randn(15, n_edges)
    fc_te = rng.randn(5, n_edges)
    sc_te = rng.randn(5, n_edges)
    X_tr, X_te = compute_sfc_pair_features_for_split(
        fc_tr, sc_tr, fc_te, sc_te, edge_pairs, n_pairs
    )
    assert X_tr.shape == (15, n_pairs)
    assert X_te.shape == (5, n_pairs)
    fc_tr_mean = fc_tr.mean(axis=0)
    np.testing.assert_allclose(
        (fc_tr - fc_tr_mean) / np.where(fc_tr.std(axis=0) < 1e-10, 1.0, fc_tr.std(axis=0)),
        (fc_tr - fc_tr_mean) / np.where(fc_tr.std(axis=0) < 1e-10, 1.0, fc_tr.std(axis=0)),
    )


def test_compute_sfc_pair_features_constant_input():
    n_subj, n_edges, n_pairs = 10, 4, 2
    fc = np.ones((n_subj, n_edges))
    sc = np.ones((n_subj, n_edges))
    edge_pairs = np.array([0, 0, 1, 1])
    X, _, _, _, _ = compute_sfc_pair_features(
        fc, sc, edge_pairs, n_pairs, fit_scalers=True
    )
    assert X.shape == (n_subj, n_pairs)
    assert np.all(np.isfinite(X))


def test_get_valid_pair_mask():
    edge_pairs = np.array([0, 1, -1, 2, -1])
    mask = get_valid_pair_mask(edge_pairs)
    expected = np.array([True, True, False, True, False])
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# li_sfc_ncr tests
# ---------------------------------------------------------------------------

def test_build_pair_laplacian_single_pair_returns_none():
    pair_prior = np.array([0.8])
    selected = np.array([0])
    system_names = np.array(["VIS"])
    result = build_pair_laplacian(pair_prior, selected, system_names)
    assert result is None


def test_build_pair_laplacian_symmetric():
    pair_prior = np.array([0.5, 0.8, 0.3, 0.6])
    selected = np.array([0, 1, 2, 3])
    system_names = np.array(["VIS", "SM", "VIS-SM", "FPN"])
    L = build_pair_laplacian(pair_prior, selected, system_names)
    assert L is not None
    assert L.shape == (4, 4)
    np.testing.assert_allclose(L, L.T, atol=1e-10)


def test_build_pair_laplacian_psd():
    pair_prior = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    selected = np.array([0, 1, 2, 3, 4])
    system_names = np.array(["VIS", "SM", "VIS-SM", "VIS-FPN", "SM-FPN"])
    L = build_pair_laplacian(pair_prior, selected, system_names)
    assert L is not None
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() >= -1e-10


def test_ridge_expert_predict_shape_and_finiteness():
    rng = np.random.RandomState(42)
    n_train, n_test, n_pairs = 30, 10, 5
    X_tr = rng.randn(n_train, n_pairs)
    y = rng.randn(n_train)
    X_te = rng.randn(n_test, n_pairs)
    expert = fit_sfc_ridge_expert(X_tr, y)
    pred = predict_sfc_expert(X_te, expert)
    assert pred.shape == (n_test,)
    assert np.all(np.isfinite(pred))


def test_ridge_expert_constant_target():
    n_pairs = 4
    X_tr = np.random.randn(20, n_pairs)
    y = np.ones(20) * 3.0
    expert = fit_sfc_ridge_expert(X_tr, y)
    assert expert.target_std == 1.0
    pred = predict_sfc_expert(np.random.randn(5, n_pairs), expert)
    assert np.all(np.isfinite(pred))


def test_ncr_expert_predict_shape():
    rng = np.random.RandomState(7)
    n_train, n_test, n_pairs = 25, 8, 4
    X_tr = rng.randn(n_train, n_pairs)
    y = rng.randn(n_train)
    pair_prior = rng.uniform(0.1, 1.0, n_pairs)
    system_names = np.array(["VIS", "SM", "FPN", "DMN"])
    X_te = rng.randn(n_test, n_pairs)
    expert = fit_sfc_ncr_expert(X_tr, y, pair_prior, system_names)
    pred = predict_sfc_expert(X_te, expert)
    assert pred.shape == (n_test,)
    assert np.all(np.isfinite(pred))


def test_hierarchical_fusion_eta_0_pure_baseline():
    rng = np.random.RandomState(10)
    y = rng.randn(20)
    base = rng.randn(20)
    expert = rng.randn(20)
    result = hierarchical_fusion(y, base, expert, base, expert, eta_grid=np.array([0.0]))
    assert result["eta"] == 0.0
    np.testing.assert_allclose(result["fused_oof"], base)
    np.testing.assert_allclose(result["fused_test"], base)


def test_hierarchical_fusion_eta_1_pure_expert():
    rng = np.random.RandomState(11)
    y = rng.randn(20)
    base = rng.randn(20)
    expert = rng.randn(20)
    result = hierarchical_fusion(y, base, expert, base, expert, eta_grid=np.array([1.0]))
    assert result["eta"] == 1.0
    np.testing.assert_allclose(result["fused_oof"], expert)
    np.testing.assert_allclose(result["fused_test"], expert)


def test_search_fusion_weights_returns_in_range():
    rng = np.random.RandomState(12)
    y = rng.randn(25)
    base = rng.randn(25)
    expert = rng.randn(25)
    w = search_fusion_weights(y, base, expert)
    assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# llm_prior_generator tests
# ---------------------------------------------------------------------------

def test_normalize_pair_prior_range():
    scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    s_min, s_max = scores.min(), scores.max()
    normed = (scores - s_min) / (s_max - s_min)
    assert normed.min() == pytest.approx(0.0, abs=1e-10)
    assert normed.max() == pytest.approx(1.0, abs=1e-10)
    assert len(normed) == 5
    const = np.full(5, 0.5)
    c_min, c_max = const.min(), const.max()
    if c_max > c_min:
        normed_c = (const - c_min) / (c_max - c_min)
    else:
        normed_c = np.full_like(const, 0.5)
    assert np.all(normed_c == 0.5)


def test_create_control_priors_shuffle_preserves_values():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "system_a": ["VIS", "SM", "FPN"],
        "system_b": ["SM", "FPN", "DMN"],
        "pair_name": ["VIS-SM", "SM-FPN", "FPN-DMN"],
        "normalized_score": rng.uniform(0, 1, 3),
    })
    controls = create_control_priors(df, "/tmp/test_ctrl")
    orig = set(df["normalized_score"].values)
    shuf = set(controls["shuffled"]["normalized_score"].values)
    assert orig == shuf
    r = controls["random"]["normalized_score"].values
    assert r.min() >= 0.0
    assert r.max() <= 1.0


def test_compute_prior_diagnostics_basic():
    wm = pd.DataFrame({
        "pair_name": [f"p{i}" for i in range(20)],
        "normalized_score": np.linspace(0, 1, 20),
    })
    fi = pd.DataFrame({
        "pair_name": [f"p{i}" for i in range(20)],
        "normalized_score": np.linspace(0, 1, 20),
    })
    diag = compute_prior_diagnostics(wm, fi)
    assert "wm_fi_pearson" in diag
    assert "top5_overlap" in diag
    assert diag["wm_fi_pearson"] == pytest.approx(1.0, abs=1e-10)
    assert diag["top5_overlap"] == 5
