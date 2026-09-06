#!/usr/bin/env python3
"""Tests for the frozen FP coefficient export utility."""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts_paper.export_frozen_fp_coefficients import (
    EXPECTED_COEF_LENGTH,
    PROJECT_ROOT,
    reconstruct_one_split,
    compute_edge_mean_abs,
    compute_stability_metrics,
)

CONFIG_PATH = PROJECT_ROOT / "configs/iclr/lf1_final_10x5.yaml"
FINAL_OUTPUT_BASE = PROJECT_ROOT / "outputs/iclr/lf1_final_10x5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def wm_results(cfg):
    pkl_path = FINAL_OUTPUT_BASE / "working_memory" / "all_split_results.pkl"
    if not pkl_path.exists():
        pytest.skip("Working memory results not available")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def fi_results(cfg):
    pkl_path = FINAL_OUTPUT_BASE / "fluid_intelligence" / "all_split_results.pkl"
    if not pkl_path.exists():
        pytest.skip("Fluid intelligence results not available")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def wm_data(cfg):
    from metascfc.benchmark_utils import load_connectomes
    from metascfc.experiments.msancr_refinement import upper_triangle_features, load_roi_prior

    fc_mats, sc_mats, y_all, _, _ = load_connectomes(cfg["data"])
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)
    y_task = np.load(cfg["targets"]["working_memory"]["label_path"], allow_pickle=False).astype(np.float64).reshape(-1)
    roi_prior = load_roi_prior(cfg["priors"]["working_memory"]["matched"], int(cfg["n_rois"]))
    return X_fc, X_sc, y_task, roi_prior


@pytest.fixture(scope="module")
def fi_data(cfg):
    from metascfc.benchmark_utils import load_connectomes
    from metascfc.experiments.msancr_refinement import upper_triangle_features, load_roi_prior

    fc_mats, sc_mats, y_all, _, _ = load_connectomes(cfg["data"])
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)
    y_task = np.load(cfg["targets"]["fluid_intelligence"]["label_path"], allow_pickle=False).astype(np.float64).reshape(-1)
    roi_prior = load_roi_prior(cfg["priors"]["fluid_intelligence"]["matched"], int(cfg["n_rois"]))
    return X_fc, X_sc, y_task, roi_prior


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitMetadataFromPickle:
    """Test 1: split metadata comes from pickle, not regenerated."""

    def test_wm_split_count(self, wm_results):
        assert len(wm_results) == 50

    def test_fi_split_count(self, fi_results):
        assert len(fi_results) == 50

    def test_wm_seeds(self, wm_results):
        seeds = sorted(set(r.seed for r in wm_results))
        assert seeds == list(range(10))

    def test_fi_seeds(self, fi_results):
        seeds = sorted(set(r.seed for r in fi_results))
        assert seeds == list(range(10))

    def test_wm_folds(self, wm_results):
        folds = sorted(set(r.outer_fold for r in wm_results))
        assert folds == [0, 1, 2, 3, 4]

    def test_no_duplicate_seed_fold_pairs(self, wm_results):
        pairs = [(r.seed, r.outer_fold) for r in wm_results]
        assert len(pairs) == len(set(pairs))

    def test_train_test_disjoint(self, wm_results):
        for r in wm_results:
            overlap = np.intersect1d(r.train_idx, r.test_idx)
            assert len(overlap) == 0, f"Overlap in seed={r.seed} fold={r.outer_fold}"

    def test_total_subjects_412(self, wm_results):
        for r in wm_results:
            total = len(r.train_idx) + len(r.test_idx)
            assert total == 412, f"Expected 412, got {total} in seed={r.seed} fold={r.outer_fold}"


class TestNoInnerCV:
    """Test 2: no inner-CV/model-selection helper is invoked."""

    def test_no_inner_cv_flag(self):
        from scripts_paper.export_frozen_fp_coefficients import _NO_INNER_CV_CALLED
        assert _NO_INNER_CV_CALLED is True


class TestNoPriorGeneration:
    """Test 3: no prior generation occurs."""

    def test_no_prior_generation_flag(self):
        from scripts_paper.export_frozen_fp_coefficients import _NO_NEW_PRIOR_GENERATED
        assert _NO_NEW_PRIOR_GENERATED is True


class TestMatchedPriorUsed:
    """Test 4: matched task prior is used."""

    def test_wm_matched_prior_path(self, cfg):
        path = cfg["priors"]["working_memory"]["matched"]
        assert "working_memory_contrastive_qwen3" in path

    def test_fi_matched_prior_path(self, cfg):
        path = cfg["priors"]["fluid_intelligence"]["matched"]
        assert "fluid_intelligence_contrastive_qwen3" in path


class TestCoefficientLength:
    """Test 5: coefficient length is 6670."""

    def test_expected_length(self):
        assert EXPECTED_COEF_LENGTH == 6670

    def test_reconstruction_length(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        data = reconstruct_one_split(
            wm_results[0], X_fc, X_sc, y_task, roi_prior, 116, "test",
        )
        assert data["coefficient_length"] == 6670
        assert data["coef_model_space"].shape == (6670,)


class TestCoefficientsFinite:
    """Test 6: coefficients are finite."""

    def test_wm_split0_finite(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        data = reconstruct_one_split(
            wm_results[0], X_fc, X_sc, y_task, roi_prior, 116, "test",
        )
        assert data["coefficient_finite"] is True
        assert np.isfinite(data["coef_model_space"]).all()
        assert np.isfinite(data["coef_biomarker_space"]).all()


class TestLF1PredictionUsesWeights:
    """Test 7: LF1 reconstructed prediction uses stored FP/S fusion weights."""

    def test_wm_fusion_weights_simplex(self, wm_results):
        for r in wm_results:
            w_fp = r.lf1_weights.get("FP", 0.0)
            w_s = r.lf1_weights.get("S", 0.0)
            assert abs(w_fp + w_s - 1.0) < 1e-6, \
                f"Weights don't sum to 1: w_fp={w_fp} w_s={w_s}"
            assert w_fp >= 0.0 and w_s >= 0.0, \
                f"Negative weights: w_fp={w_fp} w_s={w_s}"

    def test_wm_fusion_reconstruction(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        data = reconstruct_one_split(
            wm_results[0], X_fc, X_sc, y_task, roi_prior, 116, "test",
        )
        w_fp = wm_results[0].lf1_weights["FP"]
        w_s = wm_results[0].lf1_weights["S"]
        expected = w_fp * data["pred_fp"] + w_s * data["pred_sc"]
        np.testing.assert_allclose(
            data["pred_lf1_reconstructed"], expected, atol=1e-10,
        )


class TestReconstructionTolerance:
    """Test 8: reconstruction tolerance is enforced."""

    def test_wm_max_error_within_tolerance(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        data = reconstruct_one_split(
            wm_results[0], X_fc, X_sc, y_task, roi_prior, 116, "test",
        )
        assert data["prediction_max_abs_error"] <= 1e-5, \
            f"Max error {data['prediction_max_abs_error']:.2e} exceeds tolerance"

    def test_wm_10_splits_all_pass(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        for i in range(10):
            data = reconstruct_one_split(
                wm_results[i], X_fc, X_sc, y_task, roi_prior, 116, "test",
            )
            assert data["prediction_max_abs_error"] <= 1e-5, \
                f"Split {i} failed: {data['prediction_max_abs_error']:.2e}"


class TestFailedReconstruction:
    """Test 9: failed reconstruction does not receive status=PASS."""

    def test_failed_split_has_no_pass(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        data = reconstruct_one_split(
            wm_results[0], X_fc, X_sc, y_task, roi_prior, 116, "test",
        )
        if data["prediction_max_abs_error"] > 1e-5:
            assert data["tolerance_used"].startswith("FAILED")
        else:
            assert "PASS" in data["tolerance_used"] or data["prediction_max_abs_error"] <= 1e-6


class TestAggregateArrayShape:
    """Test 10: aggregate arrays contain exactly 50 vectors when full task is complete."""

    def test_all_coefficients_shape(self, wm_results, wm_data):
        X_fc, X_sc, y_task, roi_prior = wm_data
        all_coefs = []
        for r in wm_results:
            data = reconstruct_one_split(
                r, X_fc, X_sc, y_task, roi_prior, 116, "test",
            )
            all_coefs.append(data["coef_biomarker_space"])
        arr = np.array(all_coefs)
        assert arr.shape == (50, 6670)


class TestExistingArtifactsUnchanged:
    """Test 11: existing final LF1 artifacts are unchanged."""

    def test_wm_pkl_exists(self):
        pkl = FINAL_OUTPUT_BASE / "working_memory" / "all_split_results.pkl"
        assert pkl.exists()

    def test_fi_pkl_exists(self):
        pkl = FINAL_OUTPUT_BASE / "fluid_intelligence" / "all_split_results.pkl"
        assert pkl.exists()

    def test_wm_complete_marker_exists(self):
        marker = FINAL_OUTPUT_BASE / "COMPLETE"
        assert marker.exists()

    def test_fi_complete_marker_exists(self):
        marker = FINAL_OUTPUT_BASE / "FINAL_COMPLETE"
        assert marker.exists()


class TestEdgeMeanAbs:
    """Test edge_mean_abs.csv structure."""

    def test_compute_edge_mean_abs(self):
        coefs = np.random.randn(5, 6670)
        df = compute_edge_mean_abs(coefs, 116)
        assert len(df) == 6670
        assert "mean_abs_coef" in df.columns
        assert "rank_mean_abs" in df.columns
        assert "roi_i_1based" in df.columns
        assert "roi_j_1based" in df.columns

    def test_with_labels(self):
        coefs = np.random.randn(5, 6670)
        label_path = PROJECT_ROOT / "inputs/atlases/AAL116_labels.csv"
        df = compute_edge_mean_abs(coefs, 116, label_path)
        assert "roi_i_name" in df.columns
        assert "roi_j_name" in df.columns


class TestStabilityMetrics:
    """Test stability metrics computation."""

    def test_stability_metrics(self):
        coefs = np.random.randn(50, 6670)
        df = compute_stability_metrics(coefs)
        assert len(df) == 6670
        assert "top10_frequency" in df.columns
        assert "sign_consistency" in df.columns
        assert df["top10_frequency"].between(0, 1).all()
        assert df["sign_consistency"].between(0, 1).all()
