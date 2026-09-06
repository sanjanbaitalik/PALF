#!/usr/bin/env python3
"""Plotting regression tests for final ICLR figures."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = REPO_ROOT / "configs/iclr/lf1_final_10x5.yaml"
FINAL_OUTPUT_BASE = REPO_ROOT / "outputs/iclr/lf1_final_10x5"
EVIDENCE_AUDIT = REPO_ROOT / "outputs/iclr/lf1_final_evidence_audit"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def wm_seed_metrics():
    return pd.read_csv(FINAL_OUTPUT_BASE / "working_memory" / "seed_metrics.csv")


@pytest.fixture(scope="module")
def fi_seed_metrics():
    return pd.read_csv(FINAL_OUTPUT_BASE / "fluid_intelligence" / "seed_metrics.csv")


@pytest.fixture(scope="module")
def wm_bio():
    return pd.read_csv(EVIDENCE_AUDIT / "biomarker_seed_metrics_working_memory.csv")


@pytest.fixture(scope="module")
def fi_bio():
    return pd.read_csv(EVIDENCE_AUDIT / "biomarker_seed_metrics_fluid_intelligence.csv")


@pytest.fixture(scope="module")
def wm_results():
    pkl = FINAL_OUTPUT_BASE / "working_memory" / "all_split_results.pkl"
    with pkl.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def fi_results():
    pkl = FINAL_OUTPUT_BASE / "fluid_intelligence" / "all_split_results.pkl"
    with pkl.open("rb") as f:
        return pickle.load(f)


def _normalize_model(name):
    s = str(name).strip().lower()
    if s == "lf1" or "lf1" in s or "prior_substitution" in s:
        return "PALF"
    if s == "lf0" or "lf0" in s or "no_prior_late" in s:
        return "No-prior late fusion"
    return str(name)


def _compute_deltas(df):
    df = df.copy()
    df["paper_model"] = df["model"].map(_normalize_model)
    keep = df[df["paper_model"].isin(["PALF", "No-prior late fusion"])].copy()
    pivot = keep.pivot_table(index="seed", columns="paper_model", values="pearson", aggfunc="mean")
    pivot = pivot.dropna(subset=["PALF", "No-prior late fusion"]).sort_index()
    return pivot["PALF"] - pivot["No-prior late fusion"]


class TestSeedDeltaSourceCalculations:
    """Test 1: seed-delta source calculations reproduce frozen means."""

    def test_wm_mean_delta(self, wm_seed_metrics):
        delta = _compute_deltas(wm_seed_metrics)
        assert abs(delta.mean() - 0.0170) < 0.001, f"WM mean Δr = {delta.mean():.6f}"

    def test_wm_positive_seeds(self, wm_seed_metrics):
        delta = _compute_deltas(wm_seed_metrics)
        assert int((delta > 0).sum()) == 8

    def test_fi_mean_delta(self, fi_seed_metrics):
        delta = _compute_deltas(fi_seed_metrics)
        assert abs(delta.mean() - 0.0079) < 0.001, f"FI mean Δr = {delta.mean():.6f}"

    def test_fi_positive_seeds(self, fi_seed_metrics):
        delta = _compute_deltas(fi_seed_metrics)
        assert int((delta > 0).sum()) == 7


class TestSeedDeltaPlotting:
    """Test 2: seed-delta plotting does not use a connected line."""

    def test_no_line_in_source(self):
        src = (REPO_ROOT / "scripts_paper" / "plot_seed_deltas.py").read_text()
        assert "ax.plot(" not in src.split("ax.scatter")[0].split("def main")[-1], \
            "plot_seed_deltas.py should not call ax.plot() before scatter"


class TestBiomarkerUsesEvidenceAudit:
    """Test 3: biomarker plots use evidence-audit data."""

    def test_evidence_audit_file_exists(self):
        for task in ["working_memory", "fluid_intelligence"]:
            path = EVIDENCE_AUDIT / f"biomarker_seed_metrics_{task}.csv"
            assert path.exists(), f"Missing: {path}"

    def test_evidence_audit_has_model_column(self, wm_bio):
        assert "model" in wm_bio.columns

    def test_evidence_audit_has_alignment_columns(self, wm_bio):
        for col in ["matched_alignment", "unrelated_alignment", "shuffled_alignment", "random_alignment"]:
            assert col in wm_bio.columns, f"Missing column: {col}"


class TestBiomarkerMeans:
    """Test 4: biomarker means reproduce final audited values."""

    def test_wm_no_prior(self, wm_bio):
        vals = wm_bio[wm_bio["model"] == "FC_no_prior"]["matched_alignment"]
        assert abs(vals.mean() - 0.0005) < 0.01

    def test_wm_matched(self, wm_bio):
        vals = wm_bio[wm_bio["model"] == "FP_matched"]["matched_alignment"]
        assert abs(vals.mean() - 0.6699) < 0.01

    def test_wm_unrelated(self, wm_bio):
        vals = wm_bio[wm_bio["model"] == "FP_unrelated"]["matched_alignment"]
        assert abs(vals.mean() - 0.4974) < 0.01

    def test_wm_shuffled(self, wm_bio):
        vals = wm_bio[wm_bio["model"] == "FP_shuffled"]["matched_alignment"]
        assert abs(vals.mean() - (-0.1006)) < 0.01

    def test_wm_random(self, wm_bio):
        vals = wm_bio[wm_bio["model"] == "FP_random"]["matched_alignment"]
        assert abs(vals.mean() - 0.0873) < 0.01

    def test_fi_matched(self, fi_bio):
        vals = fi_bio[fi_bio["model"] == "FP_matched"]["matched_alignment"]
        assert abs(vals.mean() - 0.7615) < 0.01

    def test_fi_unrelated(self, fi_bio):
        vals = fi_bio[fi_bio["model"] == "FP_unrelated"]["matched_alignment"]
        assert abs(vals.mean() - 0.5831) < 0.01


class TestFusionWeightCount:
    """Test 5: fusion plots contain exactly 50 values per task."""

    def test_wm_50_splits(self, wm_results):
        assert len(wm_results) == 50

    def test_fi_50_splits(self, fi_results):
        assert len(fi_results) == 50


class TestFusionWeightMeans:
    """Test 6: fusion-weight means reproduce ~0.76 / ~0.37."""

    def test_wm_mean_weight(self, wm_results):
        w_fp = [float(r.lf1_weights["FP"]) for r in wm_results]
        mean_w = np.mean(w_fp)
        assert abs(mean_w - 0.76) < 0.05, f"WM mean w_FP = {mean_w:.4f}"

    def test_fi_mean_weight(self, fi_results):
        w_fp = [float(r.lf1_weights["FP"]) for r in fi_results]
        mean_w = np.mean(w_fp)
        assert abs(mean_w - 0.37) < 0.05, f"FI mean w_FP = {mean_w:.4f}"


class TestPriorMapScaling:
    """Test 7: prior maps use identical [0,1] scaling."""

    def test_wm_prior_range(self):
        df = pd.read_csv(REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")
        scores = pd.to_numeric(df["prior_score"], errors="coerce")
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_fi_prior_range(self):
        df = pd.read_csv(REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv")
        scores = pd.to_numeric(df["prior_score"], errors="coerce")
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestTopEdgePlotTopK:
    """Test 8: top-edge plot uses exactly top 20 for both tasks."""

    def test_wm_top20(self):
        df = pd.read_csv(REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv")
        top20 = df.sort_values("mean_abs_coef", ascending=False).head(20)
        assert len(top20) == 20

    def test_fi_top20(self):
        df = pd.read_csv(REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/fluid_intelligence/stable_top_edges.csv")
        top20 = df.sort_values("mean_abs_coef", ascending=False).head(20)
        assert len(top20) == 20


class TestTopEdgeSource:
    """Test 9: top-edge source is frozen reconstructed coefficients."""

    def test_wm_stable_top_exists(self):
        path = REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv"
        assert path.exists()

    def test_fi_stable_top_exists(self):
        path = REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/fluid_intelligence/stable_top_edges.csv"
        assert path.exists()

    def test_wm_has_signed_coef(self):
        df = pd.read_csv(REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv")
        assert "mean_signed_coef" in df.columns
        assert "mean_abs_coef" in df.columns


class TestPlottingScriptsNoArtifactModification:
    """Test 10: plotting scripts do not modify frozen experiment artifacts."""

    def test_seed_deltas_no_write_to_outputs(self):
        src = (REPO_ROOT / "scripts_paper" / "plot_seed_deltas.py").read_text()
        assert "all_split_results.pkl" not in src or "read" in src.lower()

    def test_biomarker_no_write_to_outputs(self):
        src = (REPO_ROOT / "scripts_paper" / "plot_biomarker_alignment.py").read_text()
        assert "all_split_results.pkl" not in src

    def test_fusion_weights_read_only(self):
        src = (REPO_ROOT / "scripts_paper" / "plot_fusion_weights.py").read_text()
        assert "pickle.load" in src
        assert "pickle.dump" not in src

    def test_roi_prior_no_experiment_files(self):
        src = (REPO_ROOT / "scripts_paper" / "plot_roi_prior.py").read_text()
        assert "all_split_results.pkl" not in src
