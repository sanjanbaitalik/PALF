"""Biomarker stability correction tests (v36, reporting-only).

T1-T16 per the correction protocol: modality-specific importance definitions,
ranking/zero handling, inactive-fold exclusion, edge ordering, coefficient
reconstruction, no fitting, frozen prediction files, holdout seal, paper-ready
table/figure agreement, and top-10 traceability.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import final510_wm_biomarker_correction as C  # noqa: E402
import final510_wm_pilot as P  # noqa: E402
import final510_wm_report as R  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
PR = OUT / "paper_ready"
_CACHE = {}


def entries():
    if "entries" not in _CACHE:
        _CACHE["entries"] = C.load_coefficient_entries()[1]
    return _CACHE["entries"]


def entry(c_fc, c_sc):
    imp_fc = C.incident_importance(c_fc)
    imp_sc = C.incident_importance(c_sc)
    return {"c_fc": c_fc, "c_sc": c_sc, "c_multi": c_fc + c_sc,
            "imp_fc": imp_fc, "imp_sc": imp_sc, "imp_multi": imp_fc + imp_sc,
            "fc_active": bool(np.any(c_fc != 0)),
            "sc_active": bool(np.any(c_sc != 0)),
            "multi_active": bool(np.any((c_fc + c_sc) != 0))}


def _edge(i, j, n=6670):
    v = np.zeros(n)
    k = np.where((C.IU[0] == i) & (C.IU[1] == j))[0][0]
    v[k] = 1.0
    return v


# ── T1-T3 importance definitions ──────────────────────────────────────

def test_t1_fc_importance_uses_fc_only():
    c_fc = _edge(0, 1); c_sc = _edge(2, 3)
    e = entry(c_fc, c_sc)
    assert e["imp_fc"][0] == 1 and e["imp_fc"][1] == 1
    assert e["imp_fc"][2] == 0 and e["imp_fc"][3] == 0
    assert not np.allclose(e["imp_fc"], e["imp_sc"])


def test_t2_sc_importance_uses_sc_only():
    c_fc = _edge(0, 1); c_sc = _edge(2, 3)
    e = entry(np.zeros(6670), c_sc)
    assert e["imp_sc"][2] == 1 and e["imp_sc"][3] == 1
    assert np.all(e["imp_fc"] == 0)


def test_t3_multimodal_equals_fc_plus_sc():
    rng = np.random.RandomState(0)
    c_fc = rng.randn(6670); c_sc = rng.randn(6670)
    e = entry(c_fc, c_sc)
    assert np.allclose(e["imp_multi"], e["imp_fc"] + e["imp_sc"])
    for real in entries():
        if real["multi_active"]:
            assert np.allclose(real["imp_multi"],
                               real["imp_fc"] + real["imp_sc"])
            break


# ── T4-T6 rankings and zero handling ──────────────────────────────────

def test_t4_fc_and_sc_top10_can_differ():
    found = False
    for e in entries():
        if e["model"] == "R-MATCHED" and e["fc_active"] and e["sc_active"]:
            rf = set(C.rank_rois(e["imp_fc"], 10).tolist())
            rs = set(C.rank_rois(e["imp_sc"], 10).tolist())
            if rf != rs:
                found = True
                break
    assert found


def test_t5_jaccard_correct():
    assert C.jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert C.jaccard([1, 2], [1, 2]) == 1.0
    assert C.jaccard([1], [2]) == 0.0


def test_t6_all_zero_modality_has_no_ranking():
    assert C.rank_rois(np.zeros(116), 10) is None
    e1 = entry(_edge(0, 1), np.zeros(6670))
    st = C.corrected_stability([e1])
    assert st["sc_top10_roi_jaccard_n_pairs"] == 0
    assert np.isnan(st["sc_top10_roi_jaccard"])
    assert st["fc_top10_roi_jaccard_n_pairs"] == 0  # single entry, no pairs


# ── T7-T8 inactive-fold exclusion ─────────────────────────────────────

def test_t7_inactive_modality_fold_excluded():
    e1 = entry(_edge(0, 1), np.zeros(6670))
    e2 = entry(_edge(0, 2), np.zeros(6670))
    st = C.corrected_stability([e1, e2])
    assert st["fc_top10_roi_jaccard_n_pairs"] == 1
    assert st["sc_top10_roi_jaccard_n_pairs"] == 0
    assert st["sc_top10_roi_jaccard_n_excluded_inactive"] == 1
    assert st["sc_edge_spearman_n_pairs"] == 0


def test_t8_multimodal_valid_with_one_inactive_modality():
    e1 = entry(_edge(0, 1), np.zeros(6670))
    e2 = entry(_edge(0, 2), np.zeros(6670))
    st = C.corrected_stability([e1, e2])
    assert st["multimodal_top10_roi_jaccard_n_pairs"] == 1
    assert np.isfinite(st["multimodal_top10_roi_jaccard"])
    assert st["multimodal_sign_consistency_n_pairs"] == 1


# ── T9-T10 ordering and reconstruction ────────────────────────────────

def test_t9_edge_ordering_unchanged():
    assert P.N_EDGE == 6670
    assert np.array_equal(C.IU[0], np.triu_indices(116, 1)[0])
    assert np.array_equal(C.IU[1], np.triu_indices(116, 1)[1])
    rng = np.random.RandomState(1)
    c = rng.randn(6670)
    imp = C.incident_importance(c)
    M = np.zeros((116, 116))
    M[C.IU[0], C.IU[1]] = np.abs(c)
    M = M + M.T
    assert np.allclose(imp, M.sum(1))


def test_t10_coefficient_reconstruction_exact():
    err = C.verify_reconstruction()
    assert err <= 1e-8, err


# ── T11-T13 no fitting, frozen files, holdout ─────────────────────────

def test_t11_no_model_fitting_in_correction_script():
    src = inspect.getsource(C)
    for banned in (".fit(", "fit_expert", "evaluate_ablation_split",
                   "NetworkConstrainedRidge(", "Ridge("):
        assert banned not in src, banned
    assert "load_folds" in src and "coefficient_maps" in src


def test_t12_no_prediction_file_modified():
    seal = json.loads((OUT / "BIOMARKER_CORRECTION_SEAL.json").read_text())
    assert seal["prediction_files_unchanged"] is True
    for f, h in seal["frozen_file_hashes"].items():
        assert C.sha256_file(OUT / f) == h, f
    assert "primary_metrics.csv" in seal["frozen_file_hashes"]


def test_t13_no_holdout_access():
    ids = (ROOT / "data_splits" / "phase3_holdout_98.txt").read_text().strip().split("\n")
    sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert len(ids) == 98 and sha == C.HOLDOUT_SHA
    seal = json.loads((OUT / "BIOMARKER_CORRECTION_SEAL.json").read_text())
    assert seal["holdout_sha256"] == C.HOLDOUT_SHA
    assert seal["holdout_id_scan_hits"] == {}
    for f in seal["files_written"]:
        p = OUT / f
        if p.exists() and p.suffix in (".csv", ".md", ".json"):
            txt = p.read_text(errors="ignore")
            assert not any(i in txt for i in ids), f


# ── T14-T16 paper-ready agreement, figure, top-10 ─────────────────────

def test_t14_paper_ready_table_equals_corrected_csv():
    tab = pd.read_csv(PR / "table_main_wm_biomarkers.csv")
    cor = pd.read_csv(OUT / "biomarker_stability_corrected.csv").set_index("model")
    for _, r in tab.iterrows():
        model = f"{'R' if r['arch'] == 'ridge' else 'N'}-{r['prior'].upper()}"
        row = cor.loc[model]
        for legacy, src in (("fc_top10_roi_jaccard", "fc_top10_roi_jaccard"),
                            ("sc_top10_roi_jaccard", "sc_top10_roi_jaccard"),
                            ("multimodal_top10_roi_jaccard",
                             "multimodal_top10_roi_jaccard")):
            assert abs(row[src] - r[legacy]) < 1e-12, (model, src)


def test_t15_figure_uses_corrected_values():
    src = inspect.getsource(R.fig_stability)
    for col in ("fc_top10_roi_jaccard", "sc_top10_roi_jaccard",
                "multimodal_top10_roi_jaccard", "fc_edge_spearman",
                "sc_edge_spearman"):
        assert col in src
    csv_mtime = (OUT / "biomarker_stability_corrected.csv").stat().st_mtime
    for d in (OUT / "plots", PR):
        f = d / "fig5_wm_biomarker_stability.pdf"
        assert f.exists() and f.stat().st_mtime >= csv_mtime


def test_t16_top10_traces_to_saved_coefficients():
    out = pd.read_csv(PR / "wm_top10_biomarkers_R-MATCHED.csv")
    ent = [e for e in entries() if e["model"] == "R-MATCHED" and e["multi_active"]]
    imp = np.mean([e["imp_multi"] for e in ent], axis=0)
    order = np.lexsort((np.arange(116), -imp))[:10]
    assert list(out["atlas_index"]) == [int(i) + 1 for i in order]
    for _, r in out.iterrows():
        i = int(r["atlas_index"]) - 1
        assert abs(r["multimodal_importance"] - imp[i]) < 1e-10


def test_t17_required_columns_present():
    df = pd.read_csv(OUT / "biomarker_stability_corrected.csv")
    for c in ("model", "task", "n_valid_outer_folds", "n_inactive_outer_folds",
              "fc_edge_spearman", "sc_edge_spearman",
              "fc_top100_edge_jaccard", "sc_top100_edge_jaccard",
              "fc_top300_edge_jaccard", "sc_top300_edge_jaccard",
              "fc_top10_roi_jaccard", "sc_top10_roi_jaccard",
              "multimodal_top10_roi_jaccard", "multimodal_top20_roi_jaccard",
              "fc_sign_consistency", "sc_sign_consistency",
              "multimodal_sign_consistency"):
        assert c in df.columns, c
    assert len(df) == 8
    assert set(df["model"]) == set(P.ALL_MODELS)


def test_t18_biomarker_level_unchanged_b0():
    st = json.loads((OUT / "FINAL_STATUS.json").read_text())
    assert st["biomarker_levels"]["R"]["level"] == "B0"
    assert st["prediction_level"]["RIDGE_PREDICTION_LEVEL"] == "P1"
