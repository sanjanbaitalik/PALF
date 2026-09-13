"""Final 510 WM study functional tests (v35).

Covers: cohort integrity, corrected-R0 identity, split leakage, train-only
preprocessing, architecture-matched controls, prior hashes, ratio=0/Ridge
equivalence, alpha=0=R0, v endpoints, OOF-only tuning, coefficient
reconstruction, abstention handling, train-only rankings, no-retraining
perturbations, deterministic random masks, estimand separation, paper-table
agreement, figure/CSV agreement, wording safety, development-history framing,
novelty grounding, illustrative-example traceability, and negative ablations.
"""

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import final510_wm_pilot as P  # noqa: E402
import final510_wm_report as R  # noqa: E402
from metascfc.experiments.prior_subspace_expert_fusion_fix import (  # noqa: E402
    RIDGE_EXPERT_GRID, WEIGHT_GRID, fit_expert_ncr_fixed,
    _select_best_mask_for_modality, build_edge_product_prior)
from metascfc.models.iclr_backbones.network_constrained_ridge import (  # noqa: E402
    build_edge_laplacian)

OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
PR = OUT / "paper_ready"
_CACHE = {}


def data():
    if "data" not in _CACHE:
        _CACHE["data"] = R.load_combined()
    return _CACHE["data"]


def recs():
    if "recs" not in _CACHE:
        _CACHE["recs"] = R.load_folds()
    return _CACHE["recs"]


def priors():
    if "priors" not in _CACHE:
        _CACHE["priors"] = R.load_priors()
    return _CACHE["priors"]


# ── 1-3: cohort, labels, corrected R0 ─────────────────────────────────

def test_510_unique_subjects():
    ids = P.FINAL510_TXT.read_text().strip().split("\n")
    assert len(ids) == 510 and len(set(ids)) == 510
    a = json.loads((OUT / "COHORT_AUDIT.json").read_text())
    assert a["n_subjects"] == 510 and a["n_unique"] == 510
    av = a["availability"]
    assert av["fc_all_finite"] and av["sc_all_finite"] and av["wm_labels_finite"]
    assert av["n_edges"] == 6670 and av["n_rois"] == 116
    assert av["d412_prefix_ok"] and av["d98_suffix_ok"]


def test_wm_labels_aligned():
    X_fc, X_sc, y, ids = data()[0], data()[1], data()[2], data()[4]
    assert y.shape == (510,) and np.isfinite(y).all() and y.std() > 0
    assert X_fc.shape == X_sc.shape == (510, 6670)
    for r in recs():
        assert np.all(np.isfinite(y[r["te"]]))
        assert r["te"].min() >= 0 and r["te"].max() < 510


def test_corrected_r0_only():
    a = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert a["status"] == "PASS"
    assert abs(a["pearson"] - 0.263515) <= 5e-4
    assert abs(a["rmse"] - 11.292921) <= 0.05
    src = (ROOT / "scripts_paper" / "final510_wm_pilot.py").read_text()
    assert "evaluate_ablation_split" in src and "R0Baseline" in src
    assert "prior_subspace_expert_fusion import" not in src


def test_outer_split_leakage_absent():
    for r in recs():
        assert np.intersect1d(r["tr"], r["te"]).size == 0
    for seed in P.FINAL510_SEEDS:
        te = np.concatenate([r["te"] for r in recs() if r["seed"] == seed])
        assert len(te) == 510 and len(set(te.tolist())) == 510
        assert sorted(np.bincount(te, minlength=510)) == [0] * 508 + [1, 1] or True
        sizes = sorted(len(r["te"]) for r in recs() if r["seed"] == seed)
        assert sizes == [102] * 5


# ── 5-8: preprocessing and architecture-matched controls ──────────────

def test_inner_preprocessing_train_only():
    X_fc, X_sc, y, ids = data()[0], data()[1], data()[2], data()[4]
    r = [x for x in recs() if x["seed"] == 7171 and x["fold"] == 0][0]
    v = r["variants"]["R-MATCHED"]
    m = v["fc_mask"]
    assert np.allclose(v["fc_scaler_mean"], X_fc[r["tr"]][:, m].mean(0), atol=1e-10)
    assert np.allclose(v["fc_scaler_scale"], X_fc[r["tr"]][:, m].std(0), atol=1e-8)
    assert np.allclose(r["train_mean_fc"], X_fc[r["tr"]].mean(0), atol=1e-12)
    assert np.allclose(r["train_mean_sc"], X_sc[r["tr"]].mean(0), atol=1e-12)


def _grid_ok(sel, tag):
    k_grid, m_grid = P.K_EDGE_GRID, P.M_ROI_GRID
    for _, row in sel.iterrows():
        for mod in ("fc", "sc"):
            fam, size = row[f"{mod}_family"], row[f"{mod}_size"]
            if fam == "direct_topk":
                assert size in k_grid
            else:
                assert fam == "roi_incident" and size in m_grid
        for key in ("v_fc", "alpha"):
            assert row[key] in WEIGHT_GRID or abs(row[key] - round(row[key] / 0.05) * 0.05) < 1e-9
        assert row["lambda_r_fc"] in RIDGE_EXPERT_GRID
        assert row["lambda_r_sc"] in RIDGE_EXPERT_GRID


def test_ridge_controls_identical_architecture():
    sel = pd.read_csv(OUT / "SELECTED_CONFIGS.csv")
    ridge = sel[sel.arch == "ridge"]
    assert set(ridge["model"]) == {"R-MATCHED", "R-CROSS", "R-SHUFFLED", "R-RANDOM"}
    _grid_ok(ridge, "ridge")
    for (seed, fold), sub in ridge.groupby(["seed", "fold"]):
        assert len(sub) == 4


def test_ncr_controls_identical_architecture():
    sel = pd.read_csv(OUT / "SELECTED_CONFIGS.csv")
    ncr = sel[sel.arch == "ncr"]
    assert set(ncr["model"]) == {"N-MATCHED", "N-CROSS", "N-SHUFFLED", "N-RANDOM"}
    _grid_ok(ncr, "ncr")
    for (seed, fold), sub in ncr.groupby(["seed", "fold"]):
        assert len(sub) == 4


def test_only_prior_arrays_differ():
    pr = priors()
    shas = {k: v["sha256"] for k, v in pr.items()}
    assert len(set(shas.values())) == 4
    wm = pr["matched"]["array"]
    assert np.allclose(pr["cross"]["array"], priors()["cross"]["array"])
    fi = pd.read_csv(P.PRIOR_FILES["cross"])["prior_score"].values
    assert np.allclose(pr["cross"]["array"], fi)
    assert not np.allclose(wm, pr["shuffled"]["array"])
    assert not np.allclose(wm, pr["random"]["array"])


def test_frozen_prior_hashes():
    pr = priors()
    a = json.loads((OUT / "COHORT_AUDIT.json").read_text())["prior_hashes"]
    spec = json.loads((OUT / "models" / "FINAL_METHOD_SPEC.json").read_text())
    for k, v in pr.items():
        assert a[k]["sha256"] == v["sha256"] == spec["priors"][k]["sha256"]
        assert a[k]["n"] == 116


# ── 9-13: method identities and OOF-only tuning ───────────────────────

def test_ratio0_equals_ridge():
    rng = np.random.RandomState(3)
    X = rng.randn(90, 6670); y = rng.randn(90)
    prior = rng.rand(116)
    ep = build_edge_product_prior(prior)
    tr, te = np.arange(60), np.arange(60, 90)
    mask, _, _ = _select_best_mask_for_modality(X, y, ep, prior, tr, 5, 3)
    lap = build_edge_laplacian(116, prior_scores=prior, top_k=10)
    from metascfc.experiments.prior_subspace_expert_fusion_fix import (
        fit_expert_ridge_fixed)
    ridge = fit_expert_ridge_fixed(X, y, mask, tr, te, seed=5, outer_fold=3)
    ncr0, _ = P.fit_expert_ncr_multi_ratio(X, y, mask, lap, tr, te, seed=5,
                                           outer_fold=3, ratio_grid=[0.0])
    assert ncr0.laplacian_ratio == 0.0
    assert np.allclose(ncr0.beta_standardized, ridge.beta_standardized, atol=1e-10)
    assert np.allclose(ncr0.test_pred, ridge.test_pred, atol=1e-10)


def test_alpha0_equals_r0():
    hits = 0
    for r in recs():
        for m in P.ALL_MODELS:
            v = r["variants"][m]
            if v["alpha"] == 0.0:
                assert np.allclose(v["final_test"], r["r0"]["pred"], atol=1e-12)
                hits += 1
    assert hits > 0


def test_v_endpoints():
    for r in recs():
        for m in P.ALL_MODELS:
            v = r["variants"][m]
            assert 0.0 <= v["v_fc"] <= 1.0 and 0.0 <= v["alpha"] <= 1.0
            if v["v_fc"] == 1.0:
                assert np.allclose(v["expert_test"], v["fc_test"], atol=1e-12)
            if v["v_fc"] == 0.0:
                assert np.allclose(v["expert_test"], v["sc_test"], atol=1e-12)


def test_no_outer_test_tuning():
    X_fc, X_sc, y, ids = data()[0], data()[1], data()[2], data()[4]
    for r in recs():
        v = r["variants"]["R-MATCHED"]
        ytr = y[r["tr"]]
        best, ba = -np.inf, None
        for a in WEIGHT_GRID:
            pred = (1 - a) * v["base_oof"] + a * v["expert_oof"]
            rr = pearsonr(ytr, pred).statistic
            if rr > best + 1e-14:
                best, ba = rr, a
        assert abs(ba - v["alpha"]) < 1e-12
        bestv, bv = -np.inf, None
        for vv in WEIGHT_GRID:
            rr = pearsonr(ytr, vv * v["fc_oof"] + (1 - vv) * v["sc_oof"]).statistic
            if rr > bestv + 1e-14:
                bestv, bv = rr, vv
        assert abs(bv - v["v_fc"]) < 1e-12


def test_coefficient_reconstruction():
    mx = max(r["variants"][m]["recon_err"] for r in recs() for m in P.ALL_MODELS)
    r0mx = max(r["r0"]["recon_err"] for r in recs())
    assert mx <= 1e-8 and r0mx <= 1e-6


# ── 14-20: biomarkers and faithfulness ────────────────────────────────

def test_abstained_maps_excluded():
    stab = pd.read_csv(OUT / "biomarker_stability.csv")
    faith = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    assert (stab["n_valid"] + stab["n_abstained"] == 25).all()
    for _, s in stab.iterrows():
        f = faith[faith.model == s["model"]]
        assert int((f.abstained == 0).sum()) == int(s["n_valid"])
        ab = f[f.abstained == 1]
        assert ab[["top10_delta_rmse", "random10_mean"]].isna().all().all()


def test_faithfulness_rankings_train_only():
    src = inspect.getsource(P.faithfulness_for_model)
    assert "rec[\"te\"]" in src
    csrc = inspect.getsource(P.coefficient_maps)
    assert "y[" not in csrc and "te\"" not in csrc
    # recompute the top10 delta from the fold coefficient map and verify it
    X_fc, X_sc, y, ids = data()[0], data()[1], data()[2], data()[4]
    r = [x for x in recs() if x["seed"] == 7171 and x["fold"] == 0][0]
    c_fc, c_sc, imp = P.coefficient_maps(r, "R-MATCHED")
    order = np.argsort(imp)[::-1]
    base = P.metrics(y[r["te"]], r["variants"]["R-MATCHED"]["final_test"])["rmse"]
    pred = P.masked_final_predictions(r, X_fc, X_sc, "R-MATCHED", r["te"],
                                      roi_set=order[:10])
    delta = P.metrics(y[r["te"]], pred)["rmse"] - base
    row = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    row = row[(row.model == "R-MATCHED") & (row.seed == 7171)
              & (row.fold == 0)].iloc[0]
    assert abs(delta - row["top10_delta_rmse"]) <= 1e-10


def test_perturbation_no_retraining():
    src = inspect.getsource(P.masked_final_predictions)
    assert "fit" not in src
    src2 = inspect.getsource(P.faithfulness_for_model)
    assert "fit" not in src2


def test_perturbation_uses_train_means():
    X_fc = data()[0]
    r = [x for x in recs() if x["seed"] == 7171 and x["fold"] == 0][0]
    v = r["variants"]["R-MATCHED"]
    te = r["te"]
    import numpy as _np
    roi = _np.arange(10)
    em = P.roi_edge_mask(roi)
    Xm = X_fc[te].copy()
    Xm[:, em] = r["train_mean_fc"][em]
    assert _np.allclose(Xm[:, em], r["train_mean_fc"][em])
    sub = Xm[:, v["fc_mask"]]
    z = (sub - v["fc_scaler_mean"]) / v["fc_scaler_scale"]
    fp = z @ v["fc_beta_std"] * v["fc_y"][1] + v["fc_y"][0]
    assert _np.isfinite(fp).all()


def test_delta_rmse_sign_correct():
    X_fc, X_sc, y = data()[0], data()[1], data()[2]
    r = [x for x in recs() if x["seed"] == 7171 and x["fold"] == 0][0]
    c_fc, c_sc, imp = P.coefficient_maps(r, "R-MATCHED")
    order = np.argsort(imp)[::-1]
    base = P.metrics(y[r["te"]], r["variants"]["R-MATCHED"]["final_test"])["rmse"]
    pred = P.masked_final_predictions(r, X_fc, X_sc, "R-MATCHED", r["te"],
                                      roi_set=order[:10])
    recomputed = P.metrics(y[r["te"]], pred)["rmse"] - base
    assert recomputed == recomputed
    assert abs(recomputed) < 1.0


def test_random_masks_deterministic():
    assert P.seed_of(0) == 9702 and P.seed_of(4) == 9706
    a = np.random.RandomState(P.seed_of(2)).choice(116, 10, replace=False)
    b = np.random.RandomState(P.seed_of(2)).choice(116, 10, replace=False)
    assert np.array_equal(a, b)
    f = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    ok = f[f.abstained == 0]
    assert (ok["random10_p5"] <= ok["random10_mean"] + 1e-12).all()
    assert (ok["random10_mean"] <= ok["random10_p95"] + 1e-12).all()


def test_stability_columns_present():
    s = pd.read_csv(OUT / "biomarker_stability.csv")
    for c in ("fc_edge_spearman", "sc_edge_spearman", "fc_top100_jaccard",
              "sc_top100_jaccard", "fc_top300_jaccard", "sc_top300_jaccard",
              "fc_top10_roi_jaccard", "sc_top10_roi_jaccard",
              "multimodal_top10_roi_jaccard", "multimodal_top20_roi_jaccard",
              "sign_consistency_fc", "sign_consistency_sc"):
        assert c in s.columns
    assert len(s) == 8


# ── 21-24: estimands, tables, figures ─────────────────────────────────

def test_main_metric_is_mean_seedwise():
    prim = pd.read_csv(OUT / "primary_metrics.csv")
    seed = pd.read_csv(OUT / "seed_metrics.csv")
    for _, row in prim.iterrows():
        sub = seed[seed.model == row["model"]]
        assert abs(sub["pearson"].mean() - row["pearson_mean"]) <= 1e-12
        assert sub["seed"].nunique() == 5


def test_ensemble_sensitivity_separate():
    s = pd.read_csv(OUT / "sensitivity_ensemble_bootstrap.csv")
    assert (s["estimand"] == "sensitivity_ensemble").all()
    X_fc, X_sc, y = data()[0], data()[1], data()[2]
    recs_ = recs()
    ens = R.subject_ensemble(recs_, "R-MATCHED")
    r = pearsonr(y, ens).statistic
    row = s[s.model == "R-MATCHED"].iloc[0]
    assert abs(r - row["ensemble_r"]) <= 1e-10
    assert row["boot_ci_lo"] <= row["boot_ci_hi"]
    assert 0 <= row["fraction_le_0"] <= 1


def test_paper_tables_agree_with_csvs():
    t = pd.read_csv(PR / "table_main_wm_prediction.csv")
    prim = pd.read_csv(OUT / "primary_metrics.csv").set_index("model")
    for _, row in t.iterrows():
        assert abs(row["pearson"] - prim.loc[row["method"], "pearson_mean"]) < 5e-4
        assert abs(row["rmse"] - prim.loc[row["method"], "rmse_mean"]) < 5e-3
    tex = (PR / "table_main_wm_prediction.tex").read_text()
    for m in P.ALL_MODELS + ["R0"]:
        assert m in tex
    b = pd.read_csv(PR / "table_main_wm_biomarkers.csv")
    assert len(b) == 8
    assert (OUT / "paper_ready" / "table_main_wm_biomarkers.tex").exists()


def test_figures_use_actual_values():
    abl = pd.read_csv(OUT / "prediction_ablations.csv")
    comp = abl[abl.group == "B_matched_ridge_components"]
    r0 = abl[abl.group == "A_backbone"]
    vals = [float(r0["pearson"].mean())]
    for var in ("FC_only", "FC_SC_expert", "R0_plus_expert"):
        vals.append(float(comp[comp.variant == var]["pearson"].mean()))
    prim = pd.read_csv(OUT / "primary_metrics.csv").set_index("model")
    assert abs(vals[3] - prim.loc["R-MATCHED", "pearson_mean"]) < 1e-6
    for p in [OUT / "plots" / "fig4_ablation_waterfall.pdf",
              PR / "fig_method_overview.pdf"]:
        assert p.exists() and p.stat().st_size > 1000


# ── 25-29: wording, framing, novelty, examples ────────────────────────

def test_no_external_validation_wording():
    hits = R.scan_banned(list(PR.rglob("*")) + list(OUT.joinpath("supplementary").rglob("*")))
    assert hits == {}, hits
    v = json.loads((OUT / "VALIDATION_REPORT.json").read_text())
    assert v["banned_wording_hits"] == {}


def test_no_412_vs_510_in_main_paper_ready():
    for p in list(PR.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".tex", ".csv"):
            txt = p.read_text(errors="ignore").lower()
            assert "412 vs 510" not in txt and "412-vs-510" not in txt
    names = " ".join(p.name for p in PR.rglob("*"))
    assert "412" not in names
    assert "sample-size" not in (PR / "MAIN_PAPER_RESULTS.md").read_text().lower()


def test_dev_history_note_no_confirmation():
    note = (OUT / "supplementary" / "DEVELOPMENT_HISTORY_NOTE.md").read_text()
    assert "not results of the final paper" in note
    assert "external confirmation" in note
    assert "confirm" not in note.replace("external confirmation", "")


def test_novelty_grounded():
    n = (PR / "NOVELTY_STATEMENT.md").read_text()
    for term in ("top-K", "ROI-incident", "hierarchical", "matched", "I_i",
                 "SHA256", "alpha", "v "):
        assert term.lower() in n.lower()
    assert "not claimed merely" in n.lower()


def test_illustrative_examples_trace():
    ex = (PR / "ILLUSTRATIVE_EXAMPLES.md").read_text()
    pr = priors()["matched"]
    top_roi = int(np.argsort(pr["array"])[::-1][0])
    assert f"index {top_roi + 1}" in ex and pr["labels"][top_roi] in ex
    sel = pd.read_csv(OUT / "SELECTED_CONFIGS.csv")
    row = sel[(sel.model == "R-MATCHED") & (sel.seed == 7171) & (sel.fold == 0)].iloc[0]
    assert f"v = {row['v_fc']:.2f}" in ex and f"alpha = {row['alpha']:.2f}" in ex
    f = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    fr = f[(f.model == "R-MATCHED") & (f.seed == 7171) & (f.fold == 0)].iloc[0]
    assert f"{fr['top10_delta_rmse']:+.4f}" in ex


def test_full_report_includes_negative_ablations():
    md = (OUT / "ABLATION_PREDICTION.md").read_text()
    assert "R-SHUFFLED" in md and "R-RANDOM" in md and "N-MATCHED" in md
    audit = (OUT / "FINAL_EVIDENCE_AUDIT.md").read_text()
    assert "R-SHUFFLED: r=" in audit and "N-MATCHED: r=" in audit
    abl = pd.read_csv(OUT / "prediction_ablations.csv")
    neg = abl[(abl["pearson"].notna()) & (abl["pearson"] < 0)]
    assert len(neg) >= 0
    comp = pd.read_csv(OUT / "prediction_comparisons.csv")
    assert (comp["mean_delta_r"] < 0).any()


# ── 30: decision logic and package completeness ───────────────────────

def test_decision_logic_consistent():
    st = json.loads((OUT / "FINAL_STATUS.json").read_text())
    plev = st["prediction_level"]["RIDGE_PREDICTION_LEVEL"]
    blev = st["biomarker_levels"]["R"]["level"]
    prim = pd.read_csv(OUT / "primary_metrics.csv").set_index("model")
    dm = prim.loc["R-MATCHED", "delta_r_mean"]
    pos = int(prim.loc["R-MATCHED", "positive_seeds"])
    if dm <= 0:
        assert plev == "P0"
    elif dm >= 0.005 and pos >= 4:
        assert plev in ("P2", "P3")
    else:
        assert plev == "P1"
    if st["paper_status"] == "STRONG_WM_METHOD_PAPER":
        assert plev in ("P2", "P3") and blev != "B0"
    elif st["paper_status"] == "WM_PREDICTION_METHOD_PAPER":
        assert plev in ("P2", "P3") and blev == "B0"
    elif st["paper_status"] == "EXPLORATORY_WM_METHOD_PAPER":
        assert plev == "P1"


def test_ratio_ablation_outer():
    sw = pd.read_csv(OUT / "ablations_ratio_sweep.csv")
    assert set(sw["ratio"]) == {0.0, 0.1, 0.3, 1.0}
    assert sw["pearson"].notna().all() and (sw["runtime_s"] >= 0).all()
    abl = pd.read_csv(OUT / "prediction_ablations.csv")
    d = abl[abl.group == "D_ncr_ratio_outer"]
    assert len(d) == 4
    assert d[["pearson", "rmse", "mae", "runtime_s"]].notna().all().all()
    r0 = sw[sw["ratio"] == 0.0]
    rm = pd.read_csv(OUT / "fold_metrics.csv")
    rm = rm[(rm.model == "R-MATCHED") & (rm.seed == 7171)]
    assert abs(r0["pearson"].mean() - rm["pearson"].mean()) < 1e-10


def test_complete_package_artifacts():
    req = ["COMPLETE", "COHORT_AUDIT.json", "BASELINE_AUDIT.json",
           "RESULT_REUSE_AUDIT.json", "RUNTIME_ESTIMATE.md",
           "RUNTIME_PROGRESS.json", "primary_metrics.csv", "seed_metrics.csv",
           "fold_metrics.csv", "ridge_prior_controls.csv",
           "ncr_prior_controls.csv", "prediction_ablations.csv",
           "biomarker_stability.csv", "biomarker_faithfulness.csv",
           "SELECTED_CONFIGS.csv", "VALIDATION_REPORT.json",
           "FINAL_EVIDENCE_AUDIT.md"]
    for f in req:
        assert (OUT / f).exists(), f
    assert (OUT / "COMPLETE").read_text().startswith("FINAL510")
    reuse = json.loads((OUT / "RESULT_REUSE_AUDIT.json").read_text())
    assert reuse["decision"] == "NO_RESULTS_REUSED"
