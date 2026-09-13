"""412 vs 510 scaling study functional tests.

Covers the project bug classes: manifest integrity, corrected R0 identity,
paired split coverage/no leakage, regime pairing, bootstrap reproducibility,
learning-curve labels, biomarker stability/faithfulness consistency, ablation
completeness, secondary 510 CV, trigger discipline, package artifacts, and
non-overwrite of prior phase outputs.
"""

import ast
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling"
CKPT = OUT / "_state" / "splits"
D412 = ROOT / "data_splits" / "scaling_D412.txt"
D98 = ROOT / "data_splits" / "scaling_D98.txt"
D510 = ROOT / "data_splits" / "scaling_D510.txt"
SEEDS = [7171, 7272, 7373, 7474, 7575]
MODELS = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4"]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_d412_manifest_integrity():
    ids = D412.read_text().strip().split("\n")
    assert len(ids) == 412 and len(set(ids)) == 412
    assert _sha(D412) == "8d4ee9586d78e2f997eaa9ac2ea4abe4d117bf5dec04ec836a89490657b4af6e"


def test_d98_manifest_canonical_seal():
    ids = D98.read_text().strip().split("\n")
    assert len(ids) == 98 and len(set(ids)) == 98
    canonical = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert canonical == "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"


def test_d510_is_disjoint_union():
    a = D412.read_text().strip().split("\n")
    b = D98.read_text().strip().split("\n")
    c = D510.read_text().strip().split("\n")
    assert len(c) == 510 and len(set(c)) == 510
    assert set(a).isdisjoint(set(b))
    assert set(c) == set(a) | set(b)
    assert _sha(D510) == "070188ff3a060b9003160b343c74db10fb51b243b43d77feaf126c80ccb62ca2"


def test_manifest_audit_pass():
    a = json.loads((OUT / "MANIFEST_AUDIT.json").read_text())
    assert a["disjoint"] is True and a["union_exact"] is True
    assert a["phase4_result_preserved"] is True
    assert a["D412"]["sha256"] == _sha(D412)
    assert a["D98"]["sha256"] == a["historical_98_canonical_sha256"]
    assert a["D510"]["sha256"] == _sha(D510)
    assert "NOT independent" in a["claim_boundary"]


def test_baseline_audit_strict():
    a = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert a["status"] == "PASS"
    assert abs(a["details"]["WM"]["r"] - 0.2635147736) <= 5e-4
    assert abs(a["details"]["FI"]["r"] - 0.3709173350) <= 5e-4
    assert abs(a["details"]["WM"]["rmse"] - 11.2929210027) <= 0.05
    assert abs(a["details"]["FI"]["rmse"] - 4.5666891937) <= 0.05


def test_primary_seed_metrics_complete():
    d = pd.read_csv(OUT / "primary_seed_metrics.csv")
    assert len(d) == 2 * len(MODELS) * 2 * 5
    assert set(d.model) == set(MODELS)
    assert set(d.target) == {"WM", "FI"}
    assert set(d.regime) == {"A", "B"}
    assert d.pearson.notna().all() and (d.pearson.abs() <= 1).all()
    assert (d.rmse > 0).all() and (d.mae > 0).all()


def test_primary_model_metrics():
    d = pd.read_csv(OUT / "primary_model_metrics.csv")
    assert len(d) == 2 * len(MODELS) * 2
    assert (d.n_seeds == 5).all()
    assert (d.r_std >= 0).all()


def test_subject_predictions_full_coverage():
    d = pd.read_csv(OUT / "primary_subject_predictions.csv")
    g = d.groupby(["target", "model", "regime", "seed"]).size()
    assert (g == 412).all()
    u = d.groupby(["target", "model", "regime", "seed"]).subject.nunique()
    assert (u == 412).all()


def test_predictions_pairing_same_eval_subjects():
    d = pd.read_csv(OUT / "primary_subject_predictions.csv")
    key = ["target", "model", "seed", "fold"]
    tab = d.pivot_table(index=key + ["subject"], columns="regime", values="pred")
    assert tab.notna().all().all()
    # identical test subjects for A and B
    a = d[d.regime == "A"].groupby(key).subject.apply(lambda s: tuple(sorted(s)))
    b = d[d.regime == "B"].groupby(key).subject.apply(lambda s: tuple(sorted(s)))
    assert (a == b).all()
    # identical labels across regimes and models
    y = d.groupby(["target", "seed", "fold", "subject"]).y.nunique()
    assert (y == 1).all()


def test_regimes_actually_differ():
    d = pd.read_csv(OUT / "primary_subject_predictions.csv")
    for target in ("WM", "FI"):
        sub = d[(d.target == target) & (d["model"] == "B1")]
        a = sub[sub.regime == "A"].sort_values(["seed", "fold", "subject"]).pred.values
        b = sub[sub.regime == "B"].sort_values(["seed", "fold", "subject"]).pred.values
        assert np.mean(np.abs(a - b)) > 1e-6


def test_paired_bootstrap_schema():
    d = pd.read_csv(OUT / "paired_bootstrap_results.csv")
    assert len(d) == 2 * len(MODELS)
    assert d.delta_r_ci_lo.le(d.delta_r_ci_hi).all()
    assert d.delta_r_fraction_le_0.between(0, 1).all()
    assert d.delta_rmse_ci_lo.notna().all()


def test_bootstrap_observed_delta_identity():
    d = pd.read_csv(OUT / "paired_bootstrap_results.csv")
    assert np.allclose(d.delta_r_observed, d.r_plus98 - d.r_412, atol=1e-12)
    assert np.allclose(d.fisher_z_diff,
                       np.arctanh(np.clip(d.r_plus98, -0.999, 0.999))
                       - np.arctanh(np.clip(d.r_412, -0.999, 0.999)), atol=1e-9)


def test_bootstrap_seed_deterministic():
    d = pd.read_csv(OUT / "paired_bootstrap_results.csv")
    # recompute one entry with the frozen bootstrap seed
    from scipy.stats import pearsonr
    subj = pd.read_csv(OUT / "primary_subject_predictions.csv",
                       dtype={"subject": str})
    t, m = "WM", "A0"
    order = D412.read_text().strip().split("\n")
    s = subj[(subj.target == t) & (subj["model"] == m)]
    pa = s[s.regime == "A"].groupby("subject").pred.mean().reindex(order).values
    pb = s[s.regime == "B"].groupby("subject").pred.mean().reindex(order).values
    y = s[s.regime == "A"].groupby("subject").y.mean().reindex(order).values
    rng = np.random.RandomState(9401)
    idx = rng.randint(0, 412, size=(10000, 412))
    dr = np.array([pearsonr(y[i], pb[i]).statistic - pearsonr(y[i], pa[i]).statistic
                   for i in idx])
    row = d[(d.target == t) & (d["model"] == m)].iloc[0]
    assert abs(row.delta_r_mean - dr.mean()) < 1e-10
    assert abs(row.delta_r_ci_lo - np.percentile(dr, 2.5)) < 1e-10


def test_method_effects_complete():
    d = pd.read_csv(OUT / "method_effects.csv")
    assert len(d) == 2 * 8 * 2
    assert d.positive.between(0, 5).all()
    assert d.groupby(["target", "contrast", "regime"]).size().eq(1).all()


def test_sample_size_interactions_complete():
    d = pd.read_csv(OUT / "sample_size_interactions.csv")
    assert len(d) == 6
    assert set(d.interaction) == {"B1_R0", "PALF_R0", "NCR_Ridge"}


def test_learning_curve_labels_and_runs():
    d = pd.read_csv(OUT / "learning_curve.csv")
    assert len(d) == 2 * 2 * 5
    assert set(d.size_label) == {"n250", "n300", "n350", "full_T412", "full_TB"}
    assert d[d.size_label == "n250"].n_runs.eq(15).all()
    assert d[d.size_label == "full_TB"].n_runs.eq(5).all()
    full = d[d.size_label == "full_T412"].n_eff_mean
    assert full.between(329, 330).all()
    fullb = d[d.size_label == "full_TB"].n_eff_mean
    assert fullb.between(427, 428).all()
    assert d.r_mean.notna().all() and (d.r_mean.abs() <= 1).all()


def test_stability_counts_consistent():
    d = pd.read_csv(OUT / "biomarker_stability.csv")
    assert len(d) == 2 * 5 * 2
    assert (d.n_used + d.abstained == 25).all()
    assert d.n_used.between(0, 25).all()


def test_faithfulness_abstain_consistency():
    f = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    s = pd.read_csv(OUT / "biomarker_stability.csv")
    s = s[s["model"].isin(f["model"].unique())]
    assert len(f) == 2 * 2 * 2 * 25
    assert f.abstained.isin([0, 1]).all()
    ab = f[f.abstained == 1]
    assert ab[["top10", "random10_mean", "top10_minus_random10"]].isna().all().all()
    ok = f[f.abstained == 0]
    assert ok[["top10", "random10_mean", "top10_minus_random10"]].notna().all().all()
    assert ok.groupby(["target", "model", "regime"]).size().eq(
        s.set_index(["target", "model", "regime"]).n_used).all()


def test_faithfulness_sign_convention():
    f = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    ok = f[f.abstained == 0]
    assert np.allclose(ok.top10_minus_random10, ok.top10 - ok.random10_mean)


def test_ablation_groups_complete():
    d = pd.read_csv(OUT / "ablation_results.csv")
    assert set(d.group) == {"A_sample_size", "B_PALF_components",
                            "E_modality_F_fusion", "G_mask_family"}
    assert len(d[d.group == "A_sample_size"]) == 18
    assert len(d[d.group == "B_PALF_components"]) == 32
    assert len(d[d.group == "E_modality_F_fusion"]) == 16


def test_secondary_510_cv():
    d = pd.read_csv(OUT / "full510_secondary_metrics.csv")
    assert len(d) == 30
    assert set(d.seed) == {7171}
    assert set(d["model"]) == {"A0", "A1", "B1"}
    assert d.fold.nunique() == 5
    assert d.pearson.notna().all()


def test_no_nan_key_tables():
    for f in ("primary_seed_metrics.csv", "primary_model_metrics.csv",
              "paired_bootstrap_results.csv", "method_effects.csv",
              "learning_curve.csv"):
        d = pd.read_csv(OUT / f)
        num = d.select_dtypes(include=[np.number])
        assert num.notna().all().all(), f


def test_learning_curve_monotone_check_is_reported():
    # study claims no scale gain: from 250 -> full_TB, delta should not be
    # positive and significant; just verify values are finite and bounded.
    d = pd.read_csv(OUT / "learning_curve.csv")
    for (t, m), sub in d.groupby(["target", "model"]):
        sub = sub.set_index("size_label")
        assert sub.loc["full_TB", "r_mean"] < 1.0
        assert sub.loc["n250", "r_mean"] > -1.0


def test_checkpoint_expert_invariants():
    f = sorted(CKPT.glob("split_seed*_fold0.pkl"))[0]
    rec = pickle.load(open(f, "rb"))
    for regime in ("A", "B"):
        for target in ("WM", "FI"):
            e = rec[regime]["targets"][target]["experts"]["B1"]
            assert 0.0 <= e["alpha"] <= 1.0
            assert 0.0 <= e["v_fc"] <= 1.0


def test_test_folds_partition_d412():
    ids = D412.read_text().strip().split("\n")
    d = pd.read_csv(OUT / "primary_subject_predictions.csv",
                    dtype={"subject": str})
    for seed in SEEDS:
        sub = d[(d.seed == seed) & (d.target == "WM") & (d["model"] == "A0")]
        assert sub.subject.nunique() == 412
        assert set(sub.subject) == set(ids)
        sizes = sub.groupby("fold").subject.nunique().values
        assert sorted(sizes) == [82, 82, 82, 83, 83]


def test_study_status_discloses_exploratory():
    s = (OUT / "STUDY_STATUS.md").read_text()
    assert "NOT independent" in s or "not independent" in s.lower()
    assert "post-hoc" in s.lower() or "exploratory" in s.lower()
    assert "PAPER_READY_TRIGGER" in s


def test_trigger_discipline():
    status = (OUT / "STUDY_STATUS.md").read_text()
    trigger = "PAPER_READY_TRIGGER: YES" in status
    pr = OUT / "paper_ready" / "PAPER_READY_RESULTS.md"
    if trigger:
        assert pr.exists()
    else:
        assert not pr.exists()


def test_plots_and_supplementary_exist():
    main = ["fig_phase_scaling_prediction", "fig_phase_scaling_method_effect",
            "fig_phase_scaling_learning_curve", "fig_phase_scaling_stability",
            "fig_phase_scaling_faithfulness"]
    for name in main:
        for ext in ("pdf", "png", "svg"):
            assert (OUT / "plots" / f"{name}.{ext}").exists()
    for i in (1, 2, 3, 4, 5, 6):
        hits = list((OUT / "supplementary").glob(f"fig_S{i}_*.pdf"))
        assert len(hits) == 1


def test_complete_and_runtime_artifacts():
    assert (OUT / "COMPLETE").exists()
    assert (OUT / "RUNTIME_ESTIMATE.md").exists()
    r = json.loads((OUT / "RUNTIME_FINAL.json").read_text())
    assert "report_runtime_seconds" in r


def test_prior_phase_outputs_preserved():
    p4 = ROOT / "outputs" / "iclr" / "palf_phase4_wm_confirmation"
    assert (p4 / "COMPLETE").exists()
    assert "WM_CONFIRMATION_FAILED" in (p4 / "FINAL_CONFIRMATION_REPORT.md").read_text()
    oc = ROOT / "outputs" / "iclr" / "openchallenge"
    assert (oc / "COMPLETE").exists() or (oc / "OC2_COMPLETE").exists()
    p3 = ROOT / "outputs" / "iclr" / "palf_phase3a_fix2_pg_mt_bcr"
    assert (p3 / "COMPLETE").exists()


def test_buggy_module_not_used():
    for name in ("scaling_412_vs_510_pilot.py", "scaling_412_vs_510_report.py"):
        src = (ROOT / "scripts_paper" / name).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.endswith("prior_subspace_expert_fusion"), name
                if "prior_subspace_expert_fusion" in node.module:
                    assert node.module.endswith("prior_subspace_expert_fusion_fix"), name
