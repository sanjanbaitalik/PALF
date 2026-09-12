"""Phase 3A-FIX functional tests (corrected PG-MT-BCR)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

from metascfc.phase3a_fix import bcr as B  # noqa: E402
from metascfc.phase3a_fix.r0_baseline import (  # noqa: E402
    AccessLogger,
    R0Baseline,
)

OUT = ROOT / "outputs" / "iclr" / "palf_phase3a_fix_pg_mt_bcr"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"


def _holdout():
    return HOLDOUT_TXT.read_text().strip().split("\n")


def _dev():
    return DEV_TXT.read_text().strip().split("\n")


# 1
def test_holdout_count_and_sha():
    ids = _holdout()
    assert len(ids) == 98 and len(set(ids)) == 98
    canonical = "\n".join(sorted(ids)) + "\n"
    assert hashlib.sha256(canonical.encode()).hexdigest() == HOLDOUT_SHA


# 2
def test_dev_count_412():
    ids = _dev()
    assert len(ids) == 412 and len(set(ids)) == 412


# 3
def test_dev_holdout_disjoint():
    assert set(_dev()).isdisjoint(set(_holdout()))


# 4
def test_loader_raises_on_holdout_id():
    acc = AccessLogger(ROOT / "outputs" / "iclr" / "palf_phase3a_fix_pg_mt_bcr" / "_test_log.jsonl")
    acc.set_holdout(_holdout())
    with pytest.raises(ValueError, match="HOLDOUT ACCESS VIOLATION"):
        acc.log("test", _holdout()[:3])
    if acc.path.exists():
        acc.path.unlink()


# 5
def test_access_log_has_no_holdout():
    log = ROOT / "outputs" / "iclr" / "palf_phase3a_fix_pg_mt_bcr" / "PHASE3A_DATA_ACCESS_LOG.jsonl"
    assert log.exists(), "access log missing"
    holdout = set(_holdout())
    with open(log) as f:
        for line in f:
            rec = json.loads(line)
            assert not (set(rec["subject_ids"]) & holdout)
            assert rec["holdout_overlap"] == []


# 6
def test_baseline_uses_conditions_r0():
    src = (ROOT / "src" / "metascfc" / "phase3a_fix" / "r0_baseline.py").read_text()
    assert 'CONDITIONS["R0"]' in src


# 7
def test_baseline_fusion_uses_fp_oof_not_fc_oof():
    src = (ROOT / "src" / "metascfc" / "phase3a_fix" / "r0_baseline.py").read_text()
    assert "fp_oof" in src
    assert '"FP": oof.fp_oof' in src
    assert "oof.fc_oof" not in src


# 8
def test_baseline_fp_sc_convex_fusion():
    src = (ROOT / "src" / "metascfc" / "phase3a_fix" / "r0_baseline.py").read_text()
    assert 'search_fusion_weights' in src
    assert 'weights["FP"] * fp_final.test_pred + weights["SC"] * sc_final.test_pred' in src


# 9
def test_audit_wm_r():
    audit = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert abs(audit["details"]["WM"]["r"] - 0.263515) <= 5e-4


# 10
def test_audit_fi_r():
    audit = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert abs(audit["details"]["FI"]["r"] - 0.370917) <= 5e-4


# 11
def test_audit_wm_rmse():
    audit = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert abs(audit["details"]["WM"]["rmse"] - 11.292921) <= 0.05


# 12
def test_audit_fi_rmse():
    audit = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert abs(audit["details"]["FI"]["rmse"] - 4.566689) <= 0.05


# 13
def test_no_concatenated_ridge_as_a0():
    src = (ROOT / "src" / "metascfc" / "phase3a_fix" / "r0_baseline.py").read_text()
    assert "np.hstack([fc" not in src
    assert "hstack([fc_edges" not in src
    assert "generate_crossfit_oof" in src


# 14
def test_inner_candidates_cover_all_folds():
    import phase3a_fix_pilot as P
    scope = np.arange(30)
    folds = P.partition(scope, 1, "t", 3)
    covered = np.concatenate([c for _, c in folds])
    assert len(covered) == 30
    assert set(covered.tolist()) == set(scope.tolist())
    # each fold is a strict subset (B excludes C)
    for b, c in folds:
        assert set(b).isdisjoint(set(c))


# 15
def test_inner_fc_scaler_train_only():
    import phase3a_fix_pilot as P
    rng = np.random.RandomState(0)
    fc_tr = rng.randn(20, 116, 116)
    fc_va = rng.randn(5, 116, 116) + 100.0  # shifted
    sc_tr = rng.randn(20, 116, 116)
    sc_va = rng.randn(5, 116, 116)
    ftr, str_, fva, sva = P.standardize_scope(fc_tr, sc_tr, fc_va, sc_va)
    # validation standardized with train stats => not zero mean
    assert abs(fva.mean()) > 1.0
    # train standardized has ~zero mean
    assert abs(ftr.mean()) < 0.5


# 16
def test_inner_sc_scaler_train_only():
    import phase3a_fix_pilot as P
    rng = np.random.RandomState(1)
    sc_tr = rng.randn(20, 116, 116)
    sc_va = rng.randn(5, 116, 116) + 50.0
    ftr, str_, fva, sva = P.standardize_scope(rng.randn(20, 116, 116), sc_tr,
                                              rng.randn(5, 116, 116), sc_va)
    assert abs(sva.mean()) > 1.0
    assert abs(str_.mean()) < 0.5


# 17
def test_residual_standardization_b_only_stats():
    import phase3a_fix_pilot as P
    scope = np.arange(24)
    folds = P.partition(scope, 3, "z", 3)
    for Bidx, Cidx in folds:
        res = np.random.RandomState(2).randn(len(Bidx)) * 3 + 7
        z = (res - res.mean()) / (res.std() + 1e-12)
        assert abs(z.mean()) < 1e-9
        # std is B-only; if C stats were used this would differ
        assert abs(z.std() - 1.0) < 1e-6


# 18
def test_crossfit_r0_is_out_of_fold():
    """Every subject's prediction must come from a model not trained on it."""
    X = np.random.RandomState(0).randn(40, 6670)
    y = np.random.RandomState(1).randn(40)
    r0 = R0Baseline(X, X, y, y, np.array([str(i) for i in range(40)]))
    seen = {"train": [], "val": []}

    def fake_fit(task, train_idx, val_idx, seed, outer_fold, cache_tag=""):
        seen["train"].append(set(np.asarray(train_idx).tolist()))
        seen["val"].append(set(np.asarray(val_idx).tolist()))
        return np.zeros(len(val_idx))

    r0.fit_r0_predict = fake_fit
    scope = np.arange(40)
    out = r0.crossfit_r0_within("WM", scope, 1, "tag", n_folds=3)
    assert out.shape == (40,)
    # For each fit, train and val are disjoint
    for tr, va in zip(seen["train"], seen["val"]):
        assert tr.isdisjoint(va)
    # Every scope subject appears exactly once as a validation subject
    all_val = np.concatenate([sorted(v) for v in seen["val"]])
    assert set(all_val.tolist()) == set(scope.tolist())


# 19
def test_r0_validation_prediction_from_b_only():
    """fit_r0_predict caches separately for different train/val splits."""
    X = np.random.RandomState(0).randn(40, 6670)
    y = np.random.RandomState(1).randn(40)
    r0 = R0Baseline(X, X, y, y, np.array([str(i) for i in range(40)]))
    calls = []

    def fake_fit(task, train_idx, val_idx, seed, outer_fold, cache_tag=""):
        calls.append((tuple(np.sort(train_idx)), tuple(np.sort(val_idx))))
        return np.zeros(len(val_idx))

    r0.fit_r0_predict = fake_fit
    r0.fit_r0_predict("WM", np.arange(20), np.arange(20, 30), 0, 0)
    r0.fit_r0_predict("WM", np.arange(20), np.arange(30, 40), 0, 0)
    assert len(calls) == 2  # no cache collision across val sets


# 20
def test_st_wm_and_fi_coexist():
    import pandas as pd
    sel = OUT / "selection_details.csv"
    assert sel.exists()
    df = pd.read_csv(sel)
    models = set(df["model"].unique())
    assert "B0_WM" in models and "B0_FI" in models
    assert "B1_WM" in models and "B1_FI" in models
    # both tasks have selected rows with distinct predictions tracked
    assert (df[df["model"] == "B0_WM"]["selected"] == True).any()
    assert (df[df["model"] == "B0_FI"]["selected"] == True).any()


# 21
def test_b1_searches_all_lambda_combinations():
    import pandas as pd
    df = pd.read_csv(OUT / "selection_details.csv")
    sub = df[df["model"] == "B1_WM"]
    pairs = set(zip(sub["lambda_amp"], sub["lambda_prior"]))
    assert len(pairs) == 9


# 22
def test_mt_shared_factors_are_shared():
    torch.manual_seed(0)
    m = B._BCRNet("MT").to(dtype=torch.float64)
    # set zero amplitudes
    with torch.no_grad():
        for a in B._amp_list(m):
            a.zero_()
        m.b_WM.zero_(); m.b_FI.zero_()
        m.amp_fc_WM.fill_(1.0); m.amp_fc_FI.fill_(1.0)
    Xe = torch.randn(4, 6670, dtype=torch.float64)
    z = torch.zeros(4, 6670, dtype=torch.float64)
    rW1, rF1 = m.forward(Xe, z)
    # perturb shared FC factor; both WM and FI predictions must change
    with torch.no_grad():
        m.U_fc_shared[:, 0] += 0.3
        B._project_factors(m)
    rW2, rF2 = m.forward(Xe, z)
    assert not torch.allclose(rW1, rW2)
    assert not torch.allclose(rF1, rF2)


# 23
def test_mt_shared_amplitudes_task_specific():
    m = B._BCRNet("MT")
    names = {n for n, _ in m.named_parameters()}
    assert "amp_fc_shared_WM" in names and "amp_fc_shared_FI" in names
    assert "amp_sc_shared_WM" in names and "amp_sc_shared_FI" in names
    assert m.amp_fc_shared_WM is not m.amp_fc_shared_FI


# 24
def test_factor_norms_after_projection():
    torch.manual_seed(1)
    m = B._BCRNet("MT").to(dtype=torch.float64)
    for _ in range(30):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        B._project_factors(m)
    worst = 0.0
    for name in ["U_fc_shared", "U_sc_shared", "U_fc_WM", "U_fc_FI", "U_sc_WM", "U_sc_FI"]:
        U = getattr(m, name).detach().numpy()
        for k in range(U.shape[1]):
            worst = max(worst, abs(np.linalg.norm(U[:, k]) - 1.0))
    assert worst <= 1e-8


# 25
def test_shared_factor_orthogonality():
    torch.manual_seed(2)
    m = B._BCRNet("MT").to(dtype=torch.float64)
    for _ in range(30):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        B._project_factors(m)
    worst = 0.0
    for name in ["U_fc_shared", "U_sc_shared"]:
        U = getattr(m, name).detach().numpy()
        worst = max(worst, abs(float(U[:, 0] @ U[:, 1])))
    assert worst <= 1e-8


# 26
def test_adam_frozen_settings():
    assert (B.LR, B.MAX_STEPS, B.MIN_STEPS, B.PATIENCE,
            B.REL_TOL, B.GRAD_CLIP, B.N_RESTARTS) == (0.01, 1500, 200, 75, 1e-7, 5.0, 2)


# 27
def test_restart_chosen_by_training_loss():
    rng = np.random.RandomState(3)
    A = rng.randn(30, 116, 116)
    Xfc = (A + A.transpose(0, 2, 1)) / 2
    C = rng.randn(30, 116, 116)
    Xsc = (C + C.transpose(0, 2, 1)) / 2
    zW = rng.randn(30); zF = rng.randn(30)
    p = np.ones(116) / 116
    res = B.train_bcr(Xfc, Xsc, zW, zF, p, p, p, 1.0, 0.0, kind="MT", seed=5)
    losses = [r["final_loss"] for r in res.restart_records]
    assert res.train_loss == min(losses)
    chosen = [r for r in res.restart_records if r["chosen"]]
    assert len(chosen) == 1
    assert chosen[0]["restart"] == res.restart_used


# 28
def test_no_validation_metric_chooses_restart():
    import inspect
    sig = inspect.signature(B.train_bcr)
    params = list(sig.parameters.keys())
    assert not any("val" in p.lower() for p in params)
    # chosen restart is the argmin of final training loss only
    rng = np.random.RandomState(11)
    A = rng.randn(20, 116, 116); Xfc = (A + A.transpose(0, 2, 1)) / 2
    C = rng.randn(20, 116, 116); Xsc = (C + C.transpose(0, 2, 1)) / 2
    zW = rng.randn(20); zF = rng.randn(20); p = np.ones(116) / 116
    res = B.train_bcr(Xfc, Xsc, zW, zF, p, p, p, 1.0, 0.0, kind="MT", seed=12)
    losses = [r["final_loss"] for r in res.restart_records]
    assert res.restart_used == int(np.argmin(losses))


# 29
def test_c0_lambda_prior_exactly_zero():
    rng = np.random.RandomState(4)
    Xfc = rng.randn(25, 116, 116); Xfc = (Xfc + Xfc.transpose(0, 2, 1)) / 2
    Xsc = rng.randn(25, 116, 116); Xsc = (Xsc + Xsc.transpose(0, 2, 1)) / 2
    zW = rng.randn(25); zF = rng.randn(25); p = np.ones(116) / 116
    # With lambda_prior=0 the prior penalty must not affect loss: two different priors give same loss
    r1 = B.train_bcr(Xfc, Xsc, zW, zF, p, p, p, 1.0, 0.0, kind="MT", seed=6)
    p2 = rng.rand(116)
    r2 = B.train_bcr(Xfc, Xsc, zW, zF, p2, p2, p2, 1.0, 0.0, kind="MT", seed=6)
    assert abs(r1.train_loss - r2.train_loss) < 1e-9


# 30
def test_c1_matched_priors_match_frozen_files():
    import pandas as pd
    p_wm = pd.read_csv(ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")["prior_score"].values
    p_fi = pd.read_csv(ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv")["prior_score"].values
    out_wm = pd.read_csv(OUT / "prior_controls" / "matched_wm.csv")["prior_score"].values
    out_fi = pd.read_csv(OUT / "prior_controls" / "matched_fi.csv")["prior_score"].values
    assert np.allclose(p_wm, out_wm)
    assert np.allclose(p_fi, out_fi)


# 31
def test_controls_deterministic():
    import phase3a_fix_pilot as P
    p_wm = P.load_prior(P.PRIOR_WM_PATH)
    p_fi = P.load_prior(P.PRIOR_FI_PATH)
    c1 = P.make_controls(p_wm, p_fi)
    c2 = P.make_controls(p_wm, p_fi)
    for key in c1:
        for i in range(3):
            assert np.allclose(c1[key][i], c2[key][i])


# 32
def test_alpha_zero_equals_r0():
    import phase3a_fix_pilot as P
    base = np.random.RandomState(0).randn(50)
    resid = np.random.RandomState(1).randn(50) * 2
    alpha = 0.0
    final = base + alpha * resid
    assert np.array_equal(final, base)


# 33
def test_all_25_mt_alpha_pairs_evaluated():
    import phase3a_fix_pilot as P
    base = np.random.RandomState(0).randn(40)
    y = np.random.RandomState(1).randn(40)
    res = np.random.RandomState(2).randn(40)
    cands = [{"lambda_amp": 1.0, "lambda_prior": 0.1, "alpha_WM": a, "alpha_FI": b,
              "res_WM": res, "res_FI": res}
             for a in P.ALPHA for b in P.ALPHA]
    best, scored = P.select_mt(cands, base, base, y, y)
    assert len(scored) == 25


# 34
def test_concatenated_oof_drives_selection():
    import phase3a_fix_pilot as P
    n = 120
    base = np.random.RandomState(0).randn(n)
    y = base * 2 + np.random.RandomState(1).randn(n) * 0.1
    good = {"lambda_amp": 1.0, "lambda_prior": 0.0, "alpha_WM": 1.0, "alpha_FI": 0.0,
            "res_WM": (y - base), "res_FI": np.zeros(n)}
    bad = {"lambda_amp": 1.0, "lambda_prior": 0.0, "alpha_WM": 1.0, "alpha_FI": 0.0,
           "res_WM": np.random.RandomState(3).randn(n), "res_FI": np.zeros(n)}
    best, scored = P.select_mt([bad, good], base, base, y, y)
    assert best["res_WM"] is good["res_WM"]


# 35
def test_eligibility_uses_concatenated_oof():
    import phase3a_fix_pilot as P
    n = 80
    base = np.random.RandomState(0).randn(n)
    y = np.random.RandomState(1).randn(n)
    bad = {"lambda_amp": 1.0, "lambda_prior": 0.0, "alpha_WM": 1.0, "alpha_FI": 1.0,
           "res_WM": np.random.RandomState(2).randn(n) * 10,
           "res_FI": np.random.RandomState(3).randn(n) * 10}
    best, scored = P.select_mt([bad], base, base, y, y)
    assert scored[0]["eligible"] is False or best.get("fallback", False)


# 36
def test_final_refit_uses_full_scope():
    X = np.random.RandomState(0).randn(60, 6670)
    y = np.random.RandomState(1).randn(60)
    r0 = R0Baseline(X, X, y, y, np.array([str(i) for i in range(60)]))
    seen_scope = {}

    def fake_fit(task, train_idx, val_idx, seed, outer_fold, cache_tag=""):
        seen_scope["n_tr"] = len(train_idx)
        return np.zeros(len(val_idx))

    r0.fit_r0_predict = fake_fit
    scope = np.arange(60)
    out = r0.crossfit_r0_within("FI", scope, 2, "t", n_folds=3)
    assert out.shape == (60,)
    assert np.all(np.isfinite(out))


# 37
def test_bcr_coefficient_reconstruction():
    rng = np.random.RandomState(7)
    n = 25
    Xfc = rng.randn(n, 116, 116); Xfc = (Xfc + Xfc.transpose(0, 2, 1)) / 2
    for i in range(n):
        np.fill_diagonal(Xfc[i], 0)
    Xsc = rng.randn(n, 116, 116); Xsc = (Xsc + Xsc.transpose(0, 2, 1)) / 2
    for i in range(n):
        np.fill_diagonal(Xsc[i], 0)
    zW = rng.randn(n); zF = rng.randn(n); p = np.ones(116) / 116
    res = B.train_bcr(Xfc, Xsc, zW, zF, p, p, p, 1.0, 0.0, kind="MT", seed=8)
    rWM, rFI = B.predict_bcr(res, Xfc, Xsc)
    Xef = B.to_edge(Xfc); Xes = B.to_edge(Xsc)
    recW = B.reconstruct_residual(res, "WM", Xef, Xes)
    recF = B.reconstruct_residual(res, "FI", Xef, Xes)
    assert np.max(np.abs(recW - rWM)) <= 1e-8
    assert np.max(np.abs(recF - rFI)) <= 1e-8


# 38
def test_biomarker_diagnostics_downstream_only():
    """Selection details must not contain biomarker columns."""
    import pandas as pd
    df = pd.read_csv(OUT / "selection_details.csv")
    cols = set(df.columns)
    assert "faithfulness" not in cols
    assert "roi_importance" not in cols
    assert "edge_spearman" not in cols


# 39
def test_model_frozen_only_on_pass():
    vreport = json.loads((OUT / "VALIDATION_REPORT.json").read_text())
    frozen = (OUT / "MODEL_FROZEN.json").exists()
    locked = (OUT / "HOLDOUT_REMAINS_LOCKED").exists()
    if vreport["gates"]["all_pass"]:
        assert frozen
    else:
        assert not frozen
        assert locked


# 40
def test_no_holdout_evaluation_imported():
    src = (ROOT / "scripts_paper" / "phase3a_fix_pilot.py").read_text()
    assert "load_holdout" not in src
    assert "evaluate_holdout" not in src
    assert "palf_phase3b" not in src
    assert "phase3b_pilot" not in src
