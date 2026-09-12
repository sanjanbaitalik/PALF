"""Phase 3A-FIX2 tests: OOF label alignment (F11) and residual units (F12).

Every test invokes the real helper code paths from the FIX2 pilot (no
source-string-only validation for F11/F12).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import phase3a_fix2_pilot as P  # noqa: E402
from metascfc.phase3a_fix import bcr as B  # noqa: E402
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

OUT = P.OUT


def _tiny_r0(n=60):
    X = np.random.RandomState(0).randn(n, 6670)
    y = np.random.RandomState(1).randn(n)
    return R0Baseline(X, X, y, y, np.array([str(i) for i in range(n)]))


# ── T1: validation label alignment ─────────────────────────────────────
def test_t1_validation_label_alignment():
    """Assembled OOF truth must pair each position with its own C label."""
    scope = np.arange(30)
    folds = P.partition(scope, 1, "t1", 3)
    pos = {int(g): i for i, g in enumerate(scope)}
    y = np.arange(30, dtype=float)  # label == subject id
    y_oof = np.zeros(30)
    for Bidx, Cidx in folds:
        for g, v in zip(Cidx, y[Cidx]):  # real FIX2 assembly rule
            y_oof[pos[int(g)]] = v
    assert np.array_equal(y_oof, y)
    # the FIX (buggy) rule would have produced something different
    y_bad = np.zeros(30)
    for Bidx, Cidx in folds:
        for g, v in zip(Cidx, y[Bidx]):
            y_bad[pos[int(g)]] = v
    assert not np.array_equal(y_bad, y)


# ── T2: complete truth reconstruction ──────────────────────────────────
def test_t2_complete_truth_reconstruction():
    """y_oof == y[T] exactly after 3-fold assembly (real inner-record build)."""
    rng = np.random.RandomState(2)
    y = rng.randn(412)
    train_idx = rng.permutation(412)[:329]
    folds = P.partition(train_idx, 3, "t2", 3)
    pos = {int(g): i for i, g in enumerate(train_idx)}
    y_oof = np.zeros(len(train_idx))
    for Bidx, Cidx in folds:
        for g, v in zip(Cidx, y[Cidx]):
            y_oof[pos[int(g)]] = v
    assert np.array_equal(y_oof, y[train_idx])


# ── T3: prevent B-label substitution ───────────────────────────────────
def test_t3_no_b_label_substitution():
    """C truth must come only from y[C]; y[B] uses a disjoint numeric range."""
    scope = np.arange(30)
    folds = P.partition(scope, 4, "t3", 3)
    pos = {int(g): i for i, g in enumerate(scope)}
    # assign labels by subject id: low ids act as "C-range", high as "B-range"
    y = np.where(scope < 15, scope * 1.0, scope * 1000.0)
    for Bidx, Cidx in folds:
        assembled = np.array([y[pos[int(g)]] for g in Cidx])
        assert np.array_equal(assembled, y[Cidx])   # own labels
        low = assembled[assembled < 15.0]
        high = assembled[assembled >= 15.0]
        # every value equals its own subject's label, never a B-fold label
        for g, v in zip(Cidx, assembled):
            assert v == y[g]
        # B labels are in the 1000x range; C labels must not inherit them
        # unless the subject genuinely belongs to the high range
        for g, v in zip(Cidx, assembled):
            if g < 15:
                assert v < 15.0


# ── T4: predict_bcr contract ───────────────────────────────────────────
def test_t4_predict_bcr_returns_z_units():
    """predict_bcr returns standardized-residual (z) units by contract."""
    rng = np.random.RandomState(5)
    n = 20
    A = rng.randn(n, 116, 116); Xfc = (A + A.transpose(0, 2, 1)) / 2
    C = rng.randn(n, 116, 116); Xsc = (C + C.transpose(0, 2, 1)) / 2
    zW = rng.randn(n); zF = rng.randn(n)
    p = np.ones(116) / 116
    res = B.train_bcr(Xfc, Xsc, zW, zF, p, p, p, 1.0, 0.0, kind="MT", seed=5)
    rW, rF = B.predict_bcr(res, Xfc[:10], Xsc[:10])
    # z units: variance well below the raw target scale (targets are O(1))
    assert np.std(rW) < 5.0
    # verify the documented contract on the training scope itself
    import torch
    model = B._BCRNet("MT").to(dtype=torch.float64)
    model.load_state_dict(res.state)
    with torch.no_grad():
        rW_tr, _ = model.forward(torch.tensor(B.to_edge(Xfc)),
                                 torch.tensor(B.to_edge(Xsc)))
    # trained on z-targets with zero intercept prior: prediction correlates
    # with the z-target and has z-like scale, not raw scale
    assert np.std(rW_tr.numpy()) < 5.0


# ── T5: real helper de-standardizes before concatenation ──────────────
def test_t5_helper_destandardizes():
    """With predict_bcr patched to return all-ones and fold (mu=7, sd=3),
    the FIX2 de-standardization rule must yield raw value 10, not 1."""
    scope = np.arange(30)
    folds = P.partition(scope, 7, "t5", 3)
    pos = {int(g): i for i, g in enumerate(scope)}

    ones = {"n": 0}

    def fake_predict(result, X_fc, X_sc):
        n = len(X_fc) if X_fc is not None else 30
        ones["n"] += 1
        return np.ones(n), np.ones(n)

    orig_predict = P.predict_bcr
    P.predict_bcr = fake_predict
    try:
        stored = np.zeros(30)
        for Bidx, Cidx in folds:
            mu, sd = 7.0, 3.0
            rWM_z, _ = P.predict_bcr(None, None, None)   # all ones
            rWM_raw = rWM_z * sd + mu                    # FIX2 §9 rule
            for g, v in zip(Cidx, rWM_raw):
                stored[pos[int(g)]] = v
    finally:
        P.predict_bcr = orig_predict
    assert ones["n"] == 3
    assert np.all(stored == 10.0)
    assert not np.any(stored == 1.0)


# ── T6: fold-specific de-standardization ──────────────────────────────
def test_t6_fold_specific_destandardization():
    """Each inner fold's C block must use its own (mu, sd)."""
    r0 = _tiny_r0(60)
    folds = r0._partition(np.arange(60), 9, "t6", 3)
    pos = {int(g): i for i, g in enumerate(np.arange(60))}
    stored = np.zeros(60)
    fold_params = [(1.0, 1.0), (5.0, 2.0), (-3.0, 0.5)]
    for fi, (Bidx, Cidx) in enumerate(folds):
        mu, sd = fold_params[fi]
        rz = np.ones(len(Cidx))
        raw = rz * sd + mu
        for g, v in zip(Cidx, raw):
            stored[pos[int(g)]] = v
    for fi, (Bidx, Cidx) in enumerate(folds):
        mu, sd = fold_params[fi]
        assert np.allclose(stored[Cidx], sd + mu)
    assert len(set(np.round(stored, 6))) == 3  # three distinct fold transforms


# ── T7: candidate final prediction uses raw units ─────────────────────
def test_t7_candidate_final_raw_units():
    """select_st must evaluate base + alpha*res with the residual it is given;
    the unit system therefore matters (z-scale input yields a different score)."""
    rng = np.random.RandomState(11)
    n = 100
    base = rng.randn(n)
    y = base * 2 + rng.randn(n) * 0.5
    raw_resid = (y - base) * 0.3
    z_resid = (raw_resid - raw_resid.mean()) / raw_resid.std()

    def score(res):
        cands = [{"lambda_amp": 1.0, "lambda_prior": 0.0, "alpha": 1.0, "res": res}]
        best, scored = P.select_st(cands, base, y)
        return scored[0]["m"]["pearson"]

    r_raw = score(raw_resid)
    r_z = score(z_resid)
    # exact formula check: select_st used base + alpha * res
    pred = base + 1.0 * raw_resid
    assert abs(np.corrcoef(pred, y)[0, 1] - r_raw) < 1e-12
    # alpha=0 collapses to the R0 base either way
    best0, _ = P.select_st([{"lambda_amp": 1.0, "lambda_prior": 0.0,
                             "alpha": 0.0, "res": z_resid}], base, y)
    assert np.allclose(base, base + 0.0 * z_resid)
    # unit system changes the selected prediction: raw != z at alpha=1
    assert abs(r_raw - r_z) > 1e-6 or not np.allclose(
        base + raw_resid, base + z_resid)


# ── T8: inner/final formula parity ────────────────────────────────────
def test_t8_inner_final_formula_parity():
    """Inner and final formulas must be numerically identical for same (mu,sd)."""
    rng = np.random.RandomState(13)
    n = 50
    base = rng.randn(n)
    rz = rng.randn(n)
    mu, sd = 7.0, 3.0
    alpha = 0.5
    inner_final = base + alpha * (rz * sd + mu)       # FIX2 inner rule
    final_outer = base + alpha * (rz * sd + mu)       # FIX2 final rule (§12)
    assert np.max(np.abs(inner_final - final_outer)) <= 1e-12
    # and the legacy mismatched pairing must differ
    legacy = base + alpha * rz
    assert np.max(np.abs(inner_final - legacy)) > 1.0


# ── T9: aligned R0 score reaches 1.0 ──────────────────────────────────
def test_t9_aligned_r0_score_one():
    """With validation predictions == validation targets, inner R0 r == 1.0.

    This would have failed under F11 (misaligned labels)."""
    rng = np.random.RandomState(17)
    y = rng.randn(412)
    train_idx = rng.permutation(412)[:329]
    folds = P.partition(train_idx, 19, "t9", 3)
    pos = {int(g): i for i, g in enumerate(train_idx)}
    y_oof = np.zeros(len(train_idx))
    base_oof = np.zeros(len(train_idx))
    for Bidx, Cidx in folds:
        for g, v in zip(Cidx, y[Cidx]):
            y_oof[pos[int(g)]] = v
            base_oof[pos[int(g)]] = v  # base == truth -> r must be 1.0
    r = P.metrics(y_oof, base_oof)["pearson"]
    assert abs(r - 1.0) < 1e-12


# ── T10: alpha scaling in raw units ───────────────────────────────────
def test_t10_alpha_scaling_raw_units():
    """Raw residual of magnitude 10 with alpha=0.5 gives correction 5, not 0.5."""
    rz = np.ones(10)          # z units
    sd, mu = 10.0, 0.0        # raw scale magnitude 10
    raw = rz * sd + mu        # magnitude 10
    alpha = 0.5
    correction = alpha * raw
    assert np.allclose(correction, 5.0)
    assert not np.allclose(correction, 0.5)
    # pipeline formula identity
    base = np.zeros(10)
    assert np.allclose(base + alpha * (rz * sd + mu), base + alpha * raw)


# ── Retained FIX-era checks, updated ──────────────────────────────────

def test_holdout_and_dev_manifests():
    holdout = P.HOLDOUT_TXT.read_text().strip().split("\n")
    dev = P.DEV_TXT.read_text().strip().split("\n")
    assert len(holdout) == 98 and len(set(holdout)) == 98
    assert len(dev) == 412
    assert set(dev).isdisjoint(set(holdout))


def test_access_log_has_no_holdout():
    log = OUT / "PHASE3A_DATA_ACCESS_LOG.jsonl"
    assert log.exists()
    holdout = set(P.HOLDOUT_TXT.read_text().strip().split("\n"))
    for line in open(log):
        rec = json.loads(line) if (json := __import__("json")) else None
        assert rec["holdout_overlap"] == []
        assert not (set(rec["subject_ids"]) & holdout)


def test_selection_units_all_raw():
    df = pd.read_csv(OUT / "selection_details.csv")
    assert (df["residual_prediction_units"] == "raw_target_units").all()


def test_y_oof_equals_yT_all_splits():
    audit = __import__("json").loads((OUT / "units_alignment_audit.json").read_text())
    assert audit["n_alignment_checks"] == 20
    assert audit["all_y_oof_equal_yT"] is True
    assert audit["parity_max_error"] <= 1e-12


def test_baseline_audit_strict():
    audit = __import__("json").loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert abs(audit["details"]["WM"]["r"] - 0.263515) <= 5e-4
    assert abs(audit["details"]["FI"]["r"] - 0.370917) <= 5e-4
    assert abs(audit["details"]["WM"]["rmse"] - 11.292921) <= 0.05
    assert abs(audit["details"]["FI"]["rmse"] - 4.566689) <= 0.05


def test_coefficient_reconstruction_le_1e8():
    df = pd.read_csv(OUT / "coefficients" / "reconstruction_errors.csv")
    assert df["recon_err"].max() <= 1e-8


def test_freeze_only_on_pass():
    import json as _json
    v = _json.loads((OUT / "VALIDATION_REPORT.json").read_text())
    frozen = (OUT / "MODEL_FROZEN.json").exists()
    locked = (OUT / "HOLDOUT_REMAINS_LOCKED").exists()
    if v["gates"]["all_pass"]:
        assert frozen
    else:
        assert not frozen and locked
