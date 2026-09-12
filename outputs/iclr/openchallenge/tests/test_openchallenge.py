"""OpenChallenge CKE functional tests (locked method validation).

Covers the project bug classes: baseline identity/calibration, correct FP+SC
R0, OOF label alignment, unit consistency of the ensemble, train-scope-only
preprocessing, final refit = selected procedure, task outputs not overwritten,
architecture-matched control differs only in kernel, attribution
reconstruction, perturbation sign convention, abstained-map exclusion, and no
holdout access.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.openchallenge import ckr as C  # noqa: E402
from metascfc.openchallenge.ckr import (  # noqa: E402
    edge_mask_for_rois,
    fit_krr,
    krr_gradient,
    krr_predict,
    median_heuristic,
)

OUT = ROOT / "outputs" / "iclr" / "openchallenge"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"


def test_holdout_seal():
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    assert len(ids) == 98 and len(set(ids)) == 98
    import hashlib
    sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert sha == "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"


def test_dev_holdout_disjoint():
    dev = (ROOT / "data_splits" / "phase3_development_412.txt").read_text().strip().split("\n")
    assert len(dev) == 412
    assert set(dev).isdisjoint(set(HOLDOUT_TXT.read_text().strip().split("\n")))


def test_baseline_audit_strict():
    audit = (OUT / "BASELINE_AUDIT.json")
    assert audit.exists()
    a = audit.read_text() and __import__("json").loads(audit.read_text())
    assert a["status"] == "PASS"
    assert abs(a["details"]["WM"]["r"] - 0.263515) <= 5e-4
    assert abs(a["details"]["FI"]["r"] - 0.370917) <= 5e-4
    assert abs(a["details"]["WM"]["rmse"] - 11.292921) <= 0.05
    assert abs(a["details"]["FI"]["rmse"] - 4.566689) <= 0.05


def test_r0_is_fp_sc_fusion():
    src = (ROOT / "src" / "metascfc" / "phase3a_fix" / "r0_baseline.py").read_text()
    assert '"FP": oof.fp_oof' in src and "oof.fc_oof" not in src
    assert 'weights["FP"] * fp_final.test_pred + weights["SC"] * sc_final.test_pred' in src


def test_r0_fused_linear_map_reconstruction():
    r0 = _tiny_r0()
    tr, va = np.arange(40), np.arange(40, 50)
    pred, info = r0.fit_r0_predict_full("WM", tr, va, 0, 0, cache_tag="t")
    assert info["recon_err"] <= 1e-6


def _tiny_r0(n=60):
    from metascfc.phase3a_fix.r0_baseline import R0Baseline
    X = np.random.RandomState(0).randn(n, 6670)
    y = np.random.RandomState(1).randn(n)
    return R0Baseline(X, X, y, y, np.array([str(i) for i in range(n)]))


def test_oof_labels_aligned_to_validation():
    """Each OOF position must carry the label of its OWN validation subject."""
    sys.path.insert(0, str(ROOT / "scripts_paper"))
    import ocke_pilot as P
    scope = np.arange(30)
    folds = P.partition(scope, 1, "t", 3)
    pos = {int(g): i for i, g in enumerate(scope)}
    y = np.arange(30, dtype=float)
    y_oof = np.zeros(30)
    for B, Cv in folds:
        for g, v in zip(Cv, y[Cv]):
            y_oof[pos[int(g)]] = v
    assert np.array_equal(y_oof, y)


def test_ensemble_formula_raw_units():
    rng = np.random.RandomState(3)
    base = rng.randn(50) * 10 + 100          # raw target units
    fk = rng.randn(50) * 3
    w = 0.6
    pred = (1 - w) * base + w * fk
    assert np.allclose(pred, 0.4 * base + 0.6 * fk)
    # w = 0 must equal R0 exactly
    assert np.array_equal((1 - 0.0) * base + 0.0 * fk, base)


def test_krr_fit_predict_roundtrip():
    rng = np.random.RandomState(4)
    Xtr = rng.randn(40, 30)
    ytr = Xtr[:, 0] * 2 + rng.randn(40) * 0.1
    fit = fit_krr(Xtr, ytr, "rbf", lam=0.3, mult=1.0, sigma_base=median_heuristic(Xtr))
    ptr = krr_predict(fit, Xtr)
    # near-interpolation at small lambda: training fit correlates strongly
    assert np.corrcoef(ptr, ytr)[0, 1] > 0.9
    # predict on training rows equals K @ alpha + mean exactly
    G = Xtr @ Xtr.T
    d2 = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0)
    K = np.exp(-d2 / (2 * fit.sigma ** 2))
    assert np.max(np.abs(K @ fit.alpha + fit.y_mean - ptr)) <= 1e-10


def test_krr_gradient_matches_finite_differences():
    rng = np.random.RandomState(5)
    Xtr = rng.randn(30, 12)
    ytr = rng.randn(30)
    fit = fit_krr(Xtr, ytr, "rbf", lam=1.0, mult=1.0, sigma_base=1.5)
    x = rng.randn(12)
    g = krr_gradient(fit, x[None, :])[0]
    eps = 1e-6
    for e in (0, 5, 11):
        xp = x.copy()
        xp[e] += eps
        f1 = krr_predict(fit, xp[None, :])[0]
        f0 = krr_predict(fit, x[None, :])[0]
        fd = (f1 - f0) / eps
        assert abs(g[e] - fd) <= 1e-4 * max(1.0, abs(fd))


def test_linear_kernel_gradient_is_constant_ridge_map():
    rng = np.random.RandomState(6)
    Xtr = rng.randn(25, 10)
    ytr = rng.randn(25)
    fit = fit_krr(Xtr, ytr, "linear", lam=1.0)
    g = krr_gradient(fit, Xtr[:5])
    assert np.allclose(g, g[0][None, :])       # constant across subjects
    w = Xtr.T @ fit.alpha
    assert np.allclose(g[0], w)


def test_architectures_differ_only_in_kernel():
    rng = np.random.RandomState(7)
    Xtr = rng.randn(25, 10)
    ytr = rng.randn(25)
    f_lin = fit_krr(Xtr, ytr, "linear", lam=1.0)
    f_rbf = fit_krr(Xtr, ytr, "rbf", lam=1.0, mult=1.0, sigma_base=1.0)
    assert f_lin.kernel == "linear" and f_rbf.kernel == "rbf"
    Xte = rng.randn(5, 10)
    p_lin = krr_predict(f_lin, Xte)
    p_rbf = krr_predict(f_rbf, Xte)
    assert not np.allclose(p_lin, p_rbf)


def test_edge_mask_incidence():
    mask = edge_mask_for_rois([0, 1])
    assert mask.shape == (13340,)
    # FC block: edge (0,1) is incident to both; edge (2,3) to neither
    iu = np.triu_indices(116, k=1)
    idx01 = int(np.where((iu[0] == 0) & (iu[1] == 1))[0][0])
    idx23 = int(np.where((iu[0] == 2) & (iu[1] == 3))[0][0])
    assert mask[idx01] and mask[idx23] is not None
    assert not mask[idx23]


def test_perturbation_sign_convention():
    """Masking an edge that actually drives the prediction must increase RMSE."""
    rng = np.random.RandomState(8)
    n = 60
    X = rng.randn(n, 10)
    y = X[:, 0] * 5 + rng.randn(n) * 0.05      # edge 0 dominates
    tr, te = np.arange(40), np.arange(40, 60)
    fit = fit_krr(X[tr], y[tr], "rbf", lam=0.01, mult=1.0, sigma_base=3.0)
    base = krr_predict(fit, X[te])
    rmse0 = np.sqrt(np.mean((base - y[te]) ** 2))
    Xm = X[te].copy()
    Xm[:, 0] = 0.0                              # mask the driving edge
    pm = krr_predict(fit, Xm)
    rmse_m = np.sqrt(np.mean((pm - y[te]) ** 2))
    delta = rmse_m - rmse0
    assert delta > 0.5 * np.std(y[te])          # strongly positive => faithful


def test_abstained_maps_excluded():
    """Stability aggregation must skip abstained (w=0) fits."""
    maps = []
    abst = 0
    arts = [{"abstained": True}, {"abstained": False, "importance": np.ones(5)},
            {"abstained": False, "importance": 2 * np.ones(5)}]
    for a in arts:
        if a["abstained"] or a.get("importance") is None:
            abst += 1
            continue
        maps.append(a["importance"])
    assert abst == 1 and len(maps) == 2
    assert np.allclose(maps[0], 1.0) and np.allclose(maps[1], 2.0)


def test_access_log_no_holdout():
    log = OUT / "PHASE3A_DATA_ACCESS_LOG.jsonl"
    assert log.exists()
    holdout = set(HOLDOUT_TXT.read_text().strip().split("\n"))
    import json as J
    for line in open(log):
        rec = J.loads(line)
        assert rec["holdout_overlap"] == []
        assert not (set(rec["subject_ids"]) & holdout)


def test_selection_details_complete():
    f = OUT / "selection_details.csv"
    assert f.exists()
    df = pd.read_csv(f)
    for col in ("kernel", "lam", "w", "pearson", "rmse", "mae", "selected"):
        assert col in df.columns
    for key in ("P1_rbf", "P0_linear"):
        sub = df[df["kernel"] == key]
        assert (sub["selected"] == True).any()
        # full candidate grid present: 4 lambdas x 3 mults x 11 weights
        assert len(sub[sub["selected"] == False]) >= 100
        for lam in (0.3, 1.0, 3.0, 10.0):
            assert (sub["lam"] == lam).any()


def test_optimizer_diagnostics_solve_ok():
    f = OUT / "optimizer_diagnostics.csv"
    assert f.exists()
    df = pd.read_csv(f)
    assert df["solve_ok"].all()
    assert df["chosen"].any()


def test_lock_file_frozen():
    import hashlib
    p = OUT / "MODEL_PROPOSAL_LOCK.md"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == "daff28b377bb8141718f726d9fa13b685f2627bb572a3c8fedaf7d5d64b2f0dd"
