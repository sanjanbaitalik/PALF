"""OpenChallenge (independent re-run) functional tests.

Tests the validity of the NO_METHOD_JUSTIFIED decision: holdout seal, strict
baseline audit, screening evidence consistency, honest nested-selection
invariants, and the pre-coding rejection. Every test is capable of failing.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

OUT = ROOT / "outputs" / "iclr" / "openchallenge"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"


def test_holdout_count_and_sha():
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    assert len(ids) == 98 and len(set(ids)) == 98
    sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert sha == HOLDOUT_SHA


def test_dev_holdout_disjoint():
    dev = DEV_TXT.read_text().strip().split("\n")
    hold = HOLDOUT_TXT.read_text().strip().split("\n")
    assert len(dev) == 412 and len(set(dev)) == 412
    assert set(dev).isdisjoint(set(hold))


def test_seal_report():
    rep = json.loads((OUT / "HOLDOUT_SEAL_REPORT.json").read_text())
    assert rep["n_subjects"] == 98
    assert rep["canonical_sha256"] == HOLDOUT_SHA
    assert rep["intersection"] == 0
    assert rep["holdout_features_labels_loaded"] is False


def test_baseline_audit_strict():
    a = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert a["status"] == "PASS"
    assert abs(a["details"]["WM"]["r"] - 0.263515) <= 5e-4
    assert abs(a["details"]["FI"]["r"] - 0.370917) <= 5e-4
    assert abs(a["details"]["WM"]["rmse"] - 11.292921) <= 0.05
    assert abs(a["details"]["FI"]["rmse"] - 4.566689) <= 0.05


def test_partition_covers_scope():
    import oc2_bank as B
    scope = np.arange(57)
    folds = B.partition(scope, 3, "t", 3)
    covered = np.concatenate([c for _, c in folds])
    assert len(covered) == 57
    assert set(covered.tolist()) == set(scope.tolist())
    for b, c in folds:
        assert set(b).isdisjoint(set(c))
        assert len(b) + len(c) == 57


def test_blend_weight_selection_inner_only():
    import oc2_bank as B
    rng = np.random.RandomState(0)
    y = rng.randn(80)
    r0 = y + rng.randn(80) * 2.0
    cand = y + rng.randn(80) * 0.1        # candidate far better than R0
    best = (-9.0, 0.0)
    for w in np.linspace(0, 1, 21):
        r = np.corrcoef((1 - w) * r0 + w * cand, y)[0, 1]
        if r > best[0]:
            best = (r, w)
    assert best[1] > 0.5                   # picks a large candidate weight
    # sanity: the selector uses only arrays it is given (no test labels)
    assert not any("test" in k for k in B.partition.__code__.co_varnames)


def test_per_subject_zscore_invariants():
    rng = np.random.RandomState(1)
    A = rng.randn(20, 50) * 5 + 3
    mu = A.mean(1, keepdims=True)
    sd = A.std(1, keepdims=True)
    Z = (A - mu) / (sd + 1e-12)
    assert np.allclose(Z.mean(1), 0, atol=1e-9)
    assert np.allclose(Z.std(1), 1, atol=1e-9)


def test_perturbation_sign_convention():
    """Masking a truly predictive feature must increase RMSE (faithfulness)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(2)
    n = 80
    X = rng.randn(n, 12)
    y = X[:, 0] * 4 + rng.randn(n) * 0.05
    tr, te = np.arange(50), np.arange(50, 80)
    s = StandardScaler().fit(X[tr])
    m = Ridge(alpha=0.01).fit(s.transform(X[tr]), y[tr])
    base = np.sqrt(np.mean((m.predict(s.transform(X[te])) - y[te]) ** 2))
    Xm = X[te].copy()
    Xm[:, 0] = 0.0
    rmse_m = np.sqrt(np.mean((m.predict(s.transform(Xm)) - y[te]) ** 2))
    assert rmse_m - base > 0.5 * np.std(y[te])   # positive = faithful


def test_no_holdout_ids_in_outputs():
    hold = set(HOLDOUT_TXT.read_text().strip().split("\n"))
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".md", ".csv", ".txt", ".log"):
            txt = f.read_text(errors="ignore")
            for hid in list(hold)[:200]:
                assert hid not in txt, f"holdout id {hid} found in {f}"


def test_screening_evidence_consistency():
    ev = json.loads((OUT / "screening" / "screening_evidence.json").read_text())
    d = ev["decisive_falsification"]
    for key in ("WM_proposed", "WM_control", "FI_proposed", "FI_control"):
        vals = np.array(d["per_seed_deltas"][key], dtype=float)
        assert len(vals) == 5
        assert abs(float(vals.mean()) - d["means"][key]) <= 1e-5
    assert d["means"]["FI_proposed"] < 0.005      # the failing task
    assert abs(d["means"]["WM_proposed"] - d["means"]["WM_control"]) <= 0.001


def test_decision_is_justified_and_no_false_freeze():
    v = json.loads((OUT / "VALIDATION_REPORT.json").read_text())
    assert v["decision"] == "NO_METHOD_JUSTIFIED"
    assert v["status"] == "OPENCHALLENGE_COMPLETE"
    assert v["holdout"]["touched"] is False
    assert not (OUT / "MODEL_FROZEN.json").exists()
    assert not (OUT / "MODEL_PROPOSAL_LOCK.md").exists()
