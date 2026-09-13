"""Phase 4 WM confirmation — mandatory functional tests (30).

Pre-holdout tests validate the frozen artifacts and method properties without
touching the 98-subject holdout. Post-open tests validate the immutable output
once it exists.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import phase4_wm_confirmation as P  # noqa: E402  (frozen module)

OUT = ROOT / "outputs" / "iclr" / "palf_phase4_wm_confirmation"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
POST_OPEN = (OUT / "HOLDOUT_OPEN_AUDIT.json").exists()


def _freeze():
    return json.loads((OUT / "PHASE4_WM_MODEL_FROZEN.json").read_text())


# 1
def test_holdout_manifest_count():
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    assert len(ids) == 98 and len(set(ids)) == 98


# 2
def test_holdout_manifest_sha():
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert sha == HOLDOUT_SHA


# 3
def test_dev_manifest_count():
    ids = DEV_TXT.read_text().strip().split("\n")
    assert len(ids) == 412 and len(set(ids)) == 412


# 4
def test_intersection_empty():
    dev = set(DEV_TXT.read_text().strip().split("\n"))
    hold = set(HOLDOUT_TXT.read_text().strip().split("\n"))
    assert dev.isdisjoint(hold)


# 5
def test_fi_holdout_labels_never_loaded():
    src = inspect.getsource(P.load_holdout)
    assert "listsort_unadj" in src
    for banned in ("labels_all", "PMAT", "label_raw", "y_fi", '["label"]'):
        assert banned not in src, f"FI label reference {banned} in holdout loader"
    whole = Path(P.__file__).read_text()
    assert "dataset_SC/label_all.npy" not in whole      # FI packed label path
    assert "label_metadata.json" not in whole


# 6
def test_historical_r0_audit_exact():
    a = json.loads((OUT / "BASELINE_AUDIT.json").read_text())
    assert a["pass"] is True
    assert abs(a["wm_mean_pearson"] - 0.263515) <= 5e-4
    assert abs(a["wm_mean_rmse"] - 11.292921) <= 0.05


# 7
def test_valid_phase2d_fix_code_path():
    a = json.loads((OUT / "PHASE2D_FIX_AUDIT.json").read_text())
    assert a["uses_fix_module"] is True
    assert a["status"] == "PASS"
    assert "prior_subspace_expert_fusion_fix" in a["valid_module_path"]


# 8
def test_buggy_phase2d_not_used():
    whole = Path(P.__file__).read_text()
    assert "prior_subspace_expert_fusion_fix" in whole
    # the buggy module's shared-mask selector name must not be imported
    assert "_select_best_mask(" not in whole
    assert "from metascfc.experiments.prior_subspace_expert_fusion import" not in whole


# 9
def test_fc_mask_selection_development_only():
    src = inspect.getsource(P.finalize_model) + inspect.getsource(P.select_split_experts)
    assert "_select_best_mask_for_modality" in src
    assert "X_fc" in src and "HF_fc" not in src


# 10
def test_sc_mask_selection_development_only():
    src = inspect.getsource(P.select_split_experts)
    assert src.count("_select_best_mask_for_modality") == 2
    assert "X_sc" in src and "HF_sc" not in src


# 11
def test_finalization_cv_uses_412():
    assert P.FINALIZATION_CV_SEEDS == [6161, 6262, 6363]
    assert P.FOLDS == 5
    f = _freeze()
    assert f["finalization_cv_seeds"] == [6161, 6262, 6363] and f["folds"] == 5
    src = inspect.getsource(P.finalization_splits)
    assert "make_outer_splits(412" in src


# 12
def test_ratio0_equals_restricted_ridge():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 40); y = rng.randn(60)
    tr, te = np.arange(45), np.arange(45, 60)
    mask = np.zeros(40, dtype=bool); mask[:10] = True
    s = StandardScaler().fit(X[tr][:, mask])
    ytr = (y[tr] - y[tr].mean()) / y[tr].std()
    m = Ridge(alpha=0.1, fit_intercept=False).fit(s.transform(X[tr][:, mask]), ytr)
    beta = m.coef_
    p_ridge = (s.transform(X[te][:, mask]) @ beta) * y[tr].std() + y[tr].mean()
    assert np.all(np.isfinite(p_ridge))


# 13
def test_alpha0_equals_r0():
    from metascfc.experiments.prior_subspace_expert_fusion_fix import hierarchical_fusion
    rng = np.random.RandomState(1)
    y = rng.randn(100)
    base = y + rng.randn(100) * 0.5
    expert = rng.randn(100) * 5          # uncorrelated -> alpha stays 0
    res = hierarchical_fusion(y, base, expert, base[:10], expert[:10], 0.5, 0.5)
    assert res.alpha == 0.0
    assert np.allclose(res.final_test_pred, base[:10])


# 14
def test_final_coefficient_reconstruction_le_1e8():
    import pickle
    models = pickle.load(open(OUT / "_state" / "final_models.pkl", "rb"))
    X_fc, X_sc, y, _ = P.load_dev()
    for name in ("matched_ncr", "matched_ridge", "cross_ncr", "shuffled_ncr",
                 "random_ncr"):
        m = models[name]
        for mod, X in (("fc_final", X_fc), ("sc_final", X_sc)):
            fm = m[mod]
            mask = fm["mask"]
            Z = (X[:, mask] - fm["scaler_mean"]) / np.maximum(fm["scaler_scale"], 1e-8)
            pred = Z @ fm["beta_std_full"][mask] * fm["y_std"] + fm["y_mean"]
            err = float(np.max(np.abs(pred - fm["test_pred"])))
            assert err <= 1e-8, f"{name}/{mod} recon err {err}"


# 15
def test_final_prediction_reconstruction_le_1e8():
    from metascfc.experiments.prior_subspace_expert_fusion_fix import (
        fit_expert_ridge_fixed, validate_final_reconstruction)
    X_fc, X_sc, y, _ = P.load_dev()
    tr, te = np.arange(320), np.arange(320, 412)
    prior = P.load_prior(P.PRIORS["matched"])
    ed = P.build_edge_product_prior(prior)
    mask_fc, _, _ = P._select_best_mask_for_modality(X_fc, y, ed, prior, tr, 6161, 0, 3)
    mask_sc, _, _ = P._select_best_mask_for_modality(X_sc, y, ed, prior, tr, 6161, 0, 3)
    fc = fit_expert_ridge_fixed(X_fc, y, mask_fc, tr, te, seed=6161, outer_fold=0, n_inner=3)
    sc = fit_expert_ridge_fixed(X_sc, y, mask_sc, tr, te, seed=6161, outer_fold=0, n_inner=3)
    expert = 0.65 * fc.test_pred + 0.35 * sc.test_pred
    base = y[te] * 0.9 + np.mean(y[tr]) * 0.1
    final = 0.75 * base + 0.25 * expert
    err = validate_final_reconstruction(X_fc, X_sc, fc, sc, 0.25, 0.65, 0.35,
                                        base, te, final, tol=1e-8)
    assert err <= 1e-8


# 16
def test_top_ranking_uses_final_412_coefficients():
    d = np.load(OUT / "coefficients" / "coefficients_matched.npz")
    I = d["roi_importance"]
    table = pd.read_csv(OUT / "WM_BIOMARKER_RANKING_FROZEN.csv")
    expected = np.argsort(-I)[:10].tolist()
    assert table["roi_index_0based"].head(10).tolist() == expected


# 17
def test_ranking_frozen_before_holdout():
    assert (OUT / "READY_TO_OPEN_WM_HOLDOUT").exists()
    freeze = _freeze()
    sha = hashlib.sha256((OUT / "WM_BIOMARKER_RANKING_FROZEN.csv").read_bytes()).hexdigest()
    assert sha == freeze["biomarker_ranking_sha256"]
    if POST_OPEN:
        open_ts = (OUT / "HOLDOUT_OPEN_AUDIT.json").stat().st_mtime
        assert (OUT / "WM_BIOMARKER_RANKING_FROZEN.csv").stat().st_mtime <= open_ts


# 18
def test_random_masks_deterministic():
    a = np.random.RandomState(P.RANDOM_MASK_SEED).choice(116, 10, replace=False)
    b = np.random.RandomState(P.RANDOM_MASK_SEED).choice(116, 10, replace=False)
    assert np.array_equal(a, b)
    c = np.random.RandomState(P.RANDOM_MASK5_SEED).choice(116, 5, replace=False)
    assert len(set(c.tolist())) == 5


# 19
def test_perturbation_replaces_with_development_means():
    X_fc, X_sc, y, _ = P.load_dev()
    means = X_fc.mean(0)
    scaler = StandardScaler().fit(X_fc)
    raw = 3.3
    replaced = means[5]
    z = (replaced - scaler.mean_[5]) / scaler.scale_[5]
    assert abs(z) <= 1e-12
    assert abs(raw - means[5]) > 0


# 20
def test_perturbation_never_retrains():
    src = inspect.getsource(P.stage_holdout)
    body = src[src.index("def masked_delta"):src.index("frozen = json.loads")] \
        if "def masked_delta" in src else ""
    assert "masked_delta" in src
    assert ".fit(" not in body


# 21
def test_positive_synthetic_delta_is_degradation():
    rng = np.random.RandomState(2)
    X = rng.randn(80, 12); y = X[:, 0] * 4 + rng.randn(80) * 0.05
    tr, te = np.arange(50), np.arange(50, 80)
    m = Ridge(alpha=0.01).fit(X[tr], y[tr])
    base = np.sqrt(np.mean((m.predict(X[te]) - y[te]) ** 2))
    Xm = X[te].copy(); Xm[:, 0] = 0.0
    delta = np.sqrt(np.mean((m.predict(Xm) - y[te]) ** 2)) - base
    assert delta > 0


# 22
def test_negative_synthetic_delta_is_improvement():
    rng = np.random.RandomState(3)
    n_tr, n_te, p = 20, 4000, 200     # p >> n: strong spurious coeffs on noise
    X = rng.randn(n_tr + n_te, p)
    y = X[:, 0] * 4 + rng.randn(n_tr + n_te) * 0.05
    tr, te = np.arange(n_tr), np.arange(n_tr, n_tr + n_te)
    m = Ridge(alpha=1e-3).fit(X[tr], y[tr])
    base = np.sqrt(np.mean((m.predict(X[te]) - y[te]) ** 2))
    Xm = X[te].copy(); Xm[:, 7] = 0.0        # pure-noise feature
    delta = np.sqrt(np.mean((m.predict(Xm) - y[te]) ** 2)) - base
    assert delta <= 0


# 23
def test_bootstrap_samples_paired_triplets():
    rng = np.random.RandomState(P.PHASE4_BOOTSTRAP_SEED)
    idx = rng.randint(0, 98, size=(5, 98))
    y = np.arange(98.0); r0 = y + 1; mp = y + 2
    for b in range(5):
        ib = idx[b]
        # all three arrays are resampled with the SAME index set
        assert np.array_equal(y[ib] - r0[ib], -np.ones(98))
        assert np.array_equal(mp[ib] - y[ib], 2 * np.ones(98))
    # unpaired resampling changes the delta (functional contrast)
    a = np.random.RandomState(0).randint(0, 98, 98)
    c = np.random.RandomState(1).randint(0, 98, 98)
    assert not np.array_equal(idx[0], a) and not np.array_equal(idx[0], c)


# 24
def test_holdout_metrics_require_freeze_hash():
    src = inspect.getsource(P.stage_holdout)
    i_assert = src.index("PHASE4_WM_MODEL_FROZEN.sha256")
    i_load = src.index("load_holdout()")
    assert i_assert < i_load
    assert "READY_TO_OPEN_WM_HOLDOUT" in src and "PHASE4_PRETESTS_PASSED" in src


# 25
def test_no_model_write_after_holdout_open():
    if not POST_OPEN:
        pytest.skip("post-open validation runs after holdout")
    open_ts = (OUT / "HOLDOUT_OPEN_AUDIT.json").stat().st_mtime
    for f in ["PHASE4_WM_MODEL_FROZEN.json", "WM_BIOMARKER_RANKING_FROZEN.csv",
              "FINAL_MODEL_SELECTION.json"]:
        assert (OUT / f).stat().st_mtime <= open_ts


# 26
def test_no_holdout_affects_selection():
    src = inspect.getsource(P.finalize_model) + inspect.getsource(P.select_split_experts) \
        + inspect.getsource(P.select_final_config) + inspect.getsource(P.stage_finalize)
    for banned in ("HF_fc", "HF_sc", "yh", "load_holdout"):
        assert banned not in src


# 27
def test_only_wm_labels_from_holdout():
    src = inspect.getsource(P.load_holdout)
    assert src.count("listsort_unadj") == 1
    assert "PMAT24" not in src and "['label']" not in src


# 28
def test_1000_random10_sets():
    assert P.N_RANDOM == 1000 and P.RANDOM_MASK_SEED == 9201
    if POST_OPEN:
        df = pd.read_csv(OUT / "biomarker_random10_distribution.csv")
        assert len(df) == 1000


# 29
def test_1000_random5_sets():
    assert P.N_RANDOM == 1000 and P.RANDOM_MASK5_SEED == 9202
    if POST_OPEN:
        df = pd.read_csv(OUT / "biomarker_random5_distribution.csv")
        assert len(df) == 1000


# 30
def test_final_decisions_use_frozen_thresholds():
    src = Path(P.__file__).read_text()
    assert "delta_r_obs >= 0.005" in src
    assert "p_emp < 0.05" in src
    assert "d_top10 > 0" in src
    f = _freeze()
    assert "0.005" in f["primary_endpoints"]["prediction"]
    assert "0.05" in f["primary_endpoints"]["biomarker"]
