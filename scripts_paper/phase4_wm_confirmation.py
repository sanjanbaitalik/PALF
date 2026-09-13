#!/usr/bin/env python3
"""PALF Phase 4: WM-only frozen PS-NCR-EF confirmation on the 98 holdout.

Stages:
  audit     - integrity, strict R0 audit, Phase-2D-FIX audit, runtime ETA
  finalize  - final 412-subject selection/fits, biomarker rankings, freeze
  pretests  - pre-holdout functional self-checks
  holdout   - one-time 98-subject prediction + biomarker perturbation + reports

The 98-subject holdout is opened ONLY in the holdout stage, after the freeze
artifacts and pre-holdout tests exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (  # noqa: E402
    CONDITIONS,
    N_EDGE,
    N_ROI,
    generate_crossfit_oof,
    make_outer_splits,
    reselect_and_fit_final,
    search_fusion_weights,
)
from metascfc.experiments.prior_subspace_expert_fusion_fix import (  # noqa: E402
    K_EDGE_GRID,
    LAPLACIAN_RATIO_GRID,
    M_ROI_GRID,
    RIDGE_EXPERT_GRID,
    WEIGHT_GRID,
    _build_sub_laplacian,
    _select_best_mask_for_modality,
    build_edge_product_prior,
    build_control_prior,
    direct_topk_mask,
    fit_expert_candidate_on_split,
    fit_expert_ncr_fixed,
    fit_expert_ridge_fixed,
    hierarchical_fusion,
    roi_incident_mask,
    search_fusion_weights_simple,
    validate_expert_reconstruction,
    validate_final_reconstruction,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (  # noqa: E402
    NetworkConstrainedRidge,
    build_edge_laplacian,
)
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────
OUT = ROOT / "outputs" / "iclr" / "palf_phase4_wm_confirmation"
STATE = OUT / "_state"
DEV_FC = ROOT / "inputs" / "dataset_FC" / "FC_all.npy"
DEV_SC = ROOT / "inputs" / "dataset_SC" / "SC_all.npy"
DEV_YWM = ROOT / "inputs" / "dataset_SC" / "task_labels" / "ListSort_Unadj" / "label_all.npy"
DEV_SUBJECTS = ROOT / "inputs" / "dataset_SC" / "hcp_subjects_used.csv"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"
HOLDOUT_FC_DIR = ROOT / "data" / "hcp" / "processed" / "fc"
HOLDOUT_SC_DIR = ROOT / "data" / "hcp" / "processed" / "sc"
HOLDOUT_LABELS = ROOT / "data" / "hcp" / "processed" / "labels.csv"

PRIORS = {
    "matched": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv",
    "cross_task": ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv",
    "shuffled": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3_shuffled/roi_prior.csv",
    "random": ROOT / "outputs/priors/random_prior/aal116/roi_prior.csv",
}
VALID_PHASE2D_MODULE = ROOT / "src/metascfc/experiments/prior_subspace_expert_fusion_fix.py"
BUGGY_PHASE2D_MODULE = ROOT / "src/metascfc/experiments/prior_subspace_expert_fusion.py"
R0_MODULE = ROOT / "src/metascfc/experiments/palf_crossfit_ablation.py"
NCR_MODULE = ROOT / "src/metascfc/models/iclr_backbones/network_constrained_ridge.py"

# ── Frozen constants ───────────────────────────────────────────────────
FINALIZATION_CV_SEEDS = [6161, 6262, 6363]
FOLDS = 5
N_INNER = 3
PHASE4_BOOTSTRAP_SEED = 9101
RANDOM_MASK_SEED = 9201
RANDOM_MASK5_SEED = 9202
BIOMARKER_BOOTSTRAP_SEED = 9301
N_BOOT = 10000
N_RANDOM = 1000
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
DEV_SHA = "8d4ee9586d78e2f997eaa9ac2ea4abe4d117bf5dec04ec836a89490657b4af6e"
EXPECTED_WM_R, EXPECTED_WM_RMSE = 0.263515, 11.292921
TOL_R, TOL_RMSE = 5e-4, 0.05
PLACEHOLDER_PRIOR = np.ones(N_ROI) / N_ROI

MODEL_SPECS = {
    "R0": ("matched", "r0"),
    "matched_ridge": ("matched", "ridge"),
    "matched_ncr": ("matched", "ncr"),
    "cross_ncr": ("cross_task", "ncr"),
    "shuffled_ncr": ("shuffled", "ncr"),
    "random_ncr": ("random", "ncr"),
}


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def progress(stage: str, elapsed: float, est_remaining: float, pct: float):
    p = OUT / "RUNTIME_PROGRESS.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec[stage] = {"elapsed_seconds": round(elapsed, 1),
                  "estimated_remaining_seconds": round(est_remaining, 1),
                  "percent_complete": round(pct, 1),
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.write_text(json.dumps(rec, indent=2))


def load_prior(path) -> np.ndarray:
    return pd.read_csv(path)["prior_score"].values.astype(np.float64)


def load_dev():
    FC = np.load(DEV_FC).astype(np.float64)
    SC = np.load(DEV_SC).astype(np.float64)
    y = np.load(DEV_YWM).astype(np.float64)
    subs = pd.read_csv(DEV_SUBJECTS, dtype={"subject": str})["subject"].tolist()
    assert FC.shape == (412, 116, 116) and SC.shape == (412, 116, 116)
    assert y.shape == (412,) and len(subs) == 412
    iu = np.triu_indices(116, 1)
    return FC[:, iu[0], iu[1]], SC[:, iu[0], iu[1]], y, subs


def preprocess_subject(fc_raw, sc_raw):
    """Exact scripts/24_pack_hcp_arrays.py preprocessing."""
    fc = np.nan_to_num(np.asarray(fc_raw, dtype=np.float32))
    sc = np.nan_to_num(np.asarray(sc_raw, dtype=np.float32))
    sc = np.log1p(np.clip(sc, 0, None))
    if sc.max() > 0:
        sc = sc / sc.max()
    fc = (fc + fc.T) / 2
    sc = (sc + sc.T) / 2
    np.fill_diagonal(fc, 0)
    np.fill_diagonal(sc, 0)
    return fc, sc


def load_holdout():
    """Load ONLY the 98 holdout FC, SC, WM labels (never FI)."""
    assert (OUT / "READY_TO_OPEN_WM_HOLDOUT").exists(), "freeze marker missing"
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    assert len(ids) == 98 and len(set(ids)) == 98
    labels = pd.read_csv(HOLDOUT_LABELS, dtype={"subject": str})
    wm_map = dict(zip(labels["subject"], labels["listsort_unadj"]))
    iu = np.triu_indices(116, 1)
    FC = np.zeros((98, len(iu[0]))); SC = np.zeros((98, len(iu[0])))
    y = np.zeros(98)
    for i, s in enumerate(ids):
        fcr = np.load(HOLDOUT_FC_DIR / f"{s}_fc.npy")
        scr = np.loadtxt(HOLDOUT_SC_DIR / s / "sc_116.csv", delimiter=",")
        fc, sc = preprocess_subject(fcr, scr)
        FC[i] = fc[iu[0], iu[1]]
        SC[i] = sc[iu[0], iu[1]]
        y[i] = float(wm_map[s])
    assert np.all(np.isfinite(FC)) and np.all(np.isfinite(SC)) and np.all(np.isfinite(y))
    return FC, SC, y, ids


def metrics(y, p):
    return {"pearson": float(pearsonr(y, p).statistic),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


# ══════════════════════════════════════════════════════════════════════
# Stage A: integrity + audits + ETA
# ══════════════════════════════════════════════════════════════════════

def run_integrity():
    hold = HOLDOUT_TXT.read_text().strip().split("\n")
    dev = DEV_TXT.read_text().strip().split("\n")
    sha = sha256_bytes(("\n".join(sorted(hold)) + "\n").encode())
    assert len(hold) == 98 and len(set(hold)) == 98 and sha == HOLDOUT_SHA
    assert len(dev) == 412 and len(set(dev)) == 412
    assert set(dev).isdisjoint(set(hold))
    prior_shas = {k: sha256_file(v) for k, v in PRIORS.items()}
    audit = {
        "n_holdout": 98, "n_dev": 412, "holdout_sha256": sha,
        "dev_manifest_sha256": sha256_bytes(("\n".join(sorted(dev)) + "\n").encode()),
        "intersection": 0, "prior_sha256": prior_shas,
        "holdout_accessed": False,
        "fi_holdout_labels_loaded": False,
        "valid_phase2d_module": str(VALID_PHASE2D_MODULE.relative_to(ROOT)),
        "buggy_phase2d_module_present": BUGGY_PHASE2D_MODULE.exists(),
        "status": "PREOPEN_OK",
    }
    (OUT / "HOLDOUT_PREOPEN_AUDIT.json").write_text(json.dumps(audit, indent=2))
    return audit


def run_r0_audit(X_fc, X_sc, y, r0):
    rows = []
    t0 = time.time()
    for seed in range(10):
        for _, fold, tr, te in make_outer_splits(412, [seed], FOLDS):
            p = r0.fit_r0_predict("WM", tr, te, seed, fold, cache_tag="p4audit")
            rows.append({"seed": seed, "fold": fold, **metrics(y[te], p)})
    df = pd.DataFrame(rows)
    r, rmse = float(df["pearson"].mean()), float(df["rmse"].mean())
    ok = abs(r - EXPECTED_WM_R) <= TOL_R and abs(rmse - EXPECTED_WM_RMSE) <= TOL_RMSE
    res = {"wm_mean_pearson": r, "wm_mean_rmse": rmse,
           "expected_pearson": EXPECTED_WM_R, "expected_rmse": EXPECTED_WM_RMSE,
           "pass": bool(ok), "n_splits": len(df), "runtime_seconds": time.time() - t0}
    (OUT / "BASELINE_AUDIT.json").write_text(json.dumps(res, indent=2))
    df.to_csv(OUT / "baseline_audit_splits.csv", index=False)
    return res


def run_phase2d_fix_audit(X_fc, X_sc, y, roi_prior):
    """Functional checks of the valid Phase-2D-FIX implementation."""
    checks = {}
    # 1. valid module in use; buggy module not imported
    import metascfc.experiments.prior_subspace_expert_fusion_fix as fixmod
    checks["valid_module_path"] = str(Path(fixmod.__file__).resolve())
    checks["uses_fix_module"] = "prior_subspace_expert_fusion_fix" in fixmod.__file__
    assert "prior_subspace_expert_fusion_fix" in fixmod.__file__
    # 2. grids exact
    checks["K_EDGE_GRID"] = K_EDGE_GRID
    checks["M_ROI_GRID"] = M_ROI_GRID
    checks["RIDGE_EXPERT_GRID"] = RIDGE_EXPERT_GRID
    checks["LAPLACIAN_RATIO_GRID"] = LAPLACIAN_RATIO_GRID
    # 3. ratio=0 == restricted Ridge on a real dev split
    tr, te = np.arange(320), np.arange(320, 412)
    edge_prior = build_edge_product_prior(roi_prior)
    mask, family, size = _select_best_mask_for_modality(
        X_fc, y, edge_prior, roi_prior, tr, 6161, 0, 3)
    ridge_res = fit_expert_ridge_fixed(X_fc, y, mask, tr, te, seed=6161,
                                       outer_fold=0, n_inner=3)
    # force ratio=0 through NCR path with same lambda
    from metascfc.models.iclr_backbones.network_constrained_ridge import (
        build_edge_laplacian)
    lap = build_edge_laplacian(N_ROI, prior_scores=roi_prior, top_k=10)
    ncr_res = fit_expert_ncr_fixed(X_fc, y, mask, lap, tr, te,
                                   lambda_grid=[ridge_res.lambda_r],
                                   ratio_grid=[0.0], seed=6161, outer_fold=0,
                                   n_inner=3)
    err_ratio0 = float(np.max(np.abs(ncr_res.test_pred - ridge_res.test_pred)))
    checks["ratio0_equals_ridge_max_abs_err"] = err_ratio0
    checks["ratio0_pass"] = err_ratio0 <= 1e-10
    # 4. reconstruction
    sc_mask, _, _ = _select_best_mask_for_modality(
        X_sc, y, edge_prior, roi_prior, tr, 6161, 0, 3)
    sc_res = fit_expert_ridge_fixed(X_sc, y, sc_mask, tr, te, seed=6161,
                                    outer_fold=0, n_inner=3)
    w = {"fc": 0.7, "sc": 0.3}
    expert_test = w["fc"] * ridge_res.test_pred + w["sc"] * sc_res.test_pred
    err_expert = validate_expert_reconstruction(
        X_fc, X_sc, ridge_res, sc_res, w["fc"], w["sc"], te, expert_test, tol=1e-8)
    fusion = hierarchical_fusion(y[tr], np.full(len(tr), np.nan), np.full(len(tr), np.nan),
                                 ridge_res.test_pred, expert_test, w["fc"], w["sc"])
    # alpha=0 == R0 hard identity (pure algebra on any baseline)
    base_test = np.random.RandomState(0).randn(len(te))
    fusion0 = hierarchical_fusion(y[tr], np.full(len(tr), np.nan), np.full(len(tr), np.nan),
                                  base_test, expert_test, w["fc"], w["sc"])
    checks["alpha0_equals_r0"] = bool(fusion0.alpha == 0.0 and
                                      np.allclose(fusion0.final_test_pred, base_test))
    checks["expert_reconstruction_max_err"] = err_expert
    checks["reconstruction_pass"] = err_expert <= 1e-8
    checks["status"] = "PASS" if all([
        checks["uses_fix_module"], checks["ratio0_pass"],
        checks["reconstruction_pass"], checks["alpha0_equals_r0"]]) else "FAIL"
    (OUT / "PHASE2D_FIX_AUDIT.json").write_text(json.dumps(checks, indent=2, default=str))
    return checks


def run_benchmark(X_fc, X_sc, y, roi_prior):
    """Development-only timing of representative Phase-4 workloads."""
    edge_prior = build_edge_product_prior(roi_prior)
    lap = build_edge_laplacian(N_ROI, prior_scores=roi_prior, top_k=10)
    tr, te = np.arange(329), np.arange(329, 412)
    times = {}
    t0 = time.time()
    mask_fc, fam_fc, size_fc = _select_best_mask_for_modality(
        X_fc, y, edge_prior, roi_prior, tr, 6161, 0, N_INNER)
    times["mask_select_fc"] = time.time() - t0
    t0 = time.time()
    mask_sc, fam_sc, size_sc = _select_best_mask_for_modality(
        X_sc, y, edge_prior, roi_prior, tr, 6161, 0, N_INNER)
    times["mask_select_sc"] = time.time() - t0
    t0 = time.time()
    r_ncr = fit_expert_ncr_fixed(X_fc, y, mask_fc, lap, tr, te,
                                 seed=6161, outer_fold=0, n_inner=N_INNER)
    times["fit_ncr_fc"] = time.time() - t0
    t0 = time.time()
    r_ncr_sc = fit_expert_ncr_fixed(X_sc, y, mask_sc, lap, tr, te,
                                    seed=6161, outer_fold=0, n_inner=N_INNER)
    times["fit_ncr_sc"] = time.time() - t0
    t0 = time.time()
    fit_expert_ridge_fixed(X_fc, y, mask_fc, tr, te, seed=6161, outer_fold=0, n_inner=N_INNER)
    times["fit_ridge_fc"] = time.time() - t0
    t0 = time.time()
    fit_expert_candidate_on_split(X_fc, y, mask_fc, tr, te, r_ncr.lambda_r,
                                  r_ncr.laplacian_ratio, lap)
    times["fixed_fit_fc"] = time.time() - t0
    t0 = time.time()
    r0 = R0Baseline(X_fc, X_sc, y, y, np.array([str(i) for i in range(412)]))
    r0.fit_r0_predict("WM", tr, te, 6161, 0, cache_tag="bench")
    times["r0_split"] = time.time() - t0
    (OUT / "benchmark_times.json").write_text(json.dumps(times, indent=2))

    n_mask_sel = 15 * 2 * 5          # splits x modalities x models
    n_expert_fit = 15 * 2 * 5
    n_ridge_extra = 15 * 2 * 1
    n_fixed = 15 * 2 * 5 * 8         # distinct-config pooled eval, generous
    n_r0 = 50 + 15                   # audit + finalization
    est = {
        "A_integrity_audits": 50 * times["r0_split"] + 120,
        "B_finalization": n_mask_sel * (times["mask_select_fc"] + times["mask_select_sc"]) / 2
        + n_expert_fit * (times["fit_ncr_fc"] + times["fit_ncr_sc"]) / 2
        + n_ridge_extra * times["fit_ridge_fc"]
        + 15 * times["r0_split"],
        "C_selection_refits": n_fixed * times["fixed_fit_fc"],
        "D_pretests": 180,
        "E_holdout_inference": 600,
        "F_perturbation_bootstrap": 900,
        "G_reports_packaging": 300,
    }
    total = sum(est.values())
    (OUT / "RUNTIME_ESTIMATE.md").write_text(
        "# Phase 4 Runtime Estimate (machine-specific)\n\n"
        "## Benchmark times (development-only)\n\n"
        + "\n".join(f"- {k}: {v:.2f}s" for k, v in times.items())
        + "\n\n## Estimated stage times\n\n"
        + "\n".join(f"- {k}: {v/60:.1f} min" for k, v in est.items())
        + f"\n\n**Total: {total/3600:.2f} hours ({total/60:.0f} min)**\n")
    print(f"PHASE4_ESTIMATED_TOTAL_TIME: {total/3600:.2f} hours ({total/60:.0f} minutes)")
    print(f"PHASE4_ESTIMATED_FINISH_FROM_START: {total/3600:.2f} hours")
    return est


def stage_audit():
    t0 = time.time()
    print("=" * 70); print("PHASE 4 STAGE A: integrity + audits + ETA"); print("=" * 70)
    run_integrity()
    print("  integrity OK (98/412 disjoint, priors hashed)")
    X_fc, X_sc, y, subs = load_dev()
    r0 = R0Baseline(X_fc, X_sc, y, y, np.array(subs), access_logger=None)
    audit = run_r0_audit(X_fc, X_sc, y, r0)
    print(f"  R0 audit: WM r={audit['wm_mean_pearson']:.10f} "
          f"RMSE={audit['wm_mean_rmse']:.10f} -> {'PASS' if audit['pass'] else 'FAIL'}")
    if not audit["pass"]:
        (OUT / "COMPLETE").write_text("PHASE4_BASELINE_AUDIT_FAILED\n")
        raise SystemExit("STATUS: PHASE4_BASELINE_AUDIT_FAILED")
    p2d = run_phase2d_fix_audit(X_fc, X_sc, y, load_prior(PRIORS["matched"]))
    print(f"  Phase-2D-FIX audit: ratio0 err={p2d['ratio0_equals_ridge_max_abs_err']:.2e} "
          f"recon err={p2d['expert_reconstruction_max_err']:.2e} -> {p2d['status']}")
    assert p2d["status"] == "PASS", "Phase-2D-FIX audit failed"
    est = run_benchmark(X_fc, X_sc, y, load_prior(PRIORS["matched"]))
    progress("A_audit", time.time() - t0, sum(est.values()), 10.0)
    print(f"  Stage A done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage B/C: finalization + final fits + freeze
# ══════════════════════════════════════════════════════════════════════

def finalization_splits():
    splits = []
    for seed in FINALIZATION_CV_SEEDS:
        for _, fold, tr, te in make_outer_splits(412, [seed], FOLDS):
            splits.append((seed, fold, tr, te))
    return splits


def r0_pooled_predictions(X_fc, X_sc, y, splits):
    """Per-split R0 held-out predictions (cached)."""
    path = STATE / "r0_finalization.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    out = []
    for seed, fold, tr, te in splits:
        oof = generate_crossfit_oof(X_fc, X_sc, y, tr, CONDITIONS["R0"],
                                    PLACEHOLDER_PRIOR, seed, fold,
                                    n_fusion_folds=3, n_inner=N_INNER)
        w, _ = search_fusion_weights(y[tr], {"FP": oof.fp_oof, "SC": oof.sc_oof},
                                     ["FP", "SC"])
        _, sc_f, fp_f = reselect_and_fit_final(
            X_fc, X_sc, y, tr, te, CONDITIONS["R0"], PLACEHOLDER_PRIOR,
            seed, fold, n_final_cv=N_INNER)
        pred = w["FP"] * fp_f.test_pred + w["SC"] * sc_f.test_pred
        out.append({"seed": seed, "fold": fold, "train": tr, "test": te,
                    "pred": pred, "y": y[te]})
        print(f"    R0 split seed={seed} fold={fold} done", flush=True)
    pickle.dump(out, open(path, "wb"))
    return out


def select_split_experts(X_fc, X_sc, y, prior, edge_lap, seed, fold, tr, te,
                         expert_type):
    """Phase-2D-FIX nested selection inside training; predict held-out fold."""
    edge_prior = build_edge_product_prior(prior)
    mask_fc, fam_fc, size_fc = _select_best_mask_for_modality(
        X_fc, y, edge_prior, prior, tr, seed, fold, N_INNER)
    mask_sc, fam_sc, size_sc = _select_best_mask_for_modality(
        X_sc, y, edge_prior, prior, tr, seed, fold, N_INNER)
    if expert_type == "ncr":
        fc = fit_expert_ncr_fixed(X_fc, y, mask_fc, edge_lap, tr, te,
                                  seed=seed, outer_fold=fold, n_inner=N_INNER)
        sc = fit_expert_ncr_fixed(X_sc, y, mask_sc, edge_lap, tr, te,
                                  seed=seed, outer_fold=fold, n_inner=N_INNER)
    else:
        fc = fit_expert_ridge_fixed(X_fc, y, mask_fc, tr, te,
                                    seed=seed, outer_fold=fold, n_inner=N_INNER)
        sc = fit_expert_ridge_fixed(X_sc, y, mask_sc, tr, te,
                                    seed=seed, outer_fold=fold, n_inner=N_INNER)
    fc.mask_family, fc.mask_size = fam_fc, size_fc
    sc.mask_family, sc.mask_size = fam_sc, size_sc
    return {"fc_pred": fc.test_pred, "sc_pred": sc.test_pred,
            "fc_cfg": (fam_fc, size_fc, fc.lambda_r, fc.laplacian_ratio),
            "sc_cfg": (fam_sc, size_sc, sc.lambda_r, sc.laplacian_ratio),
            "fc_lambda": fc.lambda_r, "sc_lambda": sc.lambda_r,
            "fc_ratio": fc.laplacian_ratio, "sc_ratio": sc.laplacian_ratio}


def rebuild_mask(prior, cfg):
    fam, size = cfg[0], cfg[1]
    if fam == "direct_topk":
        return direct_topk_mask(build_edge_product_prior(prior), size)
    mask, _ = roi_incident_mask(prior, size)
    return mask


def pooled_config_eval(X, y, mask, lam, ratio, splits, edge_lap):
    preds = []
    for seed, fold, tr, te in splits:
        p = fit_expert_candidate_on_split(X, y, mask, tr, te, lam, ratio, edge_lap)
        preds.append(p)
    pred = np.concatenate(preds)
    ypool = np.concatenate([y[s[3]] for s in splits])
    m = metrics(ypool, pred)
    return {"pearson": m["pearson"], "rmse": m["rmse"], "mae": m["mae"], "pred": pred}


def select_final_config(X, y, candidate_cfgs, splits, prior, edge_lap):
    """Pooled CV selection over distinct candidate configs with tie-breaks."""
    scored = []
    for cfg in sorted(set(candidate_cfgs)):
        fam, size, lam, ratio = cfg
        mask = rebuild_mask(prior, cfg)
        ev = pooled_config_eval(X, y, mask, lam, ratio, splits, edge_lap)
        scored.append({"cfg": cfg, "n_mask": int(mask.sum()), **{k: ev[k] for k in
                                                               ("pearson", "rmse", "mae")}})
    scored.sort(key=lambda r: (-r["pearson"], r["rmse"], r["mae"],
                               r["n_mask"], r["cfg"]))
    return scored[0], scored


def fit_expert_fixed_config(X, y, mask, lam, ratio, train_idx, test_idx,
                            edge_laplacian):
    """Deterministic final fit with the frozen config (no re-selection)."""
    X_sub = X[:, mask]
    n_sub = int(mask.sum())
    scaler = StandardScaler().fit(X_sub[train_idx])
    X_tr = scaler.transform(X_sub[train_idx])
    X_te = scaler.transform(X_sub[test_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_tr = (y[train_idx] - y_mean) / y_std
    sub_lap = None
    if ratio > 0 and edge_laplacian is not None:
        sub_lap = _build_sub_laplacian(mask, edge_laplacian, n_sub)
    if ratio == 0 or sub_lap is None:
        model = Ridge(alpha=lam, fit_intercept=False).fit(X_tr, y_tr)
        beta_std = model.coef_.copy()
        test_pred = model.predict(X_te) * y_std + y_mean
        used_ratio = 0.0
    else:
        X_tr2 = np.hstack([X_tr, np.zeros_like(X_tr)])
        X_te2 = np.hstack([X_te, np.zeros_like(X_te)])
        ncr = NetworkConstrainedRidge(alpha1=lam, alpha2=ratio * lam,
                                      edge_laplacian=sub_lap, standardize=False)
        ncr.fit(X_tr2, y_tr)
        beta_std = ncr.beta()[:n_sub].copy()
        test_pred = ncr.predict(X_te2) * y_std + y_mean
        used_ratio = ratio
    beta_full = np.zeros(X.shape[1])
    beta_full[mask] = beta_std
    return {"mask": mask, "n_selected": n_sub, "lambda_r": lam,
            "laplacian_ratio": used_ratio, "scaler_mean": scaler.mean_.copy(),
            "scaler_scale": scaler.scale_.copy(), "y_mean": y_mean,
            "y_std": y_std, "beta_std_full": beta_full, "test_pred": test_pred}


def finalize_model(model_name, X_fc, X_sc, y, splits, r0_splits, cache):
    prior_type, expert_type = MODEL_SPECS[model_name]
    prior = load_prior(PRIORS[prior_type])
    edge_lap = build_edge_laplacian(N_ROI, prior_scores=prior, top_k=10)
    print(f"  [{model_name}] finalization CV ...", flush=True)
    fc_preds, sc_preds, fc_cfgs, sc_cfgs = [], [], [], []
    for k, (seed, fold, tr, te) in enumerate(splits):
        r = select_split_experts(X_fc, X_sc, y, prior, edge_lap, seed, fold,
                                 tr, te, expert_type)
        fc_preds.append(r["fc_pred"]); sc_preds.append(r["sc_pred"])
        fc_cfgs.append(r["fc_cfg"]); sc_cfgs.append(r["sc_cfg"])
        if (k + 1) % 5 == 0:
            print(f"    split {k+1}/{len(splits)} done", flush=True)
    fc_oof = np.concatenate(fc_preds); sc_oof = np.concatenate(sc_preds)
    y_pool = np.concatenate([y[s[3]] for s in splits])
    r0_pool = np.concatenate([r["pred"] for r in r0_splits])
    v, _ = search_fusion_weights_simple(y_pool, fc_oof, sc_oof)
    expert_fused = v["fc"] * fc_oof + v["sc"] * sc_oof
    zeros = np.zeros_like(r0_pool)
    fusion = hierarchical_fusion(y_pool, r0_pool, expert_fused, zeros, zeros,
                                 v["fc"], v["sc"])
    alpha = fusion.alpha
    final_oof = (1 - alpha) * r0_pool + alpha * expert_fused
    dev = {"v": v, "alpha": alpha,
           "expert_fused_oof_r": float(pearsonr(y_pool, expert_fused).statistic),
           "final_oof_r": float(pearsonr(y_pool, final_oof).statistic),
           "fc_oof_r": float(pearsonr(y_pool, fc_oof).statistic),
           "sc_oof_r": float(pearsonr(y_pool, sc_oof).statistic),
           "r0_pool_r": float(pearsonr(y_pool, r0_pool).statistic)}
    print(f"    pooled expert r={dev['expert_fused_oof_r']:.4f} "
          f"v_fc={v['fc']:.2f} alpha={alpha:.2f} final r={dev['final_oof_r']:.4f}")

    # Final config selection on all 412 via pooled CV over distinct configs
    if expert_type == "ncr":
        fc_best, fc_scored = select_final_config(X_fc, y, fc_cfgs, splits, prior, edge_lap)
        sc_best, sc_scored = select_final_config(X_sc, y, sc_cfgs, splits, prior, edge_lap)
    else:
        fc_best, fc_scored = select_final_config(X_fc, y, fc_cfgs, splits, prior, None)
        sc_best, sc_scored = select_final_config(X_sc, y, sc_cfgs, splits, prior, None)
    print(f"    final FC cfg={fc_best['cfg']} r={fc_best['pearson']:.4f} | "
          f"SC cfg={sc_best['cfg']} r={sc_best['pearson']:.4f}")

    # Final all-412 fits with the frozen configs
    all_idx = np.arange(412)
    fc_final = fit_expert_fixed_config(X_fc, y, rebuild_mask(prior, fc_best["cfg"]),
                                       fc_best["cfg"][2], fc_best["cfg"][3],
                                       all_idx, all_idx, edge_lap)
    sc_final = fit_expert_fixed_config(X_sc, y, rebuild_mask(prior, sc_best["cfg"]),
                                       sc_best["cfg"][2], sc_best["cfg"][3],
                                       all_idx, all_idx, edge_lap)
    model = {"name": model_name, "prior_type": prior_type, "expert_type": expert_type,
             "prior": prior, "v": v, "alpha": alpha, "dev": dev,
             "fc_final": fc_final, "sc_final": sc_final,
             "fc_config": fc_best["cfg"], "sc_config": sc_best["cfg"],
             "fc_scored": fc_scored, "sc_scored": sc_scored}
    cache[model_name] = model
    return model


def linear_final_map(model, r0_info):
    """Full standardized-space linear map (W, c) of a frozen final predictor.

    Implements the frozen formula final = (1-alpha) * R0 + alpha * expert_fused.
    """
    alpha = float(model.get("alpha", 0.0))
    W = (1.0 - alpha) * r0_info["w_r0"].copy()
    c = (1.0 - alpha) * float(r0_info["c_r0"])
    if alpha > 0:
        v = model["v"]
        fc, sc = model["fc_final"], model["sc_final"]
        W[:N_EDGE] += alpha * v["fc"] * fc["y_std"] * fc["beta_std_full"]
        W[N_EDGE:] += alpha * v["sc"] * sc["y_std"] * sc["beta_std_full"]
        c += alpha * (v["fc"] * fc["y_mean"] + v["sc"] * sc["y_mean"])
    return W, float(c)


def biomarker_importance(model):
    """I_i = sum_j |c^FC_ij| + sum_j |c^SC_ij| with c = alpha * w_m * beta_std."""
    alpha = model["alpha"]; v = model["v"]
    cf = alpha * v["fc"] * model["fc_final"]["beta_std_full"]
    cs = alpha * v["sc"] * model["sc_final"]["beta_std_full"]
    iu = np.triu_indices(N_ROI, 1)
    def roi_score(vec):
        M = np.zeros((N_ROI, N_ROI))
        M[iu[0], iu[1]] = vec
        M = M + M.T
        return np.abs(M).sum(axis=1)
    I = roi_score(cf) + roi_score(cs)
    return I, cf, cs


def ranking_table(I, n_top=10):
    order = np.argsort(-I)
    labels = pd.read_csv(ROOT / "inputs/atlases/AAL116_labels.csv")
    names = labels["roi_label"].tolist() if "roi_label" in labels.columns else None
    rows = []
    for rank, idx in enumerate(order):
        rows.append({"rank": rank + 1, "roi_index_0based": int(idx),
                     "roi_index_1based": int(idx + 1),
                     "roi_name": names[idx] if names else "",
                     "importance": float(I[idx])})
    return pd.DataFrame(rows)


def stage_finalize():
    t0 = time.time()
    print("=" * 70); print("PHASE 4 STAGE B/C: finalize 412 + freeze"); print("=" * 70)
    X_fc, X_sc, y, subs = load_dev()
    splits = finalization_splits()
    r0_splits = r0_pooled_predictions(X_fc, X_sc, y, splits)

    cache = {}
    for name in ["matched_ncr", "matched_ridge", "cross_ncr", "shuffled_ncr",
                 "random_ncr"]:
        finalize_model(name, X_fc, X_sc, y, splits, r0_splits, cache)
        progress("B_finalization", time.time() - t0, 0, 40.0)

    # Final R0 on 412 with linear map (train on dev, predict dev for recon check)
    r0_full = R0Baseline(X_fc, X_sc, y, y, np.array(subs))
    r0_pred_dev, r0_info = r0_full.fit_r0_predict_full(
        "WM", np.arange(412), np.arange(412), 6161, 0, cache_tag="p4final")
    recon_dev = float(r0_info["recon_err"])
    assert recon_dev <= 1e-6, f"R0 linear map recon err {recon_dev}"
    cache["R0"] = {"name": "R0", "prior_type": "matched", "expert_type": "r0",
                   "prior": load_prior(PRIORS["matched"]), "v": {"fc": 0.0, "sc": 1.0},
                   "alpha": 0.0, "dev": {}, "info": r0_info}

    # Biomarker rankings from matched NCR (primary) and all controls
    ranking_inputs = {
        "matched": cache["matched_ncr"],
        "cross_task": cache["cross_ncr"],
        "shuffled": cache["shuffled_ncr"],
        "random": cache["random_ncr"],
        "matched_ridge": cache["matched_ridge"],
    }
    rankings = {}
    frozen_sets = {}
    for key, m in ranking_inputs.items():
        I, cf, cs = biomarker_importance(m)
        tab = ranking_table(I)
        rankings[key] = tab
        order = np.argsort(-I)
        frozen_sets[key] = {"top5": order[:5].tolist(), "top10": order[:10].tolist(),
                            "bottom5": order[-5:].tolist(), "bottom10": order[-10:].tolist()}
        np.savez(OUT / "coefficients" / f"coefficients_{key}.npz",
                 roi_importance=I, beta_fc_weighted=cf, beta_sc_weighted=cs)

    tab = rankings["matched"]
    tab.to_csv(OUT / "WM_BIOMARKER_RANKING_FROZEN.csv", index=False)
    tab_sha = sha256_file(OUT / "WM_BIOMARKER_RANKING_FROZEN.csv")
    (OUT / "WM_BIOMARKER_RANKING_FROZEN.sha256").write_text(
        f"{tab_sha}  WM_BIOMARKER_RANKING_FROZEN.csv\n")
    ctrl_rows = []
    for key, t in rankings.items():
        if key == "matched":
            continue
        for _, r in t.head(10).iterrows():
            ctrl_rows.append({"control": key, **r.to_dict()})
    pd.DataFrame(ctrl_rows).to_csv(OUT / "CONTROL_BIOMARKER_RANKINGS_FROZEN.csv",
                                   index=False)

    # FINALIZATION_CV_RESULTS + FINAL_MODEL_SELECTION
    rows = []
    for name, m in cache.items():
        if name == "R0":
            continue
        rows.append({"model": name, "prior": m["prior_type"], "expert": m["expert_type"],
                     "v_fc": m["v"]["fc"], "v_sc": m["v"]["sc"], "alpha": m["alpha"],
                     "fc_cfg": str(m["fc_config"]), "sc_cfg": str(m["sc_config"]),
                     **m["dev"]})
    pd.DataFrame(rows).to_csv(OUT / "FINALIZATION_CV_RESULTS.csv", index=False)
    sel = {name: {"v": m["v"], "alpha": m["alpha"], "fc_config": list(m["fc_config"]),
                  "sc_config": list(m["sc_config"])}
           for name, m in cache.items() if name != "R0"}
    (OUT / "FINAL_MODEL_SELECTION.json").write_text(json.dumps(sel, indent=2, default=str))

    # Freeze file
    diff = subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, text=True).stdout
    freeze = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
        "working_tree_diff_sha256": sha256_bytes(diff.encode()),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "platform": platform.platform()},
        "dev_manifest_sha256": DEV_SHA,
        "holdout_manifest_sha256": HOLDOUT_SHA,
        "wm_prior_sha256": sha256_file(PRIORS["matched"]),
        "control_prior_sha256": {k: sha256_file(v) for k, v in PRIORS.items()},
        "r0_code_sha256": sha256_file(R0_MODULE),
        "phase2d_fix_code_sha256": sha256_file(VALID_PHASE2D_MODULE),
        "ncr_code_sha256": sha256_file(NCR_MODULE),
        "phase4_code_sha256": sha256_file(Path(__file__).resolve()),
        "finalization_cv_seeds": FINALIZATION_CV_SEEDS,
        "folds": FOLDS,
        "selected_models": sel,
        "biomarker_ranking_sha256": tab_sha,
        "frozen_roi_sets": frozen_sets,
        "random_perturbation_seeds": {"random10": RANDOM_MASK_SEED,
                                      "random5": RANDOM_MASK5_SEED},
        "bootstrap_seeds": {"prediction": PHASE4_BOOTSTRAP_SEED,
                            "biomarker": BIOMARKER_BOOTSTRAP_SEED},
        "primary_endpoints": {
            "prediction": "delta_r = r_matched_ncr - r_R0; PASS if delta_r >= +0.005 "
                          "and one-sided 95% bootstrap lower bound > 0",
            "biomarker": "delta_RMSE_top10 > 0 and > mean(random10) and empirical p < 0.05",
        },
        "runtime_estimate_path": str((OUT / "RUNTIME_ESTIMATE.md").relative_to(ROOT)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "PHASE4_WM_MODEL_FROZEN.json").write_text(json.dumps(freeze, indent=2, default=str))
    (OUT / "PHASE4_WM_MODEL_FROZEN.sha256").write_text(
        f"{sha256_file(OUT / 'PHASE4_WM_MODEL_FROZEN.json')}  PHASE4_WM_MODEL_FROZEN.json\n")
    with open(STATE / "final_models.pkl", "wb") as f:
        pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "prior"}
                     for k, v in cache.items()}, f)
    (OUT / "READY_TO_OPEN_WM_HOLDOUT").write_text("READY\n")
    progress("C_freeze", time.time() - t0, 0, 70.0)
    print(f"  freeze written; ranking SHA {tab_sha[:16]}...")
    print(f"  Stage B/C done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage D: pre-holdout functional self-checks
# ══════════════════════════════════════════════════════════════════════

def stage_pretests():
    t0 = time.time()
    print("=" * 70); print("PHASE 4 STAGE D: pre-holdout functional checks"); print("=" * 70)
    freeze = json.loads((OUT / "PHASE4_WM_MODEL_FROZEN.json").read_text())
    sha_line = (OUT / "PHASE4_WM_MODEL_FROZEN.sha256").read_text().split()[0]
    assert sha_line == sha256_file(OUT / "PHASE4_WM_MODEL_FROZEN.json")
    checks = {
        "freeze_sha_verifies": True,
        "ready_marker": (OUT / "READY_TO_OPEN_WM_HOLDOUT").exists(),
        "ranking_frozen": (OUT / "WM_BIOMARKER_RANKING_FROZEN.csv").exists(),
        "ranking_rows_116": len(pd.read_csv(OUT / "WM_BIOMARKER_RANKING_FROZEN.csv")) == 116,
        "holdout_txt_sha": sha256_bytes(
            ("\n".join(sorted(HOLDOUT_TXT.read_text().strip().split("\n"))) + "\n")
            .encode()) == HOLDOUT_SHA,
    }
    # synthetic perturbation sign convention
    from sklearn.linear_model import Ridge as R
    rng = np.random.RandomState(0)
    X = rng.randn(80, 12); y = X[:, 0] * 4 + rng.randn(80) * 0.05
    tr, te = np.arange(50), np.arange(50, 80)
    m = R(alpha=0.01).fit(X[tr], y[tr])
    base = np.sqrt(np.mean((m.predict(X[te]) - y[te]) ** 2))
    Xm = X[te].copy(); Xm[:, 0] = 0.0
    d = np.sqrt(np.mean((m.predict(Xm) - y[te]) ** 2)) - base
    checks["synthetic_positive_delta"] = bool(d > 0)
    checks["sign_convention"] = "RMSE_masked - RMSE_unmasked; positive=faithful"
    checks["status"] = "PASS" if all(v for k, v in checks.items()
                                     if isinstance(v, bool)) else "FAIL"
    (OUT / "PRETEST_CHECKS.json").write_text(json.dumps(checks, indent=2))
    assert checks["status"] == "PASS"
    (OUT / "PHASE4_PRETESTS_PASSED").write_text("PASS\n")
    progress("D_pretests", time.time() - t0, 0, 80.0)
    print(f"  pre-holdout checks PASS ({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════════
# Stage E/F/G: one-time holdout evaluation
# ══════════════════════════════════════════════════════════════════════

def delta_ci(deltas, alpha=0.05):
    lo = float(np.percentile(deltas, 100 * alpha))
    hi = float(np.percentile(deltas, 100 * (1 - alpha)))
    return {"mean": float(np.mean(deltas)), "ci95": [lo, hi],
            "lower_one_sided": lo,
            "fraction_le_0": float(np.mean(deltas <= 0))}


def steiger_williams(r_y1, r_y2, r_12, n):
    """Williams/Steiger test for two dependent correlations sharing y."""
    det = 1 - r_y1 ** 2 - r_y2 ** 2 - r_12 ** 2 + 2 * r_y1 * r_y2 * r_12
    rbar = (r_y1 + r_y2) / 2
    num = (r_y1 - r_y2) * np.sqrt((n - 1) * (1 + r_12))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r_12) ** 3)
    t = num / den if den > 0 else 0.0
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 3))
    return {"t": float(t), "df": n - 3, "p_two_sided": float(p)}


def stage_holdout():
    t0 = time.time()
    print("=" * 70); print("PHASE 4 STAGE E/F/G: one-time holdout evaluation"); print("=" * 70)
    assert (OUT / "READY_TO_OPEN_WM_HOLDOUT").exists()
    assert (OUT / "PHASE4_PRETESTS_PASSED").exists()
    assert (OUT / "PHASE4_WM_MODEL_FROZEN.sha256").exists()
    freeze_sha = (OUT / "PHASE4_WM_MODEL_FROZEN.sha256").read_text().split()[0]
    assert freeze_sha == sha256_file(OUT / "PHASE4_WM_MODEL_FROZEN.json")

    open_audit = {"HOLDOUT_OPEN_TIMESTAMP": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "freeze_sha256": freeze_sha, "stage": "holdout",
                  "models_frozen": True}
    (OUT / "HOLDOUT_OPEN_AUDIT.json").write_text(json.dumps(open_audit, indent=2))

    with open(STATE / "final_models.pkl", "rb") as f:
        models = pickle.load(f)
    iu = np.triu_indices(N_ROI, 1)
    X_fc, X_sc, ydev, subs = load_dev()
    HF_fc, HF_sc, yh, hids = load_holdout()
    # freeze manifest: never change anything below this line based on holdout y

    # Development scalers (shared by all components) for the holdout
    fc_scaler = StandardScaler().fit(X_fc); sc_scaler = StandardScaler().fit(X_sc)
    Z = np.hstack([fc_scaler.transform(HF_fc), sc_scaler.transform(HF_sc)])

    # Verify stored per-mask expert scalers equal the shared development
    # scalers on the selected columns (validator fix; model unchanged).
    for name, m in models.items():
        if name == "R0":
            continue
        fm, sm = m["fc_final"], m["sc_final"]
        assert np.allclose(fm["scaler_mean"], fc_scaler.mean_[fm["mask"]], atol=1e-9)
        assert np.allclose(fm["scaler_scale"], fc_scaler.scale_[fm["mask"]], atol=1e-9)
        assert np.allclose(sm["scaler_mean"], sc_scaler.mean_[sm["mask"]], atol=1e-9)
        assert np.allclose(sm["scaler_scale"], sc_scaler.scale_[sm["mask"]], atol=1e-9)

    # R0 map trained on all 412, validated on holdout
    r0_full = R0Baseline(np.vstack([X_fc, HF_fc]), np.vstack([X_sc, HF_sc]),
                         np.concatenate([ydev, yh]), np.concatenate([ydev, yh]),
                         np.array(subs + hids))
    r0_pred_h, r0_info = r0_full.fit_r0_predict_full(
        "WM", np.arange(412), np.arange(412, 510), 6161, 0, cache_tag="p4holdout")
    r0_map_pred = Z @ r0_info["w_r0"] + r0_info["c_r0"]
    recon_r0 = float(np.max(np.abs(r0_map_pred - r0_pred_h)))
    assert recon_r0 <= 1e-6, f"R0 holdout map recon {recon_r0}"

    # Build linear maps and predictions for all frozen models
    preds = {"R0": r0_map_pred}
    maps = {"R0": (r0_info["w_r0"], float(r0_info["c_r0"]))}
    for name in ["matched_ridge", "matched_ncr", "cross_ncr", "shuffled_ncr",
                 "random_ncr"]:
        W, c = linear_final_map(models[name], r0_info)
        preds[name] = Z @ W + c
        maps[name] = (W, c)

    metrics_rows = []
    for name, p in preds.items():
        metrics_rows.append({"model": name, **metrics(yh, p)})
    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(OUT / "holdout_model_metrics.csv", index=False)
    pd.DataFrame({"subject": hids, "y_wm": yh, **{k: v for k, v in preds.items()}}
                 ).to_csv(OUT / "holdout_predictions.csv", index=False)
    print(mdf.to_string(index=False))

    # ── Primary prediction inference ──
    r0p = preds["R0"]; mp = preds["matched_ncr"]
    delta_r_obs = float(pearsonr(yh, mp).statistic - pearsonr(yh, r0p).statistic)
    rng = np.random.RandomState(PHASE4_BOOTSTRAP_SEED)
    idx = rng.randint(0, 98, size=(N_BOOT, 98))
    boot_r = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        ib = idx[b]
        boot_r[b] = (pearsonr(yh[ib], mp[ib]).statistic
                     - pearsonr(yh[ib], r0p[ib]).statistic)
    pred_inf = {"delta_r_observed": delta_r_obs, **delta_ci(boot_r)}
    pred_inf["gate_pass"] = bool(delta_r_obs >= 0.005 and
                                 pred_inf["lower_one_sided"] > 0)
    # sensitivity
    r12 = float(pearsonr(r0p, mp).statistic)
    pred_inf["williams_steiger"] = steiger_williams(
        float(pearsonr(yh, mp).statistic), float(pearsonr(yh, r0p).statistic), r12, 98)
    dm = np.zeros(N_BOOT); da = np.zeros(N_BOOT)
    rm0 = np.zeros(N_BOOT); rm1 = np.zeros(N_BOOT)
    mae0 = np.zeros(N_BOOT); mae1 = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        ib = idx[b]
        rm0[b] = np.sqrt(np.mean((r0p[ib] - yh[ib]) ** 2))
        rm1[b] = np.sqrt(np.mean((mp[ib] - yh[ib]) ** 2))
        mae0[b] = np.mean(np.abs(r0p[ib] - yh[ib]))
        mae1[b] = np.mean(np.abs(mp[ib] - yh[ib]))
    pred_inf["delta_rmse"] = delta_ci(rm1 - rm0)
    pred_inf["delta_mae"] = delta_ci(mae1 - mae0)
    (OUT / "prediction_inference.json").write_text(json.dumps(pred_inf, indent=2))
    np.save(OUT / "prediction_bootstrap.npy", boot_r)
    print(f"  delta_r={delta_r_obs:+.4f} lower95={pred_inf['lower_one_sided']:+.4f} "
          f"frac<=0={pred_inf['fraction_le_0']:.3f} -> "
          f"{'PASS' if pred_inf['gate_pass'] else 'FAIL'}")

    # ── Biomarker perturbation (matched final predictor fixed) ──
    Wm, cm = maps["matched_ncr"]
    base_pred = preds["matched_ncr"]
    contrib = Z * Wm[None, :]
    base_rmse = np.sqrt(np.mean((base_pred - yh) ** 2))

    def masked_delta(roi_set):
        fc_idx = np.where(np.isin(iu[0], roi_set) | np.isin(iu[1], roi_set))[0]
        sc_idx = fc_idx + N_EDGE
        p = base_pred - contrib[:, fc_idx].sum(1) - contrib[:, sc_idx].sum(1)
        return float(np.sqrt(np.mean((p - yh) ** 2)) - base_rmse)

    frozen = json.loads((OUT / "PHASE4_WM_MODEL_FROZEN.json").read_text())["frozen_roi_sets"]
    primary_top10 = frozen["matched"]["top10"]
    d_top10 = masked_delta(primary_top10)
    d_top5 = masked_delta(frozen["matched"]["top5"])
    d_bottom5 = masked_delta(frozen["matched"]["bottom5"])
    d_bottom10 = masked_delta(frozen["matched"]["bottom10"])
    rs = np.random.RandomState(RANDOM_MASK_SEED)
    rnd10 = np.array([masked_delta(rs.choice(116, 10, replace=False))
                      for _ in range(N_RANDOM)])
    rs5 = np.random.RandomState(RANDOM_MASK5_SEED)
    rnd5 = np.array([masked_delta(rs5.choice(116, 5, replace=False))
                     for _ in range(N_RANDOM)])
    p_emp = (1 + int(np.sum(rnd10 >= d_top10))) / (1 + N_RANDOM)
    bio_pass = bool(d_top10 > 0 and d_top10 > rnd10.mean() and p_emp < 0.05)
    bio_inf = {"unmasked_rmse": base_rmse, "top10_delta_rmse": d_top10,
               "top5_delta_rmse": d_top5,
               "bottom5_delta_rmse": d_bottom5, "bottom10_delta_rmse": d_bottom10,
               "random10_mean": float(rnd10.mean()), "random10_median": float(np.median(rnd10)),
               "random10_p95": float(np.percentile(rnd10, 95)),
               "random10_fraction_ge_top": float(np.mean(rnd10 >= d_top10)),
               "empirical_p_top10": p_emp,
               "random5_mean": float(rnd5.mean()),
               "empirical_p_top5": (1 + int(np.sum(rnd5 >= d_top5))) / (1 + N_RANDOM),
               "gate_pass": bio_pass,
               "sign_convention": "delta_RMSE = RMSE_masked - RMSE_unmasked; "
                                  "positive = faithful"}
    (OUT / "biomarker_inference.json").write_text(json.dumps(bio_inf, indent=2))
    pd.DataFrame({"random_set": np.arange(N_RANDOM), "delta_rmse": rnd10}
                 ).to_csv(OUT / "biomarker_random10_distribution.csv", index=False)
    pd.DataFrame({"random_set": np.arange(N_RANDOM), "delta_rmse": rnd5}
                 ).to_csv(OUT / "biomarker_random5_distribution.csv", index=False)

    # Ranking controls on the SAME matched final predictor
    ctrl_rows = []
    ctrl_rankings = {k: v for k, v in frozen.items()}
    for key, f in ctrl_rankings.items():
        ctrl_rows.append({"ranking": key, "top10_delta_rmse": masked_delta(f["top10"]),
                          "top5_delta_rmse": masked_delta(f["top5"])})
    pd.DataFrame(ctrl_rows).to_csv(OUT / "biomarker_mask_results.csv", index=False)
    bio_inf["ranking_controls"] = {r["ranking"]: r["top10_delta_rmse"] for r in ctrl_rows}

    # Subject-bootstrap sensitivity (precompute per-subject MSE for top10/random10)
    ms_un = (base_pred - yh) ** 2
    fc_idx = np.where(np.isin(iu[0], primary_top10) | np.isin(iu[1], primary_top10))[0]
    p_top = base_pred - contrib[:, fc_idx].sum(1) - contrib[:, fc_idx + N_EDGE].sum(1)
    ms_top = (p_top - yh) ** 2
    MS = np.zeros((98, N_RANDOM))
    rsm = np.random.RandomState(RANDOM_MASK_SEED)
    sets = [rsm.choice(116, 10, replace=False) for _ in range(N_RANDOM)]
    for j, s in enumerate(sets):
        fidx = np.where(np.isin(iu[0], s) | np.isin(iu[1], s))[0]
        pj = base_pred - contrib[:, fidx].sum(1) - contrib[:, fidx + N_EDGE].sum(1)
        MS[:, j] = (pj - yh) ** 2
    boots = np.zeros(N_BOOT)
    rbs = np.random.RandomState(BIOMARKER_BOOTSTRAP_SEED)
    bidx = rbs.randint(0, 98, size=(N_BOOT, 98))
    for b in range(N_BOOT):
        ib = bidx[b]
        top = np.sqrt(ms_top[ib].mean())
        rndmean = float(np.mean(np.sqrt(MS[ib].mean(axis=0))))
        boots[b] = top - rndmean
    bio_inf["bootstrap_sensitivity"] = delta_ci(boots)
    (OUT / "biomarker_inference.json").write_text(json.dumps(bio_inf, indent=2))

    # Development stability context from Phase-2D-FIX coefficients (dev only)
    stability_development()

    # decisions
    pred_pass = pred_inf["gate_pass"]
    bio_pass = bio_inf["gate_pass"]
    decision = ("WM_PREDICTION_AND_BIOMARKER_CONFIRMED" if pred_pass and bio_pass else
                "WM_PREDICTION_ONLY" if pred_pass else
                "WM_BIOMARKER_ONLY" if bio_pass else "WM_CONFIRMATION_FAILED")
    val = {"PREDICTION_CONFIRMATION": "PASS" if pred_pass else "FAIL",
           "BIOMARKER_CONFIRMATION": "PASS" if bio_pass else "FAIL",
           "PHASE4_DECISION": decision,
           "primary": {"delta_r_observed": delta_r_obs,
                       "delta_r_lower_one_sided": pred_inf["lower_one_sided"],
                       "top10_delta_rmse": d_top10, "empirical_p": p_emp}}
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(val, indent=2))
    (OUT / "COMPLETE").write_text("PHASE4_WM_CONFIRMATION_COMPLETE\n")
    write_reports(pred_inf, bio_inf, mdf, decision)
    make_plots(pred_inf, bio_inf, rnd10, rnd5, mdf)
    progress("E_F_G_holdout", time.time() - t0, 0, 100.0)
    print(f"\nPREDICTION_CONFIRMATION: {'PASS' if pred_pass else 'FAIL'}")
    print(f"BIOMARKER_CONFIRMATION: {'PASS' if bio_pass else 'FAIL'}")
    print(f"PHASE4_DECISION: {decision}")
    print(f"  Stage E/F/G done in {time.time()-t0:.0f}s")


def stability_development():
    import glob
    rows = []
    for pt in ["working_memory_matched", "working_memory_cross_task",
               "working_memory_shuffled", "working_memory_random"]:
        fs = sorted(glob.glob(str(ROOT /
                    f"outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion/coefficients/*{pt}.npz")))
        fc_maps, sc_maps, valid = [], [], 0
        for f in fs:
            d = np.load(f)
            if np.allclose(d["beta_fc_expert_weighted"], 0) and \
               np.allclose(d["beta_sc_expert_weighted"], 0):
                continue
            valid += 1
            fc_maps.append(d["beta_fc_expert"]); sc_maps.append(d["beta_sc_expert"])
        if valid < 2:
            continue
        from scipy.stats import spearmanr
        def mean_spear(maps):
            vals = []
            for i in range(len(maps)):
                for j in range(i + 1, len(maps)):
                    vals.append(spearmanr(np.abs(maps[i]), np.abs(maps[j])).statistic)
            return float(np.nanmean(vals))
        def top_jaccard(maps, k):
            iu = np.triu_indices(116, 1)
            sets = []
            for m in maps:
                M = np.zeros((116, 116)); M[iu[0], iu[1]] = m; M = M + M.T
                roi = np.abs(M).sum(1)
                sets.append(set(np.argsort(-roi)[:k]))
            vals = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                    for i in range(len(sets)) for j in range(i + 1, len(sets))]
            return float(np.mean(vals))
        rows.append({"prior_type": pt.replace("working_memory_", ""),
                     "n_valid_fits": valid,
                     "fc_abs_edge_spearman": mean_spear(fc_maps),
                     "sc_abs_edge_spearman": mean_spear(sc_maps),
                     "fc_top10_roi_jaccard": top_jaccard(fc_maps, 10),
                     "sc_top10_roi_jaccard": top_jaccard(sc_maps, 10)})
    pd.DataFrame(rows).to_csv(OUT / "development_biomarker_stability.csv", index=False)


def write_reports(pred_inf, bio_inf, mdf, decision):
    lines = ["# Phase 4 WM Confirmation — Final Report", "",
             f"Decision: {decision}", "",
             "## Holdout prediction", "```",
             mdf.to_string(index=False), "```", "",
             "## Primary prediction inference",
             f"- observed delta_r = {pred_inf['delta_r_observed']:+.4f}",
             f"- bootstrap 95% CI = [{pred_inf['ci95'][0]:+.4f}, {pred_inf['ci95'][1]:+.4f}]",
             f"- one-sided lower bound = {pred_inf['lower_one_sided']:+.4f}",
             f"- fraction <= 0 = {pred_inf['fraction_le_0']:.4f}",
             f"- Williams/Steiger: {pred_inf['williams_steiger']}", "",
             "## Biomarker faithfulness",
             f"- unmasked RMSE = {bio_inf['unmasked_rmse']:.4f}",
             f"- top10 delta_RMSE = {bio_inf['top10_delta_rmse']:+.4f}",
             f"- random10 mean = {bio_inf['random10_mean']:+.4f}",
             f"- empirical p = {bio_inf['empirical_p_top10']:.4f}",
             f"- bootstrap sensitivity = {bio_inf.get('bootstrap_sensitivity')}", "",
             f"PREDICTION_CONFIRMATION: "
             f"{'PASS' if pred_inf['gate_pass'] else 'FAIL'}",
             f"BIOMARKER_CONFIRMATION: "
             f"{'PASS' if bio_inf['gate_pass'] else 'FAIL'}"]
    (OUT / "FINAL_CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n")


def make_plots(pred_inf, bio_inf, rnd10, rnd5, mdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = OUT / "plots"
    # prediction comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = mdf["model"].tolist(); vals = mdf["pearson"].tolist()
    ax.bar(range(len(names)), vals); ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_ylabel("holdout Pearson r")
    fig.tight_layout(); fig.savefig(p / "fig_phase4_holdout_prediction.pdf")
    fig.savefig(p / "fig_phase4_holdout_prediction.png"); plt.close(fig)
    # delta_r bootstrap (from saved npy)
    boot = np.load(OUT / "prediction_bootstrap.npy")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(boot, bins=60, alpha=0.8); ax.axvline(0, color="r", lw=1)
    ax.axvline(pred_inf["delta_r_observed"], color="k", lw=1)
    ax.set_xlabel("bootstrap delta_r"); fig.tight_layout()
    fig.savefig(p / "fig_phase4_delta_r_bootstrap.pdf")
    fig.savefig(p / "fig_phase4_delta_r_bootstrap.png"); plt.close(fig)
    # biomarker randomization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rnd10, bins=50, alpha=0.7, label="random10")
    ax.axvline(bio_inf["top10_delta_rmse"], color="r", lw=1.5, label="matched top10")
    ax.set_xlabel("delta_RMSE"); ax.legend(); fig.tight_layout()
    fig.savefig(p / "fig_phase4_biomarker_randomization.pdf")
    fig.savefig(p / "fig_phase4_biomarker_randomization.png"); plt.close(fig)
    # top vs bottom
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["top5", "top10", "bottom5", "bottom10"],
           [bio_inf["top5_delta_rmse"], bio_inf["top10_delta_rmse"],
            bio_inf["bottom5_delta_rmse"], bio_inf["bottom10_delta_rmse"]])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("delta_RMSE"); fig.tight_layout()
    fig.savefig(p / "fig_phase4_top_vs_bottom_perturbation.pdf")
    fig.savefig(p / "fig_phase4_top_vs_bottom_perturbation.png"); plt.close(fig)
    # dev stability
    df = pd.read_csv(OUT / "development_biomarker_stability.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["fc_abs_edge_spearman"], 0.4, label="FC")
    ax.bar(x + 0.2, df["sc_abs_edge_spearman"], 0.4, label="SC")
    ax.set_xticks(x); ax.set_xticklabels(df["prior_type"]); ax.legend()
    ax.set_ylabel("abs-edge Spearman (dev)")
    fig.tight_layout(); fig.savefig(p / "fig_phase4_development_stability.pdf")
    fig.savefig(p / "fig_phase4_development_stability.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "finalize", "pretests", "holdout"])
    args = ap.parse_args()
    if args.stage == "audit":
        stage_audit()
    elif args.stage == "finalize":
        stage_finalize()
    elif args.stage == "pretests":
        stage_pretests()
    elif args.stage == "holdout":
        stage_holdout()


if __name__ == "__main__":
    main()
