#!/usr/bin/env python3
"""FINAL 510-subject WM study (v35).

SPSEF (Semantic Prior-Selected Expert Fusion) on the full 510-subject HCP
cohort with repeated nested cross-validation and architecture-matched
semantic-prior controls (Ridge and NCR).

Stages
------
audit      : cohort audit + corrected-R0 historical implementation audit
benchmark  : time one full 510 outer fold, write ETA, save checkpoint
main       : Stage B/C - all 25 outer folds x {R0 + 8 prior/arch variants}
biomarker  : Stage D/E - coefficient stability + cross-fold faithfulness
ablations  : prediction ablations A-H
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

from metascfc.experiments.palf_crossfit_ablation import (  # noqa: E402
    CONDITIONS, N_EDGE, N_ROI, evaluate_ablation_split, make_outer_splits)
from metascfc.experiments.prior_subspace_expert_fusion_fix import (  # noqa: E402
    K_EDGE_GRID, LAPLACIAN_RATIO_GRID, M_ROI_GRID, RIDGE_EXPERT_GRID,
    _select_best_mask_for_modality, build_edge_product_prior,
    direct_topk_mask, fit_expert_ncr_fixed, fit_expert_ridge_fixed,
    generate_expert_crossfit_oof_fixed, hierarchical_fusion,
    roi_incident_mask, validate_final_reconstruction)
from metascfc.models.iclr_backbones.network_constrained_ridge import (  # noqa: E402
    NetworkConstrainedRidge, build_edge_laplacian)
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402
from scaling_412_vs_510_pilot import (  # noqa: E402
    _full_beta, load_combined, prior_for, r0_linear_map, sha256_file)

OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
STATE = OUT / "_state"
FOLDS = STATE / "folds"
for d in (OUT, STATE, FOLDS, OUT / "paper_ready", OUT / "supplementary",
          OUT / "plots", OUT / "models", OUT / "coefficients", OUT / "tests"):
    d.mkdir(parents=True, exist_ok=True)

FINAL510_TXT = ROOT / "data_splits" / "final510_subjects.txt"
D412_TXT = ROOT / "data_splits" / "scaling_D412.txt"
D98_TXT = ROOT / "data_splits" / "scaling_D98.txt"
D510_TXT = ROOT / "data_splits" / "scaling_D510.txt"
FINAL510_SEEDS = [7171, 7272, 7373, 7474, 7575]
OUTER_FOLDS = 5
INNER_FOLDS = 3
N510 = 510
RANDOM_MASK_SEED = 9702
N_RANDOM_MASKS = 100
IU = np.triu_indices(N_ROI, 1)

PRIOR_FILES = {
    "matched": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv",
    "cross": ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv",
    "shuffled": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3_shuffled/roi_prior.csv",
    "random": ROOT / "outputs/priors/random_prior/aal116/roi_prior.csv",
}
PRIOR_TYPES = ["matched", "cross", "shuffled", "random"]
ARCH_NAMES = {"ridge": "R", "ncr": "N"}
MODELS = {"ridge": [f"R-{p.upper()}" for p in PRIOR_TYPES],
          "ncr": [f"N-{p.upper()}" for p in PRIOR_TYPES]}
ALL_MODELS = MODELS["ridge"] + MODELS["ncr"]
ARCH_OF = {m: ("ridge" if m.startswith("R-") else "ncr") for m in ALL_MODELS}
PRIOR_OF = {m: m.split("-", 1)[1].lower() for m in ALL_MODELS}
EXPECTED_R0 = {"WM": (0.263515, 11.292921)}
TOL_R, TOL_RMSE = 5e-4, 0.05


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_priors():
    out = {}
    for name, p in PRIOR_FILES.items():
        df = pd.read_csv(p)
        out[name] = {"array": df["prior_score"].values.astype(np.float64),
                     "labels": df["roi_label"].tolist(),
                     "path": str(p.relative_to(ROOT)),
                     "sha256": sha256_file(p),
                     "n": int(len(df))}
    return out


def progress(stage, elapsed, remaining, pct):
    (OUT / "RUNTIME_PROGRESS.json").write_text(json.dumps(
        {"stage": stage, "elapsed_seconds": elapsed, "remaining_seconds": remaining,
         "pct": round(pct, 2),
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


def metrics(y, p):
    r = float(pearsonr(y, p).statistic) if np.std(p) > 0 else 0.0
    return {"pearson": r, "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


def seed_of(x):
    """Deterministic RNG seed for a (seed, fold) pair."""
    return int(RANDOM_MASK_SEED + int(x))


# ══════════════════════════════════════════════════════════════════════
# Mask selection with candidate diagnostics (identical selection logic)
# ══════════════════════════════════════════════════════════════════════

def _eval_mask_inner_cv_metrics(X, y, mask, analysis_idx, seed, outer_fold,
                                n_inner=3, lambda_grid=RIDGE_EXPERT_GRID):
    """Replicates the module inner-CV but returns r/RMSE/MAE at the argmax."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    n_analysis = len(analysis_idx)
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 42)
    perm = rng.permutation(n_analysis)
    fold_sizes = np.full(n_inner, n_analysis // n_inner)
    fold_sizes[:n_analysis % n_inner] += 1
    best = (-np.inf, 0.0, 0.0, lambda_grid[0])
    for lam in lambda_grid:
        oof = np.full(n_analysis, np.nan)
        current = 0
        for k in range(n_inner):
            start, stop = current, current + fold_sizes[k]
            v_local = perm[start:stop]
            a_local = np.concatenate([perm[:start], perm[stop:]])
            X_sub = X[analysis_idx, :][:, mask]
            scaler = StandardScaler()
            X_a = scaler.fit_transform(X_sub[a_local])
            X_v = scaler.transform(X_sub[v_local])
            y_mean = float(y[analysis_idx[a_local]].mean())
            y_std = max(float(y[analysis_idx[a_local]].std()), 1e-8)
            model = Ridge(alpha=lam, fit_intercept=False)
            model.fit(X_a, (y[analysis_idx[a_local]] - y_mean) / y_std)
            oof[v_local] = model.predict(X_v) * y_std + y_mean
            current = stop
        valid = np.isfinite(oof)
        if valid.sum() < 10:
            continue
        r, _ = pearsonr(oof[valid], y[analysis_idx][valid])
        if r > best[0]:
            m = metrics(y[analysis_idx][valid], oof[valid])
            best = (r, m["rmse"], m["mae"], lam)
    return best


def select_mask_with_candidates(X_modality, y, edge_prior, roi_prior,
                                analysis_idx, seed, outer_fold, n_inner=3):
    """Same selection rule as the validated module, plus per-candidate rows."""
    candidates, rows = [], []
    for k in K_EDGE_GRID:
        mask = direct_topk_mask(edge_prior, k)
        r, rmse, mae, lam = _eval_mask_inner_cv_metrics(
            X_modality, y, mask, analysis_idx, seed, outer_fold, n_inner)
        candidates.append((r, mask, "direct_topk", k))
        rows.append({"family": "direct_topk", "size": k, "inner_r": r,
                     "inner_rmse": rmse, "inner_mae": mae, "inner_lambda": lam})
    for msize in M_ROI_GRID:
        mask, _ = roi_incident_mask(roi_prior, msize)
        r, rmse, mae, lam = _eval_mask_inner_cv_metrics(
            X_modality, y, mask, analysis_idx, seed, outer_fold, n_inner)
        candidates.append((r, mask, "roi_incident", msize))
        rows.append({"family": "roi_incident", "size": msize, "inner_r": r,
                     "inner_rmse": rmse, "inner_mae": mae, "inner_lambda": lam})
    candidates.sort(key=lambda x: (-x[0], -x[3]))
    _, best_mask, best_family, best_size = candidates[0]
    for row in rows:
        row["selected"] = int(row["family"] == best_family
                              and row["size"] == best_size)
        if row["family"] == "direct_topk":
            row["n_edges"] = int(direct_topk_mask(edge_prior, row["size"]).sum())
        else:
            row["n_edges"] = int(roi_incident_mask(roi_prior, row["size"])[1])
    return best_mask, best_family, best_size, rows


# ══════════════════════════════════════════════════════════════════════
# NCR expert with per-ratio inner-CV diagnostics (identical to module grid)
# ══════════════════════════════════════════════════════════════════════

def fit_expert_ncr_multi_ratio(X_modality, y, mask, edge_laplacian, train_idx,
                               test_idx, lambda_grid=RIDGE_EXPERT_GRID,
                               ratio_grid=LAPLACIAN_RATIO_GRID, seed=0,
                               outer_fold=0, n_inner=3):
    """Replicates fit_expert_ncr_fixed but records per-ratio inner scores."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from metascfc.experiments.prior_subspace_expert_fusion_fix import (
        ExpertResult, _build_sub_laplacian)
    X_sub = X_modality[:, mask]
    n_sub = int(mask.sum())
    sub_lap = _build_sub_laplacian(mask, edge_laplacian, n_sub)
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 42)
    n_train = len(train_idx)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(n_inner, n_train // n_inner)
    fold_sizes[:n_train % n_inner] += 1
    per_ratio = {}
    best = (-np.inf, 0.0, 0.0)  # r, ratio, lambda
    for ratio in ratio_grid:
        ratio_best = (-np.inf, lambda_grid[0])
        for lam in lambda_grid:
            lambda_l = ratio * lam
            oof = np.full(n_train, np.nan)
            current = 0
            for k in range(n_inner):
                start, stop = current, current + fold_sizes[k]
                v_local = perm[start:stop]
                a_local = np.concatenate([perm[:start], perm[stop:]])
                scaler = StandardScaler()
                X_a_raw = scaler.fit_transform(X_sub[train_idx[a_local]])
                X_v_raw = scaler.transform(X_sub[train_idx[v_local]])
                y_mean_inner = float(y[train_idx[a_local]].mean())
                y_std_inner = max(float(y[train_idx[a_local]].std()), 1e-8)
                y_a = (y[train_idx[a_local]] - y_mean_inner) / y_std_inner
                if sub_lap is None or lambda_l == 0:
                    model = Ridge(alpha=lam, fit_intercept=False)
                    model.fit(X_a_raw, y_a)
                    oof[v_local] = model.predict(X_v_raw) * y_std_inner + y_mean_inner
                else:
                    X_a = np.hstack([X_a_raw, np.zeros_like(X_a_raw)])
                    X_v = np.hstack([X_v_raw, np.zeros_like(X_v_raw)])
                    ncr = NetworkConstrainedRidge(
                        alpha1=lam, alpha2=lambda_l, edge_laplacian=sub_lap,
                        standardize=False)
                    ncr.fit(X_a, y_a)
                    oof[v_local] = ncr.predict(X_v) * y_std_inner + y_mean_inner
                current = stop
            valid = np.isfinite(oof)
            if valid.sum() < 10:
                continue
            r, _ = pearsonr(oof[valid], y[train_idx][valid])
            if (r > ratio_best[0] + 1e-14 or
                    (abs(r - ratio_best[0]) < 1e-14 and lam < ratio_best[1])):
                ratio_best = (r, lam)
            if (r > best[0] + 1e-14 or
                    (abs(r - best[0]) < 1e-14 and ratio < best[1]) or
                    (abs(r - best[0]) < 1e-14 and ratio == best[1]
                     and lam < best[2])):
                best = (r, ratio, lam)
        per_ratio[ratio] = ratio_best
    best_r, best_ratio, best_lambda = best

    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(X_sub[train_idx])
    X_te_raw = scaler.transform(X_sub[test_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_tr = (y[train_idx] - y_mean) / y_std
    lambda_l = best_ratio * best_lambda
    if sub_lap is None or lambda_l == 0:
        model = Ridge(alpha=best_lambda, fit_intercept=False)
        model.fit(X_tr_raw, y_tr)
        beta_std = model.coef_[:n_sub].copy()
        test_pred = model.predict(X_te_raw) * y_std + y_mean
    else:
        X_tr = np.hstack([X_tr_raw, np.zeros_like(X_tr_raw)])
        X_te = np.hstack([X_te_raw, np.zeros_like(X_te_raw)])
        ncr = NetworkConstrainedRidge(alpha1=best_lambda, alpha2=lambda_l,
                                      edge_laplacian=sub_lap, standardize=False)
        ncr.fit(X_tr, y_tr)
        beta_std = ncr.beta()[:n_sub].copy()
        test_pred = ncr.predict(X_te) * y_std + y_mean
    beta_orig = beta_std * y_std / np.maximum(scaler.scale_, 1e-8)
    res = ExpertResult(
        mask=mask.copy(), n_selected=n_sub, mask_family="", mask_size=0,
        lambda_r=best_lambda, laplacian_ratio=best_ratio, expert_type="ncr",
        scaler_mean=scaler.mean_.copy(), scaler_scale=scaler.scale_.copy(),
        y_mean=y_mean, y_std=y_std, beta_standardized=beta_std,
        beta_original=beta_orig, test_pred=test_pred)
    diag = {"best_inner_r": best_r,
            "per_ratio_best_r": {str(k): float(v[0]) for k, v in per_ratio.items()},
            "per_ratio_best_lambda": {str(k): float(v[1]) for k, v in per_ratio.items()}}
    return res, diag


# ══════════════════════════════════════════════════════════════════════
# R0 fold + variant fits
# ══════════════════════════════════════════════════════════════════════

def fit_r0(X_fc, X_sc, y, tr, te, seed, fold, prior):
    a0 = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, tr, te,
                                 CONDITIONS["R0"], prior,
                                 n_fusion_folds=3, n_inner=INNER_FOLDS,
                                 n_final_cv=3)
    w = a0.fusion_weights
    base_oof = w["FP"] * a0.fp_oof + w["SC"] * a0.sc_oof
    W, c, sf, ss = r0_linear_map(X_fc, X_sc, y, tr, a0, np.ones(N_ROI) / N_ROI)
    Zt = np.hstack([sf.transform(X_fc[te]), ss.transform(X_sc[te])])
    recon = float(np.max(np.abs(Zt @ W + c - a0.fused_test_pred)))
    assert recon <= 1e-6, f"R0 map reconstruction {recon}"
    return {"pred": a0.fused_test_pred, "metrics": a0.fused_metrics,
            "W": W, "c": c, "fc_mean": sf.mean_, "fc_scale": sf.scale_,
            "sc_mean": ss.mean_, "sc_scale": ss.scale_,
            "base_oof": base_oof, "fusion_weights": dict(w),
            "recon_err": recon}


def fit_variant(X_fc, X_sc, y, prior, arch, tr, te, seed, fold,
                base_oof, base_test, masks=None):
    """One SPSEF variant (Ridge or NCR) with a given prior identity."""
    lap = build_edge_laplacian(N_ROI, prior_scores=prior, top_k=10)
    oof = generate_expert_crossfit_oof_fixed(
        X_fc, X_sc, y, prior, tr, seed, fold, n_fusion_folds=3,
        n_inner=INNER_FOLDS, expert_type=arch,
        edge_laplacian=lap if arch == "ncr" else None)
    eprior = build_edge_product_prior(prior)
    if masks is None:
        mfc, fam_fc, sz_fc = _select_best_mask_for_modality(
            X_fc, y, eprior, prior, tr, seed, fold, INNER_FOLDS)
        msc, fam_sc, sz_sc = _select_best_mask_for_modality(
            X_sc, y, eprior, prior, tr, seed, fold, INNER_FOLDS)
    else:
        mfc, fam_fc, sz_fc = masks["fc"]
        msc, fam_sc, sz_sc = masks["sc"]
    diag = {}
    if arch == "ncr":
        fc, diag_fc = fit_expert_ncr_multi_ratio(
            X_fc, y, mfc, lap, tr, te, seed=seed, outer_fold=fold,
            n_inner=INNER_FOLDS)
        sc, diag_sc = fit_expert_ncr_multi_ratio(
            X_sc, y, msc, lap, tr, te, seed=seed, outer_fold=fold,
            n_inner=INNER_FOLDS)
        diag = {"fc": diag_fc, "sc": diag_sc}
    else:
        fc = fit_expert_ridge_fixed(X_fc, y, mfc, tr, te, seed=seed,
                                    outer_fold=fold, n_inner=INNER_FOLDS)
        sc = fit_expert_ridge_fixed(X_sc, y, msc, tr, te, seed=seed,
                                    outer_fold=fold, n_inner=INNER_FOLDS)
    fc.mask_family, fc.mask_size = fam_fc, sz_fc
    sc.mask_family, sc.mask_size = fam_sc, sz_sc
    v = oof.fc_sc_weights
    expert_test = v["fc"] * fc.test_pred + v["sc"] * sc.test_pred
    fusion = hierarchical_fusion(y[tr], base_oof, oof.expert_fused_oof,
                                 base_test, expert_test, v["fc"], v["sc"])
    recon = validate_final_reconstruction(
        X_fc, X_sc, fc, sc, fusion.alpha, v["fc"], v["sc"], base_test, te,
        fusion.final_test_pred)
    assert recon <= 1e-8, f"final reconstruction {recon}"
    return {
        "arch": arch, "v_fc": float(v["fc"]), "alpha": float(fusion.alpha),
        "fc_mask": mfc, "sc_mask": msc,
        "fc_family": fam_fc, "fc_size": sz_fc,
        "sc_family": fam_sc, "sc_size": sz_sc,
        "fc_test": fc.test_pred, "sc_test": sc.test_pred,
        "expert_test": expert_test, "final_test": fusion.final_test_pred,
        "base_oof": base_oof, "expert_oof": oof.expert_fused_oof,
        "fc_oof": oof.fc_oof, "sc_oof": oof.sc_oof,
        "fc_beta_std": fc.beta_standardized, "sc_beta_std": sc.beta_standardized,
        "fc_beta_orig": fc.beta_original, "sc_beta_orig": sc.beta_original,
        "fc_scaler_mean": fc.scaler_mean, "fc_scaler_scale": fc.scaler_scale,
        "sc_scaler_mean": sc.scaler_mean, "sc_scaler_scale": sc.scaler_scale,
        "fc_y": (fc.y_mean, fc.y_std), "sc_y": (sc.y_mean, sc.y_std),
        "lambda_r_fc": fc.lambda_r, "lambda_r_sc": sc.lambda_r,
        "ratio_fc": fc.laplacian_ratio, "ratio_sc": sc.laplacian_ratio,
        "ncr_diag": diag, "recon_err": float(recon),
    }


def run_fold(seed, fold, X_fc, X_sc, y, priors, ids):
    ts0 = time.time()
    tr, te = None, None
    for _, f, tr_i, te_i in make_outer_splits(N510, [seed], OUTER_FOLDS):
        if f == fold:
            tr, te = tr_i, te_i
    prior_arr = priors["matched"]["array"]
    r0 = fit_r0(X_fc, X_sc, y, tr, te, seed, fold, prior_arr)
    out = {"seed": seed, "fold": fold, "tr": tr, "te": te,
           "train_mean_fc": X_fc[tr].mean(0), "train_mean_sc": X_sc[tr].mean(0),
           "r0": r0, "variants": {}, "candidates": {}, "runtime": {}}
    tb = time.time()
    # matched masks computed once, shared by Ridge and NCR matched variants
    eprior = build_edge_product_prior(prior_arr)
    masks, cand_rows = {}, []
    for mod, X, key in (("fc", X_fc, "fc"), ("sc", X_sc, "sc")):
        mask, fam, size, rows = select_mask_with_candidates(
            X, y, eprior, prior_arr, tr, seed, fold, INNER_FOLDS)
        ref = _select_best_mask_for_modality(X, y, eprior, prior_arr, tr,
                                             seed, fold, INNER_FOLDS)
        assert (mask == ref[0]).all() and fam == ref[1] and size == ref[2], \
            f"mask selection mismatch {mod} {seed}/{fold}"
        masks[mod] = (mask, fam, size)
        for row in rows:
            cand_rows.append({"modality": mod, **row})
    out["candidates"]["matched"] = cand_rows
    out["runtime"]["matched_masks"] = time.time() - tb
    for arch, tag in (("ridge", "R"), ("ncr", "N")):
        for pname in PRIOR_TYPES:
            tb = time.time()
            res = fit_variant(X_fc, X_sc, y, priors[pname]["array"], arch,
                              tr, te, seed, fold, r0["base_oof"], r0["pred"],
                              masks=masks if pname == "matched" else None)
            out["variants"][f"{tag}-{pname.upper()}"] = res
            out["runtime"][f"{tag}-{pname.upper()}"] = time.time() - tb
            print(f"    {tag}-{pname.upper()} [{time.time()-tb:.0f}s] "
                  f"v={res['v_fc']:.2f} alpha={res['alpha']:.2f} "
                  f"fc={res['fc_family']}/{res['fc_size']} "
                  f"sc={res['sc_family']}/{res['sc_size']}", flush=True)
    out["runtime"]["total"] = time.time() - ts0
    return out


def fold_path(seed, fold):
    return FOLDS / f"fold_seed{seed}_f{fold}.pkl"


# ══════════════════════════════════════════════════════════════════════
# Stage: audit
# ══════════════════════════════════════════════════════════════════════

def stage_audit():
    t0 = time.time()
    print("=" * 70); print("STAGE A: cohort audit + R0 historical audit"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    assert len(ids) == 510 and len(set(ids)) == 510
    FINAL510_TXT.write_text("\n".join(ids) + "\n")
    d412 = D412_TXT.read_text().strip().split("\n")
    d98 = D98_TXT.read_text().strip().split("\n")
    priors = load_priors()
    avail = {
        "fc_all_finite": bool(np.isfinite(X_fc).all()),
        "sc_all_finite": bool(np.isfinite(X_sc).all()),
        "wm_labels_finite": bool(np.isfinite(y_wm).all()),
        "n_edges": int(X_fc.shape[1]),
        "n_rois": int(N_ROI),
        "unique_subjects": int(len(set(ids))),
        "d412_prefix_ok": ids[:412] == d412,
        "d98_suffix_ok": ids[412:] == d98,
        "fc_shape": list(X_fc.shape), "sc_shape": list(X_sc.shape),
    }
    assert avail["fc_all_finite"] and avail["sc_all_finite"]
    assert avail["wm_labels_finite"]
    assert avail["n_edges"] == 6670 and avail["n_rois"] == 116
    (OUT / "COHORT_AUDIT.json").write_text(json.dumps({
        "n_subjects": len(ids), "n_unique": len(set(ids)),
        "cohort": "final510 = D412 union D98 (combined processed subjects)",
        "availability": avail, "manifest_sha256": sha256_file(FINAL510_TXT),
        "prior_hashes": {k: {"path": v["path"], "sha256": v["sha256"],
                             "n": v["n"]} for k, v in priors.items()},
        "aal116_ordering": "AAL116 index order 1..116 as in ROI prior tables",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"  cohort n={len(ids)} unique={len(set(ids))} edges={X_fc.shape[1]} OK")

    r0 = R0Baseline(X_fc[:412], X_sc[:412], y_wm[:412], y_fi[:412],
                    np.array([str(x) for x in range(412)]))
    rows = []
    for seed in range(10):
        for _, fold, tr, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            p = r0.fit_r0_predict("WM", tr, te, seed, fold, cache_tag="f510audit")
            rows.append({"seed": seed, "fold": fold,
                         **metrics(y_wm[:412][te], p)})
    df = pd.DataFrame(rows)
    r = float(df["pearson"].mean()); rm = float(df["rmse"].mean())
    ok = abs(r - EXPECTED_R0["WM"][0]) <= TOL_R and abs(rm - EXPECTED_R0["WM"][1]) <= TOL_RMSE
    (OUT / "BASELINE_AUDIT.json").write_text(json.dumps({
        "task": "WM", "protocol": "R0Baseline (corrected same-solver) seeds 0-9, "
                                   "5 outer folds on the 412 development cohort",
        "pearson": r, "rmse": rm,
        "expected_pearson": EXPECTED_R0["WM"][0],
        "expected_rmse": EXPECTED_R0["WM"][1],
        "tolerance": {"pearson": TOL_R, "rmse": TOL_RMSE},
        "status": "PASS" if ok else "FAIL"}, indent=2))
    df.to_csv(OUT / "baseline_audit_splits.csv", index=False)
    print(f"  R0 historical audit: r={r:.10f} RMSE={rm:.10f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("STATUS: FINAL510_BASELINE_AUDIT_FAILED")
    print(f"  Stage A done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage: benchmark (runs one full fold, saved for reuse by main)
# ══════════════════════════════════════════════════════════════════════

def stage_benchmark():
    t0 = time.time()
    print("=" * 70); print("STAGE: benchmark (one full 510 fold)"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    priors = load_priors()
    seed, fold = FINAL510_SEEDS[0], 0
    f = fold_path(seed, fold)
    if not f.exists():
        rec = run_fold(seed, fold, X_fc, X_sc, y_wm, priors, ids)
        pickle.dump(rec, open(f, "wb"))
    else:
        rec = pickle.load(open(f, "rb"))
    rt = rec["runtime"]
    # biomarker perturbation benchmark on the same fold
    tb = time.time()
    two_model_time = benchmark_perturbation_workload(rec, X_fc, X_sc, y_wm)
    rt["perturbation_workload_two_models_one_fold"] = time.time() - tb
    per_fold = rt["total"]
    n_remaining = len(FINAL510_SEEDS) * OUTER_FOLDS - 1
    est_main = n_remaining * per_fold
    est_biomarker = 25 * two_model_time * 4.0  # 8 models = 4x the 2-model workload
    est_ablations = 600.0
    est_reports = 900.0
    total = est_main + est_biomarker + est_ablations + est_reports
    lines = ["# FINAL 510 WM runtime estimate", "",
             "## Benchmark (seed 7171 fold 0, n_train=408, n_test=102)", ""]
    for k, v in rt.items():
        lines.append(f"- {k}: {v:.1f}s")
    lines += ["", "## Estimated remaining stages", "",
              f"- main (24 folds): {est_main/3600:.2f} h",
              f"- biomarker (25 folds): {est_biomarker/60:.1f} min",
              f"- ablations + reports: {(est_ablations+est_reports)/60:.1f} min",
              f"- ablation D adds a fixed-ratio outer sweep (seed 7171, ~3 min).",
              f"- **Total remaining: {total/3600:.2f} h ({total/60:.0f} min)**", "",
              f"FINAL510_ESTIMATED_RUNTIME: {int(total//3600)}h{int((total%3600)//60):02d}m",
              f"FINAL510_ESTIMATED_FINISH_FROM_START: "
              f"{int(total//3600)}h{int((total%3600)//60):02d}m",
              "Estimate printed before running the missing controls."]
    (OUT / "RUNTIME_ESTIMATE.md").write_text("\n".join(lines) + "\n")
    progress("benchmark", time.time() - t0, total, 1)
    print(f"FINAL510_ESTIMATED_RUNTIME: {int(total//3600)}h{int((total%3600)//60):02d}m")
    print(f"FINAL510_ESTIMATED_FINISH_FROM_START: {int(total//3600)}h{int((total%3600)//60):02d}m")
    print(f"  benchmark done in {time.time()-t0:.0f}s; per-fold={per_fold:.0f}s")


def benchmark_perturbation_workload(rec, X_fc, X_sc, y):
    """Time one fold's faithfulness workload (R-MATCHED and N-MATCHED)."""
    t0 = time.time()
    for model in ("R-MATCHED", "N-MATCHED"):
        _ = faithfulness_for_model(rec, X_fc, X_sc, y, model)
    return time.time() - t0


# ══════════════════════════════════════════════════════════════════════
# Biomarker machinery (used by stage_biomarker and benchmark)
# ══════════════════════════════════════════════════════════════════════

def coefficient_maps(rec, model):
    """Return (c_fc_full, c_sc_full, I_roi) or None if ABSTAINED."""
    v = rec["variants"][model]
    if v["alpha"] <= 0.0:
        return None
    c_fc = np.zeros(N_EDGE); c_sc = np.zeros(N_EDGE)
    c_fc[v["fc_mask"]] = v["alpha"] * v["v_fc"] * v["fc_beta_orig"]
    c_sc[v["sc_mask"]] = v["alpha"] * (1.0 - v["v_fc"]) * v["sc_beta_orig"]
    imp = np.zeros(N_ROI)
    np.add.at(imp, IU[0], np.abs(c_fc)); np.add.at(imp, IU[1], np.abs(c_fc))
    np.add.at(imp, IU[0], np.abs(c_sc)); np.add.at(imp, IU[1], np.abs(c_sc))
    return c_fc, c_sc, imp


def roi_edge_mask(roi_set):
    return np.isin(IU[0], list(roi_set)) | np.isin(IU[1], list(roi_set))


def masked_final_predictions(rec, X_fc, X_sc, model, te, roi_set=None,
                             edge_mask_fc=None, edge_mask_sc=None,
                             n_random=0):
    """Frozen-model predictions after replacing affected raw edges by train means."""
    v = rec["variants"][model]
    if edge_mask_fc is None:
        edge_mask_fc = roi_edge_mask(roi_set)
    if edge_mask_sc is None:
        edge_mask_sc = edge_mask_fc
    Xf = X_fc[te].copy(); Xs = X_sc[te].copy()
    Xf[:, edge_mask_fc] = rec["train_mean_fc"][edge_mask_fc]
    Xs[:, edge_mask_sc] = rec["train_mean_sc"][edge_mask_sc]
    # R0 via stored linear map
    r0 = rec["r0"]
    Zf = (Xf - r0["fc_mean"]) / r0["fc_scale"]
    Zs = (Xs - r0["sc_mean"]) / r0["sc_scale"]
    base = Zf @ r0["W"][:N_EDGE] + Zs @ r0["W"][N_EDGE:] + r0["c"]
    # experts via stored scalers/betas
    fsub = Xf[:, v["fc_mask"]]
    zf = (fsub - v["fc_scaler_mean"]) / v["fc_scaler_scale"]
    fpred = zf @ v["fc_beta_std"] * v["fc_y"][1] + v["fc_y"][0]
    ssub = Xs[:, v["sc_mask"]]
    zs = (ssub - v["sc_scaler_mean"]) / v["sc_scaler_scale"]
    spred = zs @ v["sc_beta_std"] * v["sc_y"][1] + v["sc_y"][0]
    expert = v["v_fc"] * fpred + (1 - v["v_fc"]) * spred
    return (1 - v["alpha"]) * base + v["alpha"] * expert


def faithfulness_for_model(rec, X_fc, X_sc, y, model):
    te = rec["te"]; y_te = y[te]
    maps = coefficient_maps(rec, model)
    if maps is None:
        return {"model": model, "seed": rec["seed"], "fold": rec["fold"],
                "abstained": 1}
    _, _, imp = maps
    order = np.argsort(imp)[::-1]
    base_rmse = metrics(y_te, rec["variants"][model]["final_test"])["rmse"]
    rng = np.random.RandomState(seed_of(rec["fold"]))
    rand5 = [rng.choice(N_ROI, 5, replace=False) for _ in range(N_RANDOM_MASKS)]
    rand10 = [rng.choice(N_ROI, 10, replace=False) for _ in range(N_RANDOM_MASKS)]
    out = {"model": model, "seed": rec["seed"], "fold": rec["fold"],
           "abstained": 0, "base_rmse": base_rmse}
    for name, rois in (("top5", order[:5]), ("top10", order[:10]),
                       ("bottom5", order[-5:]), ("bottom10", order[-10:])):
        p = masked_final_predictions(rec, X_fc, X_sc, model, te, roi_set=rois)
        out[f"{name}_delta_rmse"] = metrics(y_te, p)["rmse"] - base_rmse
    for name, sets in (("random5", rand5), ("random10", rand10)):
        d = []
        for rois in sets:
            p = masked_final_predictions(rec, X_fc, X_sc, model, te, roi_set=rois)
            d.append(metrics(y_te, p)["rmse"] - base_rmse)
        d = np.array(d)
        out[f"{name}_mean"] = float(d.mean())
        out[f"{name}_std"] = float(d.std())
        out[f"{name}_p95"] = float(np.percentile(d, 95))
        out[f"{name}_p5"] = float(np.percentile(d, 5))
        out[f"{name}_all"] = d
    out["top10_minus_random10"] = out["top10_delta_rmse"] - out["random10_mean"]
    out["top5_minus_random5"] = out["top5_delta_rmse"] - out["random5_mean"]
    out["top10_random10_percentile"] = float(
        np.mean(out["random10_all"] < out["top10_delta_rmse"]))
    return out


def stability_for_maps(maps):
    """Pairwise stability metrics across valid coefficient maps."""
    n = len(maps)
    keys = ["fc_edge_spearman", "sc_edge_spearman", "fc_top100_jaccard",
            "sc_top100_jaccard", "fc_top300_jaccard", "sc_top300_jaccard",
            "fc_top10_roi_jaccard", "sc_top10_roi_jaccard",
            "multimodal_top10_roi_jaccard", "multimodal_top20_roi_jaccard",
            "sign_consistency_fc", "sign_consistency_sc"]
    vals = {k: [] for k in keys}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = maps[i], maps[j]
            vals["fc_edge_spearman"].append(
                spearmanr(np.abs(a[0]), np.abs(b[0])).statistic)
            vals["sc_edge_spearman"].append(
                spearmanr(np.abs(a[1]), np.abs(b[1])).statistic)
            for tag, idx in (("fc", 0), ("sc", 1)):
                for kk, key in ((100, f"{tag}_top100_jaccard"),
                                (300, f"{tag}_top300_jaccard")):
                    sa = set(np.argsort(np.abs(a[idx]))[-kk:].tolist())
                    sb = set(np.argsort(np.abs(b[idx]))[-kk:].tolist())
                    vals[key].append(len(sa & sb) / len(sa | sb))
                ra = set(np.argsort(a[2])[-10:].tolist())
                rb = set(np.argsort(b[2])[-10:].tolist())
                vals[f"{tag}_top10_roi_jaccard"].append(
                    len(ra & rb) / len(ra | rb))
                top_a = set(np.argsort(np.abs(a[idx]))[-100:].tolist())
                top_b = set(np.argsort(np.abs(b[idx]))[-100:].tolist())
                common = top_a & top_b
                if common:
                    same = sum(1 for e in common
                               if np.sign(a[idx][e]) == np.sign(b[idx][e]))
                    vals[f"sign_consistency_{tag}"].append(same / len(common))
            for kk, key in ((10, "multimodal_top10_roi_jaccard"),
                            (20, "multimodal_top20_roi_jaccard")):
                ra = set(np.argsort(a[2])[-kk:].tolist())
                rb = set(np.argsort(b[2])[-kk:].tolist())
                vals[key].append(len(ra & rb) / len(ra | rb))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in vals.items()}, n


# ══════════════════════════════════════════════════════════════════════
# Stage: main
# ══════════════════════════════════════════════════════════════════════

def stage_main():
    t0 = time.time()
    print("=" * 70); print("STAGE B/C: final 510 WM nested CV"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    priors = load_priors()
    total = len(FINAL510_SEEDS) * OUTER_FOLDS
    done = 0
    for seed in FINAL510_SEEDS:
        for _, fold, tr, te in make_outer_splits(N510, [seed], OUTER_FOLDS):
            done += 1
            f = fold_path(seed, fold)
            if f.exists():
                print(f"  [{done}/{total}] seed={seed} fold={fold} cached", flush=True)
                continue
            print(f"  [{done}/{total}] seed={seed} fold={fold} "
                  f"tr={len(tr)} te={len(te)}", flush=True)
            rec = run_fold(seed, fold, X_fc, X_sc, y_wm, priors, ids)
            pickle.dump(rec, open(f, "wb"))
            elapsed = time.time() - t0
            per = elapsed / max(done, 1)
            progress("Stage_BC_main", elapsed, per * (total - done),
                     5 + 85 * done / total)
            print(f"  [{done}/{total}] seed={seed} fold={fold} "
                  f"[{rec['runtime']['total']:.0f}s]", flush=True)
    print(f"  Stage B/C done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage: biomarker
# ══════════════════════════════════════════════════════════════════════

def load_folds():
    recs = []
    for seed in FINAL510_SEEDS:
        for fold in range(OUTER_FOLDS):
            f = fold_path(seed, fold)
            assert f.exists(), f"missing fold {f}"
            recs.append(pickle.load(open(f, "rb")))
    return recs


def load_data():
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    return X_fc, X_sc, y_wm


def stage_biomarker():
    t0 = time.time()
    print("=" * 70); print("STAGE D/E: biomarker stability + faithfulness"); print("=" * 70)
    X_fc, X_sc, y = load_data()
    recs = load_folds()
    stat_rows, faith_rows = [], []
    for model in ALL_MODELS:
        maps, valid_ids, abstain = [], [], 0
        for rec in recs:
            m = coefficient_maps(rec, model)
            if m is None:
                abstain += 1
                continue
            maps.append(m); valid_ids.append((rec["seed"], rec["fold"]))
        st, n_pairs = stability_for_maps(maps)
        stat_rows.append({"model": model, "arch": ARCH_OF[model],
                          "prior": PRIOR_OF[model], "n_folds": len(recs),
                          "n_valid": len(maps), "n_abstained": abstain,
                          "n_pairs": n_pairs, **st})
        for rec in recs:
            fr = faithfulness_for_model(rec, X_fc, X_sc, y, model)
            faith_rows.append({k: v for k, v in fr.items()
                               if k not in ("random5_all", "random10_all")})
        print(f"  {model}: valid={len(maps)}/{len(recs)}", flush=True)
    pd.DataFrame(stat_rows).to_csv(OUT / "biomarker_stability.csv", index=False)
    pd.DataFrame(faith_rows).to_csv(OUT / "biomarker_faithfulness.csv", index=False)
    # export matched coefficient maps and top ROI ranking
    rank_rows = []
    for model in ("R-MATCHED", "N-MATCHED"):
        valid = [coefficient_maps(rec, model) for rec in recs]
        valid = [m for m in valid if m is not None]
        if not valid:
            continue
        mean_imp = np.mean([m[2] for m in valid], axis=0)
        order = np.argsort(mean_imp)[::-1]
        prior_df = pd.read_csv(PRIOR_FILES["matched"])
        labels = prior_df["roi_label"].tolist()
        for rank, i in enumerate(order, 1):
            rank_rows.append({"model": model, "rank": rank, "roi_index": int(i) + 1,
                              "roi_label": labels[i],
                              "mean_importance": float(mean_imp[i])})
        np.savez_compressed(OUT / "coefficients" / f"{model}_maps.npz",
                            mean_importance=mean_imp,
                            **{f"seed_s_f_fold": m[2] for m in valid[:0]})
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(OUT / "final_wm_biomarker_ranking.csv",
                                       index=False)
    print(f"  Stage D/E done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage: ablations
# ══════════════════════════════════════════════════════════════════════

def aggregate_model(recs, model, y, pred_key="final_test"):
    """Seed-wise outer-CV metrics for a model across all folds."""
    rows = []
    for seed in FINAL510_SEEDS:
        sub = sorted([r for r in recs if r["seed"] == seed],
                     key=lambda r: r["fold"])
        idx = np.concatenate([r["te"] for r in sub])
        pred = np.concatenate([r["variants"][model][pred_key] for r in sub])
        rows.append({"model": model, "seed": seed, **metrics(y[idx], pred)})
    return rows


def stage_ablations():
    t0 = time.time()
    print("=" * 70); print("STAGE F: prediction ablations A-H"); print("=" * 70)
    X_fc, X_sc, y = load_data()
    recs = load_folds()
    rows = []

    # A: R0
    for seed in FINAL510_SEEDS:
        sub = sorted([r for r in recs if r["seed"] == seed], key=lambda r: r["fold"])
        idx = np.concatenate([r["te"] for r in sub])
        pred = np.concatenate([r["r0"]["pred"] for r in sub])
        rows.append({"group": "A_backbone", "variant": "R0", "target": "WM",
                     "seed": seed, "regime": "outer_cv", **metrics(y[idx], pred)})
    # B/C/E/H: full variants
    runtime_by_model = {}
    for r in recs:
        for m, t in r["runtime"].items():
            if m in ALL_MODELS:
                runtime_by_model.setdefault(m, []).append(t)
    for model in ALL_MODELS:
        seed_rows = aggregate_model(recs, model, y)
        rt = float(np.mean(runtime_by_model.get(model, [np.nan])))
        for a in seed_rows:
            group = ("B_matched_ridge" if model == "R-MATCHED" else
                     "C_ridge_prior_identity" if ARCH_OF[model] == "ridge" else
                     "E_ncr_prior_identity")
            rows.append({"group": group, "runtime_s": rt, **a})
    # B components (R-MATCHED): fc only, sc only, expert, final
    for comp, key in (("FC_only", "fc_test"), ("SC_only", "sc_test"),
                      ("FC_SC_expert", "expert_test"), ("R0_plus_expert", "final_test")):
        for seed in FINAL510_SEEDS:
            sub = sorted([r for r in recs if r["seed"] == seed], key=lambda r: r["fold"])
            idx = np.concatenate([r["te"] for r in sub])
            pred = np.concatenate([r["variants"]["R-MATCHED"][key] for r in sub])
            rows.append({"group": "B_matched_ridge_components", "variant": comp,
                         "target": "WM", "seed": seed, "regime": "outer_cv",
                         **metrics(y[idx], pred)})
    # D: fixed-ratio NCR outer sweep (seed 7171) + inner-CV diagnostics
    sweep_path = OUT / "ablations_ratio_sweep.csv"
    if sweep_path.exists():
        sw = pd.read_csv(sweep_path)
        for ratio, sub in sw.groupby("ratio"):
            rows.append({"group": "D_ncr_ratio_outer", "variant": f"ratio={ratio}",
                         "target": "WM", "seed": int(sub["seed"].iloc[0]),
                         "regime": "outer_cv_reduced_budget",
                         "pearson": float(sub["pearson"].mean()),
                         "rmse": float(sub["rmse"].mean()),
                         "mae": float(sub["mae"].mean()),
                         "v_fc_mean": float(sub["v_fc"].mean()),
                         "alpha_mean": float(sub["alpha"].mean()),
                         "runtime_s": float(sub["runtime_s"].mean())})
    ratio_rows = []
    for rec in recs:
        for mod in ("fc", "sc"):
            diag = rec["variants"]["N-MATCHED"]["ncr_diag"].get(mod, {})
            for ratio, r in diag.get("per_ratio_best_r", {}).items():
                ratio_rows.append({"group": "D_ncr_ratio_inner_cv",
                                   "variant": f"ratio={ratio}", "modality": mod,
                                   "seed": rec["seed"], "fold": rec["fold"],
                                   "inner_r": r})
    rdf = pd.DataFrame(ratio_rows)
    if len(rdf):
        agg = rdf.groupby(["variant", "modality"], as_index=False).agg(
            inner_r=("inner_r", "mean"))
        for _, a in agg.iterrows():
            rows.append({"group": "D_ncr_ratio_inner_cv",
                         "variant": f"{a['variant']}_{a['modality']}",
                         "target": "WM", "seed": "all", "regime": "inner_cv",
                         "pearson": a["inner_r"], "rmse": np.nan, "mae": np.nan})
    # F/G: mask family and size from candidate inner-CV diagnostics
    cand_rows = []
    for rec in recs:
        for row in rec["candidates"].get("matched", []):
            cand_rows.append({"seed": rec["seed"], "fold": rec["fold"],
                              "model": "R-MATCHED/N-MATCHED", **row})
    cdf = pd.DataFrame(cand_rows)
    if len(cdf):
        cdf.to_csv(OUT / "mask_candidate_inner_cv.csv", index=False)
        rtpath = OUT / "mask_candidate_runtime.csv"
        rtime = (pd.read_csv(rtpath).groupby(["family", "size"])["runtime_s"]
                 .mean().to_dict()) if rtpath.exists() else {}
        for (fam, size), sub in cdf.groupby(["family", "size"]):
            rows.append({"group": "F_G_mask_family_size",
                         "variant": f"{fam}_{size}", "target": "WM", "seed": "all",
                         "regime": "inner_cv_selection",
                         "pearson": float(sub["inner_r"].mean()),
                         "rmse": float(sub["inner_rmse"].mean()),
                         "mae": float(sub["inner_mae"].mean()),
                         "n_selected": int(sub["selected"].sum()),
                         "runtime_s": float(rtime.get((fam, size), np.nan))})
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT / "prediction_ablations.csv", index=False)
    adf.to_csv(OUT / "ablation_prediction.csv", index=False)
    with open(OUT / "ABLATION_PREDICTION.md", "w") as fh:
        fh.write("# Prediction ablations (final 510 WM)\n\n")
        for g, sub in adf.groupby("group"):
            fh.write(f"## {g}\n\n```\n{sub.to_string(index=False)}\n```\n\n")
        if len(cdf):
            fh.write("## F/G mask candidate inner-CV detail\n\n```\n"
                     + cdf.groupby(["family", "size"]).agg(
                         inner_r=("inner_r", "mean"),
                         selected=("selected", "sum"),
                         n_edges=("n_edges", "mean")).to_string()
                     + "\n```\n")
    print(f"  Stage F done in {time.time()-t0:.0f}s; rows={len(adf)}")


# ══════════════════════════════════════════════════════════════════════
# Stage: fixed-ratio NCR outer sweep (ablation D, reduced budget seed 7171)
# ══════════════════════════════════════════════════════════════════════

def generate_expert_crossfit_oof_ratio(X_fc, X_sc, y, roi_prior, train_idx,
                                       seed, outer_fold, n_fusion_folds=3,
                                       n_inner=3, ratio_grid=(0.0,),
                                       edge_laplacian=None):
    """Crossfit OOF for a fixed ratio grid (same masks/folds as the module)."""
    n_train = len(train_idx)
    fc_oof = np.full(n_train, np.nan); sc_oof = np.full(n_train, np.nan)
    rng = np.random.RandomState(int(seed) * 10000 + int(outer_fold) * 100 + 777)
    perm = rng.permutation(n_train)
    fold_sizes = np.full(n_fusion_folds, n_train // n_fusion_folds)
    fold_sizes[:n_train % n_fusion_folds] += 1
    folds = []
    current = 0
    for _ in range(n_fusion_folds):
        start, stop = current, current + fold_sizes[_]
        a_local = np.concatenate([perm[:start], perm[stop:]])
        v_local = perm[start:stop]
        folds.append((a_local, v_local)); current = stop
    edge_prior = build_edge_product_prior(roi_prior)
    for a_local, v_local in folds:
        a_global, v_global = train_idx[a_local], train_idx[v_local]
        mfc, _, _ = _select_best_mask_for_modality(
            X_fc, y, edge_prior, roi_prior, a_global, seed, outer_fold, n_inner)
        msc, _, _ = _select_best_mask_for_modality(
            X_sc, y, edge_prior, roi_prior, a_global, seed, outer_fold, n_inner)
        fc, _ = fit_expert_ncr_multi_ratio(
            X_fc, y, mfc, edge_laplacian, a_global, v_global, seed=seed,
            outer_fold=outer_fold, n_inner=n_inner, ratio_grid=list(ratio_grid))
        sc, _ = fit_expert_ncr_multi_ratio(
            X_sc, y, msc, edge_laplacian, a_global, v_global, seed=seed,
            outer_fold=outer_fold, n_inner=n_inner, ratio_grid=list(ratio_grid))
        fc_oof[v_local] = fc.test_pred; sc_oof[v_local] = sc.test_pred
    return fc_oof, sc_oof


def stage_ratio_sweep():
    t0 = time.time()
    print("=" * 70); print("STAGE: fixed-ratio NCR outer sweep (seed 7171)"); print("=" * 70)
    X_fc, X_sc, y, _, _ = load_combined()
    priors = load_priors()
    prior = priors["matched"]["array"]
    rows = []
    for fold in range(OUTER_FOLDS):
        rec = pickle.load(open(fold_path(FINAL510_SEEDS[0], fold), "rb"))
        tr, te = rec["tr"], rec["te"]
        vm = rec["variants"]["N-MATCHED"]
        masks = {"fc": (vm["fc_mask"], vm["fc_family"], vm["fc_size"]),
                 "sc": (vm["sc_mask"], vm["sc_family"], vm["sc_size"])}
        base_oof = rec["r0"]["base_oof"]; base_test = rec["r0"]["pred"]
        lap = build_edge_laplacian(N_ROI, prior_scores=prior, top_k=10)
        for ratio in (0.0, 0.1, 0.3, 1.0):
            ts = time.time()
            if ratio == 0.0:
                final = rec["variants"]["R-MATCHED"]["final_test"]
                v_fc = rec["variants"]["R-MATCHED"]["v_fc"]
                alpha = rec["variants"]["R-MATCHED"]["alpha"]
                lam_fc = rec["variants"]["R-MATCHED"]["lambda_r_fc"]
                lam_sc = rec["variants"]["R-MATCHED"]["lambda_r_sc"]
            else:
                t1 = time.time()
                fc_oof, sc_oof = generate_expert_crossfit_oof_ratio(
                    X_fc, X_sc, y, prior, tr, FINAL510_SEEDS[0], fold,
                    n_fusion_folds=3, n_inner=INNER_FOLDS,
                    ratio_grid=(ratio,), edge_laplacian=lap)
                from metascfc.experiments.prior_subspace_expert_fusion_fix import (
                    search_fusion_weights_simple)
                w, _ = search_fusion_weights_simple(y[tr], fc_oof, sc_oof)
                v_fc = w["fc"]
                expert_oof = v_fc * fc_oof + (1 - v_fc) * sc_oof
                fc, _ = fit_expert_ncr_multi_ratio(
                    X_fc, y, masks["fc"][0], lap, tr, te, seed=FINAL510_SEEDS[0],
                    outer_fold=fold, n_inner=INNER_FOLDS, ratio_grid=(ratio,))
                sc, _ = fit_expert_ncr_multi_ratio(
                    X_sc, y, masks["sc"][0], lap, tr, te, seed=FINAL510_SEEDS[0],
                    outer_fold=fold, n_inner=INNER_FOLDS, ratio_grid=(ratio,))
                expert_test = v_fc * fc.test_pred + (1 - v_fc) * sc.test_pred
                fusion = hierarchical_fusion(y[tr], base_oof, expert_oof,
                                             base_test, expert_test, v_fc,
                                             1 - v_fc)
                final = fusion.final_test_pred
                alpha = fusion.alpha
                lam_fc, lam_sc = fc.lambda_r, sc.lambda_r
            m = metrics(y[te], final)
            rows.append({"ratio": ratio, "seed": FINAL510_SEEDS[0], "fold": fold,
                         "v_fc": float(v_fc), "alpha": float(alpha),
                         "lambda_r_fc": float(lam_fc), "lambda_r_sc": float(lam_sc),
                         "runtime_s": time.time() - ts, **m})
            print(f"  fold={fold} ratio={ratio} r={m['pearson']:.4f} "
                  f"[{time.time()-ts:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablations_ratio_sweep.csv", index=False)
    # representative per-candidate inner-CV runtime (fold 0, both modalities)
    rec = pickle.load(open(fold_path(FINAL510_SEEDS[0], 0), "rb"))
    ep = build_edge_product_prior(prior)
    timing_rows = []
    for mod, X in (("fc", X_fc), ("sc", X_sc)):
        for k in K_EDGE_GRID:
            ts = time.time()
            _eval_mask_inner_cv_metrics(X, y, direct_topk_mask(ep, k),
                                        rec["tr"], FINAL510_SEEDS[0], 0,
                                        INNER_FOLDS)
            timing_rows.append({"modality": mod, "family": "direct_topk",
                                "size": k, "runtime_s": time.time() - ts})
        for msize in M_ROI_GRID:
            mask, _ = roi_incident_mask(prior, msize)
            ts = time.time()
            _eval_mask_inner_cv_metrics(X, y, mask, rec["tr"],
                                        FINAL510_SEEDS[0], 0, INNER_FOLDS)
            timing_rows.append({"modality": mod, "family": "roi_incident",
                                "size": msize, "runtime_s": time.time() - ts})
    pd.DataFrame(timing_rows).to_csv(OUT / "mask_candidate_runtime.csv", index=False)
    print(f"  ratio sweep done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "benchmark", "main", "biomarker", "ablations",
                             "ratio_sweep"])
    args = ap.parse_args()
    {"audit": stage_audit, "benchmark": stage_benchmark, "main": stage_main,
     "biomarker": stage_biomarker, "ablations": stage_ablations,
     "ratio_sweep": stage_ratio_sweep}[args.stage]()


if __name__ == "__main__":
    main()
