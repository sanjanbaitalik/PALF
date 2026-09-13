#!/usr/bin/env python3
"""PALF 412 vs 510 sample-size scaling study (post-hoc, exploratory).

Paired design: outer folds over D412 only; Regime A trains on T412, Regime B
trains on T412 + D98; both predict the same V412.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
sys.path.insert(0, str(ROOT / "scripts_paper"))

from metascfc.experiments.palf_crossfit_ablation import (  # noqa: E402
    CONDITIONS, N_EDGE, N_ROI, build_condition_cache, evaluate_ablation_split,
    make_outer_splits)
from metascfc.experiments.prior_subspace_expert_fusion_fix import (  # noqa: E402
    K_EDGE_GRID, LAPLACIAN_RATIO_GRID, M_ROI_GRID, RIDGE_EXPERT_GRID,
    _select_best_mask_for_modality, build_edge_product_prior, direct_topk_mask,
    fit_expert_ncr_fixed, fit_expert_ridge_fixed,
    generate_expert_crossfit_oof_fixed, hierarchical_fusion, roi_incident_mask)
from metascfc.models.iclr_backbones.network_constrained_ridge import (  # noqa: E402
    build_edge_laplacian)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from phase4_wm_confirmation import preprocess_subject  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling"
STATE = OUT / "_state"; STATE.mkdir(parents=True, exist_ok=True)
CKPT = STATE / "splits"; CKPT.mkdir(parents=True, exist_ok=True)

DEV_FC = ROOT / "inputs/dataset_FC/FC_all.npy"
DEV_SC = ROOT / "inputs/dataset_SC/SC_all.npy"
DEV_YWM = ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
DEV_YFI = ROOT / "inputs/dataset_SC/label_all.npy"
DEV_SUBJECTS = ROOT / "inputs/dataset_SC/hcp_subjects_used.csv"
H_FC = ROOT / "data/hcp/processed/fc"
H_SC = ROOT / "data/hcp/processed/sc"
H_LAB = ROOT / "data/hcp/processed/labels.csv"
D412_TXT = ROOT / "data_splits/scaling_D412.txt"
D98_TXT = ROOT / "data_splits/scaling_D98.txt"
D510_TXT = ROOT / "data_splits/scaling_D510.txt"

PRIORS = {
    "WM_matched": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv",
    "FI_matched": ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv",
    "WM_shuffled": ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3_shuffled/roi_prior.csv",
    "FI_shuffled": ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3_shuffled/roi_prior.csv",
    "WM_random": ROOT / "outputs/priors/random_prior/aal116/roi_prior.csv",
    "FI_random": ROOT / "outputs/priors/random_prior/aal116/roi_prior.csv",
}
SCALING_OUTER_SEEDS = [7171, 7272, 7373, 7474, 7575]
OUTER_FOLDS = 5
INNER_FOLDS = 3
SCALING_BOOTSTRAP_SEED = 9401
N_BOOT = 10000
SUBSET_SEEDS = [9511, 9512, 9513]
N_RANDOM = 100
RANDOM_MASK_SEED = 9601
RANDOM_MASK5_SEED = 9602
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
TASKS = ["WM", "FI"]
PALF_MODELS = {"A0": "R0", "A1": "R3", "A2": "R1", "A3": "R2"}
EXPERT_TYPE = {"B0": "ridge", "B1": "ncr", "B2": "ncr", "B3": "ncr", "B4": "ncr"}
EXPERT_PRIOR = {"B0": "matched", "B1": "matched", "B2": "cross",
                "B3": "shuffled", "B4": "random"}
ALL_MODELS = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4"]
CONDITIONS_MAP = {k: CONDITIONS[k] for k in ["R0", "R1", "R2", "R3"]}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_prior(p) -> np.ndarray:
    return pd.read_csv(p)["prior_score"].values.astype(np.float64)


def prior_for(task, prior_type):
    if prior_type == "matched":
        return load_prior(PRIORS[f"{task}_matched"])
    if prior_type == "cross":
        other = "FI" if task == "WM" else "WM"
        return load_prior(PRIORS[f"{other}_matched"])
    return load_prior(PRIORS[f"{task}_{prior_type}"])


def load_combined():
    d412 = D412_TXT.read_text().strip().split("\n")
    d98 = D98_TXT.read_text().strip().split("\n")
    assert len(d412) == 412 and len(d98) == 98
    iu = np.triu_indices(116, 1)
    dev_fc = np.load(DEV_FC).astype(np.float64)
    dev_sc = np.load(DEV_SC).astype(np.float64)
    y_wm = np.concatenate([np.load(DEV_YWM).astype(np.float64), np.zeros(98)])
    y_fi = np.concatenate([np.load(DEV_YFI).astype(np.float64), np.zeros(98)])
    dev_ids = pd.read_csv(DEV_SUBJECTS, dtype={"subject": str})["subject"].tolist()
    assert dev_ids == d412
    lab = pd.read_csv(H_LAB, dtype={"subject": str}).set_index("subject")
    hf = np.zeros((98, len(iu[0]))); hs = np.zeros((98, len(iu[0])))
    for i, s in enumerate(d98):
        fc, sc = preprocess_subject(np.load(H_FC / f"{s}_fc.npy"),
                                    np.loadtxt(H_SC / s / "sc_116.csv", delimiter=","))
        hf[i] = fc[iu[0], iu[1]]; hs[i] = sc[iu[0], iu[1]]
        y_wm[412 + i] = float(lab.loc[s, "listsort_unadj"])
        y_fi[412 + i] = float(lab.loc[s, "label"])
    X_fc = np.vstack([dev_fc[:, iu[0], iu[1]], hf])
    X_sc = np.vstack([dev_sc[:, iu[0], iu[1]], hs])
    assert X_fc.shape == (510, 6670)
    return X_fc, X_sc, y_wm, y_fi, dev_ids + d98


def progress(stage, elapsed, remaining, pct):
    p = OUT / "RUNTIME_PROGRESS.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec[stage] = {"elapsed_seconds": round(elapsed, 1),
                  "estimated_remaining_seconds": round(remaining, 1),
                  "percent_complete": round(pct, 1),
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.write_text(json.dumps(rec, indent=2))


_M_BASE = None


def _r0_operator_base():
    """M_base for the R0 FP linear kernel operator (M = M_base / lambda_fc)."""
    global _M_BASE
    if _M_BASE is None:
        cache = build_condition_cache(np.ones(N_ROI) / N_ROI, CONDITIONS["R0"], N_ROI)
        active = np.asarray(cache.active_indices, dtype=int)
        dinv = np.asarray(cache.D_inv_sqrt, dtype=np.float64)
        U = np.asarray(cache.generalized_u, dtype=np.float64)
        M = np.zeros((N_EDGE, N_EDGE))
        WA = dinv[active][:, None] * U
        M[np.ix_(active, active)] = WA @ WA.T
        inactive = np.setdiff1d(np.arange(N_EDGE), active)
        if len(inactive):
            M[inactive, inactive] = dinv[inactive] ** 2
        _M_BASE = M
    return _M_BASE


def r0_linear_map(X_fc, X_sc, y, tr, res, prior_placeholder):
    """Build the validated R0 fused linear map from branch objects."""
    w = res.fusion_weights
    fp, sc = res.fp_final, res.sc_final
    alpha_fp = np.asarray(fp.alpha, dtype=np.float64)
    lam_fc = float(fp.selected_params["lambda_fc"])
    lam_l = float(fp.selected_params.get("lambda_l", 0.0))
    assert abs(lam_l) < 1e-12, "R0 FP branch must have lambda_l = 0"
    n_edges = X_fc.shape[1]
    M = _r0_operator_base() / lam_fc
    sf = StandardScaler().fit(X_fc[tr])
    ss = StandardScaler().fit(X_sc[tr])
    Xz = sf.transform(X_fc[tr])
    w_fp = (M @ (Xz.T @ alpha_fp)) / (2.0 * n_edges)
    sc_alpha = float(sc.selected_params["alpha"])
    sc_model = Ridge(alpha=sc_alpha, fit_intercept=True).fit(
        ss.transform(X_sc[tr]), y[tr])
    W = np.zeros(2 * n_edges)
    W[:n_edges] = w["FP"] * fp.y_std * w_fp
    W[n_edges:] = w["SC"] * sc_model.coef_
    c = w["FP"] * fp.y_mean + w["SC"] * float(sc_model.intercept_)
    return W, float(c), sf, ss


def _full_beta(beta_masked, mask):
    b = np.zeros(N_EDGE)
    b[mask] = beta_masked
    return b


def fit_target(X_fc, X_sc, y, tr, te, seed, fold, task):
    """All models for one target/regime/split."""
    prior = prior_for(task, "matched")
    out = {"preds": {}, "experts": {}}
    # A0 R0 with map
    a0 = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, tr, te,
                                 CONDITIONS_MAP["R0"], prior,
                                 n_fusion_folds=3, n_inner=INNER_FOLDS,
                                 n_final_cv=3)
    w = a0.fusion_weights
    base_oof = w["FP"] * a0.fp_oof + w["SC"] * a0.sc_oof
    base_test = a0.fused_test_pred
    W0, c0, sf, ss = r0_linear_map(X_fc, X_sc, y, tr, a0, np.ones(N_ROI) / N_ROI)
    Zt = np.hstack([sf.transform(X_fc[te]), ss.transform(X_sc[te])])
    recon = float(np.max(np.abs(Zt @ W0 + c0 - base_test)))
    assert recon <= 1e-6, f"R0 map recon {recon}"
    out["preds"]["A0"] = base_test
    out["r0_map"] = (W0, c0)
    out["r0_fc_scaler"] = (sf.mean_, sf.scale_)
    out["r0_sc_scaler"] = (ss.mean_, ss.scale_)
    # PALF A1-A3
    for mid, cond in PALF_MODELS.items():
        if mid == "A0":
            continue
        r = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, tr, te,
                                    CONDITIONS_MAP[cond], prior,
                                    n_fusion_folds=3, n_inner=INNER_FOLDS,
                                    n_final_cv=3)
        out["preds"][mid] = r.fused_test_pred
    # Experts B0-B4
    for mid in ["B0", "B1", "B2", "B3", "B4"]:
        pr = prior_for(task, EXPERT_PRIOR[mid])
        et = EXPERT_TYPE[mid]
        lap = build_edge_laplacian(N_ROI, prior_scores=pr, top_k=10)
        oof = generate_expert_crossfit_oof_fixed(
            X_fc, X_sc, y, pr, tr, seed, fold, n_fusion_folds=3,
            n_inner=INNER_FOLDS, expert_type=et,
            edge_laplacian=lap if et == "ncr" else None)
        eprior = build_edge_product_prior(pr)
        mfc, fam_fc, sz_fc = _select_best_mask_for_modality(
            X_fc, y, eprior, pr, tr, seed, fold, INNER_FOLDS)
        msc, fam_sc, sz_sc = _select_best_mask_for_modality(
            X_sc, y, eprior, pr, tr, seed, fold, INNER_FOLDS)
        if et == "ncr":
            fc = fit_expert_ncr_fixed(X_fc, y, mfc, lap, tr, te, seed=seed,
                                      outer_fold=fold, n_inner=INNER_FOLDS)
            sc = fit_expert_ncr_fixed(X_sc, y, msc, lap, tr, te, seed=seed,
                                      outer_fold=fold, n_inner=INNER_FOLDS)
        else:
            fc = fit_expert_ridge_fixed(X_fc, y, mfc, tr, te, seed=seed,
                                        outer_fold=fold, n_inner=INNER_FOLDS)
            sc = fit_expert_ridge_fixed(X_sc, y, msc, tr, te, seed=seed,
                                        outer_fold=fold, n_inner=INNER_FOLDS)
        v = oof.fc_sc_weights
        expert_test = v["fc"] * fc.test_pred + v["sc"] * sc.test_pred
        fusion = hierarchical_fusion(y[tr], base_oof, oof.expert_fused_oof,
                                     base_test, expert_test, v["fc"], v["sc"])
        out["preds"][mid] = fusion.final_test_pred
        out["experts"][mid] = {
            "final_test": fusion.final_test_pred, "expert_test": expert_test,
            "fc_test": fc.test_pred, "sc_test": sc.test_pred,
            "v_fc": float(v["fc"]), "alpha": float(fusion.alpha),
            "fc_mask": mfc, "sc_mask": msc,
            "fc_beta": _full_beta(fc.beta_standardized, mfc),
            "sc_beta": _full_beta(sc.beta_standardized, msc),
            "families": (fam_fc, sz_fc, fam_sc, sz_sc),
            "fc_scale": (fc.scaler_mean, fc.scaler_scale),
            "sc_scale": (sc.scaler_mean, sc.scaler_scale),
            "fc_y": (fc.y_mean, fc.y_std), "sc_y": (sc.y_mean, sc.y_std),
        }
    return out


def run_split(seed, fold, tr412, te, X_fc, X_sc, y_wm, y_fi, regime):
    tr = tr412 if regime == "A" else np.concatenate([tr412, np.arange(412, 510)])
    out = {"seed": seed, "fold": fold, "regime": regime,
           "tr412": tr412, "tr": tr, "te": te, "targets": {}}
    out["targets"]["WM"] = fit_target(X_fc, X_sc, y_wm, tr, te, seed, fold, "WM")
    out["targets"]["FI"] = fit_target(X_fc, X_sc, y_fi, tr, te, seed, fold, "FI")
    # training means for faithfulness
    out["train_mean_fc"] = X_fc[tr].mean(0)
    out["train_mean_sc"] = X_sc[tr].mean(0)
    return out


def stage_primary():
    t0 = time.time()
    print("=" * 70); print("STAGE 1: primary paired 412 vs +98"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    total = len(SCALING_OUTER_SEEDS) * OUTER_FOLDS
    done = 0
    for seed in SCALING_OUTER_SEEDS:
        for _, fold, tr412, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            done += 1
            f = CKPT / f"split_seed{seed}_fold{fold}.pkl"
            if f.exists():
                print(f"  [{done}/{total}] seed={seed} fold={fold} cached", flush=True)
                continue
            ts = time.time()
            rec = {"A": run_split(seed, fold, tr412, te, X_fc, X_sc, y_wm, y_fi, "A"),
                   "B": run_split(seed, fold, tr412, te, X_fc, X_sc, y_wm, y_fi, "B")}
            pickle.dump(rec, open(f, "wb"))
            print(f"  [{done}/{total}] seed={seed} fold={fold} "
                  f"[{time.time()-ts:.0f}s]", flush=True)
            progress("Stage1_primary", time.time() - t0,
                     (total - done) * (time.time() - t0) / max(done, 1), 5 + 45 * done / total)
    print(f"  Stage 1 done in {time.time()-t0:.0f}s")


def stage_integrity():
    t0 = time.time()
    print("=" * 70); print("STAGE 0: integrity + audits + ETA"); print("=" * 70)
    d412 = D412_TXT.read_text().strip().split("\n")
    d98 = D98_TXT.read_text().strip().split("\n")
    d510 = D510_TXT.read_text().strip().split("\n")
    assert len(d412) == 412 and len(d98) == 98 and len(d510) == 510
    assert set(d412).isdisjoint(d98) and set(d510) == set(d412) | set(d98)
    assert sha256_bytes(("\n".join(sorted(d98)) + "\n").encode()) == HOLDOUT_SHA
    (OUT / "MANIFEST_AUDIT.json").write_text(json.dumps({
        "D412": {"n": 412, "sha256": sha256_file(D412_TXT)},
        "D98": {"n": 98, "sha256": sha256_file(D98_TXT)},
        "D510": {"n": 510, "sha256": sha256_file(D510_TXT)},
        "historical_98_canonical_sha256": HOLDOUT_SHA,
        "disjoint": True, "union_exact": True,
        "claim_boundary": "post-hoc combined-cohort exploratory analysis; NOT independent validation",
        "phase4_result_preserved": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    rows = []
    for seed in range(10):
        for _, fold, tr, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            for task, y in (("WM", y_wm), ("FI", y_fi)):
                r = evaluate_ablation_split(
                    X_fc[:412], X_sc[:412], y[:412], seed, fold, tr, te,
                    CONDITIONS_MAP["R0"], prior_for(task, "matched"),
                    n_fusion_folds=3, n_inner=INNER_FOLDS, n_final_cv=3)
                rows.append({"task": task, "seed": seed, "fold": fold,
                             "pearson": r.fused_metrics["pearson"],
                             "rmse": r.fused_metrics["rmse"]})
    adf = pd.DataFrame(rows)
    exp = {"WM": (0.263515, 11.292921), "FI": (0.370917, 4.566689)}
    det = {}; ok_all = True
    for task in TASKS:
        sub = adf[adf["task"] == task]
        r, rm = float(sub["pearson"].mean()), float(sub["rmse"].mean())
        er, erm = exp[task]
        ok = abs(r - er) <= 5e-4 and abs(rm - erm) <= 0.05
        ok_all &= ok
        det[task] = {"r": r, "rmse": rm, "expected_r": er, "expected_rmse": erm,
                     "pass": bool(ok)}
        print(f"  R0 audit {task}: r={r:.10f} RMSE={rm:.10f} -> {'PASS' if ok else 'FAIL'}")
    (OUT / "BASELINE_AUDIT.json").write_text(json.dumps(
        {"details": det, "status": "PASS" if ok_all else "FAIL",
         "tolerance": {"pearson": 5e-4, "rmse": 0.05}}, indent=2))
    adf.to_csv(OUT / "baseline_audit_splits.csv", index=False)
    if not ok_all:
        raise SystemExit("STATUS: SCALING_BASELINE_AUDIT_FAILED")
    # benchmark
    tr, te = np.arange(330), np.arange(330, 412)
    bench = {}
    tb = time.time()
    evaluate_ablation_split(X_fc, X_sc, y_wm, 7171, 0, tr, te,
                            CONDITIONS_MAP["R0"], prior_for("WM", "matched"),
                            n_fusion_folds=3, n_inner=3, n_final_cv=3)
    bench["r0_fold"] = time.time() - tb
    tb = time.time()
    evaluate_ablation_split(X_fc, X_sc, y_wm, 7171, 0, tr, te,
                            CONDITIONS_MAP["R3"], prior_for("WM", "matched"),
                            n_fusion_folds=3, n_inner=3, n_final_cv=3)
    bench["palf_full_fold"] = time.time() - tb
    tb = time.time()
    trB = np.concatenate([tr, np.arange(412, 510)])
    r = evaluate_ablation_split(X_fc, X_sc, y_wm, 7171, 0, trB, te,
                                CONDITIONS_MAP["R0"], prior_for("WM", "matched"),
                                n_fusion_folds=3, n_inner=3, n_final_cv=3)
    bench["r0_fold_plus98"] = time.time() - tb
    est = {
        "Stage0_audit": 600,
        "Stage1_primary": 25 * 2 * 2 * 9 * bench["palf_full_fold"],
        "Stage2_secondary": 5 * (bench["palf_full_fold"] + bench["r0_fold_plus98"]) * 2,
        "Stage3_learning": 25 * 3 * 3 * (bench["r0_fold"] + bench["palf_full_fold"]),
        "Stage4_biomarker": 900, "Stage5_faithfulness": 1200, "Stage6_reports": 600,
    }
    total = sum(est.values())
    (OUT / "RUNTIME_ESTIMATE.md").write_text(
        "# Scaling study runtime estimate\n\n## Benchmark (development-only)\n\n"
        + "\n".join(f"- {k}: {v:.2f}s" for k, v in bench.items())
        + "\n\n## Estimated stages\n\n"
        + "\n".join(f"- {k}: {v/60:.1f} min" for k, v in est.items())
        + f"\n\n**Total: {total/3600:.2f} h ({total/60:.0f} min)**\n\n"
        "Planning estimate before benchmark: 4-8 h.\n")
    print(f"SCALING_STUDY_ESTIMATED_TOTAL_TIME: {int(total//3600)}h{int((total%3600)//60):02d}m")
    print(f"SCALING_STUDY_ESTIMATED_FINISH_FROM_START: {int(total//3600)}h{int((total%3600)//60):02d}m")
    progress("Stage0_integrity", time.time() - t0, total, 3)
    print(f"  Stage 0 done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["integrity", "primary", "learning", "biomarkers",
                             "secondary"])
    args = ap.parse_args()
    if args.stage == "integrity":
        stage_integrity()
    elif args.stage == "primary":
        stage_primary()
    elif args.stage == "learning":
        stage_learning()
    elif args.stage == "biomarkers":
        stage_biomarkers()
    elif args.stage == "secondary":
        stage_secondary()




# ══════════════════════════════════════════════════════════════════════
# Stage 3: learning curve (A0, B1) on fixed V412
# ══════════════════════════════════════════════════════════════════════

def _inner_folds(n, seed, tag, k=3):
    h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    perm = np.random.RandomState(h).permutation(n)
    sizes = np.full(k, n // k); sizes[:n % k] += 1
    folds, cur = [], 0
    for i in range(k):
        a, b = cur, cur + sizes[i]
        folds.append((perm[np.concatenate([perm[:a], perm[b:]])], perm[a:b]))
        cur = b
    return folds


def learning_point(X_fc, X_sc, y, tr412, te, seed, fold, target, n, subset_seed):
    """A0 and B1 trained on a subset of T_B = T412 + D98; evaluate on fixed te."""
    prior = prior_for(target, "matched")
    tB = np.concatenate([tr412, np.arange(412, 510)])
    if n >= len(tB):
        sub = tB
    elif n <= len(tr412):
        rng = np.random.RandomState(subset_seed)
        idx = rng.choice(len(tr412), min(n, len(tr412)), replace=False)
        sub = tr412[np.sort(idx)]
    else:
        rng = np.random.RandomState(subset_seed)
        idx = rng.choice(len(tB), n, replace=False)
        sub = tB[np.sort(idx)]
    # R0 OOF on sub (used for A0 fusion + B1 alpha), final test pred
    from metascfc.experiments.palf_crossfit_ablation import (
        generate_crossfit_oof as gco, search_fusion_weights as sfw)
    oof = gco(X_fc, X_sc, y, sub, CONDITIONS_MAP["R0"], np.ones(N_ROI) / N_ROI,
              seed, fold, ridge_grid=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
              n_fusion_folds=3, n_inner=INNER_FOLDS, n_rois=N_ROI)
    w, _ = sfw(y[sub], {"FP": oof.fp_oof, "SC": oof.sc_oof}, ["FP", "SC"])
    base_oof = w["FP"] * oof.fp_oof + w["SC"] * oof.sc_oof
    r0 = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, sub, te,
                                 CONDITIONS_MAP["R0"], prior,
                                 n_fusion_folds=3, n_inner=INNER_FOLDS,
                                 n_final_cv=3)
    a0_test = r0.fused_test_pred
    # B1 expert
    lap = build_edge_laplacian(N_ROI, prior_scores=prior, top_k=10)
    ep = build_edge_product_prior(prior)
    mfc, _, _ = _select_best_mask_for_modality(X_fc, y, ep, prior, sub, seed, fold, INNER_FOLDS)
    msc, _, _ = _select_best_mask_for_modality(X_sc, y, ep, prior, sub, seed, fold, INNER_FOLDS)
    fc = fit_expert_ncr_fixed(X_fc, y, mfc, lap, sub, te, seed=seed,
                              outer_fold=fold, n_inner=INNER_FOLDS)
    sc = fit_expert_ncr_fixed(X_sc, y, msc, lap, sub, te, seed=seed,
                              outer_fold=fold, n_inner=INNER_FOLDS)
    inner = _inner_folds(len(sub), seed + subset_seed, f"lc{fold}", 3)
    fc_oof = np.zeros(len(sub)); sc_oof = np.zeros(len(sub))
    fmu, fsd = X_fc[sub].mean(0), X_fc[sub].std(0)
    fsd = np.where(fsd < 1e-12, 1.0, fsd)
    smu, ssd = X_sc[sub].mean(0), X_sc[sub].std(0)
    ssd = np.where(ssd < 1e-12, 1.0, ssd)
    for B, C in inner:
        f1 = fit_expert_ncr_fixed(X_fc, y, mfc, lap, sub[B], sub[C],
                                  lambda_grid=[fc.lambda_r],
                                  ratio_grid=[fc.laplacian_ratio],
                                  seed=seed, outer_fold=fold, n_inner=3)
        s1 = fit_expert_ncr_fixed(X_sc, y, msc, lap, sub[B], sub[C],
                                  lambda_grid=[sc.lambda_r],
                                  ratio_grid=[sc.laplacian_ratio],
                                  seed=seed, outer_fold=fold, n_inner=3)
        fc_oof[C] = f1.test_pred; sc_oof[C] = s1.test_pred
    # v by training OOF
    best = (-9.0, 0.5)
    for vv in [round(i * 0.05, 4) for i in range(21)]:
        r = pearsonr(vv * fc_oof + (1 - vv) * sc_oof, y[sub]).statistic
        if r > best[0] + 1e-14:
            best = (r, vv)
    v_fc = best[1]
    expert_test = v_fc * fc.test_pred + (1 - v_fc) * sc.test_pred
    # alpha on R0 OOF vs expert OOF
    expert_oof = v_fc * fc_oof + (1 - v_fc) * sc_oof
    fusion = hierarchical_fusion(y[sub], base_oof, expert_oof,
                                 a0_test, expert_test, v_fc, 1 - v_fc)
    return {"A0": a0_test, "B1": fusion.final_test_pred,
            "n_eff": len(sub), "v_fc": v_fc, "alpha": fusion.alpha,
            "n_requested": n}


def stage_learning():
    t0 = time.time()
    print("=" * 70); print("STAGE 3: learning curve"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    sizes_spec = [250, 300, 350]
    rows = []
    nfold = len(SCALING_OUTER_SEEDS) * OUTER_FOLDS
    for seed in SCALING_OUTER_SEEDS:
        for _, fold, tr412, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            f = CKPT / f"split_seed{seed}_fold{fold}.pkl"
            rec = pickle.load(open(f, "rb"))
            for target, y in (("WM", y_wm), ("FI", y_fi)):
                sizes = sizes_spec + [len(tr412), len(rec["B"]["tr"])]
                for n in sizes:
                    seeds_use = [None] if n in (len(tr412), len(rec["B"]["tr"])) else SUBSET_SEEDS
                    # exact sets for full sizes (reuse primary checkpoints)
                    if n == len(tr412):
                        for mid in ("A0", "B1"):
                            rows.append({"seed": seed, "fold": fold, "target": target,
                                         "n_requested": n, "n_eff": len(tr412),
                                         "subset_seed": -1, "model": mid,
                                         "pred": rec["A"]["targets"][target]["preds"][mid]})
                        continue
                    if n == len(rec["B"]["tr"]):
                        for mid in ("A0", "B1"):
                            rows.append({"seed": seed, "fold": fold, "target": target,
                                         "n_requested": n, "n_eff": len(rec["B"]["tr"]),
                                         "subset_seed": -1, "model": mid,
                                         "pred": rec["B"]["targets"][target]["preds"][mid]})
                        continue
                    for ss in seeds_use:
                        out = learning_point(X_fc, X_sc, y, tr412, te, seed, fold,
                                             target, n, ss)
                        for mid in ("A0", "B1"):
                            rows.append({"seed": seed, "fold": fold, "target": target,
                                         "n_requested": n, "n_eff": out["n_eff"],
                                         "subset_seed": ss, "model": mid,
                                         "pred": out[mid]})
            print(f"  seed={seed} fold={fold} done", flush=True)
            progress("Stage3_learning", time.time() - t0, 0, 55)
    with open(STATE / "learning_preds.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    print(f"  Stage 3 done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage 4/5: biomarker stability + cross-fold faithfulness
# ══════════════════════════════════════════════════════════════════════

def _roi_from_edge(e):
    iu = np.triu_indices(N_ROI, 1)
    M = np.zeros((N_ROI, N_ROI)); M[iu[0], iu[1]] = e; M = M + M.T
    return np.abs(M).sum(1)


def _jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def _pairwise(metrics, arrays, names):
    out = {n: [] for n in names}
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            for n in names:
                out[n].append(metrics[n](arrays[i], arrays[j]))
    return {n: float(np.nanmean(v)) if v else np.nan for n, v in out.items()}


def stage_biomarkers():
    t0 = time.time()
    print("=" * 70); print("STAGE 4/5: biomarker stability + faithfulness"); print("=" * 70)
    from scipy.stats import spearmanr
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    splits = []
    for seed in SCALING_OUTER_SEEDS:
        for _, fold, tr412, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            splits.append((seed, fold, tr412, te))
    # ── stability ──
    stab_rows = []
    for target in TASKS:
        for mid in ["B0", "B1", "B2", "B3", "B4"]:
            for regime in ["A", "B"]:
                fc_maps, sc_maps, multim, signs, abst = [], [], [], [], 0
                for seed, fold, tr412, te in splits:
                    rec = pickle.load(open(CKPT / f"split_seed{seed}_fold{fold}.pkl", "rb"))
                    e = rec[regime]["targets"][target]["experts"][mid]
                    if e["alpha"] == 0:
                        abst += 1
                        continue
                    cf = e["alpha"] * e["v_fc"] * e["fc_beta"]
                    cs = e["alpha"] * (1 - e["v_fc"]) * e["sc_beta"]
                    fc_maps.append(cf); sc_maps.append(cs)
                    multim.append(_roi_from_edge(cf) + _roi_from_edge(cs))
                    signs.append(np.sign(cf) + np.sign(cs))
                if len(fc_maps) < 2:
                    continue
                res = {"target": target, "model": mid, "regime": regime,
                       "n_used": len(fc_maps), "abstained": abst}
                met = {
                    "fc_edge_spearman": lambda a, b: spearmanr(np.abs(a), np.abs(b)).statistic,
                    "sc_edge_spearman": lambda a, b: spearmanr(np.abs(a), np.abs(b)).statistic,
                    "fc_top100_jaccard": lambda a, b: _jaccard(np.argsort(-np.abs(a))[:100],
                                                               np.argsort(-np.abs(b))[:100]),
                    "sc_top100_jaccard": lambda a, b: _jaccard(np.argsort(-np.abs(a))[:100],
                                                               np.argsort(-np.abs(b))[:100]),
                    "fc_top10_roi_jaccard": lambda a, b: _jaccard(np.argsort(-_roi_from_edge(a))[:10],
                                                                  np.argsort(-_roi_from_edge(b))[:10]),
                    "sc_top10_roi_jaccard": lambda a, b: _jaccard(np.argsort(-_roi_from_edge(a))[:10],
                                                                  np.argsort(-_roi_from_edge(b))[:10]),
                    "multimodal_top10_roi_jaccard": lambda a, b: _jaccard(np.argsort(-a)[:10],
                                                                          np.argsort(-b)[:10]),
                    "multimodal_top20_roi_jaccard": lambda a, b: _jaccard(np.argsort(-a)[:20],
                                                                          np.argsort(-b)[:20]),
                }
                # modality-specific metrics
                for key, arrays, mets in [
                    ("fc", fc_maps, ["fc_edge_spearman", "fc_top100_jaccard", "fc_top10_roi_jaccard"]),
                    ("sc", sc_maps, ["sc_edge_spearman", "sc_top100_jaccard", "sc_top10_roi_jaccard"]),
                ]:
                    rr = _pairwise({m: met[m] for m in mets}, arrays, mets)
                    res.update(rr)
                rr = _pairwise({m: met[m] for m in
                                ["multimodal_top10_roi_jaccard", "multimodal_top20_roi_jaccard"]},
                               multim, ["multimodal_top10_roi_jaccard",
                                        "multimodal_top20_roi_jaccard"])
                res.update(rr)
                sg = np.array(signs)
                pos = (sg > 0).sum(0); neg = (sg < 0).sum(0)
                defined = (pos + neg) > 0
                res["sign_consistency"] = float(
                    np.mean(np.maximum(pos, neg)[defined] / (pos + neg)[defined])) if defined.any() else np.nan
                stab_rows.append(res)
    sdf = pd.DataFrame(stab_rows)
    sdf.to_csv(OUT / "biomarker_stability.csv", index=False)

    # ── cross-fold faithfulness ──
    faith_rows = []
    rs10 = np.random.RandomState(RANDOM_MASK_SEED)
    rand10 = [rs10.choice(N_ROI, 10, replace=False) for _ in range(N_RANDOM)]
    rs5 = np.random.RandomState(RANDOM_MASK5_SEED)
    rand5 = [rs5.choice(N_ROI, 5, replace=False) for _ in range(N_RANDOM)]
    iu = np.triu_indices(N_ROI, 1)
    for target in TASKS:
        for mid in ["B0", "B1"]:
            for regime in ["A", "B"]:
                for seed, fold, tr412, te in splits:
                    rec = pickle.load(open(CKPT / f"split_seed{seed}_fold{fold}.pkl", "rb"))
                    t = rec[regime]["targets"][target]
                    e = t["experts"][mid]
                    if e["alpha"] == 0:
                        faith_rows.append(dict(target=target, model=mid, regime=regime,
                                               seed=seed, fold=fold, abstained=1))
                        continue
                    W0, c0 = t["r0_map"]
                    fcm, fcs = t["r0_fc_scaler"]; scm, scs = t["r0_sc_scaler"]
                    a = e["alpha"]; v = e["v_fc"]
                    fym, fys = e["fc_y"]; sym, sys_ = e["sc_y"]
                    W = (1 - a) * W0
                    c = (1 - a) * c0
                    W[:N_EDGE] += a * v * fys * e["fc_beta"]
                    W[N_EDGE:] += a * (1 - v) * sys_ * e["sc_beta"]
                    c += a * (v * fym + (1 - v) * sym)
                    Z = np.hstack([(X_fc[te] - fcm) / np.maximum(fcs, 1e-12),
                                   (X_sc[te] - scm) / np.maximum(scs, 1e-12)])
                    y = (y_wm if target == "WM" else y_fi)[te]
                    base = Z @ W + c
                    r_un = np.sqrt(np.mean((base - y) ** 2))
                    contrib = Z * W[None, :]

                    def delta(roi_set):
                        fi = np.where(np.isin(iu[0], roi_set) | np.isin(iu[1], roi_set))[0]
                        p = base - contrib[:, fi].sum(1) - contrib[:, fi + N_EDGE].sum(1)
                        return float(np.sqrt(np.mean((p - y) ** 2)) - r_un)

                    rank = np.argsort(-(_roi_from_edge(e["alpha"] * v * e["fc_beta"]) +
                                        _roi_from_edge(e["alpha"] * (1 - v) * e["sc_beta"])))
                    row = dict(target=target, model=mid, regime=regime, seed=seed, fold=fold,
                               abstained=0,
                               top10=delta(rank[:10]), top5=delta(rank[:5]),
                               bottom10=delta(rank[-10:]), bottom5=delta(rank[-5:]),
                               random10_mean=float(np.mean([delta(s) for s in rand10])),
                               random5_mean=float(np.mean([delta(s) for s in rand5])))
                    row["top10_minus_random10"] = row["top10"] - row["random10_mean"]
                    faith_rows.append(row)
                print(f"  faith {target} {mid} {regime} done", flush=True)
    fdf = pd.DataFrame(faith_rows)
    fdf.to_csv(OUT / "biomarker_faithfulness.csv", index=False)
    progress("Stage4_5_biomarkers", time.time() - t0, 0, 90)
    print(f"  Stage 4/5 done in {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════
# Stage 2: secondary full-510 descriptive CV
# ══════════════════════════════════════════════════════════════════════

def stage_secondary():
    t0 = time.time()
    print("=" * 70); print("STAGE 2: secondary full-510 CV"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    rows = []
    for seed in [SCALING_OUTER_SEEDS[0]]:
        for _, fold, tr, te in make_outer_splits(510, [seed], OUTER_FOLDS):
            for target, y in (("WM", y_wm), ("FI", y_fi)):
                prior = prior_for(target, "matched")
                for mid, cond in [("A0", "R0"), ("A1", "R3")]:
                    r = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, tr, te,
                                                CONDITIONS_MAP[cond], prior,
                                                n_fusion_folds=3, n_inner=INNER_FOLDS,
                                                n_final_cv=3)
                    rows.append({"seed": seed, "fold": fold, "target": target,
                                 "model": mid, **r.fused_metrics})
                # B1
                lap = build_edge_laplacian(N_ROI, prior_scores=prior, top_k=10)
                ep = build_edge_product_prior(prior)
                mfc, _, _ = _select_best_mask_for_modality(X_fc, y, ep, prior, tr, seed, fold, INNER_FOLDS)
                msc, _, _ = _select_best_mask_for_modality(X_sc, y, ep, prior, tr, seed, fold, INNER_FOLDS)
                fc = fit_expert_ncr_fixed(X_fc, y, mfc, lap, tr, te, seed=seed, outer_fold=fold, n_inner=INNER_FOLDS)
                sc = fit_expert_ncr_fixed(X_sc, y, msc, lap, tr, te, seed=seed, outer_fold=fold, n_inner=INNER_FOLDS)
                oof = generate_expert_crossfit_oof_fixed(X_fc, X_sc, y, prior, tr, seed, fold,
                                                         n_fusion_folds=3, n_inner=INNER_FOLDS,
                                                         expert_type="ncr", edge_laplacian=lap)
                r0 = evaluate_ablation_split(X_fc, X_sc, y, seed, fold, tr, te,
                                             CONDITIONS_MAP["R0"], prior,
                                             n_fusion_folds=3, n_inner=INNER_FOLDS,
                                             n_final_cv=3)
                v = oof.fc_sc_weights
                et = v["fc"] * fc.test_pred + v["sc"] * sc.test_pred
                w = r0.fusion_weights
                base_oof = w["FP"] * r0.fp_oof + w["SC"] * r0.sc_oof
                fus = hierarchical_fusion(y[tr], base_oof, oof.expert_fused_oof,
                                          r0.fused_test_pred, et, v["fc"], v["sc"])
                from metascfc.benchmark_utils import prediction_metrics as pmet
                rows.append({"seed": seed, "fold": fold, "target": target,
                             "model": "B1", **pmet(y[te], fus.final_test_pred)})
            print(f"  seed={seed} fold={fold} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "full510_secondary_metrics.csv", index=False)
    print(f"  Stage 2 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
