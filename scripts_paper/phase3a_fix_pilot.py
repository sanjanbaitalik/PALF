#!/usr/bin/env python3
"""Phase 3A-FIX: corrected PG-MT-BCR development on 412 subjects.

Corrects the Phase-3A implementation:
  - uses validated corrected R0 (FP+SC cross-fitted fusion)
  - strict baseline audit (5e-4 / 0.05)
  - 3-fold inner OOF candidate evaluation
  - inner-train-only scalers and residual standardization
  - task-specific shared amplitudes
  - frozen full-batch Adam with factor projection
  - exports selection/optimizer/coefficient/biomarker outputs
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.phase3a_fix.bcr import (  # noqa: E402
    N_ROI,
    coefficient_matrices,
    predict_bcr,
    reconstruct_residual,
    roi_importance,
    shared_prior,
    to_edge,
    train_bcr,
)
from metascfc.phase3a_fix.r0_baseline import AccessLogger, R0Baseline  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────
FC_PATH = ROOT / "inputs" / "dataset_FC" / "FC_all.npy"
SC_PATH = ROOT / "inputs" / "dataset_SC" / "SC_all.npy"
Y_WM_PATH = ROOT / "inputs" / "dataset_SC" / "task_labels" / "ListSort_Unadj" / "label_all.npy"
Y_FI_PATH = ROOT / "inputs" / "dataset_SC" / "label_all.npy"
SUBJECTS_CSV = ROOT / "inputs" / "dataset_SC" / "hcp_subjects_used.csv"
PRIOR_WM_PATH = ROOT / "outputs" / "priors" / "llm" / "working_memory_contrastive_qwen3" / "roi_prior.csv"
PRIOR_FI_PATH = ROOT / "outputs" / "priors" / "llm" / "fluid_intelligence_contrastive_qwen3" / "roi_prior.csv"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"

OUT = ROOT / "outputs" / "iclr" / "palf_phase3a_fix_pg_mt_bcr"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "_checkpoints"
CKPT.mkdir(exist_ok=True)

# ── Frozen configuration ───────────────────────────────────────────────
PHASE3A_FIX_DEV_SEEDS = [3939, 4040, 4141, 4242]
AUDIT_SEEDS = list(range(10))
N_OUTER = 5
N_INNER = 3
ALPHA = [0.0, 0.25, 0.50, 0.75, 1.0]
LAMBDA_AMP_GRID = [0.1, 1.0, 10.0]
LAMBDA_PRIOR_GRID = [0.01, 0.1, 1.0]

SHUFFLE_SEED_WM, SHUFFLE_SEED_FI = 8101, 8102
RANDOM_SEED_WM, RANDOM_SEED_FI = 8103, 8104

# Fast validation mode only (never used for the frozen production run)
import os as _os
if _os.environ.get("P3AFIX_FAST"):
    LAMBDA_AMP_GRID = [1.0]
    LAMBDA_PRIOR_GRID = [0.1]
    ALPHA = [0.0, 0.5, 1.0]

EXPECTED_WM_R, EXPECTED_WM_RMSE = 0.263515, 11.292921
EXPECTED_FI_R, EXPECTED_FI_RMSE = 0.370917, 4.566689
TOL_PEARSON, TOL_RMSE = 5e-4, 0.05

HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"

MODEL_IDS = ["A0", "B0", "B1", "C0", "C1", "C2", "C3", "C4"]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def load_dev_data():
    FC = np.load(FC_PATH).astype(np.float64)
    SC = np.load(SC_PATH).astype(np.float64)
    y_wm = np.load(Y_WM_PATH).astype(np.float64)
    y_fi = np.load(Y_FI_PATH).astype(np.float64)
    subjects = pd.read_csv(SUBJECTS_CSV)["subject"].astype(str).tolist()
    assert FC.shape == (412, 116, 116), FC.shape
    assert SC.shape == (412, 116, 116), SC.shape
    assert len(subjects) == 412
    iu = np.triu_indices(116, k=1)
    fc_feat = FC[:, iu[0], iu[1]].copy()
    sc_feat = SC[:, iu[0], iu[1]].copy()
    return FC, SC, fc_feat, sc_feat, y_wm, y_fi, subjects


def load_prior(path):
    df = pd.read_csv(path)
    p = df["prior_score"].values.astype(np.float64)
    assert p.shape == (N_ROI,)
    return p


def make_controls(p_wm, p_fi):
    cross_wm, cross_fi = p_fi.copy(), p_wm.copy()
    shuf_wm = p_wm[np.random.RandomState(SHUFFLE_SEED_WM).permutation(N_ROI)]
    shuf_fi = p_fi[np.random.RandomState(SHUFFLE_SEED_FI).permutation(N_ROI)]
    rand_wm = np.random.RandomState(RANDOM_SEED_WM).uniform(0, 1, N_ROI)
    rand_fi = np.random.RandomState(RANDOM_SEED_FI).uniform(0, 1, N_ROI)
    return {
        "cross": (shared_prior(cross_wm, cross_fi), cross_wm, cross_fi),
        "shuffled": (shared_prior(shuf_wm, shuf_fi), shuf_wm, shuf_fi),
        "random": (shared_prior(rand_wm, rand_fi), rand_wm, rand_fi),
    }


def upper_to_sym(X_upper, n=116):
    M = np.zeros((X_upper.shape[0], n, n), dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    for i in range(X_upper.shape[0]):
        M[i][iu] = X_upper[i]
        M[i] = M[i] + M[i].T
    return M


def standardize_scope(fc_tr, sc_tr, fc_va, sc_va):
    """Fit edge scalers on training scope only; return symmetric standardized."""
    iu = np.triu_indices(116, k=1)
    def _std(Mtr, Mva):
        a = Mtr[:, iu[0], iu[1]]
        b = Mva[:, iu[0], iu[1]]
        mu, sd = a.mean(0), a.std(0)
        sd[sd < 1e-10] = 1.0
        return upper_to_sym((a - mu) / sd), upper_to_sym((b - mu) / sd)
    ftr, fva = _std(fc_tr, fc_va)
    str_, sva = _std(sc_tr, sc_va)
    return ftr, str_, fva, sva


def metrics(y_true, y_pred):
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    return {
        "pearson": r,
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
    }


def fisher_z(r):
    return float(np.arctanh(np.clip(r, -0.999, 0.999))) if np.isfinite(r) else 0.0


def partition(scope_idx, seed, tag, n_folds=3):
    scope_idx = np.asarray(scope_idx, dtype=int)
    h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    perm = np.random.RandomState(h).permutation(len(scope_idx))
    sizes = np.full(n_folds, len(scope_idx) // n_folds)
    sizes[: len(scope_idx) % n_folds] += 1
    folds, cur = [], 0
    for k in range(n_folds):
        start, stop = cur, cur + sizes[k]
        c = perm[start:stop]
        b = np.concatenate([perm[:start], perm[stop:]])
        folds.append((scope_idx[b], scope_idx[c]))
        cur = stop
    return folds


# ══════════════════════════════════════════════════════════════════════
# Candidate selection
# ══════════════════════════════════════════════════════════════════════

def select_mt(cands, base_oof_wm, base_oof_fi, y_wm_oof, y_fi_oof):
    """Select MT candidate + alpha pair using frozen rule. Returns best + all scored."""
    r_base_wm = metrics(y_wm_oof, base_oof_wm)["pearson"]
    r_base_fi = metrics(y_fi_oof, base_oof_fi)["pearson"]
    scored = []
    for c in cands:
        pW = base_oof_wm + c["alpha_WM"] * c["res_WM"]
        pF = base_oof_fi + c["alpha_FI"] * c["res_FI"]
        mW = metrics(y_wm_oof, pW)
        mF = metrics(y_fi_oof, pF)
        dW = mW["pearson"] - r_base_wm
        dF = mF["pearson"] - r_base_fi
        eligible = (dW >= -0.002) and (dF >= -0.002)
        fz = 0.5 * (fisher_z(mW["pearson"]) + fisher_z(mF["pearson"]))
        nrm = 0.5 * (mW["rmse"] / y_wm_oof.std() + mF["rmse"] / y_fi_oof.std())
        nmae = 0.5 * (mW["mae"] / y_wm_oof.std() + mF["mae"] / y_fi_oof.std())
        scored.append({**c, "mW": mW, "mF": mF, "dW": dW, "dF": dF,
                       "eligible": eligible, "fisher": fz, "nrmse": nrm, "nmae": nmae})
    elig = [s for s in scored if s["eligible"]]
    if not elig:
        best = {"lambda_amp": 1.0, "lambda_prior": 0.0,
                "alpha_WM": 0.0, "alpha_FI": 0.0,
                "res_WM": np.zeros_like(base_oof_wm),
                "res_FI": np.zeros_like(base_oof_fi),
                "fallback": True, "selected": True}
        for s in scored:
            s["selected"] = False
        return best, scored
    elig.sort(key=lambda s: (-s["fisher"], s["nrmse"], s["nmae"],
                             s["alpha_WM"] + s["alpha_FI"], s["lambda_prior"],
                             -s["lambda_amp"],
                             s["lambda_amp"], s["lambda_prior"],
                             s["alpha_WM"], s["alpha_FI"]))
    best = {**elig[0], "fallback": False, "selected": True}
    for s in scored:
        s["selected"] = False
    elig[0]["selected"] = True
    return best, scored


def select_st(cands, base_oof, y_oof):
    r_base = metrics(y_oof, base_oof)["pearson"]
    scored = []
    for c in cands:
        p = base_oof + c["alpha"] * c["res"]
        m = metrics(y_oof, p)
        d = m["pearson"] - r_base
        eligible = d >= -0.002
        scored.append({**c, "m": m, "d": d, "eligible": eligible, "fisher": fisher_z(m["pearson"])})
    elig = [s for s in scored if s["eligible"]]
    if not elig:
        return {"lambda_amp": 1.0, "lambda_prior": 0.0, "alpha": 0.0,
                "res": np.zeros_like(base_oof), "fallback": True}, scored
    elig.sort(key=lambda s: (-s["fisher"], s["m"]["rmse"], s["m"]["mae"],
                             s["alpha"], s["lambda_prior"], -s["lambda_amp"],
                             s["lambda_amp"], s["lambda_prior"], s["alpha"]))
    best = {**elig[0], "fallback": False}
    for s in scored:
        s["selected"] = False
    elig[0]["selected"] = True
    return best, scored


# ══════════════════════════════════════════════════════════════════════
# One outer split
# ══════════════════════════════════════════════════════════════════════

def run_outer_split(FC, SC, y_wm, y_fi, r0, controls, p_wm, p_fi,
                    train_idx, test_idx, seed, fold, opt_diag):
    t0 = time.time()
    inner_folds = partition(train_idx, seed, f"outer{fold}", N_INNER)

    # ── Precompute inner R0 + standardized matrices ──
    inner = []
    for fi, (B, C) in enumerate(inner_folds):
        cf = {t: r0.crossfit_r0_within(t, B, seed, f"{seed}:{fold}:{fi}")
              for t in ["WM", "FI"]}
        base = {t: r0.fit_r0_predict(t, B, C, seed, 700 + fi,
                                     cache_tag=f"{seed}:{fold}:{fi}") for t in ["WM", "FI"]}
        fcB, scB, fcC, scC = standardize_scope(FC[B], SC[B], FC[C], SC[C])
        res_WM = y_wm[B] - cf["WM"]
        res_FI = y_fi[B] - cf["FI"]
        st = {"B": B, "C": C, "fcB": fcB, "scB": scB, "fcC": fcC, "scC": scC,
              "base_WM": base["WM"], "base_FI": base["FI"],
              "res_WM": res_WM, "res_FI": res_FI,
              "z_WM": (res_WM - res_WM.mean()) / (res_WM.std() + 1e-12),
              "z_FI": (res_FI - res_FI.mean()) / (res_FI.std() + 1e-12),
              "mu_WM": res_WM.mean(), "sd_WM": res_WM.std(),
              "mu_FI": res_FI.mean(), "sd_FI": res_FI.std(),
              "yB_WM": y_wm[B], "yB_FI": y_fi[B]}
        inner.append(st)

    # ── Concatenated base oof over T ──
    nT = len(train_idx)
    base_oof_WM = np.zeros(nT); base_oof_FI = np.zeros(nT)
    y_oof_WM = np.zeros(nT); y_oof_FI = np.zeros(nT)
    pos = {int(g): i for i, g in enumerate(train_idx)}
    for st in inner:
        for g, v in zip(st["C"], st["base_WM"]):
            base_oof_WM[pos[int(g)]] = v
        for g, v in zip(st["C"], st["base_FI"]):
            base_oof_FI[pos[int(g)]] = v
        for g, v in zip(st["C"], st["yB_WM"]):
            y_oof_WM[pos[int(g)]] = v
        for g, v in zip(st["C"], st["yB_FI"]):
            y_oof_FI[pos[int(g)]] = v

    def _fit_residuals(st, kind, lam_amp, lam_prior, prior_sh, prior_wm, prior_fi,
                       tag, extra_diag):
        ctx = {"seed": seed, "fold": fold, "scope": tag, "kind": kind,
               "alpha_WM": None, "alpha_FI": None}
        ctx.update(extra_diag)
        return train_bcr(st["fcB"], st["scB"],
                         st["z_WM"] if kind in ("MT", "ST_WM") else None,
                         st["z_FI"] if kind in ("MT", "ST_FI") else None,
                         prior_sh, prior_wm, prior_fi, lam_amp, lam_prior,
                         kind=kind, seed=seed, diagnostics=opt_diag,
                         diag_context=ctx)

    def _oof_residuals(kind, lam_amp, lam_prior, prior_sh, prior_wm, prior_fi, tag, extra):
        """Return concatenated OOF standardized-residual predictions over T."""
        res = {"WM": np.zeros(nT), "FI": np.zeros(nT)}
        for fi, st in enumerate(inner):
            r = _fit_residuals(st, kind, lam_amp, lam_prior, prior_sh, prior_wm,
                               prior_fi, f"{tag}_if{fi}", extra)
            rWM, rFI = predict_bcr(r, st["fcC"], st["scC"])
            if rWM is not None:
                for g, v in zip(st["C"], rWM):
                    res["WM"][pos[int(g)]] = v
            if rFI is not None:
                for g, v in zip(st["C"], rFI):
                    res["FI"][pos[int(g)]] = v
        return res

    split_selection = {}

    # ═══ MT models ═══
    prior_sets = {
        "C0": (np.ones(N_ROI) / N_ROI, np.ones(N_ROI) / N_ROI, np.ones(N_ROI) / N_ROI, False),
        "C1": (shared_prior(p_wm, p_fi), p_wm, p_fi, True),
        "C2": (*controls["cross"], True),
        "C3": (*controls["shuffled"], True),
        "C4": (*controls["random"], True),
    }
    mt_preds = {}
    mt_sel = {}
    for mid, (psh, pwm, pfi, use_prior) in prior_sets.items():
        cands = []
        lp_grid = LAMBDA_PRIOR_GRID if use_prior else [0.0]
        for la in LAMBDA_AMP_GRID:
            for lp in lp_grid:
                res = _oof_residuals("MT", la, lp, psh, pwm, pfi, f"{mid}_la{la}_lp{lp}",
                                     {"lambda_amp": la, "lambda_prior": lp})
                for aW in ALPHA:
                    for aF in ALPHA:
                        cands.append({"lambda_amp": la, "lambda_prior": lp,
                                      "alpha_WM": aW, "alpha_FI": aF,
                                      "res_WM": res["WM"], "res_FI": res["FI"],
                                      "model": mid})
        best, scored = select_mt(cands, base_oof_WM, base_oof_FI, y_oof_WM, y_oof_FI)
        mt_sel[mid] = best
        split_selection[mid] = (best, scored)

    # ═══ ST models ═══
    st_sel = {}
    for mid, use_prior in [("B0", False), ("B1", True)]:
        bests = {}
        for task in ["WM", "FI"]:
            if task == "WM":
                pwm_use = p_wm if use_prior else np.ones(N_ROI) / N_ROI
                psh_use, pfi_use = pwm_use, np.ones(N_ROI) / N_ROI
                kind = "ST_WM"
            else:
                pfi_use = p_fi if use_prior else np.ones(N_ROI) / N_ROI
                psh_use, pwm_use = np.ones(N_ROI) / N_ROI, np.ones(N_ROI) / N_ROI
                kind = "ST_FI"
            cands = []
            lp_grid = LAMBDA_PRIOR_GRID if use_prior else [0.0]
            for la in LAMBDA_AMP_GRID:
                for lp in lp_grid:
                    res = _oof_residuals(kind, la, lp, psh_use, pwm_use, pfi_use,
                                         f"{mid}_{task}_la{la}_lp{lp}",
                                         {"lambda_amp": la, "lambda_prior": lp})
                    resid = res[task]
                    for a in ALPHA:
                        cands.append({"lambda_amp": la, "lambda_prior": lp, "alpha": a,
                                      "res": resid, "model": mid, "task": task})
            if task == "WM":
                best, scored = select_st(cands, base_oof_WM, y_oof_WM)
            else:
                best, scored = select_st(cands, base_oof_FI, y_oof_FI)
            bests[task] = best
            split_selection[f"{mid}_{task}"] = (best, scored)
        st_sel[mid] = bests

    # ══ Final outer refit ══
    cf_T = {t: r0.crossfit_r0_within(t, train_idx, seed, f"{seed}:{fold}:T")
            for t in ["WM", "FI"]}
    base_test = {t: r0.fit_r0_predict(t, train_idx, test_idx, seed, fold,
                                      cache_tag=f"{seed}:{fold}:test")
                 for t in ["WM", "FI"]}
    fcT, scT, fcV, scV = standardize_scope(FC[train_idx], SC[train_idx],
                                           FC[test_idx], SC[test_idx])
    resT_WM = y_wm[train_idx] - cf_T["WM"]
    resT_FI = y_fi[train_idx] - cf_T["FI"]
    zT_WM = (resT_WM - resT_WM.mean()) / (resT_WM.std() + 1e-12)
    zT_FI = (resT_FI - resT_FI.mean()) / (resT_FI.std() + 1e-12)

    results = {"A0": {"pred_WM": base_test["WM"], "pred_FI": base_test["FI"],
                      "alpha_WM": 0.0, "alpha_FI": 0.0}}
    final_models = {}

    # MT final
    for mid in ["C0", "C1", "C2", "C3", "C4"]:
        best = mt_sel[mid]
        psh, pwm, pfi, use_prior = prior_sets[mid]
        r = train_bcr(fcT, scT, zT_WM, zT_FI, psh, pwm, pfi,
                      best["lambda_amp"], best["lambda_prior"], kind="MT",
                      seed=seed, diagnostics=opt_diag,
                      diag_context={"seed": seed, "fold": fold, "scope": f"{mid}_final",
                                    "kind": "MT", "lambda_amp": best["lambda_amp"],
                                    "lambda_prior": best["lambda_prior"]})
        final_models[mid] = {"result": r,
                             "alpha_WM": best["alpha_WM"], "alpha_FI": best["alpha_FI"]}
        rWM, rFI = predict_bcr(r, fcV, scV)
        aW = best["alpha_WM"]; aF = best["alpha_FI"]
        predW = base_test["WM"] + aW * (rWM * resT_WM.std() + resT_WM.mean())
        predF = base_test["FI"] + aF * (rFI * resT_FI.std() + resT_FI.mean())
        if aW == 0.0:
            assert np.allclose(predW, base_test["WM"], atol=0, rtol=0), "alpha=0 WM mismatch"
        if aF == 0.0:
            assert np.allclose(predF, base_test["FI"], atol=0, rtol=0), "alpha=0 FI mismatch"
        results[mid] = {"pred_WM": predW, "pred_FI": predF,
                        "alpha_WM": aW, "alpha_FI": aF,
                        "lambda_amp": best["lambda_amp"],
                        "lambda_prior": best["lambda_prior"]}

    # ST final
    for mid in ["B0", "B1"]:
        for task in ["WM", "FI"]:
            best = st_sel[mid][task]
            use_prior = mid == "B1"
            if task == "WM":
                pwm_use = p_wm if use_prior else np.ones(N_ROI) / N_ROI
                psh_use, pfi_use = pwm_use, np.ones(N_ROI) / N_ROI
                kind = "ST_WM"
                ztr = zT_WM
            else:
                pfi_use = p_fi if use_prior else np.ones(N_ROI) / N_ROI
                psh_use, pwm_use = np.ones(N_ROI) / N_ROI, np.ones(N_ROI) / N_ROI
                kind = "ST_FI"
                ztr = zT_FI
            r = train_bcr(fcT, scT, ztr if task == "WM" else None,
                          ztr if task == "FI" else None,
                          psh_use, pwm_use, pfi_use,
                          best["lambda_amp"], best["lambda_prior"], kind=kind,
                          seed=seed, diagnostics=opt_diag,
                          diag_context={"seed": seed, "fold": fold, "scope": f"{mid}_{task}_final",
                                        "kind": kind, "lambda_amp": best["lambda_amp"],
                                        "lambda_prior": best["lambda_prior"]})
            rWM, rFI = predict_bcr(r, fcV, scV)
            a = best["alpha"]
            if task == "WM":
                predW = base_test["WM"] + a * (rWM * resT_WM.std() + resT_WM.mean())
                predF = base_test["FI"]
            else:
                predW = base_test["WM"]
                predF = base_test["FI"] + a * (rFI * resT_FI.std() + resT_FI.mean())
            if mid not in results:
                results[mid] = {}
            results[mid].setdefault("alpha_WM", 0.0)
            results[mid].setdefault("alpha_FI", 0.0)
            if task == "WM":
                results[mid]["pred_WM"] = predW
                results[mid]["alpha_WM"] = a
            else:
                results[mid]["pred_FI"] = predF
                results[mid]["alpha_FI"] = a
        results[mid].setdefault("pred_WM", base_test["WM"])
        results[mid].setdefault("pred_FI", base_test["FI"])

    # Metrics on test
    yte_WM = y_wm[test_idx]; yte_FI = y_fi[test_idx]
    split_metrics = {}
    for mid in MODEL_IDS:
        r = results[mid]
        split_metrics[mid] = {
            "WM": metrics(yte_WM, r["pred_WM"]),
            "FI": metrics(yte_FI, r["pred_FI"]),
            "alpha_WM": r.get("alpha_WM", 0.0),
            "alpha_FI": r.get("alpha_FI", 0.0),
            "lambda_amp": r.get("lambda_amp", None),
            "lambda_prior": r.get("lambda_prior", None),
        }

    # ── Coefficient / biomarker / faithfulness artifacts (C0, C1) ──
    artifacts = _compute_artifacts(final_models, fcV, scV, yte_WM, yte_FI,
                                   base_test, resT_WM, resT_FI, seed, fold)

    return {"seed": seed, "fold": fold, "metrics": split_metrics,
            "selection": split_selection, "runtime": time.time() - t0,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "test_idx": test_idx.tolist(), "train_idx": train_idx.tolist(),
            "artifacts": artifacts}


def _compute_artifacts(final_models, fcV, scV, yte_WM, yte_FI, base_test,
                       resT_WM, resT_FI, seed, fold):
    """Coefficient maps, reconstruction error, and faithfulness diagnostics."""
    Xe_fc = to_edge(fcV); Xe_sc = to_edge(scV)
    iu = np.triu_indices(116, k=1)
    out = {}
    for mid in ["C0", "C1"]:
        fm = final_models[mid]
        r = fm["result"]
        art = {"alpha_WM": fm["alpha_WM"], "alpha_FI": fm["alpha_FI"], "tasks": {}}
        for task, alpha, yt, base, sd, mu in [
            ("WM", fm["alpha_WM"], yte_WM, base_test["WM"], resT_WM.std(), resT_WM.mean()),
            ("FI", fm["alpha_FI"], yte_FI, base_test["FI"], resT_FI.std(), resT_FI.mean()),
        ]:
            sh, sp = coefficient_matrices(r, task)
            rpred = predict_bcr(r, fcV, scV)
            rpred_t = rpred[0] if task == "WM" else rpred[1]
            rec = reconstruct_residual(r, task, Xe_fc, Xe_sc)
            recon_err = float(np.max(np.abs(rec - rpred_t)))
            total = {m: alpha * (sh[m] + sp[m]) for m in ["FC", "SC"]}
            imp = roi_importance(r, task, alpha)
            rank = np.argsort(-imp)
            final_pred = base + alpha * (rpred_t * sd + mu)
            rmse_unmasked = float(np.sqrt(np.mean((final_pred - yt) ** 2)))
            abstained = (alpha == 0.0)

            def _deg(roi_set):
                if len(roi_set) == 0:
                    return 0.0
                mask = np.isin(iu[0], list(roi_set)) | np.isin(iu[1], list(roi_set))
                Xe_fc_m = Xe_fc.copy(); Xe_sc_m = Xe_sc.copy()
                Xe_fc_m[:, mask] = 0.0; Xe_sc_m[:, mask] = 0.0
                rec_m = reconstruct_residual(r, task, Xe_fc_m, Xe_sc_m)
                fp_m = base + alpha * (rec_m * sd + mu)
                return float(np.sqrt(np.mean((fp_m - yt) ** 2))) - rmse_unmasked

            deg = {}
            for k in [5, 10]:
                top = rank[:k]; bot = rank[-k:]
                rs = np.random.RandomState(seed * 1000 + fold * 10 + k)
                rand_degs = []
                for _ in range(50):
                    rs_set = rs.choice(116, k, replace=False)
                    rand_degs.append(_deg(rs_set))
                deg[f"top{k}"] = _deg(top)
                deg[f"bottom{k}"] = _deg(bot)
                deg[f"random{k}_mean"] = float(np.mean(rand_degs))
                deg[f"random{k}_std"] = float(np.std(rand_degs))
            art["tasks"][task] = {
                "recon_err": recon_err, "abstained": bool(abstained),
                "rmse_unmasked": rmse_unmasked, "faithfulness": deg,
                "shared": {m: sh[m] for m in ["FC", "SC"]},
                "specific": {m: sp[m] for m in ["FC", "SC"]},
                "total": total,
                "roi_importance": imp,
            }
        out[mid] = art
    return out


# ══════════════════════════════════════════════════════════════════════
# Baseline audit
# ══════════════════════════════════════════════════════════════════════

def run_audit(r0, fc_feat, sc_feat):
    print("\n" + "=" * 70)
    print("STRICT CORRECTED R0 BASELINE AUDIT (seeds 0-9, 5 outer folds)")
    print("=" * 70)
    rows = []
    for seed in AUDIT_SEEDS:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(412)
        sizes = np.full(5, 412 // 5); sizes[: 412 % 5] += 1
        cur = 0
        for fold in range(5):
            start, stop = cur, cur + sizes[fold]
            te = idx[start:stop]
            tr = np.concatenate([idx[:start], idx[stop:]])
            cur = stop
            for task, y in [("WM", r0.y["WM"]), ("FI", r0.y["FI"])]:
                pred = r0.fit_r0_predict(task, tr, te, seed, fold,
                                         cache_tag="audit")
                m = metrics(y[te], pred)
                rows.append({"task": task, "seed": seed, "fold": fold, **m})
    df = pd.DataFrame(rows)
    out = {}
    for task, exp_r, exp_rmse in [("WM", EXPECTED_WM_R, EXPECTED_WM_RMSE),
                                  ("FI", EXPECTED_FI_R, EXPECTED_FI_RMSE)]:
        sub = df[df["task"] == task]
        r = sub["pearson"].mean(); rmse = sub["rmse"].mean()
        ok = (abs(r - exp_r) <= TOL_PEARSON) and (abs(rmse - exp_rmse) <= TOL_RMSE)
        out[task] = {"r": float(r), "rmse": float(rmse),
                     "expected_r": exp_r, "expected_rmse": exp_rmse,
                     "pass": bool(ok),
                     "r_err": float(abs(r - exp_r)), "rmse_err": float(abs(rmse - exp_rmse))}
        print(f"  {task}: r={r:.10f} (exp {exp_r}, err {abs(r-exp_r):.2e}) "
              f"RMSE={rmse:.10f} (exp {exp_rmse}, err {abs(rmse-exp_rmse):.2e}) "
              f"-> {'PASS' if ok else 'FAIL'}")
    df.to_csv(OUT / "audit_split_metrics.csv", index=False)
    passed = all(out[t]["pass"] for t in ["WM", "FI"])
    with open(OUT / "BASELINE_AUDIT.json", "w") as f:
        json.dump({"details": out, "status": "PASS" if passed else "FAIL",
                   "tolerance_pearson": TOL_PEARSON, "tolerance_rmse": TOL_RMSE}, f, indent=2)
    return passed, out


# ══════════════════════════════════════════════════════════════════════
# Post-processing
# ══════════════════════════════════════════════════════════════════════

def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _topk_idx(vec, k):
    return list(np.argsort(-np.abs(vec))[:k])


def export_coefficients(split_rows):
    from scipy.stats import spearmanr
    coeff_dir = OUT / "coefficients"
    coeff_dir.mkdir(exist_ok=True)
    recon_errs = []
    stab_rows = []

    for mid in ["C0", "C1"]:
        for task in ["WM", "FI"]:
            edges = []          # (n_splits, 6670) FC total
            shared_edges = []
            specific_edges = []
            roi = []
            signs = []
            for out in split_rows:
                top_art = out["artifacts"][mid]
                alpha_t = top_art["alpha_WM"] if task == "WM" else top_art["alpha_FI"]
                art = top_art["tasks"][task]
                recon_errs.append({"model": mid, "task": task, "seed": out["seed"],
                                   "fold": out["fold"], "recon_err": art["recon_err"],
                                   "abstained": art["abstained"]})
                total_fc = art["total"]["FC"]
                iu = np.triu_indices(116, k=1)
                edges.append(total_fc[iu].copy())
                shared_edges.append(art["shared"]["FC"][iu].copy())
                specific_edges.append(art["specific"]["FC"][iu].copy())
                roi.append(art["roi_importance"].copy())
                signs.append(np.sign(total_fc[iu]))
                np.savez(coeff_dir / f"{mid}_{task}_seed{out['seed']}_fold{out['fold']}.npz",
                         shared_fc=art["shared"]["FC"], shared_sc=art["shared"]["SC"],
                         specific_fc=art["specific"]["FC"], specific_sc=art["specific"]["SC"],
                         total_fc=total_fc, total_sc=art["total"]["SC"],
                         roi_importance=art["roi_importance"],
                         alpha=np.array([alpha_t]))
            edges = np.array(edges); roi = np.array(roi); signs = np.array(signs)
            shared_edges = np.array(shared_edges); specific_edges = np.array(specific_edges)
            n = len(edges)
            spear = []
            j100 = []; j300 = []
            for i in range(n):
                for j in range(i + 1, n):
                    spear.append(spearmanr(np.abs(edges[i]), np.abs(edges[j])).statistic)
                    j100.append(_jaccard(_topk_idx(edges[i], 100), _topk_idx(edges[j], 100)))
                    j300.append(_jaccard(_topk_idx(edges[i], 300), _topk_idx(edges[j], 300)))
            rj10 = []; rj20 = []
            for i in range(n):
                for j in range(i + 1, n):
                    rj10.append(_jaccard(_topk_idx(roi[i], 10), _topk_idx(roi[j], 10)))
                    rj20.append(_jaccard(_topk_idx(roi[i], 20), _topk_idx(roi[j], 20)))
            pos = (signs > 0).sum(0); neg = (signs < 0).sum(0)
            defined = (pos + neg) > 0
            sign_cons = float(np.mean(np.maximum(pos, neg)[defined] / (pos + neg)[defined])) if defined.any() else 0.0

            def _pair_spear(arr):
                vals = []
                for i in range(n):
                    for j in range(i + 1, n):
                        vals.append(spearmanr(np.abs(arr[i]), np.abs(arr[j])).statistic)
                return float(np.mean(vals))
            stab_rows.append({
                "model": mid, "task": task,
                "edge_spearman": float(np.nanmean(spear)),
                "top100_jaccard": float(np.mean(j100)),
                "top300_jaccard": float(np.mean(j300)),
                "roi_top10_jaccard": float(np.mean(rj10)),
                "roi_top20_jaccard": float(np.mean(rj20)),
                "sign_consistency": sign_cons,
                "shared_edge_spearman": _pair_spear(shared_edges),
                "specific_edge_spearman": _pair_spear(specific_edges),
            })
    pd.DataFrame(recon_errs).to_csv(OUT / "coefficients" / "reconstruction_errors.csv", index=False)
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT / "biomarkers" / "stability.csv", index=False)
    return stab, recon_errs


def export_faithfulness(split_rows):
    rows = []
    for mid in ["C0", "C1"]:
        for task in ["WM", "FI"]:
            for k in [5, 10]:
                top = [o["artifacts"][mid]["tasks"][task]["faithfulness"][f"top{k}"]
                       for o in split_rows]
                bot = [o["artifacts"][mid]["tasks"][task]["faithfulness"][f"bottom{k}"]
                       for o in split_rows]
                rnd = [o["artifacts"][mid]["tasks"][task]["faithfulness"][f"random{k}_mean"]
                       for o in split_rows]
                rows.append({"model": mid, "task": task, "k": k,
                             "top_mean": float(np.mean(top)), "top_median": float(np.median(top)),
                             "bottom_mean": float(np.mean(bot)),
                             "random_mean": float(np.mean(rnd))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "biomarkers" / "faithfulness.csv", index=False)
    return df


def make_plots(df, decomp, stab, faith):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)

    models = MODEL_IDS
    wm = [df[df["model"] == m]["r_WM"].mean() for m in models]
    fi = [df[df["model"] == m]["r_FI"].mean() for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, wm, 0.4, label="WM")
    ax.bar(x + 0.2, fi, 0.4, label="FI")
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Pearson r")
    ax.set_title("Phase 3A-FIX model comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_model_comparison.pdf")
    fig.savefig(plots / "fig_phase3a_fix_model_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["C1-A0", "C1-C0", "C1-C2", "C1-C3", "C1-C4"]:
        ax.plot(range(1, len(decomp[name]["WM"]["deltas"]) + 1),
                decomp[name]["WM"]["deltas"], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("seed index")
    ax.set_ylabel("WM delta r"); ax.set_title("Seed-level deltas (WM)"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_seed_deltas.pdf")
    fig.savefig(plots / "fig_phase3a_fix_seed_deltas.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([0, 1], [decomp["C1-C0"]["WM"]["mean"], decomp["C1-C0"]["FI"]["mean"]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WM", "FI"])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("delta r")
    ax.set_title("Architecture-matched prior effect (C1-C0)")
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_prior_effect.pdf")
    fig.savefig(plots / "fig_phase3a_fix_prior_effect.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([0, 1], [decomp["C0-B0"]["WM"]["mean"], decomp["C0-B0"]["FI"]["mean"]],
           label="C0-B0", alpha=0.7)
    ax.bar([0, 1], [decomp["C1-B1"]["WM"]["mean"], decomp["C1-B1"]["FI"]["mean"]],
           label="C1-B1", alpha=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WM", "FI"]); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("delta r"); ax.set_title("Multi-task effect"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_multitask_effect.pdf")
    fig.savefig(plots / "fig_phase3a_fix_multitask_effect.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r.model}-{r.task}" for r in stab.itertuples()]
    ax.bar(range(len(labels)), stab["edge_spearman"].values)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean pairwise Spearman"); ax.set_title("Factor/coefficient stability")
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_factor_stability.pdf")
    fig.savefig(plots / "fig_phase3a_fix_factor_stability.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    f5 = faith[faith["k"] == 5]
    labels = [f"{r.model}-{r.task}" for r in f5.itertuples()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, f5["top_mean"].values, 0.4, label="top")
    ax.bar(x + 0.2, f5["random_mean"].values, 0.4, label="random")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("RMSE degradation"); ax.set_title("Biomarker faithfulness (k=5)"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "fig_phase3a_fix_biomarker_faithfulness.pdf")
    fig.savefig(plots / "fig_phase3a_fix_biomarker_faithfulness.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("Phase 3A-FIX: corrected PG-MT-BCR development")
    print("=" * 70)
    t0 = time.time()

    # ── Holdout seal verification ──
    holdout_ids = HOLDOUT_TXT.read_text().strip().split("\n")
    dev_ids = DEV_TXT.read_text().strip().split("\n")
    canonical = "\n".join(sorted(holdout_ids)) + "\n"
    sha = hashlib.sha256(canonical.encode()).hexdigest()
    assert len(holdout_ids) == 98 and len(set(holdout_ids)) == 98
    assert len(dev_ids) == 412 and len(set(dev_ids)) == 412
    assert sha == HOLDOUT_SHA, sha
    assert set(dev_ids).isdisjoint(set(holdout_ids))
    print(f"Holdout seal OK: n=98 sha={sha[:16]}...")
    with open(OUT / "HOLDOUT_SEAL_REPORT.json", "w") as f:
        json.dump({"n_subjects": 98, "canonical_sha256": sha,
                   "dev_count": 412, "intersection": 0,
                   "status": "SEALED_UNTOUCHED_FOR_PHASE3A_FIX"}, f, indent=2)

    # ── Load development data ──
    FC, SC, fc_feat, sc_feat, y_wm, y_fi, subjects = load_dev_data()
    print(f"Loaded development cohort: {len(subjects)} subjects")

    access = AccessLogger(OUT / "PHASE3A_DATA_ACCESS_LOG.jsonl")
    access.set_holdout(holdout_ids)
    access.log("development_cohort", subjects)

    p_wm = load_prior(PRIOR_WM_PATH)
    p_fi = load_prior(PRIOR_FI_PATH)
    controls = make_controls(p_wm, p_fi)
    prior_wm_hash = hashlib.sha256(p_wm.tobytes()).hexdigest()
    prior_fi_hash = hashlib.sha256(p_fi.tobytes()).hexdigest()
    print(f"Priors: WM hash={prior_wm_hash[:16]} FI hash={prior_fi_hash[:16]}")

    # Save prior controls
    pd.DataFrame({"roi_index": np.arange(1, 117), "prior_score": p_wm}).to_csv(
        OUT / "prior_controls" / "matched_wm.csv", index=False)
    pd.DataFrame({"roi_index": np.arange(1, 117), "prior_score": p_fi}).to_csv(
        OUT / "prior_controls" / "matched_fi.csv", index=False)
    for name, (psh, pwm, pfi) in controls.items():
        pd.DataFrame({"roi_index": np.arange(1, 117), "prior_score": pwm}).to_csv(
            OUT / "prior_controls" / f"{name}_wm.csv", index=False)
        pd.DataFrame({"roi_index": np.arange(1, 117), "prior_score": pfi}).to_csv(
            OUT / "prior_controls" / f"{name}_fi.csv", index=False)

    # ── R0 baseline ──
    r0 = R0Baseline(fc_feat, sc_feat, y_wm, y_fi, np.array(subjects), access_logger=access)

    audit_pass, audit_details = run_audit(r0, fc_feat, sc_feat)
    if not audit_pass:
        print("\nSTATUS: PHASE3A_FIX_BASELINE_AUDIT_FAILED")
        with open(OUT / "COMPLETE", "w") as f:
            f.write("PHASE3A_FIX_BASELINE_AUDIT_FAILED\n")
        access.assert_no_holdout()
        return
    if args.audit_only:
        print("Audit-only mode; stopping.")
        return

    # ── Development CV ──
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else PHASE3A_FIX_DEV_SEEDS)
    if args.smoke:
        seeds = seeds[:1]
    print("\n" + "=" * 70)
    print(f"DEVELOPMENT CV seeds={seeds}")
    print("=" * 70)

    opt_diag = []
    split_rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(412)
        sizes = np.full(N_OUTER, 412 // N_OUTER); sizes[: 412 % N_OUTER] += 1
        cur = 0
        for fold in range(N_OUTER):
            ck = CKPT / f"split_seed{seed}_fold{fold}.pkl"
            start, stop = cur, cur + sizes[fold]
            te = idx[start:stop]
            tr = np.concatenate([idx[:start], idx[stop:]])
            cur = stop
            if ck.exists():
                out = pickle.load(open(ck, "rb"))
                print(f"  seed={seed} fold={fold} [cached] "
                      f"C1 WM={out['metrics']['C1']['WM']['pearson']:.4f}")
            else:
                ts = time.time()
                n0 = len(opt_diag)
                out = run_outer_split(FC, SC, y_wm, y_fi, r0, controls, p_wm, p_fi,
                                      tr, te, seed, fold, opt_diag)
                out["optimizer_rows"] = opt_diag[n0:]
                pickle.dump(out, open(ck, "wb"))
                print(f"  seed={seed} fold={fold} [{time.time()-ts:.0f}s] "
                      f"C1 WM={out['metrics']['C1']['WM']['pearson']:.4f} "
                      f"FI={out['metrics']['C1']['FI']['pearson']:.4f}")
                pd.DataFrame([r for o in split_rows + [out]
                              for r in o.get("optimizer_rows", [])]).to_csv(
                    OUT / "optimizer_diagnostics.csv", index=False)
            split_rows.append(out)

    # Save optimizer diagnostics
    opt_diag = [r for o in split_rows for r in o.get("optimizer_rows", [])]
    pd.DataFrame(opt_diag).to_csv(OUT / "optimizer_diagnostics.csv", index=False)

    # ── Aggregate split metrics ──
    rows = []
    for out in split_rows:
        for mid in MODEL_IDS:
            m = out["metrics"][mid]
            rows.append({
                "seed": out["seed"], "fold": out["fold"], "model": mid,
                "r_WM": m["WM"]["pearson"], "rmse_WM": m["WM"]["rmse"], "mae_WM": m["WM"]["mae"],
                "r_FI": m["FI"]["pearson"], "rmse_FI": m["FI"]["rmse"], "mae_FI": m["FI"]["mae"],
                "alpha_WM": m.get("alpha_WM", 0.0), "alpha_FI": m.get("alpha_FI", 0.0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_metrics.csv", index=False)

    # seed metrics
    seed_rows = []
    for seed in seeds:
        for mid in MODEL_IDS:
            sub = df[(df["seed"] == seed) & (df["model"] == mid)]
            seed_rows.append({"seed": seed, "model": mid,
                              "r_WM": sub["r_WM"].mean(), "r_FI": sub["r_FI"].mean(),
                              "rmse_WM": sub["rmse_WM"].mean(), "rmse_FI": sub["rmse_FI"].mean(),
                              "mae_WM": sub["mae_WM"].mean(), "mae_FI": sub["mae_FI"].mean()})
    pd.DataFrame(seed_rows).to_csv(OUT / "seed_metrics.csv", index=False)

    # model summary
    sm = []
    for mid in MODEL_IDS:
        sub = df[df["model"] == mid]
        sm.append({"model": mid,
                   "WM_r_mean": sub["r_WM"].mean(), "WM_r_std": sub["r_WM"].std(),
                   "WM_rmse_mean": sub["rmse_WM"].mean(), "WM_mae_mean": sub["mae_WM"].mean(),
                   "FI_r_mean": sub["r_FI"].mean(), "FI_r_std": sub["r_FI"].std(),
                   "FI_rmse_mean": sub["rmse_FI"].mean(), "FI_mae_mean": sub["mae_FI"].mean()})
    pd.DataFrame(sm).to_csv(OUT / "model_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    for s in sm:
        print(f"  {s['model']}: WM r={s['WM_r_mean']:.6f} RMSE={s['WM_rmse_mean']:.6f} | "
              f"FI r={s['FI_r_mean']:.6f} RMSE={s['FI_rmse_mean']:.6f}")

    # ── Selection details ──
    seld = []
    for out in split_rows:
        for key, (best, scored) in out["selection"].items():
            for s in scored:
                row = {"seed": out["seed"], "fold": out["fold"], "model": key,
                       "lambda_amp": s.get("lambda_amp"), "lambda_prior": s.get("lambda_prior"),
                       "alpha_WM": s.get("alpha_WM", s.get("alpha")),
                       "alpha_FI": s.get("alpha_FI", 0.0),
                       "eligible": s.get("eligible", False),
                       "selected": s.get("selected", False),
                       "fisher": s.get("fisher", None)}
                if "mW" in s:
                    row["WM_pearson"] = s["mW"]["pearson"]; row["WM_rmse"] = s["mW"]["rmse"]
                    row["WM_mae"] = s["mW"]["mae"]
                if "mF" in s:
                    row["FI_pearson"] = s["mF"]["pearson"]; row["FI_rmse"] = s["mF"]["rmse"]
                    row["FI_mae"] = s["mF"]["mae"]
                if "m" in s:
                    task = s.get("task", "WM")
                    row[f"{task}_pearson"] = s["m"]["pearson"]
                    row[f"{task}_rmse"] = s["m"]["rmse"]
                    row[f"{task}_mae"] = s["m"]["mae"]
                seld.append(row)
    pd.DataFrame(seld).to_csv(OUT / "selection_details.csv", index=False)

    # ── Seed-level decomposition ──
    def seed_deltas(m1, m2, col):
        vals = []
        for seed in seeds:
            a = df[(df["seed"] == seed) & (df["model"] == m1)][col].mean()
            b = df[(df["seed"] == seed) & (df["model"] == m2)][col].mean()
            vals.append(a - b)
        return np.array(vals)

    comparisons = [("C1-A0", "C1", "A0"), ("C1-C0", "C1", "C0"),
                   ("C0-B0", "C0", "B0"), ("C1-B1", "C1", "B1"),
                   ("B1-B0", "B1", "B0"), ("C1-C2", "C1", "C2"),
                   ("C1-C3", "C1", "C3"), ("C1-C4", "C1", "C4")]
    decomp = {}
    print("\n" + "=" * 70)
    print("SEED-LEVEL DECOMPOSITION")
    print("=" * 70)
    for name, m1, m2 in comparisons:
        dW = seed_deltas(m1, m2, "r_WM")
        dF = seed_deltas(m1, m2, "r_FI")
        decomp[name] = {
            "WM": {"deltas": dW.tolist(), "mean": float(dW.mean()),
                   "median": float(np.median(dW)), "positive": int((dW > 0).sum())},
            "FI": {"deltas": dF.tolist(), "mean": float(dF.mean()),
                   "median": float(np.median(dF)), "positive": int((dF > 0).sum())},
        }
        print(f"  {name}: WM {dW.mean():+.6f} ({int((dW>0).sum())}/{len(seeds)}) | "
              f"FI {dF.mean():+.6f} ({int((dF>0).sum())}/{len(seeds)})")

    # ── Post-processing: coefficients, biomarkers, faithfulness, plots ──
    stab, recon_errs = export_coefficients(split_rows)
    faith = export_faithfulness(split_rows)
    try:
        make_plots(df, decomp, stab, faith)
    except Exception as e:
        print(f"  [plots error: {e}]")

    # ── Selection diagnostics ──
    sel_diag = {"alpha_WM_freq": {}, "alpha_FI_freq": {}, "lambda_amp_freq": {},
                "lambda_prior_freq": {}, "fallback_freq": {}, "eligible_pct": {}}
    if len(seld):
        sdf = pd.DataFrame(seld)
        sel_only = sdf[sdf["selected"] == True]
        for col, key in [("alpha_WM", "alpha_WM_freq"), ("alpha_FI", "alpha_FI_freq"),
                         ("lambda_amp", "lambda_amp_freq"), ("lambda_prior", "lambda_prior_freq")]:
            sel_diag[key] = sel_only[col].value_counts().to_dict()
        sel_diag["eligible_pct"] = float(sdf["eligible"].mean() * 100)
    print("\nSelection diagnostics:")
    print(f"  eligible % = {sel_diag['eligible_pct']:.1f}")

    # ── Gates ──
    def _pass_g1(key):
        d = decomp[key]["WM"] if key == "WM" else decomp[key]["FI"]
        return d
    g1W = decomp["C1-A0"]["WM"]; g1F = decomp["C1-A0"]["FI"]
    gate1 = (g1W["mean"] >= 0.005 and g1W["positive"] >= 3 and
             g1F["mean"] >= 0.005 and g1F["positive"] >= 3 and
             max(g1W["mean"], g1F["mean"]) >= 0.010)
    g2W = decomp["C1-C0"]["WM"]; g2F = decomp["C1-C0"]["FI"]
    gate2 = (g2W["mean"] >= 0.003 and g2W["positive"] >= 3 and
             g2F["mean"] >= 0.003 and g2F["positive"] >= 3)
    c3W = decomp["C1-C3"]["WM"]; c3F = decomp["C1-C3"]["FI"]
    c4W = decomp["C1-C4"]["WM"]; c4F = decomp["C1-C4"]["FI"]
    crossW = decomp["C1-C2"]["WM"]; crossF = decomp["C1-C2"]["FI"]
    gate3 = (c3W["mean"] > 0 and c3F["mean"] > 0 and c4W["mean"] > 0 and c4F["mean"] > 0
             and crossW["mean"] >= 0 and crossF["mean"] >= 0)
    # Gate 4 stability
    finite = bool(np.all(np.isfinite(df[["r_WM", "r_FI", "rmse_WM", "rmse_FI"]].values)))
    a0_rmse_wm = df[df["model"] == "A0"]["rmse_WM"].mean()
    a0_rmse_fi = df[df["model"] == "A0"]["rmse_FI"].mean()
    catastrophic = bool(((df["rmse_WM"] > 2 * a0_rmse_wm).any()) or
                        ((df["rmse_FI"] > 2 * a0_rmse_fi).any()))
    conv_frac = 1.0
    if len(opt_diag):
        od = pd.DataFrame(opt_diag)
        chosen = od[od.get("chosen", False)]
        conv_frac = float(chosen["converged"].mean()) if len(chosen) else 1.0
    gate4 = finite and (not catastrophic) and conv_frac >= 0.95

    gates = {"gate1": bool(gate1), "gate2": bool(gate2), "gate3": bool(gate3),
             "gate4": bool(gate4), "all_pass": bool(gate1 and gate2 and gate3 and gate4)}
    print("\n" + "=" * 70)
    print("GATES")
    print("=" * 70)
    print(f"  Gate1={gate1} Gate2={gate2} Gate3={gate3} Gate4={gate4}")
    decision = "FREEZE_AND_UNLOCK_PHASE3B" if gates["all_pass"] else "DO_NOT_TOUCH_HOLDOUT"
    print(f"\nPHASE3A_FIX_DECISION: {decision}")

    with open(OUT / "VALIDATION_REPORT.json", "w") as f:
        json.dump({"gates": gates, "decision": decision,
                   "seed_metrics": seed_rows, "decomposition": decomp,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    with open(OUT / "COMPLETE", "w") as f:
        f.write(decision + "\n")
    if gates["all_pass"]:
        (OUT / "READY_FOR_PHASE3B").write_text(decision + "\n")
    else:
        (OUT / "HOLDOUT_REMAINS_LOCKED").write_text(decision + "\n")

    access.assert_no_holdout()
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
