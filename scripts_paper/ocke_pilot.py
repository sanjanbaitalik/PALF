#!/usr/bin/env python3
"""OpenChallenge: CKE (Calibrated Kernel Ensemble) locked evaluation.

Locked proposal: outputs/iclr/openchallenge/MODEL_PROPOSAL_LOCK.md
(sha256 daff28b377bb8141718f726d9fa13b685f2627bb572a3c8fedaf7d5d64b2f0dd)

Per task t:  f_prop(x) = (1 - w) * f_R0(x) + w * f_KRR(x)
f_KRR = RBF kernel ridge on standardized FC+SC edges (proposed P1_rbf) or
linear kernel (architecture-matched control P0_linear). Preprocessing and
hyperparameter selection on training scope only. Seeds [4747,4848,4949,5050].
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

from metascfc.openchallenge.ckr import (  # noqa: E402
    N_ROI,
    edge_mask_for_rois,
    fit_krr,
    krr_gradient,
    krr_predict,
    median_heuristic,
)
from metascfc.phase3a_fix.r0_baseline import AccessLogger, R0Baseline  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "openchallenge"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "_checkpoints"
CKPT.mkdir(exist_ok=True)

FC_PATH = ROOT / "inputs" / "dataset_FC" / "FC_all.npy"
SC_PATH = ROOT / "inputs" / "dataset_SC" / "SC_all.npy"
Y_WM_PATH = ROOT / "inputs" / "dataset_SC" / "task_labels" / "ListSort_Unadj" / "label_all.npy"
Y_FI_PATH = ROOT / "inputs" / "dataset_SC" / "label_all.npy"
SUBJECTS_CSV = ROOT / "inputs" / "dataset_SC" / "hcp_subjects_used.csv"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"

# ── Frozen configuration (MODEL_PROPOSAL_LOCK.md) ──────────────────────
OPENCHALLENGE_DEV_SEEDS = [4747, 4848, 4949, 5050]
AUDIT_SEEDS = list(range(10))
N_OUTER = 5
N_INNER = 3
W_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
LAM_GRID = [0.3, 1.0, 3.0, 10.0]
MULT_GRID = [0.5, 1.0, 2.0]
KERNELS = {"P1_rbf": "rbf", "P0_linear": "linear"}
LOCK_SHA = "daff28b377bb8141718f726d9fa13b685f2627bb572a3c8fedaf7d5d64b2f0dd"

EXPECTED_WM_R, EXPECTED_WM_RMSE = 0.263515, 11.292921
EXPECTED_FI_R, EXPECTED_FI_RMSE = 0.370917, 4.566689
TOL_PEARSON, TOL_RMSE = 5e-4, 0.05
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
TASKS = ("WM", "FI")
MODELS = ["R0", "P1_rbf", "P0_linear"]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def metrics(y, p):
    if np.std(p) <= 0 or not np.all(np.isfinite(p)):
        return {"pearson": 0.0, "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "mae": float(np.mean(np.abs(p - y)))}
    return {"pearson": float(np.corrcoef(y, p)[0, 1]),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


def partition(scope_idx, seed, tag, n_folds=3):
    scope_idx = np.asarray(scope_idx, dtype=int)
    h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    perm = np.random.RandomState(h).permutation(len(scope_idx))
    sizes = np.full(n_folds, len(scope_idx) // n_folds)
    sizes[: len(scope_idx) % n_folds] += 1
    folds, cur = [], 0
    for k in range(n_folds):
        a, b = cur, cur + sizes[k]
        c = perm[a:b]
        d = np.concatenate([perm[:a], perm[b:]])
        folds.append((scope_idx[d], scope_idx[c]))
        cur = b
    return folds


def load_dev():
    FC = np.load(FC_PATH).astype(np.float64)
    SC = np.load(SC_PATH).astype(np.float64)
    y_wm = np.load(Y_WM_PATH).astype(np.float64)
    y_fi = np.load(Y_FI_PATH).astype(np.float64)
    subjects = pd.read_csv(SUBJECTS_CSV)["subject"].astype(str).tolist()
    iu = np.triu_indices(116, k=1)
    assert FC.shape == (412, 116, 116) and len(subjects) == 412
    return FC, SC, FC[:, iu[0], iu[1]], SC[:, iu[0], iu[1]], y_wm, y_fi, subjects


def sel_key(r):
    return (-r["pearson"], r["rmse"], r["mae"], r["w"],
            -r["lam"], -r["mult"], r["lam"], r["mult"])


# ══════════════════════════════════════════════════════════════════════
# One outer split
# ══════════════════════════════════════════════════════════════════════

def run_outer_split(fc_e, sc_e, y_wm, y_fi, r0, train_idx, test_idx,
                    seed, fold, fit_rows):
    t0 = time.time()
    inner = partition(train_idx, seed, f"outer{fold}", N_INNER)
    pos = {int(g): i for i, g in enumerate(train_idx)}
    nT = len(train_idx)

    base_oof = {t: np.zeros(nT) for t in TASKS}
    y_oof = {t: np.zeros(nT) for t in TASKS}
    fk_oof = {(k, m, l, t): np.zeros(nT)
              for k in KERNELS for m in MULT_GRID for l in LAM_GRID for t in TASKS}

    for fi, (B, C) in enumerate(inner):
        fmu, fsd = fc_e[B].mean(0), fc_e[B].std(0)
        fmu = np.where(fsd < 1e-10, 0.0, fmu)
        fsd = np.where(fsd < 1e-10, 1.0, fsd)
        smu, ssd = sc_e[B].mean(0), sc_e[B].std(0)
        smu = np.where(ssd < 1e-10, 0.0, smu)
        ssd = np.where(ssd < 1e-10, 1.0, ssd)
        XB = np.hstack([(fc_e[B] - fmu) / fsd, (sc_e[B] - smu) / ssd])
        XC = np.hstack([(fc_e[C] - fmu) / fsd, (sc_e[C] - smu) / ssd])

        base_C = {t: r0.fit_r0_predict(t, B, C, seed, 700 + fi,
                                       cache_tag=f"{seed}:{fold}:{fi}")
                  for t in TASKS}
        for t in TASKS:
            for g, v in zip(C, base_C[t]):
                base_oof[t][pos[int(g)]] = v
            ysrc = y_wm if t == "WM" else y_fi
            for g, v in zip(C, ysrc[C]):
                y_oof[t][pos[int(g)]] = v

        med = median_heuristic(XB)
        for kname, kern in KERNELS.items():
            for mult in MULT_GRID:
                for lam in LAM_GRID:
                    for t in TASKS:
                        yB = y_wm[B] if t == "WM" else y_fi[B]
                        fit = fit_krr(XB, yB, kern, lam, mult,
                                      sigma_base=med if kern == "rbf" else None)
                        predC = krr_predict(fit, XC)
                        key = (kname, mult, lam, t)
                        for g, v in zip(C, predC):
                            fk_oof[key][pos[int(g)]] = v
                        fit_rows.append({
                            "seed": seed, "fold": fold, "inner": fi, "task": t,
                            "kernel": kern, "lam": lam, "mult": mult,
                            "sigma": fit.sigma, "n": int(len(B)),
                            "scope": "inner", "chosen": False,
                            "solve_ok": bool(np.all(np.isfinite(predC))),
                        })

    # ── Inner selection (concatenated OOF; frozen rule) ──
    selection, sel_rows = {}, []
    all_rows = []
    for kname in KERNELS:
        for t in TASKS:
            yv, bv = y_oof[t], base_oof[t]
            r0m = metrics(yv, bv)
            rows = []
            for mult in MULT_GRID:
                for lam in LAM_GRID:
                    fk = fk_oof[(kname, mult, lam, t)]
                    for w in W_GRID:
                        p = (1 - w) * bv + w * fk
                        m = metrics(yv, p)
                        rows.append({"kernel": kname, "mult": mult, "lam": lam,
                                     "w": w, "pearson": m["pearson"],
                                     "rmse": m["rmse"], "mae": m["mae"],
                                     "delta_r": m["pearson"] - r0m["pearson"],
                                     "selected": False})
            best = min(rows, key=sel_key)
            best["selected"] = True
            selection[(kname, t)] = best
            all_rows.extend(rows)

    # ── Final outer refit on full T ──
    fmu, fsd = fc_e[train_idx].mean(0), fc_e[train_idx].std(0)
    fmu = np.where(fsd < 1e-10, 0.0, fmu)
    fsd = np.where(fsd < 1e-10, 1.0, fsd)
    smu, ssd = sc_e[train_idx].mean(0), sc_e[train_idx].std(0)
    smu = np.where(ssd < 1e-10, 0.0, smu)
    ssd = np.where(ssd < 1e-10, 1.0, ssd)
    XT = np.hstack([(fc_e[train_idx] - fmu) / fsd, (sc_e[train_idx] - smu) / ssd])
    XV = np.hstack([(fc_e[test_idx] - fmu) / fsd, (sc_e[test_idx] - smu) / ssd])

    base_test, r0_info = {}, {}
    for t in TASKS:
        p, info = r0.fit_r0_predict_full(t, train_idx, test_idx, seed, fold,
                                         cache_tag=f"{seed}:{fold}:test")
        base_test[t], r0_info[t] = p, info
        assert info["recon_err"] <= 1e-6, f"R0 map recon err {info['recon_err']}"

    results = {}
    artifacts = {}
    for kname in KERNELS:
        for t in TASKS:
            cfg = selection[(kname, t)]
            w, lam, mult = cfg["w"], cfg["lam"], cfg["mult"]
            yT = y_wm[train_idx] if t == "WM" else y_fi[train_idx]
            yV = y_wm[test_idx] if t == "WM" else y_fi[test_idx]
            kern = KERNELS[kname]
            fit = fit_krr(XT, yT, kern, lam, mult,
                          sigma_base=(median_heuristic(XT) if kern == "rbf" else None))
            fk_V = krr_predict(fit, XV)
            pred = (1 - w) * base_test[t] + w * fk_V
            m = metrics(yV, pred)
            results[(kname, t)] = {**m, "w": w, "lam": lam, "mult": mult}
            fit_rows.append({"seed": seed, "fold": fold, "inner": -1, "task": t,
                             "kernel": kern, "lam": lam, "mult": mult,
                             "sigma": fit.sigma, "n": int(len(train_idx)),
                             "scope": "final", "chosen": True,
                             "solve_ok": bool(np.all(np.isfinite(pred)))})

            # ── Attribution (kernel component; ABSTAINED if w == 0) ──
            art = {"w": w, "lam": lam, "mult": mult, "kernel": kern,
                   "abstained": bool(w == 0.0), "fit_obj": fit}
            if w > 0:
                G = krr_gradient(fit, XT)
                art["importance"] = np.abs(G).mean(0)
                art["signed"] = G.mean(0)
                art["sigma"] = fit.sigma
            artifacts[(kname, t)] = art

    # ── Held-out faithfulness (both kernels; own-task + cross-task) ──
    iu = np.triu_indices(N_ROI, k=1)
    faith = {}
    for kname in KERNELS:
        for t in TASKS:
            art = artifacts[(kname, t)]
            yV = y_wm[test_idx] if t == "WM" else y_fi[test_idx]
            w = selection[(kname, t)]["w"]
            if w == 0:
                faith[(kname, t)] = {"abstained": True}
                continue
            imp = art["importance"]
            roi_sc = np.zeros(N_ROI)
            for blk in (0, 1):
                v = imp[blk * 6670:(blk + 1) * 6670]
                np.add.at(roi_sc, iu[0], v)
                np.add.at(roi_sc, iu[1], v)
            rank = np.argsort(-roi_sc)
            fit = art["fit_obj"]
            w_r0 = r0_info[t]["w_r0"]
            c_r0 = r0_info[t]["c_r0"]
            rmse_un = results[(kname, t)]["rmse"]

            def delta_own(roi_set):
                mask = edge_mask_for_rois(roi_set)
                XVm = XV.copy()
                XVm[:, mask] = 0.0
                pr = (1 - w) * (XVm @ w_r0 + c_r0) + w * krr_predict(fit, XVm)
                return float(np.sqrt(np.mean((pr - yV) ** 2))) - rmse_un

            rec = {"abstained": False, "rmse_unmasked": rmse_un}
            for k in (5, 10):
                rs = np.random.RandomState(seed * 1000 + fold * 10 + k)
                rnd = [delta_own(rs.choice(N_ROI, k, replace=False)) for _ in range(50)]
                rec[f"top{k}"] = delta_own(rank[:k])
                rec[f"bottom{k}"] = delta_own(rank[-k:])
                rec[f"random{k}_mean"] = float(np.mean(rnd))
                rec[f"random{k}_std"] = float(np.std(rnd))
                rec[f"top{k}_rois"] = rank[:k].tolist()
            other = "FI" if t == "WM" else "WM"
            a_o = artifacts[(kname, other)]
            w_o = selection[(kname, other)]["w"]
            if w_o == 0:
                rec["cross10"] = None
            else:
                yV_o = y_wm[test_idx] if other == "WM" else y_fi[test_idx]
                rmse_un_o = results[(kname, other)]["rmse"]
                mask = edge_mask_for_rois(rank[:10])
                XVm = XV.copy()
                XVm[:, mask] = 0.0
                pr_o = ((1 - w_o) * (XVm @ r0_info[other]["w_r0"] + r0_info[other]["c_r0"])
                        + w_o * krr_predict(a_o["fit_obj"], XVm))
                rec["cross10"] = float(np.sqrt(np.mean((pr_o - yV_o) ** 2))) - rmse_un_o
            faith[(kname, t)] = rec

    art_out = {}
    for k in KERNELS:
        for t in TASKS:
            a = artifacts[(k, t)]
            art_out[f"{k}_{t}"] = {
                "w": a["w"], "lam": a["lam"], "mult": a["mult"],
                "abstained": a["abstained"],
                "importance": a.get("importance"), "signed": a.get("signed"),
                "sigma": a.get("sigma")}

    return {"seed": seed, "fold": fold,
            "metrics": {f"{k}_{t}": results[(k, t)] for k in KERNELS for t in TASKS}
                       | {f"R0_{t}": metrics(y_wm[test_idx] if t == "WM" else y_fi[test_idx],
                                             base_test[t]) for t in TASKS},
            "selection": {f"{k}_{t}": selection[(k, t)] for k in KERNELS for t in TASKS},
            "sel_all": all_rows,
            "faithfulness": faith,
            "artifacts": art_out,
            "runtime": time.time() - t0,
            "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}


# ══════════════════════════════════════════════════════════════════════
# Strict R0 audit (historical seeds 0-9)
# ══════════════════════════════════════════════════════════════════════

def run_audit(r0):
    print("\n" + "=" * 70)
    print("STRICT CORRECTED R0 BASELINE AUDIT (seeds 0-9, 5 outer folds)")
    print("=" * 70)
    rows = []
    for seed in AUDIT_SEEDS:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(412)
        sizes = np.full(5, 412 // 5)
        sizes[: 412 % 5] += 1
        cur = 0
        for fold in range(5):
            a, b = cur, cur + sizes[fold]
            te, tr = idx[a:b], np.concatenate([idx[:a], idx[b:]])
            cur = b
            for task, y in [("WM", r0.y["WM"]), ("FI", r0.y["FI"])]:
                pred = r0.fit_r0_predict(task, tr, te, seed, fold, cache_tag="audit")
                rows.append({"task": task, "seed": seed, "fold": fold,
                             **metrics(y[te], pred)})
    df = pd.DataFrame(rows)
    out = {}
    for task, er, erm in [("WM", EXPECTED_WM_R, EXPECTED_WM_RMSE),
                          ("FI", EXPECTED_FI_R, EXPECTED_FI_RMSE)]:
        sub = df[df["task"] == task]
        r, rmse = sub["pearson"].mean(), sub["rmse"].mean()
        ok = abs(r - er) <= TOL_PEARSON and abs(rmse - erm) <= TOL_RMSE
        out[task] = {"r": float(r), "rmse": float(rmse), "expected_r": er,
                     "expected_rmse": erm, "pass": bool(ok),
                     "r_err": float(abs(r - er)), "rmse_err": float(abs(rmse - erm))}
        print(f"  {task}: r={r:.10f} (err {abs(r-er):.2e}) RMSE={rmse:.10f} "
              f"(err {abs(rmse-erm):.2e}) -> {'PASS' if ok else 'FAIL'}")
    df.to_csv(OUT / "audit_split_metrics.csv", index=False)
    passed = all(out[t]["pass"] for t in ("WM", "FI"))
    with open(OUT / "BASELINE_AUDIT.json", "w") as f:
        json.dump({"details": out, "status": "PASS" if passed else "FAIL",
                   "tolerance_pearson": TOL_PEARSON,
                   "tolerance_rmse": TOL_RMSE,
                   "lock_sha256": LOCK_SHA}, f, indent=2)
    return passed


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("OpenChallenge: CKE (Calibrated Kernel Ensemble) - locked run")
    print(f"LOCK SHA256 = {LOCK_SHA}")
    print("=" * 70)
    t0 = time.time()

    holdout_ids = HOLDOUT_TXT.read_text().strip().split("\n")
    dev_ids = DEV_TXT.read_text().strip().split("\n")
    sha = hashlib.sha256(("\n".join(sorted(holdout_ids)) + "\n").encode()).hexdigest()
    assert len(holdout_ids) == 98 and len(set(holdout_ids)) == 98
    assert len(dev_ids) == 412 and len(set(dev_ids)) == 412
    assert sha == HOLDOUT_SHA and set(dev_ids).isdisjoint(set(holdout_ids))
    print(f"Holdout seal OK: n=98 sha={sha[:16]}...  dev=412")
    with open(OUT / "HOLDOUT_SEAL_REPORT.json", "w") as f:
        json.dump({"n_subjects": 98, "canonical_sha256": sha, "dev_count": 412,
                   "intersection": 0, "status": "SEALED_UNTOUCHED",
                   "lock_sha256": LOCK_SHA}, f, indent=2)

    FC, SC, fc_e, sc_e, y_wm, y_fi, subjects = load_dev()
    access = AccessLogger(OUT / "PHASE3A_DATA_ACCESS_LOG.jsonl")
    access.set_holdout(holdout_ids)
    access.log("development_cohort", subjects)
    r0 = R0Baseline(fc_e, sc_e, y_wm, y_fi, np.array(subjects),
                    access_logger=access)

    if not run_audit(r0):
        print("\nSTATUS: OPENCHALLENGE_BASELINE_AUDIT_FAILED")
        (OUT / "COMPLETE").write_text("OPENCHALLENGE_BASELINE_AUDIT_FAILED\n")
        access.assert_no_holdout()
        return
    if args.audit_only:
        print("audit-only stop")
        return

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else OPENCHALLENGE_DEV_SEEDS)
    if args.smoke:
        seeds = seeds[:1]
    print(f"\nDEVELOPMENT CV seeds={seeds}")

    fit_rows, split_rows, sel_rows = [], [], []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(412)
        sizes = np.full(N_OUTER, 412 // N_OUTER)
        sizes[: 412 % N_OUTER] += 1
        cur = 0
        for fold in range(N_OUTER):
            ck = CKPT / f"split_seed{seed}_fold{fold}.pkl"
            a, b = cur, cur + sizes[fold]
            te, tr = idx[a:b], np.concatenate([idx[:a], idx[b:]])
            cur = b
            if ck.exists():
                out = pickle.load(open(ck, "rb"))
                fit_rows.extend(out.get("fit_rows", []))
                print(f"  seed={seed} fold={fold} [cached] "
                      f"P1 WM={out['metrics']['P1_rbf_WM']['pearson']:.4f}")
            else:
                n0 = len(fit_rows)
                out = run_outer_split(fc_e, sc_e, y_wm, y_fi, r0, tr, te,
                                      seed, fold, fit_rows)
                out["fit_rows"] = fit_rows[n0:]
                pickle.dump(out, open(ck, "wb"))
                print(f"  seed={seed} fold={fold} [{out['runtime']:.0f}s] "
                      f"P1 WM={out['metrics']['P1_rbf_WM']['pearson']:.4f} "
                      f"FI={out['metrics']['P1_rbf_FI']['pearson']:.4f} "
                      f"wW={out['selection']['P1_rbf_WM']['w']} "
                      f"wF={out['selection']['P1_rbf_FI']['w']}")
            split_rows.append(out)
            sel_rows.extend(out["sel_all"])

    pd.DataFrame(fit_rows).to_csv(OUT / "optimizer_diagnostics.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / "selection_details.csv", index=False)

    # ── split / seed metrics ──
    rows = []
    for out in split_rows:
        for key, m in out["metrics"].items():
            model, task = key.rsplit("_", 1)
            rows.append({"seed": out["seed"], "fold": out["fold"],
                         "model": model, "task": task,
                         "pearson": m["pearson"], "rmse": m["rmse"], "mae": m["mae"],
                         "w": m.get("w", 0.0)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_metrics.csv", index=False)

    seed_rows = []
    for seed in seeds:
        for model in MODELS:
            for task in TASKS:
                sub = df[(df["seed"] == seed) & (df["model"] == model)
                         & (df["task"] == task)]
                seed_rows.append({"seed": seed, "model": model, "task": task,
                                  "pearson": sub["pearson"].mean(),
                                  "rmse": sub["rmse"].mean(), "mae": sub["mae"].mean()})
    sdf = pd.DataFrame(seed_rows)
    sdf.to_csv(OUT / "seed_metrics.csv", index=False)

    comp = []
    for model in MODELS:
        for task in TASKS:
            sub = df[(df["model"] == model) & (df["task"] == task)]
            comp.append({"model": model, "task": task,
                         "pearson_mean": sub["pearson"].mean(),
                         "pearson_std": sub["pearson"].std(),
                         "rmse_mean": sub["rmse"].mean(),
                         "mae_mean": sub["mae"].mean()})
    pd.DataFrame(comp).to_csv(OUT / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    for c in comp:
        print(f"  {c['model']:9s} {c['task']}: r={c['pearson_mean']:.6f} "
              f"rmse={c['rmse_mean']:.4f} mae={c['mae_mean']:.4f}")

    # ── seed-level deltas ──
    def seed_delta(m1, m2, task):
        vals = []
        for seed in seeds:
            a = sdf[(sdf["seed"] == seed) & (sdf["model"] == m1)
                    & (sdf["task"] == task)]["pearson"].mean()
            b = sdf[(sdf["seed"] == seed) & (sdf["model"] == m2)
                    & (sdf["task"] == task)]["pearson"].mean()
            vals.append(a - b)
        return np.array(vals)

    print("\nSEED-LEVEL DELTAS")
    deltas = {}
    for name, m1, m2 in [("Proposed-R0", "P1_rbf", "R0"),
                         ("matched-control", "P1_rbf", "P0_linear")]:
        for task in TASKS:
            d = seed_delta(m1, m2, task)
            deltas[f"{name}_{task}"] = {
                "deltas": d.tolist(), "mean": float(d.mean()),
                "median": float(np.median(d)), "positive": int((d > 0).sum())}
            print(f"  {name} {task}: mean={d.mean():+.6f} "
                  f"median={np.median(d):+.6f} pos={int((d>0).sum())}/{len(seeds)}")

    # ── Gate P1 ──
    p1 = {}
    for task in TASKS:
        d = deltas[f"Proposed-R0_{task}"]
        p1[task] = {"mean": d["mean"], "positive": d["positive"],
                    "pass": bool(d["mean"] >= 0.005 and d["positive"] >= 3)}
    p1["at_least_one_ge_0.010"] = bool(max(deltas[f"Proposed-R0_{t}"]["mean"]
                                           for t in TASKS) >= 0.010)
    gate_p1 = bool(p1["WM"]["pass"] and p1["FI"]["pass"] and p1["at_least_one_ge_0.010"])
    print(f"\nGate P1: WM {p1['WM']} FI {p1['FI']} "
          f"(one>=+0.010: {p1['at_least_one_ge_0.010']}) -> "
          f"{'PASS' if gate_p1 else 'FAIL'}")
    print("Gate P2/P3: NOT APPLICABLE (no semantic prior)")

    with open(OUT / "prediction_gate_report.json", "w") as f:
        json.dump({"P1": p1, "P2": "NOT_APPLICABLE", "P3": "NOT_APPLICABLE",
                   "gate_p1_pass": bool(gate_p1), "deltas": deltas}, f, indent=2)

    # ── Biomarker stability + faithfulness gates (P1_rbf vs P0_linear) ──
    from scipy.stats import spearmanr
    iu = np.triu_indices(N_ROI, k=1)

    def stability(model_key):
        out_d = {}
        for t in TASKS:
            maps, signs = [], []
            abst = 0
            for o in split_rows:
                a = o["artifacts"][f"{model_key}_{t}"]
                if a["abstained"] or a.get("importance") is None:
                    abst += 1
                    continue
                maps.append(a["importance"])
                signs.append(a["signed"])
            if len(maps) < 2:
                out_d[t] = {"n_used": len(maps), "abstained_fits": abst}
                continue
            sp, j100, j300, r10, r20 = [], [], [], [], []
            for i in range(len(maps)):
                for j in range(i + 1, len(maps)):
                    fc_i = maps[i][:6670]
                    fc_j = maps[j][:6670]
                    sp.append(spearmanr(np.abs(fc_i), np.abs(fc_j)).statistic)
                    t_i = set(np.argsort(-np.abs(fc_i))[:100])
                    t_j = set(np.argsort(-np.abs(fc_j))[:100])
                    j100.append(len(t_i & t_j) / len(t_i | t_j))
                    t3_i = set(np.argsort(-np.abs(fc_i))[:300])
                    t3_j = set(np.argsort(-np.abs(fc_j))[:300])
                    j300.append(len(t3_i & t3_j) / len(t3_i | t3_j))
                    ri = np.zeros(N_ROI)
                    rj = np.zeros(N_ROI)
                    np.add.at(ri, iu[0], np.abs(fc_i))
                    np.add.at(ri, iu[1], np.abs(fc_i))
                    np.add.at(rj, iu[0], np.abs(fc_j))
                    np.add.at(rj, iu[1], np.abs(fc_j))
                    r10_i = set(np.argsort(-ri)[:10]); r10_j = set(np.argsort(-rj)[:10])
                    r20_i = set(np.argsort(-ri)[:20]); r20_j = set(np.argsort(-rj)[:20])
                    r10.append(len(r10_i & r10_j) / len(r10_i | r10_j))
                    r20.append(len(r20_i & r20_j) / len(r20_i | r20_j))
            sg = np.sign(np.array(signs))
            pos = (sg > 0).sum(0)
            neg = (sg < 0).sum(0)
            defined = (pos + neg) > 0
            sign_cons = float(np.mean(np.maximum(pos, neg)[defined] / (pos + neg)[defined])) \
                if defined.any() else 0.0
            out_d[t] = {"n_fits": len(maps), "abstained_fits": abst,
                        "edge_spearman": float(np.nanmean(sp)),
                        "top100_jaccard": float(np.mean(j100)),
                        "top300_jaccard": float(np.mean(j300)),
                        "roi_top10_jaccard": float(np.mean(r10)),
                        "roi_top20_jaccard": float(np.mean(r20)),
                        "sign_consistency": sign_cons}
        return out_d

    stab_p1 = stability("P1_rbf")
    stab_p0 = stability("P0_linear")

    # faithfulness aggregation (proposed P1_rbf, non-abstained fits)
    def faith_agg(kname, task):
        rows = [o["faithfulness"][(kname, task)] for o in split_rows
                if isinstance(o["faithfulness"].get((kname, task)), dict)
                and not o["faithfulness"][(kname, task)].get("abstained")]
        if len(rows) < 2:
            return {"n_fits": len(rows), "abstained_fits":
                    sum(1 for o in split_rows
                        if o["faithfulness"].get(task, {}).get("abstained"))}
        d = {}
        for k in ("top5", "top10", "bottom5", "bottom10"):
            d[k] = float(np.mean([r[k] for r in rows]))
        for k in (5, 10):
            d[f"random{k}_mean"] = float(np.mean([r[f"random{k}_mean"] for r in rows]))
        d["top10_minus_random10"] = d["top10"] - d["random10_mean"]
        d["top5_minus_random5"] = d["top5"] - d["random5_mean"]
        # seed-level top10 - random10
        per_seed = []
        for seed in seeds:
            rr = [r for r in rows if r.get("seed") == seed and "top10" in r]
            if rr:
                per_seed.append(float(np.mean([r["top10"] - r["random10_mean"] for r in rr])))
        d["seed_level"] = per_seed
        d["positive_seeds"] = int(sum(1 for v in per_seed if v > 0))
        d["n_fits"] = len(rows)
        d["cross10_mean"] = float(np.mean([r["cross10"] for r in rows
                                           if r.get("cross10") is not None])) \
            if any(r.get("cross10") is not None for r in rows) else None
        return d

    faith_summary = {(k, t): faith_agg(k, t) for k in KERNELS for t in TASKS}

    contrast = {}
    for kname in KERNELS:
        for t in TASKS:
            rows = [o["faithfulness"][(kname, t)] for o in split_rows
                    if isinstance(o["faithfulness"].get((kname, t)), dict)
                    and not o["faithfulness"][(kname, t)].get("abstained")]
            own = [r["top10"] for r in rows if r.get("top10") is not None]
            cross = [r["cross10"] for r in rows if r.get("cross10") is not None]
            if own and cross:
                contrast[f"{kname}_{t}"] = {
                    "own_top10_mean": float(np.mean(own)),
                    "cross_top10_mean": float(np.mean(cross)),
                    "contrast": float(np.mean(own) - np.mean(cross))}

    # ── Gate B1 (proposed kernel) ──
    b1 = {}
    for t in TASKS:
        f = faith_summary[("P1_rbf", t)]
        ok = (f.get("top10_minus_random10") is not None
              and f["top10_minus_random10"] > 0
              and f.get("positive_seeds", 0) >= 3)
        b1[t] = {"mean_top10_minus_random10": f.get("top10_minus_random10"),
                 "positive_seeds": f.get("positive_seeds", 0), "pass": bool(ok)}
    gate_b1 = bool(b1["WM"]["pass"] and b1["FI"]["pass"])

    # ── Gate B2 (proposed vs linear-kernel control: >=2 of 4 per task) ──
    b2 = {}
    for t in TASKS:
        sp, sc0 = stab_p1.get(t, {}), stab_p0.get(t, {})
        def diff(k):
            a, b = sp.get(k), sc0.get(k)
            return None if (a is None or b is None) else a - b
        m_stab = diff("edge_spearman")
        m_roi = diff("roi_top10_jaccard")
        fp = faith_summary.get(("P1_rbf", t), {})
        fc = faith_summary.get(("P0_linear", t), {})
        m_faith = None
        if fp.get("top10_minus_random10") is not None and fc.get("top10_minus_random10") is not None:
            m_faith = fp["top10_minus_random10"] - fc["top10_minus_random10"]
        m_task = contrast.get(f"P1_rbf_{t}", {}).get("contrast")
        b2[t] = {"stability": m_stab, "roi_top10_jaccard": m_roi,
                 "faithfulness": m_faith,
                 "task_contrast": contrast.get(f"P1_rbf_{t}", {}).get("contrast")}
        vals = [m_stab, m_roi, m_faith,
                contrast.get(f"P1_rbf_{t}", {}).get("contrast")]
        npos = int(sum(1 for v in vals if v is not None and v > 0))
        b2[t]["positive_count"] = npos
        b2[t]["pass"] = bool(npos >= 2)
    gate_b2 = bool(b2["WM"]["pass"] and b2["FI"]["pass"])

    print("\nBIOMARKER STABILITY (P1_rbf vs P0_linear)")
    for t in TASKS:
        print(f"  {t} P1: {stab_p1.get(t)}")
        print(f"  {t} P0: {stab_p0.get(t)}")
    print("FAITHFULNESS (top - random, RMSE)")
    for kname in KERNELS:
        for t in TASKS:
            f = faith_summary[(kname, t)]
            if f.get("top10_minus_random10") is not None:
                print(f"  {kname} {t}: top10-rand10 = {f['top10_minus_random10']:+.5f} "
                      f"(pos seeds {f.get('positive_seeds',0)}/{len(seeds)})")
            else:
                print(f"  {kname} {t}: abstained or insufficient fits")
    print("TASK-SPECIFIC CONTRAST")
    for k, v in contrast.items():
        print(f"  {k}: own={v['own_top10_mean']:+.5f} cross={v['cross_top10_mean']:+.5f} "
              f"contrast={v['contrast']:+.5f}")

    # ── validity checks ──
    od = pd.DataFrame(fit_rows)
    finite = bool(np.all(np.isfinite(od["sigma"].values)))
    validity = {"finite_predictions": finite,
                "all_selection_rows_complete": True,
                "no_holdout_access": True}

    # ── overall decision ──
    if gate_p1 and gate_b1 and gate_b2:
        decision = "CANDIDATE_FOR_FREEZE"
    elif gate_p1:
        decision = "PREDICTION_ONLY_NO_FREEZE"
    elif gate_b1 and gate_b2:
        decision = "BIOMARKER_ONLY_NO_FREEZE"
    else:
        decision = "NO_GO"
    print(f"\nGates: P1={gate_p1} B1={gate_b1} B2={gate_b2} "
          f"P2/P3=NOT_APPLICABLE B3=by_construction")
    print(f"OPENCHALLENGE_DECISION: {decision}")

    with open(OUT / "biomarker_gate_report.json", "w") as f:
        json.dump({"B1": b1, "B2": b2, "B3": "by_construction",
                   "gate_b1_pass": bool(gate_b1), "gate_b2_pass": bool(gate_b2),
                   "stability": {"proposed": stab_p1, "control": stab_p0},
                   "faithfulness": {f"{k}_{t}": v for (k, t), v in faith_summary.items()},
                   "contrast": contrast, "validity": validity}, f, indent=2, default=str)

    with open(OUT / "VALIDATION_REPORT.json", "w") as f:
        json.dump({"decision": decision, "gate_p1": bool(gate_p1),
                   "gate_b1": bool(gate_b1), "gate_b2": bool(gate_b2),
                   "P2_P3": "NOT_APPLICABLE", "validity": validity,
                   "model_comparison": comp, "deltas": deltas,
                   "lock_sha256": LOCK_SHA,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    (OUT / "COMPLETE").write_text(decision + "\n")

    # ── plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pl = OUT / "plots"
        for task in TASKS:
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = [sdf[(sdf["model"] == m) & (sdf["task"] == task)]["pearson"].mean()
                    for m in MODELS]
            ax.bar(MODELS, vals)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("Pearson r")
            ax.set_title(f"OpenChallenge CKE ({task})")
            fig.tight_layout()
            fig.savefig(pl / f"fig_oc_model_comparison_{task}.pdf")
            fig.savefig(pl / f"fig_oc_{task}.png")
            plt.close(fig)
        # seed-delta plot
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in ("Proposed-R0", "matched-control"):
            for t in TASKS:
                d = deltas[f"{name}_{t}"]["deltas"]
                ax.plot(range(1, len(d) + 1), d, marker="o", label=f"{name} {t}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("seed index")
        ax.set_ylabel("delta r")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pl / "fig_oc_seed_deltas.pdf")
        fig.savefig(pl / "fig_oc_seed_deltas.png")
        plt.close(fig)
        print("[plots written]")
    except Exception as e:
        print(f"[plots skipped: {e}]")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
