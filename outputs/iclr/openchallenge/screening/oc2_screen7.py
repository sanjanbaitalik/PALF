#!/usr/bin/env python3
"""OpenChallenge screen #7: validate the full candidate method.

Predeclared method (to be locked):
  Per task, select on inner OOF from a frozen grid:
    representation R in {subject-relative FC, subject-relative FC+SC}
    ridge alpha in {1, 10, 100, 1000}
    blend weight w in {0, 0.05, ..., 1.0}
  New predictor g = ridge on per-subject z-scored edges (then edge-standardized
  on the fitting scope), R0 blend f = (1-w) R0 + w g.
Report outer-test deltas across screening seeds.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import FC_E, SC_E, Y, TASKS, get_r0_cache, partition  # noqa: E402

ALPHAS = [1.0, 10.0, 100.0, 1000.0]
WGRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]
REPS = ("fc", "both")


def zsubject(A):
    mu = A.mean(1, keepdims=True)
    sd = A.std(1, keepdims=True)
    return (A - mu) / (sd + 1e-12)


def rep_features(rep, tr, te):
    """Subject-relative features, then edge-standardized on train scope only."""
    if rep == "fc":
        A, B = zsubject(FC_E[tr]), zsubject(FC_E[te])
    else:
        A = np.hstack([zsubject(FC_E[tr]), zsubject(SC_E[tr])])
        B = np.hstack([zsubject(FC_E[te]), zsubject(SC_E[te])])
    s = StandardScaler().fit(A)
    return s.transform(A), s.transform(B)


def fit_predict(rep, alpha, tr, te, y):
    A, B = rep_features(rep, tr, te)
    return Ridge(alpha=alpha).fit(A, y).predict(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12,13,14,15")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {t: [] for t in TASKS}
    sel_log = {t: [] for t in TASKS}

    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        splits = cache["splits"]
        for fold, (te, tr) in enumerate(splits):
            t0 = time.time()
            inner = partition(tr, seed, f"outer{fold}", 3)
            pos = {int(g): i for i, g in enumerate(tr)}
            for t in TASKS:
                y = Y[t]
                r0in = cache["inner"][(fold, t)]
                yin = y[tr]
                # inner OOF for every (rep, alpha)
                oof = {}
                for rep, alpha in itertools.product(REPS, ALPHAS):
                    oof[(rep, alpha)] = np.zeros(len(tr))
                for B, C in inner:
                    for rep in REPS:
                        A, Bf = rep_features(rep, B, C)
                        for alpha in ALPHAS:
                            p = Ridge(alpha=alpha).fit(A, y[B]).predict(Bf)
                            for g, v in zip(C, p):
                                oof[(rep, alpha)][pos[int(g)]] = v
                # joint selection of (rep, alpha, w) on inner OOF
                best = (-9.0, None)
                for rep, alpha, w in itertools.product(REPS, ALPHAS, WGRID):
                    p = (1 - w) * r0in + w * oof[(rep, alpha)]
                    r = np.corrcoef(p, yin)[0, 1]
                    key = (r, -w, alpha)
                    if key > (best[0], -(best[1][2] if best[1] else 0),
                              best[1][1] if best[1] else 0):
                        best = (r, (rep, alpha, w))
                rep, alpha, w = best[1]
                p_out = fit_predict(rep, alpha, tr, te, y[tr])
                blended = (1 - w) * cache["outer"][(fold, t)] + w * p_out
                r0_r = np.corrcoef(cache["outer"][(fold, t)], y[te])[0, 1]
                r_bl = np.corrcoef(blended, y[te])[0, 1]
                results[t].append((r0_r, r_bl))
                sel_log[t].append((rep, alpha, w))
                print(f"  f{fold} {t} rep={rep} alpha={alpha} w={w} "
                      f"d={r_bl-r0_r:+.4f}", flush=True)
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== METHOD VALIDATION ===")
    for t in TASKS:
        d = np.array([b - a for a, b in results[t]])
        per_seed = [d[i * 5:(i + 1) * 5].mean() for i in range(len(seeds))]
        print(f"\n{t}: pooled mean delta {d.mean():+.4f} | pos folds "
              f"{int((d>0).sum())}/{len(d)}")
        for s, p in zip(seeds, per_seed):
            print(f"  seed {s}: {p:+.4f}")


if __name__ == "__main__":
    main()
