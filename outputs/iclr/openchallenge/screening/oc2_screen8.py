#!/usr/bin/env python3
"""OpenChallenge screen #8: validate fixed-representation subject-relative
ridge blend (weight-only inner selection) plus its architecture-matched
control (global edge standardization) on fresh screening seeds."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import FC_E, SC_E, Y, TASKS, get_r0_cache, partition  # noqa: E402

WGRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def zsubject(A):
    mu = A.mean(1, keepdims=True)
    sd = A.std(1, keepdims=True)
    return (A - mu) / (sd + 1e-12)


def feats(rep, subjrel, tr, te):
    if rep == "fc":
        A, B = FC_E[tr], FC_E[te]
    else:
        A, B = np.hstack([FC_E[tr], SC_E[tr]]), np.hstack([FC_E[te], SC_E[te]])
    if subjrel:
        A, B = zsubject(A), zsubject(B)
    s = StandardScaler().fit(A)
    return s.transform(A), s.transform(B)


def fit_predict(rep, subjrel, alpha, tr, te, y):
    A, B = feats(rep, subjrel, tr, te)
    return Ridge(alpha=alpha).fit(A, y).predict(B)


def select_w(r0in, cin, yin):
    best = (-9.0, 0.0)
    for w in WGRID:
        r = np.corrcoef((1 - w) * r0in + w * cin, yin)[0, 1]
        if r > best[0]:
            best = (r, w)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="16,17,18,19,20")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # proposed (subject-relative): WM fc, FI both; control (global): same reps
    plans = {
        "WM_prop": ("WM", "fc", True),
        "WM_ctrl": ("WM", "fc", False),
        "FI_prop": ("FI", "both", True),
        "FI_ctrl": ("FI", "both", False),
    }
    outs = {k: [] for k in plans}
    ws = {k: [] for k in plans}
    best_alpha = {k: [] for k in plans}

    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        for fold, (te, tr) in enumerate(cache["splits"]):
            t0 = time.time()
            inner = partition(tr, seed, f"outer{fold}", 3)
            pos = {int(g): i for i, g in enumerate(tr)}
            for key, (task, rep, subjrel) in plans.items():
                y = Y[task]
                r0in = cache["inner"][(fold, task)]
                yin = y[tr]
                # inner OOF per alpha; select (alpha, w) jointly on inner OOF
                best = (-9.0, None)
                for alpha in ALPHAS:
                    cin = np.zeros(len(tr))
                    for B, C in inner:
                        p = fit_predict(rep, subjrel, alpha, B, C, y[B])
                        for g, v in zip(C, p):
                            cin[pos[int(g)]] = v
                    w = select_w(r0in, cin, yin)
                    r = np.corrcoef((1 - w) * r0in + w * cin, yin)[0, 1]
                    if (r, -w) > (best[0], -(best[1][1] if best[1] else 0)):
                        best = (r, (alpha, w))
                alpha, w = best[1]
                p_out = fit_predict(rep, subjrel, alpha, tr, te, y[tr])
                r0_out = cache["outer"][(fold, task)]
                delta = (np.corrcoef((1 - w) * r0_out + w * p_out, y[te])[0, 1]
                         - np.corrcoef(r0_out, y[te])[0, 1])
                outs[key].append(delta)
                ws[key].append(w)
                best_alpha[key].append(alpha)
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== FIXED-REP METHOD VALIDATION ===")
    for key in plans:
        d = np.array(outs[key])
        per_seed = [d[i * 5:(i + 1) * 5].mean() for i in range(len(seeds))]
        print(f"\n{key}: mean {d.mean():+.4f} pos folds {int((d>0).sum())}/{len(d)}")
        for s, p in zip(seeds, per_seed):
            print(f"  seed {s}: {p:+.4f}")
        print(f"  w selected: {ws[key]}")
        print(f"  alpha selected: {best_alpha[key]}")


if __name__ == "__main__":
    main()
