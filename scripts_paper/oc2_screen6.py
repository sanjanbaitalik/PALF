#!/usr/bin/env python3
"""OpenChallenge screen #6: per-subject normalization variants (focused).

persubj = z-score each subject's edge vector, then ridge. Tests alpha and
modality variants across 4 screening seeds with honest nested blending.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import (FC_E, SC_E, Y, TASKS, get_r0_cache,  # noqa: E402
                      partition, pred_ridge)


def zsubject(A):
    mu = A.mean(1, keepdims=True)
    sd = A.std(1, keepdims=True)
    return (A - mu) / (sd + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12,13,14")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    ZF = {a: zsubject(FC_E) for a in (100, 300, 1000, 3000)}
    ZS = {a: zsubject(SC_E) for a in (100, 300, 1000, 3000)}

    cands = {}
    for a in (100, 300, 1000, 3000):
        cands[f"pz_both_{a}"] = (
            lambda tr, te, y, a=a: pred_ridge(
                np.hstack([ZF[a][tr], ZS[a][tr]]), y,
                np.hstack([ZF[a][te], ZS[a][te]]), 1.0))
    for a in (100, 1000):
        cands[f"pz_fc_{a}"] = (
            lambda tr, te, y, a=a: pred_ridge(ZF[a][tr], y, ZF[a][te], 1.0))
        cands[f"pz_sc_{a}"] = (
            lambda tr, te, y, a=a: pred_ridge(ZS[a][tr], y, ZS[a][te], 1.0))
    names = list(cands)

    all_deltas = {t: {n: [] for n in names} for t in TASKS}
    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        splits = cache["splits"]
        bl = {t: {n: [] for n in names} for t in TASKS}
        r0s = {t: [] for t in TASKS}
        for fold, (te, tr) in enumerate(splits):
            t0 = time.time()
            inner = partition(tr, seed, f"outer{fold}", 3)
            pos = {int(g): i for i, g in enumerate(tr)}
            c_in = {n: {t: np.zeros(len(tr)) for t in TASKS} for n in names}
            for n, fn in cands.items():
                for B, C in inner:
                    for t in TASKS:
                        p = fn(B, C, Y[t][B])
                        for g, v in zip(C, p):
                            c_in[n][t][pos[int(g)]] = v
            for t in TASKS:
                yin = Y[t][tr]
                r0in = cache["inner"][(fold, t)]
                r0s[t].append(np.corrcoef(cache["outer"][(fold, t)], Y[t][te])[0, 1])
                for n in names:
                    p_out = cands[n](tr, te, Y[t][tr])
                    best = (-9.0, 0.0)
                    for w in np.linspace(0.0, 1.0, 21):
                        r = np.corrcoef((1 - w) * r0in + w * c_in[n][t], yin)[0, 1]
                        if r > best[0]:
                            best = (r, w)
                    blended = (1 - best[1]) * cache["outer"][(fold, t)] + best[1] * p_out
                    bl[t][n].append(np.corrcoef(blended, Y[t][te])[0, 1])
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)
        for t in TASKS:
            base = float(np.mean(r0s[t]))
            print(f"\n{t}: R0 = {base:.4f}")
            for n in names:
                b = float(np.mean(bl[t][n]))
                all_deltas[t][n].append(b - base)
                print(f"  {n:12s} blend delta {b-base:+.4f}")

    print("\n=== ACROSS-SEED SUMMARY ===")
    for t in TASKS:
        print(f"\n{t}:")
        for n in names:
            d = all_deltas[t][n]
            print(f"  {n:12s} mean {np.mean(d):+.4f}  pos {sum(1 for x in d if x>0)}/{len(d)}  "
                  f"per-seed {[round(float(x),4) for x in d]}")


if __name__ == "__main__":
    main()
