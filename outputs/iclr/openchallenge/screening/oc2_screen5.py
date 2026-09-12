#!/usr/bin/env python3
"""OpenChallenge screen #5: cross-modal edge products, precision, effective
resistance — focused on FI. Honest nested blending with cached R0."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import (FC_E, SC_E, FC, SC, Y, TASKS, get_r0_cache,  # noqa: E402
                      partition, pred_ridge)


def cross_edge_products(tr, te, y, add_raw=True, alpha=1000.0):
    """Per-edge FC*SC products (bilinear cross-modal coupling)."""
    P = FC_E * SC_E
    Xtr = np.hstack([P[tr], FC_E[tr], SC_E[tr]]) if add_raw else P[tr]
    Xte = np.hstack([P[te], FC_E[te], SC_E[te]]) if add_raw else P[te]
    return pred_ridge(Xtr, y, Xte, alpha)


def precision_features(tr, te, y, alpha=1000.0):
    """Shrinkage precision (inverse covariance) of FC per subject."""
    out = np.zeros((len(tr) + len(te), len(FC_E[0])))
    iu = np.triu_indices(116, 1)
    for k, s in enumerate(np.concatenate([tr, te])):
        lw = LedoitWolf(assume_centered=True).fit(FC[s])
        P = lw.precision_
        out[k] = P[iu]
    ntr = len(tr)
    return pred_ridge(out[:ntr], y, out[ntr:], alpha)


def effective_resistance(tr, te, y, alpha=1000.0):
    """SC effective resistance (Laplacian pseudoinverse) per subject."""
    iu = np.triu_indices(116, 1)
    out = np.zeros((len(tr) + len(te), len(FC_E[0])))
    allidx = np.concatenate([tr, te])
    for k, s in enumerate(allidx):
        W = np.clip(SC[s], 0, None).copy()
        np.fill_diagonal(W, 0)
        d = W.sum(1)
        L = np.diag(d) - W
        Lp = np.linalg.pinv(L, rcond=1e-8)
        dd = np.diag(Lp)
        R = dd[:, None] + dd[None, :] - 2 * Lp
        out[k] = R[iu]
    ntr = len(tr)
    return pred_ridge(out[:ntr], y, out[ntr:], alpha)


def sparsified_fc(tr, te, y, alpha=1.0):
    """Ridge on thresholded/sqrt-positive FC (denoised signed weights)."""
    X = np.sign(FC_E) * np.sqrt(np.abs(FC_E))
    return pred_ridge(X[tr], y, X[te], alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cands = {
        "fcxsc_raw": lambda tr, te, y: cross_edge_products(tr, te, y, True),
        "fcxsc_only": lambda tr, te, y: cross_edge_products(tr, te, y, False),
        "precFC": lambda tr, te, y: precision_features(tr, te, y),
        "effresSC": lambda tr, te, y: effective_resistance(tr, te, y),
        "signedFC": lambda tr, te, y: sparsified_fc(tr, te, y),
    }
    names = list(cands)

    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        splits = cache["splits"]
        res = {t: {"R0": []} for t in TASKS}
        bl = {t: {n: [] for n in names} for t in TASKS}
        st = {t: {n: [] for n in names} for t in TASKS}
        wsel = {t: {n: [] for n in names} for t in TASKS}
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
                res[t]["R0"].append(np.corrcoef(cache["outer"][(fold, t)], Y[t][te])[0, 1])
                for n in names:
                    p_out = cands[n](tr, te, Y[t][tr])
                    st[t][n].append(np.corrcoef(p_out, Y[t][te])[0, 1])
                    best = (-9.0, 0.0)
                    for w in np.linspace(0.0, 1.0, 21):
                        r = np.corrcoef((1 - w) * r0in + w * c_in[n][t], yin)[0, 1]
                        if r > best[0]:
                            best = (r, w)
                    wsel[t][n].append(round(best[1], 2))
                    blended = (1 - best[1]) * cache["outer"][(fold, t)] + best[1] * p_out
                    bl[t][n].append(np.corrcoef(blended, Y[t][te])[0, 1])
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)
        for t in TASKS:
            base = float(np.mean(res[t]["R0"]))
            print(f"\n{t}: R0 = {base:.4f}")
            for n in names:
                s = float(np.mean(st[t][n]))
                b = float(np.mean(bl[t][n]))
                print(f"  {n:10s} standalone {s-base:+.4f}  blended {b-base:+.4f}  w={wsel[t][n]}")


if __name__ == "__main__":
    main()
