#!/usr/bin/env python3
"""OpenChallenge screen #4: variance-reduction and sparse estimator families."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import (FC_E, SC_E, Y, N, TASKS, get_r0_cache,  # noqa: E402
                      partition, pred_ridge)


def elastic_net(tr, te, y, alpha=1.0, l1=0.5):
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    s = StandardScaler().fit(Xtr)
    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=0)
    m.fit(s.transform(Xtr), y)
    return m.predict(s.transform(Xte))


def two_alpha_ridge(tr, te, y, afc=1000.0, asc=100.0, wfc=0.5):
    """Separately regularized FC and SC ridge branches, convex fusion."""
    pf = pred_ridge(FC_E[tr], y, FC_E[te], afc)
    ps = pred_ridge(SC_E[tr], y, SC_E[te], asc)
    return wfc * pf + (1 - wfc) * ps


def rank_target_ridge(tr, te, y, alpha=1000.0):
    from scipy.stats import norm, rankdata
    yr = norm.ppf((rankdata(y) - 0.5) / len(y))
    return pred_ridge(np.hstack([FC_E[tr], SC_E[tr]]), yr,
                      np.hstack([FC_E[te], SC_E[te]]), alpha)


def random_subspace_ridge(tr, te, y, n_models=100, frac=0.1, alpha=100.0, seed=0):
    rng = np.random.RandomState(seed)
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    p = Xtr.shape[1]
    preds = np.zeros(len(te))
    k = max(2, int(frac * p))
    for i in range(n_models):
        idx = rng.choice(p, k, replace=False)
        s = StandardScaler().fit(Xtr[:, idx])
        m = Ridge(alpha=alpha).fit(s.transform(Xtr[:, idx]), y)
        preds += m.predict(s.transform(Xte[:, idx]))
    return preds / n_models


def bagged_ridge(tr, te, y, n_models=50, alpha=1000.0, seed=0):
    rng = np.random.RandomState(seed)
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    preds = np.zeros(len(te))
    n = len(tr)
    for i in range(n_models):
        b = rng.choice(n, n, replace=True)
        s = StandardScaler().fit(Xtr[b])
        m = Ridge(alpha=alpha).fit(s.transform(Xtr[b]), y[b])
        preds += m.predict(s.transform(Xte))
    return preds / n_models


def svr_rbf(tr, te, y, C=1.0, gamma="scale", eps=0.1):
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    s = StandardScaler().fit(Xtr)
    m = SVR(C=C, gamma=gamma, epsilon=eps, kernel="rbf")
    m.fit(s.transform(Xtr), y)
    return m.predict(s.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cands = {
        "enet": lambda tr, te, y: elastic_net(tr, te, y),
        "twoalpha": lambda tr, te, y: two_alpha_ridge(tr, te, y),
        "rankridge": lambda tr, te, y: rank_target_ridge(tr, te, y),
        "rsr100": lambda tr, te, y: random_subspace_ridge(tr, te, y),
        "bagged": lambda tr, te, y: bagged_ridge(tr, te, y),
        "svr": lambda tr, te, y: svr_rbf(tr, te, y),
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
