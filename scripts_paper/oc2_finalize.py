#!/usr/bin/env python3
"""OpenChallenge finalization: holdout seal, strict R0 audit, and the
NO_METHOD_JUSTIFIED evidence dossier."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "openchallenge"
OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
DEV_TXT = ROOT / "data_splits" / "phase3_development_412.txt"
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
EXPECTED = {"WM": (0.263515, 11.292921), "FI": (0.370917, 4.566689)}
TOL_R, TOL_RMSE = 5e-4, 0.05


def main():
    t0 = time.time()
    holdout = HOLDOUT_TXT.read_text().strip().split("\n")
    dev = DEV_TXT.read_text().strip().split("\n")
    sha = hashlib.sha256(("\n".join(sorted(holdout)) + "\n").encode()).hexdigest()
    assert len(holdout) == 98 and len(set(holdout)) == 98
    assert len(dev) == 412 and len(set(dev)) == 412
    assert sha == HOLDOUT_SHA
    assert set(dev).isdisjoint(set(holdout))
    print(f"Holdout seal OK: n=98 sha={sha[:16]}...  dev=412 disjoint")
    with open(OUT / "HOLDOUT_SEAL_REPORT.json", "w") as f:
        json.dump({"n_subjects": 98, "canonical_sha256": sha, "dev_count": 412,
                   "intersection": 0, "status": "SEALED_UNTOUCHED",
                   "holdout_features_labels_loaded": False}, f, indent=2)

    FC = np.load(ROOT / "inputs/dataset_FC/FC_all.npy").astype(np.float64)
    SC = np.load(ROOT / "inputs/dataset_SC/SC_all.npy").astype(np.float64)
    y = {"WM": np.load(ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64),
         "FI": np.load(ROOT / "inputs/dataset_SC/label_all.npy").astype(np.float64)}
    iu = np.triu_indices(116, 1)
    fc_e, sc_e = FC[:, iu[0], iu[1]], SC[:, iu[0], iu[1]]
    r0 = R0Baseline(fc_e, sc_e, y["WM"], y["FI"], np.array([str(x) for x in range(412)]))

    print("\nSTRICT CORRECTED R0 BASELINE AUDIT (seeds 0-9, 5 outer folds)")
    rows = []
    for seed in range(10):
        idx = np.random.RandomState(seed).permutation(412)
        sizes = np.full(5, 412 // 5); sizes[:412 % 5] += 1
        cur = 0
        for fold in range(5):
            a, b = cur, cur + sizes[fold]
            te, tr = idx[a:b], np.concatenate([idx[:a], idx[b:]])
            cur = b
            for t in ("WM", "FI"):
                p = r0.fit_r0_predict(t, tr, te, seed, fold, cache_tag="ocaudit")
                yt = y[t][te]
                rows.append({"task": t, "seed": seed, "fold": fold,
                             "pearson": np.corrcoef(p, yt)[0, 1],
                             "rmse": float(np.sqrt(np.mean((p - yt) ** 2)))})
    df = pd.DataFrame(rows)
    details = {}
    ok_all = True
    for t in ("WM", "FI"):
        sub = df[df["task"] == t]
        r, rmse = float(sub["pearson"].mean()), float(sub["rmse"].mean())
        er, erm = EXPECTED[t]
        ok = abs(r - er) <= TOL_R and abs(rmse - erm) <= TOL_RMSE
        ok_all = ok_all and ok
        details[t] = {"r": r, "rmse": rmse, "expected_r": er, "expected_rmse": erm,
                      "r_err": abs(r - er), "rmse_err": abs(rmse - erm), "pass": bool(ok)}
        print(f"  {t}: r={r:.10f} (err {abs(r-er):.2e})  RMSE={rmse:.10f} "
              f"(err {abs(rmse-erm):.2e}) -> {'PASS' if ok else 'FAIL'}")
    with open(OUT / "BASELINE_AUDIT.json", "w") as f:
        json.dump({"details": details, "status": "PASS" if ok_all else "FAIL",
                   "tolerance_pearson": TOL_R, "tolerance_rmse": TOL_RMSE}, f, indent=2)
    df.to_csv(OUT / "audit_split_metrics.csv", index=False)
    print(f"Audit {'PASS' if ok_all else 'FAIL'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
