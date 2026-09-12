#!/usr/bin/env python3
"""Export per-split predictions + attribution artifacts for OpenChallenge CKE."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.openchallenge.ckr import (  # noqa: E402
    N_ROI, fit_krr, krr_predict, median_heuristic)
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "openchallenge"
CKPT = OUT / "_checkpoints"


def main():
    FC = np.load(ROOT / 'inputs/dataset_FC/FC_all.npy').astype(np.float64)
    SC = np.load(ROOT / 'inputs/dataset_SC/SC_all.npy').astype(np.float64)
    y_wm = np.load(ROOT / 'inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy').astype(np.float64)
    y_fi = np.load(ROOT / 'inputs/dataset_SC/label_all.npy').astype(np.float64)
    iu = np.triu_indices(116, k=1)
    fc_e, sc_e = FC[:, iu[0], iu[1]], SC[:, iu[0], iu[1]]
    r0 = R0Baseline(fc_e, sc_e, y_wm, y_fi, np.array([str(i) for i in range(412)]))

    # attribution artifacts
    for ck in sorted(CKPT.glob("split_seed*_fold*.pkl")):
        o = pickle.load(open(ck, "rb"))
        seed, fold = o["seed"], o["fold"]
        tr = np.array(o["train_idx"])
        for model in ("P1_rbf", "P0_linear"):
            for task in ("WM", "FI"):
                a = o["artifacts"][f"{model}_{task}"]
                out = {k: v for k, v in a.items() if k != "fit_obj"}
                np.savez(OUT / "coefficients_or_attributions" /
                         f"{model}_{task}_seed{seed}_fold{fold}.npz", **out)
        # controls dir: linear-kernel selections
        np.savez(OUT / "controls" / f"control_sel_seed{seed}_fold{fold}.npz",
                 **{f"{k}_{t}": np.array([o["selection"][f"{k}_{t}"]["w"],
                                          o["selection"][f"{k}_{t}"]["lam"]])
                    for k in ("P0_linear",) for t in ("WM", "FI")})

    # predictions: refit final models per split
    rows = []
    t0 = time.time()
    for ck in sorted(CKPT.glob("split_seed*_fold*.pkl")):
        o = pickle.load(open(ck, "rb"))
        seed, fold = o["seed"], o["fold"]
        tr = np.array(o["train_idx"]); te = np.array(o["test_idx"])
        fmu, fsd = fc_e[tr].mean(0), fc_e[tr].std(0)
        fsd = np.where(fsd < 1e-10, 1.0, fsd)
        smu, ssd = sc_e[tr].mean(0), sc_e[tr].std(0)
        ssd = np.where(ssd < 1e-10, 1.0, ssd)
        XT = np.hstack([(fc_e[tr] - fmu) / fsd, (sc_e[tr] - smu) / ssd])
        XV = np.hstack([(fc_e[te] - fmu) / fsd, (sc_e[te] - smu) / ssd])
        for task in ("WM", "FI"):
            ytr = y_wm[tr] if task == "WM" else y_fi[tr]
            yte = y_wm[te] if task == "WM" else y_fi[te]
            base_te, _ = r0.fit_r0_predict_full(task, tr, te, seed, fold,
                                                cache_tag=f"{seed}:{fold}:test")
            for model in ("P1_rbf", "P0_linear"):
                sel = o["selection"][f"{model}_{task}"]
                w, lam, mult = sel["w"], sel["lam"], sel["mult"]
                kern = "rbf" if model == "P1_rbf" else "linear"
                fit = fit_krr(XT, ytr, kern, lam, mult,
                              sigma_base=(median_heuristic(XT) if kern == "rbf" else None))
                fk = krr_predict(fit, XV)
                pred = (1 - w) * base_te + w * fk
                for i in range(len(te)):
                    rows.append({"seed": seed, "fold": fold, "model": model, "task": task,
                                 "subject": str(tr[0]) if False else str(te[i]),
                                 "y": yte[i], "pred": pred[i],
                                 "R0_pred": base_te[i], "w": w})
        print(f"  seed={seed} fold={fold} exported ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictions" / "test_predictions.csv", index=False)
    for (seed, fold), sub in df.groupby(["seed", "fold"]):
        sub.to_csv(OUT / "predictions" / f"pred_seed{seed}_fold{fold}.csv", index=False)
    print(f"Exported {len(df)} prediction rows.")


if __name__ == "__main__":
    main()
