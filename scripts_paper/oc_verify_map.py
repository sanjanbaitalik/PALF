#!/usr/bin/env python3
"""Verify R0 fused linear map reconstruction (feasibility for CKE)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

FC = np.load(ROOT / 'inputs/dataset_FC/FC_all.npy').astype(np.float64)
SC = np.load(ROOT / 'inputs/dataset_SC/SC_all.npy').astype(np.float64)
iu = np.triu_indices(116, k=1)
fc = FC[:, iu[0], iu[1]]
sc = SC[:, iu[0], iu[1]]
y_wm = np.load(ROOT / 'inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy').astype(np.float64)
y_fi = np.load(ROOT / 'inputs/dataset_SC/label_all.npy').astype(np.float64)

r0 = R0Baseline(fc, sc, y_wm, y_fi, np.array([str(i) for i in range(412)]))
rng = np.random.RandomState(0)
idx = rng.permutation(412)
te, tr = idx[:83], np.concatenate([idx[83:]])
for task in ("WM", "FI"):
    pred, info = r0.fit_r0_predict_full(task, tr, te, 0, 0, cache_tag="verify")
    print(f"{task}: pred shape {pred.shape} recon_err = {info['recon_err']:.3e} "
          f"weights = {info['fusion_weights']}")
