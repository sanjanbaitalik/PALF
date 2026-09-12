# OLD PHASE-3A-FIX FORENSIC AUDIT

Status: `INVALID_FOR_PHASE3A_DECISION`

The Phase-3A-FIX run (`scripts_paper/phase3a_fix_pilot.py`, output
`outputs/iclr/palf_phase3a_fix_pg_mt_bcr/`) implemented the corrected R0
baseline and preserved the holdout seal, but two additional selection bugs
invalidate every B0/B1/C0/C1/C2/C3/C4 development comparison.

---

## F11 — Inner OOF targets were misaligned with validation subjects

**Location:** `scripts_paper/phase3a_fix_pilot.py`

Each inner record stored training labels (line 270):

```python
"yB_WM": y_wm[B], "yB_FI": y_fi[B]
```

but the concatenated inner-OOF truth arrays were assembled by pairing the
validation subject indices `C` with those unrelated training labels
(lines 283-286):

```python
for g, v in zip(st["C"], st["yB_WM"]):
    y_oof_WM[pos[int(g)]] = v

for g, v in zip(st["C"], st["yB_FI"]):
    y_oof_FI[pos[int(g)]] = v
```

This pairs validation subjects `C` with labels of training subjects `B`.

**Consequences:**
- candidate Pearson/RMSE/MAE were computed against wrong labels;
- eligibility/non-inferiority (delta_r >= -0.002) was computed against wrong
  labels;
- alpha selection was invalid;
- hyperparameter selection was invalid;
- all B0/B1/C0/C1/C2/C3/C4 outer results came from invalidly selected
  candidates.

**Correct rule (FIX2 §7-8):** store `"yC_WM": y_wm[C]` / `"yC_FI": y_fi[C]`
and assemble from those, then hard-assert

```python
np.array_equal(y_oof_WM, y_wm[train_idx])
np.array_equal(y_oof_FI, y_fi[train_idx])
```

in every outer split before model selection. Phase-3A-FIX had no such
assertion, so the misalignment went undetected.

---

## F12 — Inner OOF BCR residual predictions remained standardized

**Location:** `scripts_paper/phase3a_fix_pilot.py`, `_oof_residuals()`
(lines 300-313)

`predict_bcr()` returns standardized (z-scale) residual predictions. The old
helper wrote those z-scale values directly into the concatenated residual
array:

```python
rWM, rFI = predict_bcr(r, st["fcC"], st["scC"])
if rWM is not None:
    for g, v in zip(st["C"], rWM):
        res["WM"][pos[int(g)]] = v
```

Candidate selection then evaluated `base_oof + alpha * standardized_residual`,
while the final outer refit correctly evaluated
`base_test + alpha * (standardized_residual * sd + mean)`
(lines 449-450).

**Consequences:**
- alpha/hyperparameters were selected in one unit system and applied in
  another;
- z-scale residuals (std ≈ 1) under-weighted the correction relative to
  raw-unit residuals (std ≈ 10 for WM), explaining the frequent selection of
  large alphas and the severe outer RMSE inflation observed in FIX
  (e.g. B0/C0/C1 RMSE ≈ 14-16 vs R0 ≈ 11.3).

**Correct rule (FIX2 §9):** inside each inner fold de-standardize with that
fold's B-residual statistics before storage:

```python
rWM_raw = rWM_z * st["sd_WM"] + st["mu_WM"]
rFI_raw = rFI_z * st["sd_FI"] + st["mu_FI"]
```

and store only raw-unit values, so inner selection and the final outer refit
are dimensionally identical.

---

## Results invalidated

Do not interpret or reuse as evidence from Phase-3A-FIX:
- A0 vs B0/B1/C0/C1/C2/C3/C4 comparisons;
- Gate 1-3 outcomes;
- the apparent C1-C0 prior gain;
- selected alpha/lambda frequencies;
- C0/C1 biomarker stability and faithfulness statistics.

All of these were produced by the F11/F12-invalid selection criterion.

## Still valid from Phase-3A-FIX

- the strict corrected-R0 baseline audit (WM r=0.2635147736,
  FI r=0.3709173350, reproduced in FIX2);
- the 98-subject holdout seal.
