# Phase 3A-FIX2: OOF Label Alignment + Residual Units Report

Status: `PHASE3A_FIX2_COMPLETE`
Decision: `DO_NOT_TOUCH_HOLDOUT`

---

## A. Holdout seal

- count = 98; unique = 98
- SHA256 = `89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425`
- dev/holdout intersection = 0
- Access log `PHASE3A_DATA_ACCESS_LOG.jsonl`: 0 non-empty holdout overlaps
- No holdout FC/SC/labels/distributions/R0 predictions/metrics were loaded.
- Old Phase-3A and Phase-3A-FIX outputs were not modified.

Development cohort: 412 unique subjects.

---

## B. Forensic audit (`OLD_PHASE3A_FIX_FORENSIC_AUDIT.md`)

### F11 — inner OOF validation truth misalignment (CONFIRMED)
`scripts_paper/phase3a_fix_pilot.py` stored training labels `"yB_WM": y_wm[B]`
(line 270) and assembled the inner-OOF truth by zipping validation indices `C`
with those training labels (lines 283-286). Candidate Pearson/RMSE/MAE,
eligibility, alpha selection, and hyperparameter selection were therefore all
computed against wrong labels, invalidating every B0/B1/C0/C1/C2/C3/C4
comparison. FIX2 stores `"yC_WM": y_wm[C]` and hard-asserts
`np.array_equal(y_oof, y[train_idx])` in every outer split before selection.

### F12 — standardized-vs-raw residual unit mismatch (CONFIRMED)
`_oof_residuals()` (lines 300-313) stored z-scale `predict_bcr` outputs
directly, so selection evaluated `base + alpha*z` while the final outer refit
evaluated `base + alpha*(z*sd+mean)`. This explains FIX's frequent alpha=1
selection with RMSE inflation (B0/C0/C1 RMSE 14-16 vs R0 11.3). FIX2
de-standardizes per fold (`raw = z*sd_B + mu_B`) before storage; selection and
the final refit are now dimensionally identical (parity max error 0.0).

Because of F11/F12, the FIX-era Gate 1-3 outcomes, the apparent C1-C0 prior
gain, alpha/lambda frequencies, and all C0/C1 biomarker/faithfulness
statistics are retracted. The corrected R0 audit and holdout seal remain valid.

---

## C. Strict R0 audit (unchanged, seeds 0-9)

| Task | r | err | RMSE | err | result |
|------|---|-----|------|-----|--------|
| WM | 0.2635147736 | 2.3e-07 | 11.2929210027 | 2.7e-09 | PASS |
| FI | 0.3709173350 | 3.4e-07 | 4.5666891937 | 1.9e-07 | PASS |

Tolerances: Pearson <= 5e-4, RMSE <= 0.05.

---

## D. Alignment / unit validity

- `y_oof == y[T]` hard assertion executed in **20/20** outer splits
  (`units_alignment_audit.json`: `all_y_oof_equal_yT: true`)
- residual OOF units: `raw_target_units` in all 21,900 selection rows
- inner/final formula parity: max error **0.0** over 20 splits
  (alpha ∈ {0, 0.5, 1} checked per fold on the real helper path)
- inner R0 baseline Pearson (aligned): WM mean 0.2045, FI mean 0.3188
  (recorded, no threshold applied)
- Tests: T1-T10 all pass (see L)

---

## E. Development models (seeds 4343/4444/4545/4646, 5 outer folds)

| Model | WM r | WM RMSE | WM MAE | FI r | FI RMSE | FI MAE |
|-------|------|---------|--------|------|---------|--------|
| A0 corrected R0 | 0.263226 | 11.210040 | 9.143638 | 0.382205 | 4.545105 | 3.719611 |
| B0 ST-BCR-NP | 0.251509 | 11.393386 | 9.249465 | 0.366671 | 4.609254 | 3.763733 |
| B1 ST-PG-BCR | 0.244229 | 11.581389 | 9.373437 | 0.362078 | 4.625341 | 3.786634 |
| C0 MT-BCR-NP | 0.259430 | 11.419962 | 9.290737 | 0.360278 | 4.616519 | 3.764865 |
| C1 PG-MT-BCR matched | 0.240991 | 11.662031 | 9.382934 | 0.372966 | 4.606915 | 3.760703 |
| C2 cross-task | 0.249059 | 11.425606 | 9.296532 | 0.377058 | 4.589795 | 3.745062 |
| C3 shuffled | 0.246679 | 11.405347 | 9.272918 | 0.381333 | 4.559530 | 3.723667 |
| C4 random | 0.254956 | 11.362466 | 9.237082 | 0.370699 | 4.606007 | 3.755834 |

With honest selection the BCR corrections stay close to R0 and none of the
BCR variants beats it on either task.

---

## F. Seed-level decomposition (mean / median / positive)

| Comparison | WM | FI |
|------------|----|----|
| C1−A0 | −0.022235 / −0.023017 / 0 | −0.009239 / −0.006585 / 2 |
| C1−C0 | −0.018439 / −0.018475 / 1 | +0.012688 / +0.014213 / 3 |
| C0−B0 | +0.007921 / +0.006535 / 3 | −0.006394 / −0.001695 / 2 |
| C1−B1 | −0.003238 / −0.003956 / 2 | +0.010888 / +0.005831 / 2 |
| B1−B0 | −0.007280 / −0.001251 / 1 | −0.004593 / −0.004137 / 2 |
| C1−C2 | −0.008068 / −0.009439 / 1 | −0.004092 / −0.000991 / 2 |
| C1−C3 | −0.005689 / −0.007587 / 1 | −0.008367 / −0.007806 / 0 |
| C1−C4 | −0.013966 / −0.017164 / 1 | +0.002267 / +0.002984 / 2 |

---

## G. Selection diagnostics

- 21,900 candidate rows; eligible fraction 21.7%
- Selected alpha_WM: {0.0: 69, 0.25: 65, 1.0: 21, 0.5: 15, 0.75: 10}
- Selected lambda_amp: {10.0: 91, 1.0: 57, 0.1: 32}
- Selected lambda_prior: {1.0: 63, 0.0: 60, 0.01: 34, 0.1: 23}
- Abstention (alpha=0 fallback): 43.8% of final C0/C1 task fits
- inner R0 Pearson (aligned): WM mean 0.2045 (range 0.117-0.316),
  FI mean 0.3188 (range 0.157-0.432)

---

## H. Optimizer

- 7,920 fits (2 restarts × 3,960 chosen); chosen convergence 96.46%
- restart 0 = 1,920, restart 1 = 2,040; 0 non-finite losses
- runtime 24,413 s (~6.8 h)

---

## I. Biomarker readiness (recomputed from scratch)

| Model/Task | edge Spearman | top100 J | top300 J | ROI top10 | ROI top20 | sign consistency |
|------------|---------------|----------|----------|-----------|-----------|------------------|
| C0 WM | 0.081 | 0.554 | 0.557 | 0.568 | 0.589 | 0.712 |
| C0 FI | 0.099 | 0.125 | 0.143 | 0.153 | 0.206 | 0.724 |
| C1 WM | 0.164 | 0.122 | 0.139 | 0.164 | 0.218 | 0.630 |
| C1 FI | 0.240 | 0.093 | 0.110 | 0.143 | 0.200 | 0.658 |

Faithfulness (RMSE change; negative = degradation):
- C1 WM k=10: top −0.259 vs random −0.051 (top degrades more)
- C1 FI k=10: top −0.040 vs random −0.009
- C0 WM/FI: top ≈ random (many alpha=0 abstentions → maps unused)

Coefficient reconstruction max error: 1.69e-14 (<= 1e-8).

---

## J. Gate evaluation

- **Gate 1 FAIL**: C1−A0 WM −0.0222 (0/4 positive); FI −0.0092 (2/4)
- **Gate 2 FAIL**: C1−C0 WM −0.0184 (1/4)
- **Gate 3 FAIL**: C1−C3 WM −0.0057, FI −0.0084; C1−C4 WM −0.0140
- **Gate 4 PASS**: finite; convergence 96.5% >= 95%; no RMSE > 2× R0

---

## K. Freeze / lock

`HOLDOUT_REMAINS_LOCKED` — no `MODEL_FROZEN.json` created.

---

## L. Tests

- FIX2 alignment/unit tests (`tests/test_phase3a_fix2_alignment_units.py`):
  17/17 passed (T1-T10 + 7 retained checks)
- Retained Phase-3A-FIX suite (`tests/test_phase3a_fix_pg_mt_bcr.py`): 40/40 passed
- Full repository suite: 750 passed, 2 skipped, 3 failed
- Pre-existing unrelated failures: `tests/test_submission_additions.py`
  (3 tests, unrelated to Phase 3A)

---

## M. Outputs

- `outputs/iclr/palf_phase3a_fix2_pg_mt_bcr/`
- `COMPLETE`, `RUN_REPORT.md`, `OLD_PHASE3A_FIX_FORENSIC_AUDIT.md`
- `BASELINE_AUDIT.json`, `HOLDOUT_SEAL_REPORT.json`, `VALIDATION_REPORT.json`,
  `units_alignment_audit.json`
- `PHASE3A_DATA_ACCESS_LOG.jsonl`
- `split_metrics.csv`, `seed_metrics.csv`, `model_summary.csv`,
  `selection_details.csv` (all rows `residual_prediction_units=raw_target_units`),
  `optimizer_diagnostics.csv`
- `coefficients/` (81 files), `predictions/` (21 files),
  `biomarkers/stability.csv`, `biomarkers/faithfulness.csv`,
  `prior_controls/`, `plots/` (12 files)
- `outputs/iclr/palf_phase3a_fix2_pg_mt_bcr.zip`

---

```text
PHASE3A_FIX2_DECISION: DO_NOT_TOUCH_HOLDOUT
STATUS: PHASE3A_FIX2_COMPLETE
```
