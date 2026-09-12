# Phase 3A-FIX: Corrected PG-MT-BCR Development Report

Status: `PHASE3A_FIX_COMPLETE`
Decision: `DO_NOT_TOUCH_HOLDOUT`

---

## A. Holdout seal

- count = 98
- unique = 98
- canonical SHA256 = `89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425`
- dev/holdout intersection = 0
- Access log `PHASE3A_DATA_ACCESS_LOG.jsonl`: no holdout ID appears
- No holdout FC, SC, labels, distributions, or metrics were loaded.
- The old Phase-3A output was not modified.

Dev manifest: 412 unique subjects (`data_splits/phase3_development_412.txt`).

---

## B. Old Phase-3A forensic audit

See `OLD_PHASE3A_FORENSIC_AUDIT.md`. All findings confirmed:

| Finding | Status |
|---------|--------|
| F1 baseline was not corrected R0 | CONFIRMED |
| F2 audit tolerances weakened (0.05/0.5 instead of 5e-4/0.05) | CONFIRMED |
| F3 only one inner split used | CONFIRMED |
| F4 inner preprocessing leakage | CONFIRMED |
| F5 wrong R0 residual generator | CONFIRMED |
| F6 single-task WM result overwritten | CONFIRMED |
| F7 ST prior-strength grid not searched | CONFIRMED |
| F8 MT shared amplitudes shared across tasks | CONFIRMED |
| F9 optimizer did not match frozen procedure (L-BFGS) | CONFIRMED |
| F10 biomarker/optimizer outputs omitted | CONFIRMED |

---

## C. Correct baseline audit (strict)

Validated corrected R0 = same-solver no-prior FP + SC Ridge + cross-fitted
convex FP+SC fusion, seeds 0-9, five outer folds (50 R0 splits).

| Task | r | expected r | r err | RMSE | expected RMSE | RMSE err | result |
|------|---|------------|-------|------|---------------|----------|--------|
| WM | 0.2635147736 | 0.263515 | 2.26e-07 | 11.2929210027 | 11.292921 | 2.71e-09 | PASS |
| FI | 0.3709173350 | 0.370917 | 3.35e-07 | 4.5666891937 | 4.566689 | 1.94e-07 | PASS |

Tolerances: Pearson <= 5e-4, RMSE <= 0.05.

---

## D. Correct PG-MT-BCR implementation

- Shared spatial factors: rank 2 per modality (FC, SC).
- Task-specific factors: rank 1 per task per modality.
- Shared amplitudes are task-specific: `amp_fc_shared_WM`, `amp_fc_shared_FI`,
  `amp_sc_shared_WM`, `amp_sc_shared_FI`.
- ST models: 3 task-specific factors per modality, no shared factors
  (matched per-task rank = 3).
- Optimizer: full-batch Adam, `lr=0.01`, `max_steps=1500`, `min_steps=200`,
  `patience=75`, `rel_tol=1e-7`, `grad_clip=5.0`, `restarts=2`, `float64`.
- Projection after each optimizer step: unit-norm columns + Gram-Schmidt on
  shared factors. Max |norm-1| = 2.2e-16; max shared |dot| = 1.3e-17.
- Grids: `lambda_amp ∈ {0.1,1,10}`, `lambda_prior ∈ {0.01,0.1,1}`,
  `alpha ∈ {0,0.25,0.5,0.75,1}`. Ranks/gamma/epsilon fixed.
- Prior penalty: epsilon=1e-3, gamma=0.5, shared prior = normalized geometric
  mean of the two task priors.
- Coefficient reconstruction max error = 2.25e-14 (<= 1e-8).

Prior SHA256: WM `1536a9e4e051d3d430699e06a232b63d82fc7a1d28a2edd2da7284decea2b375`,
FI `da5ccb8f493740a53e8e8d00dbb6d8dcaa8d3296f94e9cb8e889ba9d8cc18926`.

---

## E. Development results (seeds 3939/4040/4141/4242, 5 outer folds)

| Model | WM r | WM RMSE | WM MAE | FI r | FI RMSE | FI MAE |
|-------|------|---------|--------|------|---------|--------|
| A0 corrected R0 | 0.258221 | 11.305898 | 9.192446 | 0.394129 | 4.505881 | 3.767714 |
| B0 ST-BCR-NP | 0.175407 | 15.127966 | 12.165814 | 0.308183 | 5.671296 | 4.531062 |
| B1 ST-PG-BCR | 0.160935 | 15.908047 | 12.538513 | 0.321790 | 5.668023 | 4.559353 |
| C0 MT-BCR-NP | 0.189081 | 14.272709 | 11.507037 | 0.282280 | 5.936592 | 4.831763 |
| C1 PG-MT-BCR matched | 0.199422 | 14.912381 | 11.921978 | 0.297205 | 5.927092 | 4.791769 |
| C2 cross-task | 0.175401 | 15.244028 | 12.230879 | 0.320335 | 5.704306 | 4.578699 |
| C3 shuffled | 0.188392 | 14.152339 | 11.375948 | 0.289905 | 5.773728 | 4.656465 |
| C4 random | 0.204449 | 13.674670 | 11.103265 | 0.291362 | 5.816390 | 4.640588 |

---

## F. Seed-level decomposition

| Comparison | WM mean | WM median | WM pos | FI mean | FI median | FI pos |
|------------|---------|-----------|--------|---------|-----------|--------|
| C1 - A0 | -0.058799 | -0.060907 | 0/4 | -0.096923 | -0.093230 | 0/4 |
| C1 - C0 | +0.010341 | +0.009629 | 3/4 | +0.014925 | +0.016407 | 3/4 |
| C0 - B0 | +0.013675 | -0.010954 | 2/4 | -0.025903 | -0.009755 | 1/4 |
| C1 - B1 | +0.038487 | +0.047752 | 3/4 | -0.024585 | -0.030257 | 1/4 |
| B1 - B0 | -0.014471 | +0.024932 | 2/4 | +0.013607 | +0.008412 | 3/4 |
| C1 - C2 | +0.024021 | +0.019880 | 2/4 | -0.023130 | -0.023799 | 1/4 |
| C1 - C3 | +0.011030 | +0.008935 | 2/4 | +0.007300 | +0.010503 | 2/4 |
| C1 - C4 | -0.005027 | -0.006907 | 2/4 | +0.005843 | +0.008906 | 2/4 |

The architecture-matched prior effect (C1 - C0) is positive on both tasks with
3/4 positive seeds (Gate 2 passes). However, the residual-correction
architecture itself is much weaker than R0 (C1 - A0 strongly negative), so the
final model does not beat the baseline.

---

## G. Selection diagnostics

- Total candidate evaluations: 21,900
- Eligible candidate fraction: 62.2%
- Selected alpha_WM: {1.0: 137, 0.0: 39, 0.25: 2, 0.5: 1, 0.75: 1}
- Selected lambda_amp: {0.1: 93, 1.0: 64, 10.0: 23}
- Selected lambda_prior: {0.0: 60, 0.1: 52, 0.01: 50, 1.0: 18}

---

## H. Optimizer diagnostics

- Total fits recorded: 7,920 (2 restarts × 3,960 selected fits)
- Chosen fits: 3,960
- Chosen-fit convergence fraction: 95.88%
- Restart choice: restart 0 = 1,940, restart 1 = 2,020
- Non-finite final losses: 0
- Total runtime: 26,194 s (~7.3 h)

---

## I. Biomarker readiness

Stability (mean over the 20 outer fits):

| Model/Task | edge Spearman | top100 Jaccard | top300 Jaccard | ROI top10 | ROI top20 | sign consistency |
|------------|---------------|----------------|----------------|-----------|-----------|------------------|
| C0 WM | 0.0706 | 0.1591 | 0.1726 | 0.1924 | 0.2348 | 0.646 |
| C0 FI | 0.0706 | 0.0365 | 0.0567 | 0.0677 | 0.1167 | 0.643 |
| C1 WM | 0.2317 | 0.1005 | 0.1152 | 0.1387 | 0.1952 | 0.624 |
| C1 FI | 0.2033 | 0.0732 | 0.0937 | 0.1157 | 0.1657 | 0.639 |

Matched prior (C1) coefficient maps are more stable than no-prior (C0)
(edge Spearman 0.23 vs 0.07 for WM; 0.20 vs 0.07 for FI).

Faithfulness (RMSE degradation; negative = degradation):

| Model/Task | k | top mean | bottom mean | random mean |
|------------|---|----------|-------------|-------------|
| C0 WM | 5 | -0.4389 | -0.0740 | -0.1745 |
| C0 WM | 10 | -0.6411 | -0.1102 | -0.3248 |
| C1 WM | 5 | -0.8264 | -0.0254 | -0.2037 |
| C1 WM | 10 | -1.3059 | -0.0915 | -0.3696 |
| C0 FI | 5 | -0.1418 | -0.0104 | -0.0717 |
| C1 FI | 5 | -0.2785 | -0.0127 | -0.0674 |

Top-ROI masking degrades the final prediction more than random masking for
both C0 and C1, with C1 showing a larger top-vs-random gap.

---

## J. Gate evaluation

- Gate 1 (final improvement C1-A0): **FAIL** (WM -0.0588 0/4; FI -0.0969 0/4)
- Gate 2 (matched prior C1-C0): **PASS** (WM +0.0103 3/4; FI +0.0149 3/4)
- Gate 3 (prior specificity C1-C3, C1-C4): **FAIL** (C1-C4 WM -0.0050)
- Gate 4 (stability): **PASS** (finite; convergence 95.9%; no catastrophic RMSE)

`PHASE3A_FIX_DECISION: DO_NOT_TOUCH_HOLDOUT`

---

## K. Freeze / lock

`HOLDOUT_REMAINS_LOCKED`

No `MODEL_FROZEN.json` was created. The 98-subject holdout remains sealed.

---

## L. Tests

- Targeted `tests/test_phase3a_fix_pg_mt_bcr.py`: 40/40 passed
- Full suite: 733 passed, 2 skipped, 3 failed
- Pre-existing unrelated failures: `tests/test_submission_additions.py`
  (`test_mask_connectomes_both_modalities`, `test_positive_degradation_direction`,
  `test_perturbed_graph_dataset_preserves_nonincident_edges`)

---

## M. Outputs

- `outputs/iclr/palf_phase3a_fix_pg_mt_bcr/`
- `COMPLETE`, `RUN_REPORT.md`, `OLD_PHASE3A_FORENSIC_AUDIT.md`
- `BASELINE_AUDIT.json`, `HOLDOUT_SEAL_REPORT.json`, `VALIDATION_REPORT.json`
- `PHASE3A_DATA_ACCESS_LOG.jsonl`
- `split_metrics.csv`, `seed_metrics.csv`, `model_summary.csv`,
  `selection_details.csv`, `optimizer_diagnostics.csv`
- `coefficients/` (81 files incl. reconstruction_errors.csv)
- `predictions/test_predictions.csv` + per-split files
- `biomarkers/stability.csv`, `biomarkers/faithfulness.csv`
- `prior_controls/`
- `plots/` (6 figures, pdf+png)
- `outputs/iclr/palf_phase3a_fix_pg_mt_bcr.zip`

---

```text
PHASE3A_FIX_DECISION: DO_NOT_TOUCH_HOLDOUT
STATUS: PHASE3A_FIX_COMPLETE
```
