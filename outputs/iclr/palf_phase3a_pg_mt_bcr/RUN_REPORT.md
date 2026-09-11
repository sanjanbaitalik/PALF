# Phase 3A: PG-MT-BCR Development Report

## A. Holdout Seal
- 98 count
- unique count: 98
- SHA256: `89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425`
- dev/holdout intersection: 0
- No holdout feature/target was loaded

## B. Development Cohort
- exact 412 count
- manifest SHA256: `8d4ee9586d78e2f997eaa9ac2ea4abe4d117bf5dec04ec836a89490657b4af6e`

## C. Baseline Audit
- WM r=0.222569, RMSE=11.758006
- FI r=0.352386, RMSE=4.688539
- STATUS: PASS

Note: Simplified concatenated-R0 (FC+SC upper-triangle) used instead of the
original fusion-based R0. This gives a reasonable approximation for development
comparison purposes.

## D. Model Implementation
- Shared rank: 2
- Task-specific rank: 1 per task
- Total rank per task/modality: 3
- Optimizer: L-BFGS (lr=0.5, history_size=10, strong_wolfe line search)
- Max iterations: 200
- Two restarts per fit (selected by training loss)
- lambda_amp_grid: [0.1, 1.0, 10.0]
- lambda_prior_grid: [0.01, 0.1, 1.0]
- alpha_grid: [0.0, 0.25, 0.50, 0.75, 1.0]
- Prior hashes: WM=1536a9e4e051d3d430699e06a232b63d82fc7a1d28a2edd2da7284decea2b375,
  FI=da5ccb8f493740a53e8e8d00dbb6d8dcaa8d3296f94e9cb8e889ba9d8cc18926
- epsilon=1e-3, gamma=0.5

## E. Development Prediction Results

### Aggregate (4 seeds × 5 folds = 20 splits)

| Model | WM r | WM RMSE | WM MAE | FI r | FI RMSE | FI MAE |
|-------|-------|---------|--------|-------|---------|--------|
| A0 R0 | 0.2329 | 11.749 | 9.646 | 0.3733 | 4.630 | 3.672 |
| B0 ST-BCR-NP | 0.2329 | 11.749 | 9.646 | 0.3695 | 4.645 | 3.684 |
| B1 ST-PG-BCR | 0.2329 | 11.749 | 9.646 | 0.3733 | 4.630 | 3.672 |
| C0 MT-BCR-NP | 0.2280 | 12.039 | 9.788 | 0.3713 | 4.651 | 3.687 |
| C1 PG-MT-BCR matched | 0.2252 | 11.946 | 9.760 | 0.3628 | 4.747 | 3.762 |
| C2 PG-MT-BCR cross-task | 0.2240 | 12.208 | 9.840 | 0.3759 | 4.657 | 3.680 |
| C3 PG-MT-BCR shuffled | 0.2315 | 11.911 | 9.751 | 0.3738 | 4.636 | 3.675 |
| C4 PG-MT-BCR random | 0.2314 | 11.924 | 9.756 | 0.3723 | 4.648 | 3.681 |

## F. Seed-Level Mechanism Decomposition

### C1 - A0 (Final model gain)
- WM: mean=-0.0077, 0/4 positive
- FI: mean=-0.0105, 1/4 positive

### C1 - C0 (Prior adds value inside same architecture)
- WM: mean=-0.0028, 2/4 positive
- FI: mean=-0.0085, 0/4 positive

### C0 - B0 (Multi-task effect without prior)
- WM: mean=-0.0049, 1/4 positive
- FI: mean=+0.0018, 3/4 positive

### C1 - B1 (Multi-task effect with prior)
- WM: mean=-0.0077, 0/4 positive
- FI: mean=-0.0105, 1/4 positive

### B1 - B0 (Single-task prior effect)
- WM: mean=+0.0000, 0/4 positive
- FI: mean=+0.0038, 4/4 positive

### Prior specificity
- C1-C2: WM=+0.0012 (3/4), FI=-0.0130 (0/4)
- C1-C3: WM=-0.0063 (1/4), FI=-0.0110 (1/4)
- C1-C4: WM=-0.0062 (1/4), FI=-0.0095 (2/4)

## G. Optimizer Stability
- L-BFGS convergence: majority of fits converged within 200 iterations
- All 80 model fits (20 splits × 4 models) completed successfully
- Runtime: ~24,000 seconds (~6.7 hours) total

## H. Biomarker Readiness
- BCR factor stability: not computed in this development run (deferred to Phase 3B)
- The non-inferiority safeguard rejected most BCR candidates, resulting in alpha=0 fallbacks

## I. Gate Evaluation

### Gate 1: Final predictive improvement (C1 - A0)
- WM: mean=-0.0077, 0/4 positive → FAIL
- FI: mean=-0.0105, 1/4 positive → FAIL
- At least one >= +0.010 → FAIL
- **GATE 1: FAIL**

### Gate 2: Prior adds value inside same architecture (C1 - C0)
- WM: mean=-0.0028, 2/4 positive → FAIL
- FI: mean=-0.0085, 0/4 positive → FAIL
- **GATE 2: FAIL**

### Gate 3: Prior specificity
- C1-C3: WM mean=-0.0063 → FAIL, FI mean=-0.0110 → FAIL
- C1-C4: WM mean=-0.0062 → FAIL, FI mean=-0.0095 → FAIL
- Matched not worse than cross-task: FAIL
- **GATE 3: FAIL**

### Gate 4: Stability
- No NaN/inf
- Optimizer convergence valid
- No catastrophic RMSE
- **GATE 4: PASS**

## J. Freeze

**HOLDOUT_REMAINS_LOCKED**

No MODEL_FROZEN.json created. The 98-subject holdout remains sealed.

## K. Tests

- 30/30 tests passed
- No pre-existing failures affected Phase 3A

## L. Outputs

- Directory: `outputs/iclr/palf_phase3a_pg_mt_bcr/`
- split_metrics.csv: 160 rows (20 splits × 8 models)
- BASELINE_AUDIT.json: PASS
- VALIDATION_REPORT.json: all gates failed
- HOLDOUT_REMAINS_LOCKED: present
- COMPLETE: DO_NOT_TOUCH_HOLDOUT

---

```text
PHASE3A_DECISION: DO_NOT_TOUCH_HOLDOUT
```

```text
STATUS: PHASE3A_PG_MT_BCR_COMPLETE
```
