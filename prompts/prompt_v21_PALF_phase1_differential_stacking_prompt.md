# PALF ICLR 2027 — Phase 1: Differential Stacking Feasibility Test (NO BRANCH REFITS)

## Purpose

We have only 2–3 days left before the ICLR submission. Before spending time on a new expensive PALF refit, run the **highest-leverage cheap experiment first**:

> Can we improve prediction simply by combining the already-frozen same-solver no-prior FP prediction, Full-PALF FP prediction, and SC prediction through a baseline-preserving differential stack?

This phase is **prediction-only feasibility**.  
Do **not** change the manuscript yet.  
Do **not** implement adaptive prior strength, new priors, or biomarker reconstruction yet.  
Do **not** rerun any FC/SC branch.

The result of this phase will determine the next OpenCode prompt.

---

# 0. Current frozen scientific state

Use the current corrected same-solver codebase and result lineage.

The authoritative corrected source is:

```text
outputs/iclr/palf_same_solver_fusion_corrected_v2/
```

The clean manuscript freeze is:

```text
outputs/iclr/palf_manuscript_freeze_v2_same_solver/
```

The relevant primary conditions are internally:

- `R0` = same-solver no prior
- `R1` = anisotropy only
- `R2` = network penalty only
- `R3` = Full PALF

Internal IDs may be used in code and diagnostic CSVs, but all manuscript-facing labels in this phase should be descriptive.

Current fused means that the corrected source should reproduce approximately:

### Working Memory
- Same-solver no prior: `0.263515`
- Anisotropy only: `0.257818`
- Network penalty only: `0.263664`
- Full PALF: `0.262500`

### Fluid Intelligence
- Same-solver no prior: `0.370917`
- Anisotropy only: `0.367844`
- Network penalty only: `0.370309`
- Full PALF: `0.370031`

If these are not reproduced from the loaded corrected checkpoint to normal floating-point/rounding tolerance, **STOP** and report the mismatch. Do not proceed with a different result lineage.

---

# 1. Scientific idea to test

The current Full PALF forces the prior-aware FC branch to replace the no-prior generalized FC branch before fusion.

Instead, preserve the strong no-prior predictor and learn only whether the semantic prior adds a useful correction.

For each outer split, define:

```text
y0_oof   = same-solver no-prior generalized FP OOF prediction
yp_oof   = Full-PALF generalized FP OOF prediction
ys_oof   = SC OOF prediction
delta_oof = yp_oof - y0_oof
```

and on the outer test set:

```text
y0_test
yp_test
ys_test
delta_test = yp_test - y0_test
```

The key semantic feature is therefore:

\[
\Delta_P = \hat y_P - \hat y_0.
\]

This isolates the prediction change introduced by semantic regularization.

---

# 2. Hard leakage rules

This experiment must remain fully outer-test clean.

For every target / seed / outer fold:

1. `R0` and `R3` must have identical:
   - `train_idx`
   - `test_idx`
   - target identity
   - subject ordering

2. The stacker may use only:
   - OOF predictions defined on the outer-training subjects
   - outer-training target values

3. The outer-test target must never be used for:
   - selecting a stack architecture
   - choosing a stack regularization value
   - choosing a residual coefficient
   - fitting stack weights
   - deciding whether to use the semantic correction

4. The outer-test set is evaluated exactly once after the meta-model is frozen.

5. Do not regenerate any base prediction.

If any alignment assertion fails, stop.

---

# 3. Required candidate meta-models

Evaluate exactly these three candidate models.

Do not add more models after looking at outer-test performance.

## Candidate A — corrected same-solver baseline

This is the existing corrected R0 convex FP+SC fusion.

Use its stored/recomputed training-OOF fusion rule and corrected test prediction.

Call it:

```text
Same-solver baseline fusion
```

This candidate guarantees that the meta-selection procedure has a safe fallback.

---

## Candidate B — residual semantic correction

Start from the corrected no-prior convex baseline:

\[
\hat y_{\mathrm{base}}
=
w_0\hat y_0 + w_S\hat y_S.
\]

Then add only a semantic differential correction:

\[
\hat y_{\mathrm{res}}
=
\hat y_{\mathrm{base}}
+
a\,(\hat y_P-\hat y_0).
\]

Require:

\[
a \ge 0.
\]

Search:

```text
a_grid = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
```

Important:

- `a = 0` exactly recovers the corrected same-solver baseline.
- The baseline convex weights are selected from R0 OOF predictions only.
- `a` is selected using only outer-training OOF predictions.

This is the lowest-variance semantic correction model.

---

## Candidate C — nonnegative differential Ridge stack

Use the three meta-features:

\[
z_1 = \hat y_0,\qquad
z_2 = \hat y_S,\qquad
z_3 = \hat y_P-\hat y_0.
\]

Fit:

\[
\hat y =
b +
w_0 z_1 +
w_S z_2 +
w_\Delta z_3,
\]

with:

\[
w_0,w_S,w_\Delta \ge 0.
\]

Do **not** force the three coefficients to sum to one.

Select the Ridge penalty from:

```text
lambda_stack_grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
```

### Implementation recommendation

Use an unpenalized intercept with nonnegative coefficients.

A clean implementation is:

1. center `Z` and `y` using the meta-training fold only;
2. solve

\[
\min_{w\ge0}
\|y_c-Z_cw\|_2^2+\lambda\|w\|_2^2
\]

using `scipy.optimize.lsq_linear` on the augmented system

```python
Z_aug = np.vstack([Z_centered, np.sqrt(lam) * np.eye(3)])
y_aug = np.concatenate([y_centered, np.zeros(3)])
```

with bounds `(0, np.inf)`;

3. recover

```python
intercept = y_mean - Z_mean @ w
```

Do not penalize the intercept.

For `lambda_stack = 0`, solve ordinary nonnegative least squares using the same bounded solver.

---

# 4. Meta-level model selection must itself be cross-validated

Do not choose Candidate B/C hyperparameters from in-sample fit on all OOF rows.

The base predictions are OOF, but the meta-model still needs a clean hyperparameter-selection procedure.

For each outer split:

## Meta-CV

Use deterministic 5-fold CV across the outer-training OOF rows.

Recommended:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=10000 + 100 * seed + outer_fold
)
```

For each meta-CV fold:

- fit any centering/intercept only on meta-train rows;
- fit Candidate B or C only on meta-train rows;
- evaluate the meta-validation rows.

For every candidate/hyperparameter compute pooled out-of-meta-fold predictions across all outer-training subjects.

Selection criterion:

1. maximize Pearson `r`;
2. if tied within `1e-12`, minimize RMSE;
3. if still tied, minimize MAE;
4. if still tied, prefer the simpler model:
   - baseline
   - residual correction
   - differential Ridge

For Candidate B, select `a`.

For Candidate C, select `lambda_stack`.

---

# 5. Final outer-split model selection

After obtaining cross-validated OOF performance for:

- Candidate A
- best Candidate B
- best Candidate C

select exactly one architecture for the outer split using the same ordering:

1. highest Pearson `r`
2. RMSE tie-break
3. MAE tie-break
4. simpler model tie-break

Then:

- refit only the selected **meta-model** on all outer-training OOF rows;
- do not refit any base FC/SC model;
- apply the selected meta-model once to outer-test predictions.

Call this selected model:

```text
Differential PALF stack
```

This candidate family contains the baseline, so the meta-layer can decline to use the semantic correction when training evidence does not support it.

---

# 6. Important scientific controls

## 6A. No direct use of R1/R2 for fitting the new stack

Do not include anisotropy-only or network-only predictions as meta-features in Phase 1.

They may be used only as current-model comparison rows in the final report.

The only meta-features are:

```text
R0 generalized FP
SC
R3 generalized FP - R0 generalized FP
```

## 6B. No test-driven selection

Do not:
- choose a different stack formula for WM and FI after seeing test results;
- choose candidate grids using outer-test performance;
- report only favorable seeds;
- remove poor outer folds;
- tune the method on the final test predictions.

The same candidate family and grids must be applied to both targets.

---

# 7. Prediction metrics to save

For every target / seed / outer fold save:

- selected meta-architecture
- selected `a` if residual candidate
- selected `lambda_stack` if Ridge candidate
- `w0`
- `w_sc`
- `w_delta`
- intercept
- meta-CV Pearson
- meta-CV RMSE
- meta-CV MAE
- outer-test Pearson
- outer-test RMSE
- outer-test MAE
- baseline outer-test metrics
- Full PALF outer-test metrics
- network-only outer-test metrics
- anisotropy-only outer-test metrics

Also save per-subject outer-test predictions:

- target
- baseline prediction
- Full-PALF prediction
- Differential-PALF-stack prediction
- semantic differential prediction component

---

# 8. Seed-level and primary statistics

As in the final paper:

1. average the five outer-fold metrics within each seed;
2. obtain exactly 10 seed summaries per target.

Compute the following paired comparisons for fused Pearson correlation:

## Primary feasibility comparison

```text
Differential PALF stack
vs.
Same-solver no-prior baseline
```

Report:

- mean model `r`
- mean baseline `r`
- mean paired delta
- median paired delta
- SD of paired delta
- positive seeds / 10
- exact two-sided paired Wilcoxon
- 10,000-resample paired bootstrap 95% CI
- paired Cohen's dz

Apply Holm correction across the two target tests.

## Current-best comparison

Also compare the new stack descriptively against the best existing fused condition:

### WM current best
```text
Network penalty only ≈ 0.263664
```

### FI current best
```text
Same-solver no prior ≈ 0.370917
```

Do not invent significance families for these secondary feasibility comparisons unless explicitly implemented and reported separately.

---

# 9. Architecture-selection diagnostics

This is important for deciding whether the semantic correction is actually useful.

For each target report across the 50 outer splits:

- number selecting baseline
- number selecting residual correction
- number selecting differential Ridge
- distribution of selected `a`
- distribution of selected `lambda_stack`
- mean/median `w_delta` for Candidate C splits
- fraction with `w_delta > 0`
- fraction of all selected outer models that use any semantic correction at all

Also report seed-level counts.

This will tell us whether the proposed semantic residual is consistently useful or whether the model mostly falls back to the baseline.

---

# 10. Cheap additional oracle-free diagnostic

Without using outer-test targets for selection, save the training/meta-CV relationship between:

```text
meta-CV improvement over baseline
```

and

```text
outer-test improvement over baseline
```

across the 50 splits.

Report only:
- Pearson correlation between those two split-level improvements
- Spearman correlation
- scatter plot

This is diagnostic only and must not be used to choose the method.

It tells us whether the meta-CV gate is predictive of actual benefit.

---

# 11. New code organization

Prefer adding a new standalone script:

```text
scripts_paper/phase1_differential_stacking_feasibility.py
```

Do not overload the frozen primary experiment runner.

Add reusable helpers in a small module only if needed.

Do not alter the scientific values in:

```text
outputs/iclr/palf_same_solver_fusion_corrected_v2/
outputs/iclr/palf_manuscript_freeze_v2_same_solver/
```

Treat both as read-only inputs.

---

# 12. New isolated output directory

Create:

```text
outputs/iclr/palf_phase1_differential_stacking/
```

Suggested structure:

```text
outputs/iclr/palf_phase1_differential_stacking/
├── README.md
├── RUN_REPORT.md
├── VALIDATION_REPORT.json
├── split_metrics.csv
├── seed_metrics.csv
├── primary_comparisons.csv
├── architecture_selection.csv
├── outer_predictions/
│   ├── working_memory.csv
│   └── fluid_intelligence.csv
├── tables/
│   ├── table_phase1_prediction.tex
│   ├── table_phase1_comparison.tex
│   └── table_architecture_selection.tex
└── plots/
    ├── fig_phase1_seed_deltas.pdf
    ├── fig_phase1_seed_deltas.png
    ├── fig_phase1_architecture_selection.pdf
    ├── fig_phase1_architecture_selection.png
    ├── fig_phase1_meta_vs_test_improvement.pdf
    └── fig_phase1_meta_vs_test_improvement.png
```

Then create:

```text
outputs/iclr/palf_phase1_differential_stacking.zip
```

Do not mix these outputs into the current manuscript freeze.

---

# 13. Validation gates

The run is valid only if every gate passes.

## Source gates
- corrected v2 source exists
- 50 splits per target per condition
- R0 and R3 train/test indices match split-by-split
- current frozen means reproduce approximately

## No-refit gates
- no base FC branch refit
- no SC refit
- no semantic-prior regeneration
- no primary checkpoint mutation

## Meta-feature gates
For every split:

```text
delta_oof  == R3.fp_oof       - R0.fp_oof
delta_test == R3.fp_test_pred - R0.fp_test_pred
```

within tight tolerance.

## Leakage gates
- no outer-test target enters meta-CV
- meta hyperparameters selected only on OOF training rows
- final meta-model fit only on outer-training OOF rows

## Statistics gates
- 5 fold metrics averaged within seed
- exactly 10 seed summaries per target
- Holm across the two primary target tests only

## Baseline-preserving gates
- Candidate B includes `a=0`
- candidate family includes exact corrected baseline
- no forced semantic contribution

---

# 14. Tests

Add:

```text
tests/test_palf_phase1_differential_stacking.py
```

At minimum test:

1. R0/R3 split alignment assertion.
2. Differential feature equals R3 FP minus R0 FP.
3. Candidate B with `a=0` exactly reproduces baseline.
4. Candidate C coefficients are nonnegative.
5. Candidate C intercept is unpenalized.
6. Meta-CV does not access outer-test target.
7. Hyperparameter selection uses only meta-CV predictions.
8. Architecture tie-break prefers simpler model.
9. No base-model fit function is called by the Phase-1 script.
10. Output has exactly 100 selected outer models total.
11. Seed aggregation yields 10 seeds per target.
12. Primary Holm family contains exactly two tests.
13. All saved outer-test predictions are finite.
14. Existing frozen source directories remain byte-identical if practical; otherwise at least verify no file mtime/content changes caused by this script.

Run targeted tests first, then the full suite.

Do not hide pre-existing unrelated failures.

---

# 15. Do NOT do these things in Phase 1

Do not yet:

- add learnable prior strength `rho`
- change `gamma`
- change top-K prior ROIs
- widen `lambda_L`
- widen SC Ridge grid
- rerun the 400 primary branch fits
- reconstruct primal FC coefficients
- run biomarker stability
- run perturbation biomarker tests
- edit the paper
- regenerate the semantic priors
- add R1/R2 predictions to the stack
- use outer-test performance to choose the candidate family

Those are possible Phase-2/Phase-3 changes and will depend on this result.

---

# 16. Final OpenCode report

At completion print exactly these sections.

## A. Source verification
- git HEAD
- corrected source path
- current reproduced WM/FI condition means
- number of aligned R0/R3 splits

## B. No-refit proof
- FC branch refits: 0
- SC branch refits: 0
- priors regenerated: 0
- frozen source files modified: 0

## C. Differential-stack results

For each target print:

```text
Same-solver baseline fused r
Anisotropy-only fused r
Network-only fused r
Full PALF fused r
Differential PALF stack fused r
Differential PALF stack RMSE
Differential PALF stack MAE
```

## D. Primary paired comparison

For each target print:

- new mean r
- baseline mean r
- delta
- median delta
- positive seeds / 10
- 95% CI
- raw Wilcoxon p
- Holm p
- Cohen dz

## E. Is it the new numerical best?

Explicitly state:

### Working Memory
- current best existing fused r
- new stack fused r
- difference

### Fluid Intelligence
- current best existing fused r
- new stack fused r
- difference

Do not use the phrase “state of the art”; just report within-study ranking.

## F. Architecture selection

For each target:

- baseline selected X/50
- residual selected X/50
- differential Ridge selected X/50
- semantic correction used X/50
- selected `a` distribution
- selected `lambda_stack` distribution
- `w_delta` mean/median where applicable

## G. Meta-CV reliability diagnostic
- Pearson correlation: meta-CV improvement vs outer-test improvement
- Spearman correlation
- plot path

## H. Tests
- targeted test result
- full-suite test result
- exact pre-existing failures if any

## I. Outputs
- directory
- ZIP
- report
- comparison CSV
- plots

End with exactly one of:

```text
STATUS: PHASE1_DIFFERENTIAL_STACKING_COMPLETE
```

or

```text
STATUS: PHASE1_NOT_VALID
```

If invalid, explain the failed gate and do not interpret performance.

---

# 17. Important interpretation rule

This phase is a **pre-registered feasibility check of one fixed meta-model family**.

Do not modify the candidate family after seeing outer-test results.

Our next decision will be based on:

1. whether the differential stack becomes the best fused predictor within the current study;
2. how consistently it improves across seeds;
3. whether the semantic correction is selected often enough to justify the PALF mechanism;
4. whether the result is strong enough to justify spending the remaining compute budget on adaptive prior-strength branch refits.

Run this phase first and stop after producing the requested output bundle and report.
