# PALF ICLR 2027 — Phase 2A.1: Corrected Matched-Selector Adaptive-Prior Pilot

## Why this corrective run is required

Do **not** launch the final 10-seed Adaptive-PALF experiment yet.

An independent audit of the Phase-2A output found that the apparent adaptive
gain is confounded by mismatched model-selection procedures and, in several
splits, different SC branches.

The current Phase-2A result bundle reports approximately:

```text
Working Memory:
baseline fused r = 0.293867
adaptive fused r = 0.307582
delta = +0.013715

Fluid Intelligence:
baseline fused r = 0.373398
adaptive fused r = 0.377044
delta = +0.003646
```

However, the exported CSVs reveal the following problems.

### Problem 1 — "rho=0, tau=0" is not actually identical to the baseline

`adaptive_selection_summary.csv` contains:

```text
32/40 splits with selected_rho=0 and selected_tau=0
```

not 29/40.

Of those 32 nominal no-semantic splits, only 15 reproduce the Model-A
baseline prediction exactly.

In many of the others, the adaptive path selected a different `lambda_F`,
different `alpha_SC`, or both.

Therefore the statement:

```text
rho=0, tau=0 -> identical to baseline
```

is not true for the executed Phase-2A comparison.

### Problem 2 — the SC branch is not shared

The Phase-2A prompt required the same expanded SC branch and selection
procedure for baseline and Adaptive PALF.

But `split_metrics.csv` contains splits in which:

```text
model_a_alpha_SC != model_b_alpha_SC
```

and the SC test predictions/metrics differ.

This means some fused improvements cannot be attributed to the adaptive
semantic FC regularizer.

### Problem 3 — baseline and adaptive FC selection are not matched

The adaptive path uses the one-standard-error semantic search, while the
baseline path can select a different no-prior `lambda_F` using a different
selection logic.

For example, several splits with adaptive `rho=0,tau=0` change
`lambda_F` from 0.01/0.001/1/100 to 0.1, producing substantial prediction
changes even though the semantic prior is completely off.

### Problem 4 — the apparent gain is not evidence that the prior helped

In the audited Phase-2A CSV:

```text
WM semantic-active splits: 5/20
FI semantic-active splits: 3/20
```

For FI, the mean fused delta over the three semantic-active splits is negative.

For WM, much of the overall mean gain also occurs on `rho=0,tau=0` splits.

Therefore Phase 2A is **interesting evidence that the robust selector/grid may
improve prediction**, but it is **not yet evidence that adaptive semantic
regularization caused the improvement**.

The goal of Phase 2A.1 is to remove these confounds.

---

# 1. Development data must remain unchanged

Use exactly the same development seeds:

```text
dev_seeds = [101, 202, 303, 404]
```

and five outer folds.

Do not use primary/final seeds 0–9.

Do not change the outer split assignments.

Do not add or remove a development seed after seeing results.

Total:

```text
2 targets × 4 seeds × 5 folds = 40 outer splits
```

---

# 2. Keep the Adaptive-PALF mathematical family unchanged

Do not expand the semantic model in this corrective run.

Use:

\[
D_{\rho}=(1-\rho)I+\rho D_p
\]

and

\[
\hat\beta =
\arg\min_{\beta}
\left\{
\|\widetilde y-\widetilde X_F\beta\|_2^2
+
c\lambda_F\beta^\top D_{\rho}\beta
+
c\tau_L\beta^\top L_p\beta
\right\}.
\]

Keep:

```text
rho_grid = [0.0, 0.25, 0.50, 0.75, 1.0]

tau_grid = [
    0.0,
    0.003,
    0.01,
    0.03,
    0.10,
    0.50,
    1.0,
    2.0,
    5.0
]

lambda_F_grid = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

c = 13340
gamma = 0.5
epsilon = 1e-3
top_k_prior_rois = 10
edge_lift = product
```

No new semantic parameter is allowed in this run.

---

# 3. Select the SC branch ONCE per outer split

This is mandatory.

For each:

```text
target / seed / outer_fold
```

perform SC Ridge hyperparameter selection exactly once using the outer-training
subjects.

Use:

```text
alpha_SC_grid = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    300.0,
    1000.0,
    3000.0
]
```

Use the current standard nested SC selection criterion:

1. maximize mean validation Pearson r;
2. RMSE tie-break;
3. MAE tie-break;
4. deterministic alpha tie-break.

Do NOT introduce a new SC one-SE rule in Phase 2A.1.

After selection:

- generate one SC OOF vector;
- fit one final SC model;
- generate one SC outer-test vector.

Then reuse these exact SC predictions for:

- matched no-prior baseline;
- Adaptive PALF;
- all shadow/counterfactual analyses.

Hard assertions:

```python
baseline_sc_oof is adaptive_sc_oof  # semantically same stored array
baseline_sc_test == adaptive_sc_test
baseline_alpha_SC == adaptive_alpha_SC
```

to numerical tolerance.

There must be no Model-A vs Model-B SC discrepancy.

---

# 4. Match the FC model-selection rule

Both the no-prior baseline and Adaptive PALF must use the **same one-SE
selection framework**.

## 4A. Matched robust no-prior baseline

Candidate family:

```text
rho = 0
tau = 0
lambda_F in lambda_F_grid
```

For each candidate save foldwise inner-CV Pearson values.

Selection:

1. find candidate with highest mean Pearson r;
2. compute `SE_best` from its foldwise Pearson values;
3. eligible if

```text
mean_r >= best_mean_r - SE_best
```

4. among eligible candidates use this exact deterministic preference:

```text
smallest abs(log10(lambda_F) - log10(0.1))
then smaller lambda_F
```

Call this:

```text
Matched robust no-prior
```

This is the only baseline for Phase 2A.1.

---

## 4B. Adaptive PALF

Candidate family:

```text
rho in rho_grid
tau in tau_grid
lambda_F in lambda_F_grid
```

Use the same one-SE procedure.

Among eligible candidates prefer:

```text
lower rho
then lower tau
then smallest abs(log10(lambda_F) - log10(0.1))
then smaller lambda_F
```

Call this:

```text
Adaptive PALF
```

Do not compare against the old mismatched Phase-2A Model-A result as the
primary pilot baseline.

It may be printed descriptively only.

---

# 5. Exact identity gate

This gate must pass before any prediction interpretation.

For every split, explicitly evaluate the no-prior configuration with:

```text
rho = 0
tau = 0
```

at a given `lambda_F`.

The adaptive implementation and baseline implementation must return identical:

- fitted generalized-FC predictions;
- OOF predictions;
- outer-test predictions;
- metrics

when `(rho,tau,lambda_F)` are identical.

Add a numerical unit test at multiple `lambda_F` values.

This must fail loudly if not true.

---

# 6. Fully cross-fitted fusion

For both baseline and Adaptive PALF:

- use the same already-selected SC OOF/test prediction;
- select convex FP+SC fusion weights using only the outer-training OOF
  predictions;
- use the same grid:

```text
w_FP = [0, 0.05, ..., 1]
w_SC = 1 - w_FP
```

Each model may select a different fusion weight because its FP prediction
differs.

Outer-test labels may not enter fusion selection.

Save:

```text
baseline_w_FP
baseline_w_SC
adaptive_w_FP
adaptive_w_SC
```

---

# 7. Add an isotropic shadow for semantic attribution

This is a crucial new diagnostic.

After Adaptive PALF selects:

```text
rho*
tau*
lambda_F*
```

fit a **shadow no-prior FP model** on the same training data with:

```text
rho = 0
tau = 0
lambda_F = lambda_F*
```

Do not reselect `lambda_F`.

This gives:

```text
adaptive FP at lambda_F*
vs
isotropic-shadow FP at the same lambda_F*
```

Therefore the direct difference is attributable to the semantic geometry
(rho/tau), not to a changed base penalty.

Save:

```text
shadow_fp_oof
shadow_fp_test
```

For semantic-effect diagnostics compute:

### Direct FP effect

```text
semantic_fp_delta =
adaptive_fp_pearson - shadow_fp_pearson
```

### Fixed-weight fused effect

Use the Adaptive-PALF fusion weights for both predictions:

```python
adaptive_fused_fixed =
adaptive_w_FP * adaptive_fp_test + adaptive_w_SC * sc_test

shadow_fused_fixed =
adaptive_w_FP * shadow_fp_test + adaptive_w_SC * sc_test
```

Then:

```text
semantic_fixed_fused_delta =
r(adaptive_fused_fixed) - r(shadow_fused_fixed)
```

This fixed-weight comparison isolates the semantic FP prediction change from
fusion-weight reselection.

If `rho*=0` and `tau*=0`, adaptive and shadow must be exactly identical and
both semantic deltas must be zero.

---

# 8. Preserve a separate prediction comparison

The **prediction** question is:

```text
Adaptive PALF
vs
Matched robust no-prior
```

This comparison permits the adaptive candidate family to select semantic
regularization when inner evidence supports it.

The **semantic-attribution** question is:

```text
Adaptive PALF
vs
isotropic shadow at the exact same lambda_F
```

Do not mix these two questions.

A prediction gain can exist because the larger adaptive model-selection family
changes which no-prior lambda is selected.

That is a valid property of the predictive procedure but not proof that the
semantic prior caused the gain.

The shadow analysis tells us whether the prior itself contributes.

---

# 9. Required outputs

Create a fresh isolated directory:

```text
outputs/iclr/palf_phase2a1_matched_adaptive_pilot/
```

Do not overwrite Phase 2A.

Create:

```text
split_metrics.csv
seed_metrics.csv
selection_summary.csv
semantic_shadow_metrics.csv
VALIDATION_REPORT.json
RUN_REPORT.md
COMPLETE
```

and:

```text
outputs/iclr/palf_phase2a1_matched_adaptive_pilot.zip
```

---

# 10. Required split-level columns

`split_metrics.csv` must include:

```text
task
seed
outer_fold
n_train
n_test

shared_alpha_SC
shared_sc_pearson
shared_sc_rmse
shared_sc_mae

baseline_lambda_F
baseline_fp_pearson
baseline_fp_rmse
baseline_fp_mae
baseline_w_FP
baseline_w_SC
baseline_fused_pearson
baseline_fused_rmse
baseline_fused_mae

adaptive_rho
adaptive_tau
adaptive_lambda_F
adaptive_fp_pearson
adaptive_fp_rmse
adaptive_fp_mae
adaptive_w_FP
adaptive_w_SC
adaptive_fused_pearson
adaptive_fused_rmse
adaptive_fused_mae

shadow_lambda_F
shadow_fp_pearson
shadow_fp_rmse
shadow_fp_mae

prediction_delta_fp
prediction_delta_fused

semantic_delta_fp
semantic_delta_fused_fixed_weight

adaptive_uses_semantic
```

Where:

```text
prediction_delta_fp =
adaptive FP - matched robust no-prior FP

prediction_delta_fused =
adaptive fused - matched robust no-prior fused

semantic_delta_fp =
adaptive FP - isotropic shadow FP

semantic_delta_fused_fixed_weight =
adaptive fixed-weight fusion - shadow fixed-weight fusion
```

---

# 11. Seed-level summary

Average the five folds within each seed.

There must be exactly four seed summaries per target.

For each target print:

### Matched robust baseline
- FP r
- fused r
- RMSE
- MAE

### Adaptive PALF
- FP r
- fused r
- RMSE
- MAE

### Prediction deltas
- FP delta
- fused delta
- positive seeds / 4

### Semantic shadow deltas
- semantic FP delta
- semantic fixed-fusion delta
- positive seeds / 4

Do not make final p-value claims from four development seeds.

You may print effect-size diagnostics, clearly labeled development-only.

---

# 12. Parameter-frequency audit

The previous OpenCode report said 29/40 selected `(rho=0,tau=0)`, while the
exported CSV contains 32/40.

Fix all counting logic.

Report directly from the canonical CSV:

For each target and total:

```text
rho=0,tau=0 count
rho>0 count
tau>0 count
any semantic count
rho distribution
tau distribution
lambda_F distribution
```

Also assert that the text report and CSV-derived counts are identical.

No manually typed count is allowed.

---

# 13. Existing Phase-2A forensic audit

Before running the corrected pilot, generate:

```text
PHASE2A_FORENSIC_AUDIT.json
```

from the old Phase-2A bundle.

Record:

- number of `(rho=0,tau=0)` rows;
- number of those exactly matching old Model-A baseline;
- number with different `lambda_F`;
- number with different `alpha_SC`;
- number with different SC test metrics;
- semantic-active split count;
- mean old Phase-2A fused delta on semantic-active splits;
- mean old Phase-2A fused delta on semantic-inactive splits.

This documents why Phase 2A should not be used as final evidence.

Expected approximate audit findings from the uploaded CSV:

```text
rho=0,tau=0: 32/40

WM:
semantic-active = 5/20
mean fused delta on semantic-active ≈ +0.0138
mean fused delta on semantic-inactive ≈ +0.0137

FI:
semantic-active = 3/20
mean fused delta on semantic-active ≈ -0.0073
mean fused delta on semantic-inactive ≈ +0.0056
```

Compute, do not hard-code.

---

# 14. Corrected development interpretation

Use these predefined labels.

## SEMANTIC_STRONG_GO

Only if:

### Prediction
For BOTH targets:

```text
mean adaptive fused - matched baseline fused >= +0.002
```

and at least 3/4 development seeds are positive,

AND

### Semantic attribution
For BOTH targets:

```text
mean semantic_delta_fp > 0
```

and at least 3/4 seed-level semantic FP deltas are nonnegative,

with semantic regularization selected in at least 25% of outer splits.

---

## WM_SEMANTIC_GO

If Working Memory satisfies:

```text
prediction fused delta >= +0.002
positive prediction seeds >= 3/4
mean semantic FP delta > 0
positive semantic FP seeds >= 3/4
```

while Fluid Intelligence is approximately neutral:

```text
adaptive fused delta >= -0.002
```

This outcome supports a target-adaptive method that uses semantic
regularization when useful and falls back to no prior for FI.

---

## PREDICTION_ONLY_GO

If Adaptive PALF improves prediction but:

- semantic shadow deltas are zero/nonpositive;
- or almost all splits choose no semantic regularization.

This means the gain comes from the robust selection procedure rather than the
semantic prior.

Do not present it as a semantic-prior success.

---

## NO_GO

If matched correction removes the predictive gain or semantic-active settings
are consistently harmful.

---

# 15. Plots

Generate:

```text
plots/fig_prediction_seed_deltas.pdf/png
plots/fig_semantic_shadow_deltas.pdf/png
plots/fig_rho_tau_counts.pdf/png
plots/fig_baseline_adaptive_shadow.pdf/png
```

### Prediction figure
Adaptive minus matched robust no-prior.

### Semantic figure
Adaptive minus isotropic shadow at the same selected lambda_F.

Do not mix these effects in one unlabeled delta.

---

# 16. Tests

Add/update tests so that they explicitly catch the Phase-2A failure mode.

Minimum tests:

1. SC is selected exactly once per split.
2. Baseline and adaptive share identical SC OOF/test arrays.
3. Baseline/adaptive `alpha_SC` is identical.
4. Baseline and adaptive use the same one-SE helper.
5. `(rho=0,tau=0,lambda=x)` produces bitwise/numerically identical FP
   predictions in both code paths.
6. Adaptive shadow uses exactly the adaptive selected `lambda_F`.
7. If adaptive rho=tau=0, adaptive == shadow.
8. Fixed-weight semantic fused comparison uses Adaptive fusion weights for both.
9. No outer-test labels enter selection.
10. Four dev seeds × five folds × two tasks are completed.
11. Reported semantic counts are generated from CSV, not hand-coded.
12. Old Phase-2A outputs remain unchanged.
13. Current final/frozen seeds 0–9 are not used.

Run targeted tests and the full suite.

---

# 17. Efficiency

Reuse as much of the Phase-2A implementation as possible, but do not reuse
incorrect split-level results as scientific evidence.

If saved candidate-level inner scores are sufficiently complete to reconstruct
the matched baseline and shared SC branch exactly without refitting, first
validate that reconstruction.

Otherwise rerun the 40 development splits.

Checkpoint every split.

This corrected pilot is more important than saving a few hours.

---

# 18. Final OpenCode report

Print:

## A. Phase-2A forensic audit
- zero-semantic count
- exact baseline matches
- mismatched lambda count
- mismatched SC count
- semantic-active count by target
- semantic-active vs inactive old deltas

## B. Corrected matched protocol
- dev seeds
- shared SC proof
- matched one-SE selector proof
- exact rho=tau=0 identity proof

## C. Corrected prediction results
For each target:

```text
Matched robust no-prior FP r
Adaptive FP r
FP delta

Matched robust no-prior fused r
Adaptive fused r
Fused delta

Positive fused seeds / 4
```

## D. Semantic attribution
For each target:

```text
Adaptive FP r
Isotropic-shadow FP r
semantic FP delta
positive semantic FP seeds / 4

Adaptive fixed-weight fused r
Shadow fixed-weight fused r
semantic fixed-fusion delta
```

## E. Semantic parameter use
Print directly from CSV:
- rho/tau counts
- any-semantic count
- parameter distributions

## F. Tests
- targeted
- full suite
- pre-existing failures

## G. Outputs
- directory
- ZIP
- forensic audit
- split CSV
- seed CSV
- plots

Then print exactly one:

```text
PHASE2A1_DECISION: SEMANTIC_STRONG_GO
```

```text
PHASE2A1_DECISION: WM_SEMANTIC_GO
```

```text
PHASE2A1_DECISION: PREDICTION_ONLY_GO
```

or

```text
PHASE2A1_DECISION: NO_GO
```

Finally end with:

```text
STATUS: PHASE2A1_MATCHED_ADAPTIVE_PILOT_COMPLETE
```

If any validation gate fails:

```text
STATUS: PHASE2A1_NOT_VALID
```

and do not interpret performance.

---

# 19. Do not proceed to the final paper yet

Stop after Phase 2A.1.

The next prompt will depend on whether the corrected gain is:

- genuinely semantic,
- prediction-only from robust selection,
- Working-Memory-specific,
- or absent.

Do not spend the final-seed compute budget until this distinction is resolved.
