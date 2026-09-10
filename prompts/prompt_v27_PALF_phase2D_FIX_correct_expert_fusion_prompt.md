# PALF ICLR 2027 — Phase 2D-FIX: Correct PS-NCR Expert Fusion Implementation

## Purpose

The Phase-2D `NO_GO` result must NOT be treated as valid scientific evidence.

A direct code/results audit found multiple implementation errors that violate
the Phase-2D protocol and materially affect prediction/fusion.

The correction must preserve the Phase-2D scientific hypothesis and grids.
Do not invent a new model in this run.

---

# 0. Mandatory forensic findings to reproduce

Before changing code, confirm these issues in the current Phase-2D implementation.

## F1. Wrong baseline branch

The current Phase-2D baseline helper fits ordinary:

```python
Ridge(...)
```

on FC and SC.

That is NOT the corrected paper baseline.

The correct baseline is:

```text
R0 generalized same-solver FP branch
+
SC Ridge
+
fully cross-fitted FP+SC convex fusion
```

using the existing implementation in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

Specifically, R0 must use:

```python
CONDITIONS["R0"]
generate_crossfit_oof(...)
oof.fp_oof
oof.sc_oof
search_fusion_weights(..., ["FP", "SC"])
reselect_and_fit_final(...)
final_fp.test_pred
final_sc.test_pred
```

Do NOT use `fc_oof` as the baseline FC-side fused branch.

---

## F2. Final baseline predictions are in the wrong calibration

The current helper:

```python
_fit_and_predict_ridge_on_subset(...)
```

fits:

```python
Ridge(fit_intercept=False)
```

on centered/scaled X but raw y.

This causes the current Phase-2D output to have absurd errors such as:

```text
WM baseline RMSE ~ 100+
FI baseline RMSE ~ 17+
```

instead of the corrected-paper scale near:

```text
WM RMSE ~ 11.3
FI RMSE ~ 4.57
```

The corrected R0 implementation already handles target centering/scaling and
final prediction units correctly.

Delete the custom Phase-2D baseline fitting path.

---

## F3. Inner-CV preprocessing leakage in expert selection

Current expert functions fit `StandardScaler` on the complete expert training
scope BEFORE dividing it into inner CV folds.

This occurs in:
- `fit_expert_ridge`
- `fit_expert_ncr`
- `_evaluate_mask_inner_cv`

For every inner validation fold, fit:
- X scaler on inner-train only;
- target mean/std on inner-train only.

Then transform/predict inner validation.

After hyperparameters are selected, fit a new scaler on the complete current
training scope for the final fit.

---

## F4. SC mask selection is not independent

Current `_select_best_mask(...)` evaluates candidate masks only using `X_fc`
and then reuses that mask for SC.

This violates the intended design, especially because FI is SC-dominant.

Correct behavior:

```text
FC expert mask -> selected using FC inner CV only
SC expert mask -> selected using SC inner CV only
```

The prior/ranking family is the same, but:
- mask family may differ;
- mask size may differ;
- selected lambda may differ.

---

## F5. Broken primal/final reconstruction validation

Current `validate_primal_reconstruction(...)` reconstructs:

```python
(1-alpha) * fc_expert_pred + alpha * expert_pred
```

and compares it to the FINAL baseline+expert prediction.

That is not the final model.

Correct validation must be split into:

### Expert validation
Reconstruct FC expert test prediction from its saved primal coefficients.

Reconstruct SC expert test prediction from its saved primal coefficients.

Fuse using saved expert FC/SC weights and verify:

```text
reconstructed_expert_fused == direct_expert_fused
```

### Final prediction validation
Use the stored/direct corrected baseline prediction:

```text
reconstructed_final =
    (1-alpha) * baseline_test
    + alpha * reconstructed_expert_fused
```

Then verify:

```text
reconstructed_final == direct_final
```

with:

```text
max_abs_error <= 1e-8
```

The previous Phase-2D `max_recon_error` values around tens/hundreds are a failed
validation, not a passing result.

---

## F6. Weak baseline audit test

The current audit accepts deviations up to:

```text
abs(delta r) < 0.01
```

which allowed:

```text
WM 0.257854 to PASS against 0.263515
FI 0.365037 to PASS against 0.370917
```

This is too loose for a correctness audit.

The corrected audit must use the actual R0 code path and reproduce:

```text
Working Memory fused r ≈ 0.263515
Fluid Intelligence fused r ≈ 0.370917
Working Memory RMSE ≈ 11.2929
Fluid Intelligence RMSE ≈ 4.5667
```

Use tolerances:

```text
Pearson mean absolute tolerance <= 5e-4
RMSE mean absolute tolerance <= 0.05
```

If not reproduced:

```text
STATUS: PHASE2D_FIX_BASELINE_AUDIT_FAILED
```

and stop.

---

# 1. Do not change the scientific Phase-2D model

Keep the same prior-selected expert hypothesis.

## Mask families

```text
direct_topk:
K = [100, 300, 600, 1200]

roi_incident:
M = [5, 10, 15]
```

## Ridge expert grid

```text
lambda_R = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]
```

## NCR ratio grid

```text
laplacian_ratio = [0.0, 0.1, 0.3, 1.0]
lambda_L = laplacian_ratio * lambda_R
```

## Prior types

```text
matched
cross_task
shuffled
random
```

Do not add new prior forms or tune new grids.

---

# 2. Correct baseline implementation

Create one helper that wraps the already validated R0 code.

For each:

```text
task / seed / outer fold
```

do:

```python
condition = CONDITIONS["R0"]
roi_placeholder = np.ones(116) / 116
```

Then:

```python
oof = generate_crossfit_oof(
    X_fc,
    X_sc,
    y,
    train_idx,
    condition,
    roi_placeholder,
    seed,
    outer_fold,
    ...
)
```

Use only:

```python
oof.fp_oof
oof.sc_oof
```

for the corrected no-prior fusion.

Select:

```python
base_weights, _ = search_fusion_weights(
    y[train_idx],
    {"FP": oof.fp_oof, "SC": oof.sc_oof},
    ["FP", "SC"],
)
```

For the final fit:

```python
_, final_sc, final_fp = reselect_and_fit_final(
    ...
    condition=CONDITIONS["R0"],
    ...
)
```

Then:

```python
baseline_test =
    base_weights["FP"] * final_fp.test_pred
    + base_weights["SC"] * final_sc.test_pred
```

This baseline computation must happen ONCE per task/seed/fold and be reused
unchanged across all four prior controls.

Do not recompute a slightly different baseline per prior type.

---

# 3. Correct expert inner-CV fitting

Implement a fold-local expert fitting primitive:

```python
fit_expert_candidate_on_split(
    X,
    y,
    mask,
    inner_train_idx,
    inner_val_idx,
    lambda_R,
    laplacian_ratio,
    ...
)
```

Inside it:

```python
scaler = StandardScaler().fit(X[inner_train_idx][:, mask])

X_train_z = scaler.transform(...)
X_val_z = scaler.transform(...)

y_mean = y[inner_train_idx].mean()
y_std = y[inner_train_idx].std()

y_train_z = (y_train - y_mean) / y_std
```

Predict validation and convert back:

```python
pred = pred_z * y_std + y_mean
```

No statistic may be fit using inner-validation rows.

Use this primitive for:
- mask selection;
- Ridge lambda selection;
- NCR lambda/ratio selection.

---

# 4. Correct FC and SC mask selection

For each fusion-analysis scope and modality independently:

## Step A — choose the prior-selected subspace using restricted Ridge

Evaluate every:

```text
direct_topk K
roi_incident M
```

with the Ridge lambda grid using the corrected fold-local inner CV.

Choose by:

1. highest pooled/mean Pearson;
2. RMSE;
3. MAE;
4. smaller selected edge count;
5. deterministic family/size tie-break.

This produces:

```text
fc_mask
sc_mask
```

separately.

## Step B — fit Ridge expert on the selected modality-specific mask

Tune `lambda_R` on the fixed selected mask.

## Step C — fit NCR expert on the SAME modality-specific mask

Tune:

```text
lambda_R
laplacian_ratio
```

on the same selected mask.

This gives a fair NCR-vs-Ridge comparison without allowing NCR to win by
changing feature cardinality.

---

# 5. Correct OOF expert generation

Use the exact same fusion folds as the corrected R0 baseline whenever possible.

For each outer-training split:

1. obtain `make_fusion_folds(...)` from the existing PALF code;
2. for each analysis/validation fusion fold:
   - independently select FC mask on analysis only;
   - independently select SC mask on analysis only;
   - fit FC/SC Ridge experts on analysis;
   - fit FC/SC NCR experts on analysis;
   - predict validation;
3. concatenate OOF predictions.

All mask, scaler, target scaling, lambda and NCR ratio decisions occur without
the held-out fusion fold.

---

# 6. Hierarchical fusion remains unchanged conceptually

Use corrected raw-unit predictions.

## Stage A

For Ridge and NCR separately:

```text
FC expert + SC expert
```

using the same two-branch convex weight grid:

```text
0, 0.05, ..., 1
```

Use the validated:

```python
search_fusion_weights(...)
```

or an equivalent implementation with the same:
- Pearson;
- RMSE tie-break;
- MAE tie-break.

## Stage B

Fuse:

```text
corrected R0 baseline
+
prior expert
```

with:

```text
alpha in [0, 0.05, ..., 1]
```

Again:
- Pearson;
- RMSE;
- MAE.

`alpha=0` must reproduce the corrected baseline exactly.

---

# 7. New development seeds for the corrected decision

The old seeds:

```text
1313, 1414, 1515, 1616
```

have now been inspected under the buggy implementation.

Do NOT use them for the corrected go/no-go decision.

Use exactly:

```text
FIX_DEV_SEEDS = [1717, 1818, 1919, 2020]
```

with five outer folds.

Use seeds 0–9 ONLY for the correctness audit.

Do not change these seeds after results are seen.

---

# 8. Forensic re-evaluation on old Phase-2D seeds

After the corrected fresh-seed pilot is complete, optionally run the corrected
implementation on:

```text
1313, 1414, 1515, 1616
```

for a forensic comparison only.

Save separately:

```text
FORENSIC_OLD_SEEDS/
```

These rows must NOT enter the Phase-2D-FIX decision.

Report how much the corrected baseline/expert results changed relative to the
buggy Phase-2D outputs.

---

# 9. Required scale/calibration gates

For every split assert:

```text
all predictions finite
```

and save:
- y test mean/std;
- baseline prediction mean/std;
- FC expert prediction mean/std;
- SC expert prediction mean/std;
- final prediction mean/std.

Add a hard warning/failure if:

```text
baseline RMSE > 3 * y_train_std
```

or if:

```text
abs(mean(baseline_test) - mean(y_train)) > 2 * y_train_std
```

unless a documented reason exists.

The previous RMSE ~100 WM result must never pass silently again.

---

# 10. Correct coefficient/reconstruction gates

For every final expert fit save:

```text
mask indices
beta standardized
beta original
intercept/or target mean handling
scaler mean
scaler scale
y mean
y std
```

Validate each branch independently.

Then validate expert fusion.

Then validate final baseline+expert fusion using the direct baseline prediction.

Hard gate:

```text
max branch reconstruction error <= 1e-8
max expert-fusion reconstruction error <= 1e-8
max final reconstruction error <= 1e-8
```

Any failure => invalid split.

---

# 11. Report NCR correctly

The previous final report printed `NaN` for NCR-vs-Ridge although
`diagnostics.json` contained a finite comparison.

Create canonical split-level values:

```text
ncr_final_pearson
ridge_final_pearson
ncr_minus_ridge
```

Aggregate:

- fold -> seed;
- then four development seeds.

Report:

```text
mean NCR-Ridge
median NCR-Ridge
positive seeds / 4
```

Also report selected NCR ratio frequencies separately for FC and SC.

No NaN is acceptable unless a split is explicitly invalid.

---

# 12. Required outputs

Create a NEW directory:

```text
outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion/
```

Do not modify the buggy Phase-2D output.

Required:

```text
COMPLETE
VALIDATION_REPORT.json
BASELINE_AUDIT.json
BUG_FORENSIC_REPORT.md
RUN_REPORT.md

split_metrics.csv
seed_metrics.csv
expert_selection.csv
prior_control_summary.csv

coefficients/
predictions/
plots/
```

ZIP:

```text
outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion.zip
```

---

# 13. Tests that must replace the weak structural tests

Add/update:

```text
tests/test_prior_subspace_expert_fusion_fix.py
```

Minimum tests:

1. baseline OOF uses `fp_oof`, not `fc_oof`.
2. baseline condition is exactly `CONDITIONS["R0"]`.
3. final baseline uses `final_fp.test_pred + final_sc.test_pred`.
4. no custom plain-FC Ridge baseline helper is used.
5. frozen audit WM Pearson within 5e-4 of 0.263515.
6. frozen audit FI Pearson within 5e-4 of 0.370917.
7. frozen audit WM RMSE within 0.05 of 11.2929.
8. frozen audit FI RMSE within 0.05 of 4.5667.
9. inner expert scaler is fit only on inner-train indices.
10. inner target mean/std use inner-train indices only.
11. FC mask selection uses FC data only.
12. SC mask selection uses SC data only.
13. FC and SC masks are allowed to differ.
14. NCR and Ridge use the same selected mask within a modality.
15. NCR ratio=0 equals Ridge for same lambda/mask.
16. alpha=0 reproduces corrected R0 test prediction exactly.
17. baseline computed once per split and reused across priors.
18. expert primal reconstruction <= 1e-8.
19. final reconstruction includes the actual corrected baseline prediction.
20. old Phase-2D output directory is unchanged.
21. fresh fix seeds are exactly 1717/1818/1919/2020.
22. no previous development seeds enter the decision CSV.

Run targeted tests, then full suite.

---

# 14. Prediction comparison and controls

For each target report:

```text
Corrected R0 baseline
Matched prior Ridge-expert fusion
Matched PS-NCR-EF
Cross-task PS-NCR-EF
Shuffled PS-NCR-EF
Random PS-NCR-EF
```

Report:
- Pearson;
- RMSE;
- MAE.

For the proposed model compute seed-level:

```text
matched PS-NCR-EF - corrected baseline
```

For prior specificity:

```text
matched - cross-task
matched - shuffled
matched - random
```

For NCR:

```text
matched NCR expert fusion - matched Ridge expert fusion
```

---

# 15. Predefined corrected decision rule

## STRONG_GO

```text
PHASE2D_FIX_DECISION: STRONG_GO
```

if both tasks have:

```text
mean matched PS-NCR-EF delta >= +0.003
positive seeds >= 3/4
```

and at least one has:

```text
delta >= +0.005
```

and matched beats shuffled/random on both tasks.

## PROMISING_GO

```text
PHASE2D_FIX_DECISION: PROMISING_GO
```

if one task has:

```text
delta >= +0.004
positive seeds >= 3/4
```

and the other is nonnegative, with matched specificity in the positive task.

## RIDGE_EXPERT_GO

```text
PHASE2D_FIX_DECISION: RIDGE_EXPERT_GO
```

if restricted-Ridge expert fusion succeeds but NCR does not add benefit.

## NO_GO

```text
PHASE2D_FIX_DECISION: NO_GO
```

only after all correctness gates pass and the corrected results fail the above
criteria.

---

# 16. Final OpenCode report

Print:

## A. Bug forensic audit
Confirm F1–F6 with exact old code locations and old output symptoms.

## B. Correct baseline audit
- WM r / RMSE
- FI r / RMSE
- tolerances
- PASS/FAIL

## C. Corrected implementation
- baseline uses R0 FP+SC
- fold-local expert preprocessing
- separate FC/SC mask selection
- coefficient reconstruction status

## D. Fresh development setup
- seeds 1717/1818/1919/2020
- folds
- runtime

## E. Corrected prediction results
For both tasks:
- R0 baseline
- matched Ridge expert
- matched NCR expert
- three prior controls

## F. Seed deltas
- four matched-vs-baseline seed deltas
- mean
- median
- positive count

## G. NCR contribution
- four NCR-vs-Ridge seed deltas
- mean
- selected FC/SC NCR ratio frequencies

## H. Prior specificity
- matched-cross
- matched-shuffled
- matched-random

## I. Fusion mechanism
- expert FC/SC weight
- final alpha
- alpha=0 fraction
- base/expert prediction correlation
- base/expert error correlation

## J. Calibration and reconstruction
- prediction means/stds
- RMSE sanity
- max branch/expert/final reconstruction error

## K. Tests
- targeted
- full suite
- pre-existing failures

## L. Outputs
- directory
- ZIP
- reports/CSVs/plots/coefficients

Then exactly one:

```text
PHASE2D_FIX_DECISION: STRONG_GO
```

```text
PHASE2D_FIX_DECISION: PROMISING_GO
```

```text
PHASE2D_FIX_DECISION: RIDGE_EXPERT_GO
```

or

```text
PHASE2D_FIX_DECISION: NO_GO
```

Finally:

```text
STATUS: PHASE2D_FIX_COMPLETE
```

If ANY validity gate fails:

```text
STATUS: PHASE2D_FIX_NOT_VALID
```

and do not interpret performance.

---

# 17. Stop after the corrected rerun

Do not introduce another architecture until Phase-2D-FIX is complete.

The current Phase-2D NO_GO must be marked:

```text
INVALID_FOR_SCIENTIFIC_INTERPRETATION
```

because the baseline, nested preprocessing, SC mask selection, and coefficient
validation did not match the requested protocol.
