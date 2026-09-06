# Task: Reconstruct and Export Frozen Final LF1 FC Coefficients

We need to add a **reproducibility-only coefficient export utility** to the current `metaSFC_extends` repository.

This is NOT a new experiment.

Do not tune anything.
Do not change any model.
Do not change any prior.
Do not change any split.
Do not perform inner-CV selection.
Do not alter the final LF1 results.
Do not overwrite any existing final output.

The purpose is solely to reconstruct the fitted matched-prior FC coefficient vectors that were used in the already-completed final LF1 10×5 experiment, because the original final runner saved predictions and selected hyperparameters but did not serialize the 6670-dimensional FP coefficient vectors.

---

## Existing final artifacts

For Working Memory:

```text
outputs/iclr/lf1_final_10x5/working_memory/all_split_results.pkl
```

For Fluid Intelligence:

```text
outputs/iclr/lf1_final_10x5/fluid_intelligence/all_split_results.pkl
```

Each pickle contains exactly 50 `OuterSplitResult` objects corresponding to:

```text
10 seeds × 5 outer folds
```

Each object contains:

```text
seed
outer_fold
train_idx
test_idx

level1_results["S"]
level1_results["F0"]
level1_results["FP"]

lf1_weights
lf1_test_pred

lf1_pearson
lf1_rmse
lf1_mae
```

The FP `Level1Result` contains the frozen selected hyperparameters through the existing `inner_metrics.best_params` / equivalent structure.

Observed example:

```text
lambda_fc
lambda_sc
lambda_l
gamma
lifting
```

The final LF1 weight dictionary contains:

```text
{"FP": ..., "S": ...}
```

The `Level1Result` itself does NOT serialize the fitted FC coefficient vector.

---

# Objective

Create:

```text
scripts_paper/export_frozen_fp_coefficients.py
```

This script must reconstruct only the final outer refit for the matched-prior FP branch and export its exact FC coefficient vector.

It must also reconstruct the SC branch needed to verify the final fused LF1 prediction.

The reconstruction must use the same data transforms, target normalization, prior construction, model solver, and prediction code as the final LF1 experiment.

Do not reimplement the mathematical model independently if existing repository functions/classes can be reused.

---

# 1. Inspect and reuse the actual final pipeline

Before writing code, inspect at minimum:

```text
scripts/114_run_lf1_final_10x5.py

src/metascfc/experiments/prior_aware_late_fusion.py

src/metascfc/models/iclr_backbones/fc_only_msancr.py
```

and every helper they call for:

```text
loading FC
loading SC
loading behavioral targets
feature extraction
standardization
target normalization
prior loading
prior lifting
Laplacian construction
FP fitting
SC Ridge fitting
outer refitting
inverse target transform
```

The exporter must use these exact helpers.

Do NOT duplicate preprocessing logic unless absolutely unavoidable.

---

# 2. No inner-CV rerun

The exporter must NOT repeat model selection.

For every saved outer split:

```python
result = saved_results[i]
```

read the already-selected FP hyperparameters from:

```python
result.level1_results["FP"]
```

Use the existing repository representation exactly.

For example, if the current object exposes:

```python
result.level1_results["FP"].inner_metrics["best_params"]
```

or:

```python
result.level1_results["FP"].inner_metrics.best_params
```

handle the actual structure present in the repository.

Likewise reuse the exact SC selected parameter already stored for that split.

No grid search may be invoked.

No inner folds may be generated.

---

# 3. Exact split reconstruction

For each saved `OuterSplitResult` use exactly:

```python
train_idx = result.train_idx
test_idx  = result.test_idx
seed      = result.seed
fold      = result.outer_fold
```

Do NOT regenerate KFold splits.

Verify:

```text
50 records per task
10 unique seeds
5 unique folds per seed
no duplicate (seed, fold)
train/test disjoint
train + test = 412 subjects
```

Where fold sizes differ by one subject, preserve them exactly.

---

# 4. Exact matched prior

Use only:

```text
working_memory_contrastive_qwen3
```

for Working Memory and:

```text
fluid_intelligence_contrastive_qwen3
```

for Fluid Intelligence.

Use the exact prior loading function/path from the final LF1 runner.

Do not regenerate the LLM prior.

Do not use:

```text
unrelated
shuffled
random
FIP
```

priors for coefficient export.

---

# 5. Reconstruct the FP outer refit

For each saved split:

1. load the exact FC feature matrix used by the final experiment;
2. use `train_idx` and `test_idx`;
3. reproduce the exact training-only FC scaler;
4. reproduce the exact target standardization used in the original outer refit;
5. load the matched prior;
6. use the exact stored values of:

```text
lambda_fc
lambda_l
gamma
lifting
```

and any other required fixed parameter;
7. fit the existing FC-only MS-A-NCR implementation on the full outer training set;
8. obtain:
   - the outer-test FP prediction;
   - the final primal FC coefficient vector.

The exported coefficient vector must correspond to the model that produced the prediction.

Expected coefficient dimensionality:

```text
6670
```

because:

```text
116 × 115 / 2 = 6670
```

If the existing solver internally solves a dual problem, retrieve/reconstruct the exact primal coefficient vector using the repository's existing method.

Do not approximate coefficients.

---

# 6. Reconstruct the SC branch

The saved LF1 output contains only the fused test prediction, not the standalone FP test prediction.

Therefore reconstruct the SC branch with its already-selected frozen hyperparameters using the exact final-pipeline helper.

No SC hyperparameter selection.

Obtain:

```text
pred_fp_reconstructed
pred_sc_reconstructed
```

in the original target units.

---

# 7. Reproduce the stored LF1 prediction

Use the saved final weights:

```python
w_fp = result.lf1_weights["FP"]
w_sc = result.lf1_weights["S"]
```

Compute:

```python
pred_lf1_reconstructed = (
    w_fp * pred_fp_reconstructed
    + w_sc * pred_sc_reconstructed
)
```

Compare against:

```python
result.lf1_test_pred
```

for the same split.

Required diagnostics:

```text
max_abs_prediction_error
mean_abs_prediction_error
RMSE_between_predictions
Pearson_between_predictions
```

Primary acceptance tolerance:

```text
max_abs_prediction_error <= 1e-6
```

If floating-point/backend differences make this impossible despite verified identical code paths, permit a secondary tolerance:

```text
max_abs_prediction_error <= 1e-5
```

but clearly mark which tolerance was needed.

A split must NOT be accepted if:

```text
max_abs_prediction_error > 1e-5
```

Do not silently save its coefficients.

---

# 8. Metric reproduction

For every reconstructed split also recompute:

```text
LF1 Pearson
LF1 RMSE
LF1 MAE
```

using the exact test target values.

Compare to:

```text
result.lf1_pearson
result.lf1_rmse
result.lf1_mae
```

Require numerical agreement within reasonable floating-point tolerance.

The script must fail if prediction-array agreement or metric agreement fails.

---

# 9. Coefficient scale must be documented

Determine whether the model's returned primal coefficient vector is defined in:

```text
standardized FC feature space
```

or:

```text
original FC feature space
```

Do not guess.

Read the implementation.

Store this explicitly in every export as:

```text
coefficient_space
```

For qualitative ranking, retain the exact coefficient representation used by the biomarker/evidence-audit code.

If the audit converted coefficients back to original feature scale before calculating alignment/rank metrics, perform the same transformation and save BOTH:

```text
coef_model_space
coef_biomarker_space
```

Otherwise save only the actual representation used by the final biomarker analysis and document it.

---

# 10. Output structure

Create a new directory only:

```text
outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/
```

Inside:

```text
working_memory/
fluid_intelligence/
```

For every split save:

```text
seed00_fold00.npz
seed00_fold01.npz
...
seed09_fold04.npz
```

Each NPZ must include at minimum:

```text
coef
seed
outer_fold
train_idx
test_idx

lambda_fc
lambda_l
gamma
lifting

w_fp
w_sc

pred_fp
pred_sc
pred_lf1_reconstructed
pred_lf1_stored

prediction_max_abs_error
prediction_mean_abs_error

coefficient_space
prior_name
```

If coefficient model-space and biomarker-space vectors differ, store:

```text
coef_model_space
coef_biomarker_space
```

instead of an ambiguous `coef`.

---

# 11. Aggregate export

For each task also write:

```text
all_coefficients.npz
```

containing coefficient arrays ordered by:

```text
seed ascending
outer_fold ascending
```

Expected:

```text
coefficients shape = (50, 6670)
seeds shape        = (50,)
folds shape        = (50,)
```

Also write:

```text
reconstruction_audit.csv
selected_hyperparameters.csv
```

`reconstruction_audit.csv` columns:

```text
task
seed
outer_fold

n_train
n_test

w_fp
w_sc

prediction_max_abs_error
prediction_mean_abs_error
prediction_rmse_difference
prediction_pearson

stored_lf1_pearson
reconstructed_lf1_pearson
pearson_abs_error

stored_lf1_rmse
reconstructed_lf1_rmse
rmse_abs_error

stored_lf1_mae
reconstructed_lf1_mae
mae_abs_error

coefficient_length
coefficient_finite
status
```

---

# 12. Aggregate coefficient summaries for visualization

After all 50 splits pass reconstruction:

compute three coefficient summaries for each task:

```text
mean_signed_coef.npy
mean_abs_coef.npy
median_abs_coef.npy
```

Important:

For ranking candidate biomarkers, the default qualitative visualization should use:

```text
mean absolute coefficient magnitude across 50 splits
```

unless the existing final biomarker code uses another explicitly defined aggregation.

Do NOT change the scientific definition merely for visualization.

Also calculate:

```text
edge_mean_abs.csv
```

with 6670 rows:

```text
edge_index
roi_i_0based
roi_j_0based
roi_i_1based
roi_j_1based
mean_signed_coef
mean_abs_coef
median_abs_coef
rank_mean_abs
```

If:

```text
inputs/atlases/AAL116_labels.csv
```

is available, join the actual ROI names and add:

```text
roi_i_name
roi_j_name
```

Do not invent names if the label file is unavailable.

---

# 13. Stability-aware qualitative edge list

A large coefficient in one split should not automatically become a qualitative example.

For every edge additionally calculate:

```text
top10_frequency
top20_frequency
top50_frequency
sign_consistency
```

Definitions:

```text
top10_frequency =
fraction of the 50 split coefficient vectors in which the edge
is among the 10 largest absolute coefficients

top20_frequency =
fraction in top 20

top50_frequency =
fraction in top 50

sign_consistency =
max(
    fraction coefficient > 0,
    fraction coefficient < 0
)
```

Write:

```text
stable_top_edges.csv
```

Sort primarily by:

```text
mean_abs_coef descending
```

but include all stability quantities.

Do not call these edges "validated biomarkers".

Recommended terminology:

```text
high-weight candidate FC edges
```

or:

```text
illustrative prior-aware FC coefficients
```

---

# 14. Strong safeguards

The utility must contain hard assertions:

```text
assert no inner CV function is called
assert no hyperparameter grid search is called
assert no new prior is generated
assert split identity is exact
assert coefficient length == 6670
assert all coefficients are finite
assert prediction reconstruction passes tolerance
assert all 50 splits pass before COMPLETE is written
```

If one split fails:

```text
STOP
```

and print:

```text
FROZEN_COEFFICIENT_RECONSTRUCTION_FAILED
```

Do not continue and do not write a success marker.

---

# 15. Frozen-source integrity

Before running reconstruction, hash:

```text
all_split_results.pkl
final config
matched prior files
packed FC input
packed SC input
target input
```

Store hashes in:

```text
source_integrity.json
```

The exporter may create new files only under:

```text
frozen_fp_coefficients/
```

It must not modify any existing final experiment artifact.

---

# 16. Tests

Add:

```text
tests/test_export_frozen_fp_coefficients.py
```

At minimum test:

1. split metadata comes from pickle, not regenerated;
2. no inner-CV/model-selection helper is invoked;
3. no prior generation occurs;
4. matched task prior is used;
5. coefficient length is 6670;
6. coefficients are finite;
7. LF1 reconstructed prediction uses stored FP/S fusion weights;
8. reconstruction tolerance is enforced;
9. failed reconstruction does not receive `status=PASS`;
10. aggregate arrays contain exactly 50 vectors when full task is complete;
11. existing final LF1 artifacts are unchanged.

---

# 17. CLI

The script should support:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task working_memory
```

and:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task fluid_intelligence
```

Also support a smoke check:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task working_memory \
  --seed 0 \
  --fold 0 \
  --smoke
```

Smoke mode may export to:

```text
outputs/iclr/lf1_final_10x5/frozen_fp_coefficients_smoke/
```

and must never contaminate the final export directory.

---

# 18. Recommended execution order

First:

```bash
pytest -q tests/test_export_frozen_fp_coefficients.py
```

Then:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task working_memory \
  --seed 0 \
  --fold 0 \
  --smoke
```

Inspect reproduction error.

Only if smoke PASS:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task working_memory
```

Then:

```bash
PYTHONPATH=src python scripts_paper/export_frozen_fp_coefficients.py \
  --task fluid_intelligence
```

Finally:

```bash
pytest -q
```

---

# 19. Final console report

Print:

```text
FROZEN FP COEFFICIENT EXPORT COMPLETE

Working Memory:
splits reconstructed = 50/50
max prediction reproduction error = ...
mean prediction reproduction error = ...
coefficients shape = (50, 6670)
all finite = YES

Fluid Intelligence:
splits reconstructed = 50/50
max prediction reproduction error = ...
mean prediction reproduction error = ...
coefficients shape = (50, 6670)
all finite = YES

No hyperparameter selection performed.
No prior regenerated.
No final artifact modified.
```

If only one task was requested, print only that task.

Do not claim success unless every requested split passes.

---

# 20. Create a plotting utility after successful export

Also create:

```text
scripts_paper/plot_frozen_top_edges.py
```

It must take:

```text
--edge-summary stable_top_edges.csv
--atlas inputs/atlases/AAL116.nii.gz
--top-k 20
--output ...
```

It must NOT select `top-k` by prediction performance.

Default:

```text
top-k = 20
```

is purely an illustrative display threshold.

The figure should use the mean absolute coefficient magnitude to determine edge thickness.

If sign consistency is shown, use line style or another non-misleading visual mechanism.

The generated caption text printed by the script should state:

> Highest-magnitude prior-aware FC coefficients aggregated across the 50 frozen outer fits. The visualization is descriptive and does not imply causal connectivity.

Generate one figure per task.

---

# FINAL RULE

This task is artifact reconstruction only.

The final LF1 predictions, statistics, priors, hyperparameters, fusion weights, and scientific conclusions are frozen.

Do not optimize anything based on the reconstructed coefficients.