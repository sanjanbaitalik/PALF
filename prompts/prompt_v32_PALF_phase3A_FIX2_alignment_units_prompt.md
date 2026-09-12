# PALF ICLR 2027 — Phase 3A-FIX2
# Correct OOF Label Alignment + Residual Units in PG-MT-BCR; Keep 98 Holdout Sealed

## Executive status

The Phase-3A-FIX result must be marked:

```text
INVALID_FOR_PHASE3A_DECISION
```

The corrected R0 implementation and the 98-subject holdout seal were valid,
but a direct code audit found TWO additional selection bugs that invalidate all
B0/B1/C0/C1/C2/C3/C4 development comparisons.

This run is a correction of the SAME PG-MT-BCR architecture and SAME
predeclared gates. It is not a new architecture search.

The 98-subject holdout must remain sealed.

If Phase-3A-FIX2 is valid and fails the original gates, stop model development
on the 412-subject cohort.

---

# 0. Preserve the holdout seal

Required:

```text
holdout count  = 98
holdout unique = 98
holdout SHA256 =
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425

development count = 412
development ∩ holdout = empty
```

Do NOT load:
- holdout FC,
- holdout SC,
- holdout labels,
- holdout target distributions,
- holdout R0 predictions,
- any holdout metric.

Continue using the access logger.

Create a new directory only:

```text
outputs/iclr/palf_phase3a_fix2_pg_mt_bcr/
```

Do not modify old Phase-3A or Phase-3A-FIX outputs.

---

# 1. Mandatory forensic audit of Phase-3A-FIX

Create:

```text
OLD_PHASE3A_FIX_FORENSIC_AUDIT.md
```

Confirm the following two critical bugs with exact file/line references.

## F11 — inner OOF targets were misaligned with validation subjects

In the Phase-3A-FIX runner, each inner record stored:

```python
"yB_WM": y_wm[B],
"yB_FI": y_fi[B]
```

but later assembled validation targets by:

```python
for g, v in zip(st["C"], st["yB_WM"]):
    y_oof_WM[pos[int(g)]] = v

for g, v in zip(st["C"], st["yB_FI"]):
    y_oof_FI[pos[int(g)]] = v
```

This pairs validation subject indices `C` with unrelated training labels `B`.

Consequences:
- candidate Pearson/RMSE/MAE were computed against wrong labels;
- eligibility/non-inferiority was computed against wrong labels;
- alpha selection was invalid;
- hyperparameter selection was invalid;
- all B0/B1/C0/C1/C2/C3/C4 outer results were generated from invalid selected
  candidates.

Correct rule:

```python
"yC_WM": y_wm[C]
"yC_FI": y_fi[C]
```

and concatenate those values.

After assembly hard assert:

```python
np.array_equal(y_oof_WM, y_wm[train_idx])
np.array_equal(y_oof_FI, y_fi[train_idx])
```

up to exact index/order semantics. If floating copies require tolerance, use
`np.allclose(..., atol=0, rtol=0)`.

This assertion must execute in every outer split before model selection.

---

## F12 — inner OOF BCR residual predictions remained standardized

`predict_bcr()` explicitly returns:

```text
standardized residual predictions
```

The old `_oof_residuals()` wrote those z-scale predictions directly into the
concatenated residual array and candidate selection then evaluated:

```python
base_oof + alpha * standardized_residual
```

But the final outer model correctly evaluated:

```python
base_test + alpha * (
    standardized_residual * training_residual_sd
    + training_residual_mean
)
```

Thus alpha/hyperparameters were selected in one unit system and applied in
another.

This is a material scale mismatch and explains the frequent selection of
`alpha=1` followed by large outer-test RMSE inflation.

Correct rule inside EACH inner fold:

```python
rWM_z, rFI_z = predict_bcr(...)

rWM_raw = rWM_z * st["sd_WM"] + st["mu_WM"]
rFI_raw = rFI_z * st["sd_FI"] + st["mu_FI"]
```

Store only `r*_raw` in the candidate OOF residual arrays.

Then candidate final OOF prediction is:

```python
base_oof_WM + alpha_WM * residual_oof_WM_raw
base_oof_FI + alpha_FI * residual_oof_FI_raw
```

The inner and final outer formulas must therefore be dimensionally identical.

---

# 2. Phase-3A-FIX results to invalidate

Do not interpret or reuse as evidence:

```text
A0  vs B0/B1/C0/C1/C2/C3/C4 comparisons from Phase-3A-FIX
Gate 1–3 outcomes
C1-C0 apparent prior gain
selected alpha frequencies
selected lambda frequencies
C0/C1 biomarker stability
C0/C1 faithfulness
```

Why:
all BCR model selection was based on the F11/F12-invalid OOF criterion.

The corrected R0 audit itself remains valid.

The holdout seal remains valid.

---

# 3. Correct R0 baseline — unchanged

Continue to use the validated:

```text
CONDITIONS["R0"]
generate_crossfit_oof
oof.fp_oof
oof.sc_oof
search_fusion_weights
reselect_and_fit_final
```

from:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

The strict audit must again reproduce:

```text
WM r    = 0.263515 ± 5e-4
WM RMSE = 11.292921 ± 0.05

FI r    = 0.370917 ± 5e-4
FI RMSE = 4.566689 ± 0.05
```

If not:

```text
STATUS: PHASE3A_FIX2_BASELINE_AUDIT_FAILED
```

and stop.

---

# 4. Scientific architecture — unchanged

Do NOT alter the PG-MT-BCR model.

Multi-task model per modality:

```text
2 shared spatial factors
1 WM-specific factor
1 FI-specific factor
```

Single-task model:

```text
3 task-specific factors per task/modality
```

Keep:
- gamma = 0.5
- epsilon = 1e-3
- same frozen ROI priors
- same shuffled/random controls
- same Adam optimizer
- same factor projection
- same ranks
- same amplitude penalty
- same prior penalty

No architecture modifications are allowed.

---

# 5. Frozen optimizer — unchanged

Use:

```text
Adam
float64
lr = 0.01
max_steps = 1500
min_steps = 200
patience = 75
relative training-loss tolerance = 1e-7
gradient clip = 5
2 deterministic restarts
```

Restart chosen by final TRAINING penalized objective only.

---

# 6. Hyperparameter grids — unchanged

No-prior:

```text
lambda_amp = [0.1, 1.0, 10.0]
lambda_prior = 0
```

Prior-aware:

```text
lambda_amp   = [0.1, 1.0, 10.0]
lambda_prior = [0.01, 0.1, 1.0]
```

Alpha:

```text
[0.0, 0.25, 0.50, 0.75, 1.0]
```

MT evaluates all 25 predefined `(alpha_WM, alpha_FI)` combinations.

No additional hyperparameters.

---

# 7. Correct inner record construction

For every outer split T/V and inner fold B/C construct:

```python
cf = {
    task: r0.crossfit_r0_within(task, B, ...)
    for task in ["WM","FI"]
}

base = {
    task: r0.fit_r0_predict(task, B, C, ...)
    for task in ["WM","FI"]
}
```

Training residual targets:

```python
res_WM_B = y_wm[B] - cf["WM"]
res_FI_B = y_fi[B] - cf["FI"]
```

Store:

```python
{
    "B": B,
    "C": C,

    "base_WM_C": base["WM"],
    "base_FI_C": base["FI"],

    "yC_WM": y_wm[C],
    "yC_FI": y_fi[C],

    "res_WM_B": res_WM_B,
    "res_FI_B": res_FI_B,

    "mu_WM": res_WM_B.mean(),
    "sd_WM": res_WM_B.std(),
    "mu_FI": res_FI_B.mean(),
    "sd_FI": res_FI_B.std(),

    "z_WM_B": (res_WM_B-mu_WM)/sd_WM,
    "z_FI_B": (res_FI_B-mu_FI)/sd_FI,

    ... fold-local standardized FC/SC matrices ...
}
```

Do NOT store `yB_*` for use as validation truth.

---

# 8. Correct concatenated inner OOF truth/base arrays

Initialize arrays aligned exactly with `train_idx`.

For every validation fold C:

```python
y_oof_WM[pos[C]]    = y_wm[C]
y_oof_FI[pos[C]]    = y_fi[C]

base_oof_WM[pos[C]] = base_WM_C
base_oof_FI[pos[C]] = base_FI_C
```

After all 3 inner folds, mandatory hard gates:

```python
assert np.all(np.isfinite(y_oof_WM))
assert np.all(np.isfinite(y_oof_FI))
assert np.all(np.isfinite(base_oof_WM))
assert np.all(np.isfinite(base_oof_FI))

assert np.array_equal(y_oof_WM, y_wm[train_idx])
assert np.array_equal(y_oof_FI, y_fi[train_idx])
```

Also save for diagnostics:

```text
inner_base_r_WM
inner_base_r_FI
```

computed from these correctly aligned arrays.

Do not set any fixed minimum correlation threshold; just record them.

---

# 9. Correct raw-unit BCR OOF residual predictions

Rename the helper:

```text
_oof_residuals_raw(...)
```

For each inner fold:

1. train BCR on standardized residual target `z_*_B`;
2. predict C, obtaining z-scale residual prediction;
3. de-standardize using B residual statistics:

```python
pred_WM_raw = pred_WM_z * sd_WM + mu_WM
pred_FI_raw = pred_FI_z * sd_FI + mu_FI
```

4. store the raw residual prediction aligned to subjects C.

The returned arrays MUST be in original cognitive-target units.

Do not return standardized residual predictions to selection.

Add explicit metadata:

```text
residual_prediction_units = "raw_target_units"
```

to selection output.

---

# 10. Dimensional-consistency hard test in the running pipeline

For every inner fold and task, choose a deterministic tiny sample of
validation subjects and verify:

```python
candidate_final_raw
==
base_raw + alpha * (
    pred_z * training_residual_sd + training_residual_mean
)
```

for alpha in:

```text
0.0
0.5
1.0
```

to <= 1e-12.

This test must run against the real helper path, not merely a standalone
algebra toy.

---

# 11. Candidate selection — same scientific rules

ST candidate:

```text
final_oof =
base_oof_raw + alpha * residual_oof_raw
```

Tie-break:
1. Pearson
2. RMSE
3. MAE
4. smaller alpha
5. lower lambda_prior
6. larger lambda_amp
7. deterministic tuple

Prior-aware ST candidate eligibility:

```text
delta_r >= -0.002
```

relative to aligned R0 OOF.

MT:

```text
final_WM =
base_oof_WM_raw + alpha_WM * residual_oof_WM_raw

final_FI =
base_oof_FI_raw + alpha_FI * residual_oof_FI_raw
```

Eligibility:

```text
delta_r_WM >= -0.002
delta_r_FI >= -0.002
```

Then:
1. mean Fisher-z
2. normalized RMSE
3. normalized MAE
4. smaller alpha sum
5. lower lambda_prior
6. larger lambda_amp
7. deterministic tuple

If no eligible candidate:

```text
alpha_WM=0
alpha_FI=0
```

and model prediction must exactly equal R0.

---

# 12. Final outer refit — retain already-correct unit conversion

For complete outer training T:

```python
cf_T_WM = crossfit_r0_within("WM", T, ...)
cf_T_FI = crossfit_r0_within("FI", T, ...)

resT_WM = y_wm[T] - cf_T_WM
resT_FI = y_fi[T] - cf_T_FI
```

Fit BCR to:

```text
zT = (resT - mean(resT)) / sd(resT)
```

Predict V in z units and convert:

```python
res_test_WM_raw = pred_z_WM * sd(resT_WM) + mean(resT_WM)
res_test_FI_raw = pred_z_FI * sd(resT_FI) + mean(resT_FI)
```

Final:

```python
pred_WM =
R0_test_WM + alpha_WM * res_test_WM_raw

pred_FI =
R0_test_FI + alpha_FI * res_test_FI_raw
```

This is the SAME formula used in inner selection after F12 is fixed.

---

# 13. Fresh development seeds

The Phase-3A-FIX seeds have now been inspected under an invalid selector:

```text
3939
4040
4141
4242
```

Do not use them for the corrected decision.

Use exactly:

```text
PHASE3A_FIX2_DEV_SEEDS = [4343, 4444, 4545, 4646]
```

with:
- 5 outer folds
- 3 inner folds

Historical seeds 0–9 remain baseline-audit only.

Do not alter seeds after results.

The 98 holdout remains sealed.

---

# 14. Models — unchanged

Report exactly:

```text
A0  corrected R0

B0  ST-BCR-NP
B1  ST-PG-BCR

C0  MT-BCR-NP
C1  PG-MT-BCR matched

C2  PG-MT-BCR cross-task
C3  PG-MT-BCR shuffled
C4  PG-MT-BCR random
```

No Phase 3C.

No new architecture.

---

# 15. Selection diagnostics — mandatory additions

`selection_details.csv` must now include:

```text
seed
fold
model
lambda_amp
lambda_prior
alpha_WM
alpha_FI

inner_base_r_WM
inner_base_r_FI

candidate_r_WM
candidate_r_FI
candidate_delta_WM
candidate_delta_FI

candidate_rmse_WM
candidate_rmse_FI

candidate_residual_sd_WM_raw
candidate_residual_sd_FI_raw

residual_prediction_units

eligible
selected
```

For ST use task-relevant columns.

At completion assert:

```text
residual_prediction_units == "raw_target_units"
```

for every row.

---

# 16. New tests specifically targeting F11/F12

Add:

```text
tests/test_phase3a_fix2_alignment_units.py
```

Required tests:

## T1 — validation label alignment
Construct deterministic y values equal to subject IDs.

Run the real inner-record/OOF assembly helper.

Assert every position gets the label from its own C subject.

## T2 — complete truth reconstruction
For a real/synthetic T split:

```python
y_oof == y[T]
```

exactly after 3-fold assembly.

## T3 — prevent B-label substitution
Monkeypatch `y[B]` and `y[C]` to non-overlapping numeric ranges.

Assert assembled C truth comes only from `y[C]`.

## T4 — predict_bcr contract
Assert/document:

```text
predict_bcr returns standardized-residual units
```

## T5 — real helper de-standardizes before concatenation
Monkeypatch `predict_bcr` to return all ones.

Set fold residual:

```text
mu = 7
sd = 3
```

The real OOF helper must return exactly:

```text
10
```

for that fold, not `1`.

## T6 — fold-specific de-standardization
Give each inner fold a different `(mu,sd)`.

Assert each C block receives its own fold's raw-unit transform.

## T7 — candidate final prediction units
Using the real selection-input builder, verify:

```text
base + alpha * raw_residual
```

and reject standardized residual input.

## T8 — inner/final formula parity
For fixed synthetic model output and same `(mu,sd)`, verify the inner and final
prediction helper return numerically identical formula results.

## T9 — aligned R0 score
Assemble deterministic validation predictions equal to validation targets.
The inner R0 Pearson must equal 1.0.

This would have failed under F11.

## T10 — alpha scaling
For a raw residual of magnitude 10 and alpha .5, correction must be magnitude 5,
not .5.

---

# 17. Retain all prior Phase-3A-FIX functional tests

Keep the previous 40 tests where still relevant, but update any test that
encoded the invalid F11/F12 behavior.

In particular add a test that actually invokes the real outer-inner assembly
and proves:

```text
corr(y_oof, base_oof)
```

uses the correct y values.

Do not use source-string-only tests as the sole validation for F11/F12.

Run targeted tests and full suite.

---

# 18. Coefficients/biomarkers

Recompute from scratch because previous selected models were invalid.

Do not reuse old C0/C1 coefficients or biomarker statistics.

Keep:
- coefficient reconstruction <= 1e-8;
- shared/task-specific maps;
- FC/SC maps;
- alpha-weighted maps;
- edge/ROI stability;
- faithfulness diagnostics.

Biomarker diagnostics remain downstream only.

---

# 19. Original scientific gates remain unchanged

Do not weaken them.

## Gate 1

Both tasks:

```text
mean(C1-A0) >= +0.005
positive seeds >= 3/4
```

and at least one task:

```text
mean(C1-A0) >= +0.010
```

## Gate 2

Both tasks:

```text
mean(C1-C0) >= +0.003
positive seeds >= 3/4
```

## Gate 3

Both tasks:

```text
mean(C1-C3) > 0
mean(C1-C4) > 0
```

and matched must not be worse than cross-task on BOTH tasks.

## Gate 4

- finite;
- >=95% final-fit optimizer validity;
- no catastrophic RMSE > 2× R0.

All pass:

```text
PHASE3A_FIX2_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

Otherwise:

```text
PHASE3A_FIX2_DECISION: DO_NOT_TOUCH_HOLDOUT
```

---

# 20. Output directory

Create:

```text
outputs/iclr/palf_phase3a_fix2_pg_mt_bcr/
```

with:

```text
COMPLETE
OLD_PHASE3A_FIX_FORENSIC_AUDIT.md
BASELINE_AUDIT.json
HOLDOUT_SEAL_REPORT.json
VALIDATION_REPORT.json
RUN_REPORT.md
PHASE3A_DATA_ACCESS_LOG.jsonl

split_metrics.csv
seed_metrics.csv
model_summary.csv
selection_details.csv
optimizer_diagnostics.csv
units_alignment_audit.json

coefficients/
predictions/
biomarkers/
prior_controls/
plots/
```

If all gates pass:

```text
MODEL_FROZEN.json
MODEL_FROZEN.sha256
READY_FOR_PHASE3B
```

Otherwise:

```text
HOLDOUT_REMAINS_LOCKED
```

ZIP:

```text
outputs/iclr/palf_phase3a_fix2_pg_mt_bcr.zip
```

---

# 21. Final report

Print:

## A. Holdout seal
- count / SHA / intersection
- access-log result
- explicit confirmation holdout untouched

## B. Forensic audit
- F11 exact label-alignment bug
- F12 standardized/raw-unit bug
- why each invalidated Phase-3A-FIX selection

## C. Strict R0 audit
- WM r/RMSE
- FI r/RMSE
- PASS/FAIL

## D. Alignment/unit validity
- `y_oof == y[T]` check count
- residual OOF units
- inner/final formula parity error
- all relevant tests

## E. Development models
A0/B0/B1/C0/C1/C2/C3/C4:
- Pearson
- RMSE
- MAE

## F. Seed-level decomposition
- C1-A0
- C1-C0
- C0-B0
- C1-B1
- B1-B0
- C1-C2/C3/C4

four seed deltas, mean, median, positive count.

## G. Selection diagnostics
- selected alpha distributions
- lambda distributions
- abstention count
- eligibility fraction
- inner R0 Pearson distribution

## H. Optimizer
- convergence
- failures
- runtime

## I. Biomarker readiness
- C0/C1 stability
- ROI consistency
- faithfulness

## J. Gate evaluation
Gates 1–4.

## K. Freeze/lock
Exactly one:
```text
READY_FOR_PHASE3B
```
or
```text
HOLDOUT_REMAINS_LOCKED
```

## L. Tests
- FIX2 alignment/unit tests
- full Phase-3A-FIX/FIX2 suite
- full repository suite
- unrelated pre-existing failures

Then exactly one:

```text
PHASE3A_FIX2_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

or:

```text
PHASE3A_FIX2_DECISION: DO_NOT_TOUCH_HOLDOUT
```

Finally:

```text
STATUS: PHASE3A_FIX2_COMPLETE
```

If any validity check fails:

```text
STATUS: PHASE3A_FIX2_NOT_VALID
```

and holdout remains sealed.

---

# 22. Integrity stop condition

This FIX2 rerun is justified only because Phase-3A-FIX selected models using:
1. misaligned validation labels, and
2. mismatched standardized-vs-raw residual units.

If Phase-3A-FIX2 is valid and returns `DO_NOT_TOUCH_HOLDOUT`, stop.

Do not:
- create Phase 3C;
- change PG-MT-BCR ranks;
- alter the prior;
- loosen gates;
- inspect the 98 holdout.

Only a valid model that passes the original development gates may unlock the
98-subject confirmatory holdout.
