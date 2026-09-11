# PALF ICLR 2027 — Phase 3A
# PG-MT-BCR Development on 412 Subjects + Automatic Method Freeze + Sealed 98-Subject Holdout

## Executive objective

We now have a genuinely new set of **98 processed HCP subjects** that were not
part of the previous 412-subject development cohort.

This changes the evaluation strategy fundamentally.

The previous 412 subjects must now be treated as a **development cohort**.
The new 98 subjects must be treated as a **sealed confirmatory holdout**.

Phase 3A has exactly three jobs:

1. implement and evaluate one final genuinely different model family on the
   412-subject development cohort only;
2. decide, using predefined criteria, whether the model is strong enough to
   justify unlocking the 98-subject holdout later;
3. if and only if the gate passes, freeze the complete training/evaluation
   procedure for Phase 3B.

Phase 3A must **never load the 98 subjects' connectomes, targets, predictions,
or any derived feature**.

Do not run Phase 3B in this prompt.

---

# 1. Sealed holdout manifest

Create:

```text
data_splits/phase3_holdout_98.txt
```

with EXACTLY these subject IDs:

```text
137936
138130
138231
138332
138534
138837
139233
139435
139637
139839
140117
140319
140420
140824
140925
141119
141422
141826
142828
143224
143325
143426
143830
146533
147030
147636
147737
148032
148133
148335
148436
148840
149236
149337
149539
149741
149842
150524
150625
150726
150928
151021
151526
151728
152225
152427
152831
153025
153126
153227
153429
153631
154330
154431
199655
199958
200008
200109
200210
200311
200513
200614
200917
201111
201414
201515
201818
202113
202719
202820
203418
203923
204016
204218
204319
204420
204521
204622
205119
205220
205725
205826
206222
206323
206525
206727
206828
206929
207123
207426
208024
208125
208226
208327
208428
208630
209127
209228
```

Canonical representation for hashing:

```text
sort IDs lexicographically
join with "\n"
append one final "\n"
```

Expected:

```text
count  = 98
unique = 98
SHA256 = 89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

Create:

```text
data_splits/phase3_holdout_98.sha256
data_splits/PHASE3_HOLDOUT_LOCK.json
```

`PHASE3_HOLDOUT_LOCK.json` must contain at least:

```json
{
  "n_subjects": 98,
  "canonical_sha256": "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425",
  "status": "SEALED_UNTOUCHED_FOR_PHASE3A",
  "allowed_phase": "Phase3B_only_after_MODEL_FROZEN",
  "labels_may_be_loaded_in_phase3a": false,
  "features_may_be_loaded_in_phase3a": false
}
```

Do not inspect holdout target distributions.

Do not compute holdout FC/SC statistics.

Do not check holdout model performance.

Existence of processed files is not needed for Phase 3A.

---

# 2. Freeze the 412-subject development cohort

Use the exact subject list/order underlying the corrected 412-subject R0
experiments.

Create:

```text
data_splits/phase3_development_412.txt
data_splits/phase3_development_412.sha256
```

Hard gates:

```text
len(dev_ids) == 412
len(unique(dev_ids)) == 412
set(dev_ids) ∩ set(holdout_ids) == empty
```

If the current repository now exposes a combined 510-subject dataset, every
Phase-3A loader must explicitly subset to the frozen 412 IDs before any feature
or target access.

Add a central guard:

```python
assert_phase3a_no_holdout_access(subject_ids)
```

that raises immediately if any holdout ID is requested.

All Phase-3A modeling functions must call this guard.

---

# 3. Scientific hypothesis

All previous direct edge-penalty, residual NCR, expert-fusion, and SFC-pair
variants have failed to deliver a robust prior-specific predictive gain.

The remaining model tests a genuinely different representation:

> The LLM prior may be useful as a prior on **low-rank node factors** that
> generate predictive connectome patterns, rather than as a fixed edge prior.

Working name:

```text
PG-MT-BCR
Prior-Guided Multi-Task Bilinear Connectome Regression
```

The model is trained as a residual correction on top of the exact corrected R0
FC+SC late-fusion predictor so that the strong baseline remains available as an
exact special case.

This is the final development architecture before the sealed holdout.

---

# 4. Correct R0 baseline

Use only the validated corrected same-solver implementation from:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

R0 is:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Required audit on the frozen historical protocol:

```text
WM r    ≈ 0.263515
WM RMSE ≈ 11.292921

FI r    ≈ 0.370917
FI RMSE ≈ 4.566689
```

Tolerances:

```text
Pearson abs error <= 5e-4
RMSE abs error    <= 0.05
```

If audit fails:

```text
STATUS: PHASE3A_BASELINE_AUDIT_FAILED
```

and stop.

Do not weaken, replace, or retune R0 to make the proposed method look better.

---

# 5. Frozen ROI priors

Use only the already frozen task-specific ROI priors:

```text
outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv
outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv
```

Record SHA256 hashes.

No LLM calls in Phase 3A.

No regeneration.

No Phase-2E pair prior is used in the primary PG-MT-BCR model.

Let:

```text
p_WM ∈ [0,1]^116
p_FI ∈ [0,1]^116
```

Define the shared prior:

\[
p_{\mathrm{shared},i}
=
\sqrt{(p_{\mathrm{WM},i}+\epsilon)(p_{\mathrm{FI},i}+\epsilon)}
\]

with:

```text
epsilon = 1e-6
```

then min-max normalize to `[0,1]`.

Task-specific factor priors are simply:

```text
p_WM
p_FI
```

Do NOT construct `p_WM - p_FI` or another new prior in Phase 3A.

---

# 6. Input connectome representation

Use the same AAL116 FC and SC matrices underlying the 412-subject final PALF
experiments.

For every fitting scope:

1. start from the existing 6670 upper-triangle FC feature vector;
2. fit per-edge FC mean/std on TRAINING subjects only;
3. standardize training/validation rows;
4. reconstruct a symmetric 116×116 matrix with zero diagonal;

and independently repeat for SC.

Thus for subject s:

```text
M_FC[s] ∈ R^(116×116)
M_SC[s] ∈ R^(116×116)
```

No global edge scaling.

For an outer-validation subject, all scaling statistics come only from the
corresponding outer-training subjects.

For inner validation, all scaling comes only from inner training.

---

# 7. Leakage-safe R0 residuals

The proposed model predicts only what R0 misses.

Within every outer-training split, obtain true R0 OOF predictions using the
validated cross-fitting code:

```text
base_oof_WM
base_oof_FI
```

Every outer-training subject's base prediction must come from a model that did
not train on that subject.

Define:

\[
r_{\mathrm{WM}} = y_{\mathrm{WM}} - \hat y_{\mathrm{R0,WM}}^{\mathrm{OOF}}
\]

\[
r_{\mathrm{FI}} = y_{\mathrm{FI}} - \hat y_{\mathrm{R0,FI}}^{\mathrm{OOF}}
\]

Do NOT use in-sample R0 predictions to form residual targets.

For the outer-test fold, use the normal final R0 test prediction.

---

# 8. PG-MT-BCR parameterization

The connectomes are symmetric, so use symmetric low-rank bilinear components.

For modality:

```text
m ∈ {FC, SC}
```

learn:

```text
U_m_shared  ∈ R^(116×2)
U_m_WM      ∈ R^(116×1)
U_m_FI      ∈ R^(116×1)
```

The ranks are FIXED:

```text
shared rank        = 2
task-specific rank = 1 per task
```

Do not tune rank in Phase 3A.

For a subject matrix `M_m`, define component scores:

\[
g_{m,k}(M)
=
u_{m,k}^\top M u_{m,k}.
\]

For task `t`, the standardized residual prediction is:

\[
\hat r_t
=
b_t
+
\sum_{m\in\{FC,SC\}}
\left[
\sum_{k=1}^2
a^{\mathrm{shared}}_{m,t,k}
\,u^{\mathrm{shared}\top}_{m,k} M_m u^{\mathrm{shared}}_{m,k}
+
a^{\mathrm{specific}}_{m,t}
\,u^{\mathrm{specific}\top}_{m,t} M_m u^{\mathrm{specific}}_{m,t}
\right].
\]

Important:

- shared spatial factors `U_m_shared` are common to WM and FI;
- shared-factor amplitudes are task-specific;
- each task also has one task-specific spatial factor;
- FC and SC have separate factors;
- the model is linear in explicit coefficient matrices after training.

---

# 9. Factor normalization

There is otherwise a scale ambiguity between `u` and its amplitude.

After every optimizer update, normalize every factor column:

```text
u <- u / max(||u||_2, 1e-12)
```

Do not normalize the amplitude coefficients.

No orthogonality penalty is tuned.

For the two shared factors, after column normalization use deterministic
Gram-Schmidt orthogonalization.

Then renormalize.

Add a unit test proving factor norms are approximately 1.

---

# 10. Prior-weighted factor penalty

For any ROI prior `p`, define:

\[
d_i(p)
=
\frac{(\epsilon+p_i)^{-\gamma}}
{\frac1{116}\sum_j(\epsilon+p_j)^{-\gamma}}
\]

with fixed:

```text
epsilon = 1e-3
gamma   = 0.5
```

and:

\[
D(p)=\operatorname{diag}(d_1,\ldots,d_{116}).
\]

Prior penalty:

\[
R_{\mathrm{prior}}
=
\sum_m
\left[
\sum_{k=1}^2
u_{m,k}^{\mathrm{shared}\top}
D(p_{\mathrm{shared}})
u_{m,k}^{\mathrm{shared}}
+
u_{m,\mathrm{WM}}^\top D(p_{\mathrm{WM}})u_{m,\mathrm{WM}}
+
u_{m,\mathrm{FI}}^\top D(p_{\mathrm{FI}})u_{m,\mathrm{FI}}
\right].
\]

This is the only way the LLM prior enters the primary model.

---

# 11. Complete training objective

Within a fitting scope, standardize each residual task using TRAINING residual
mean/std only.

Optimize:

\[
\mathcal L =
\frac1n
\sum_s\sum_{t\in\{\mathrm{WM,FI}\}}
(r_{s,t}-\hat r_{s,t})^2
+
\lambda_{\mathrm{amp}}R_{\mathrm{amp}}
+
\lambda_{\mathrm{prior}}R_{\mathrm{prior}}
\]

where:

\[
R_{\mathrm{amp}}
=
\sum a^2.
\]

Clarification:
- amplitudes are penalized;
- intercepts are unpenalized.

Do not add other loss terms in Phase 3A.

---

# 12. Architecture-matched no-prior model

The architecture-matched comparator is:

```text
MT-BCR-NP
```

It is EXACTLY the same model:
- same ranks;
- same initialization;
- same optimizer;
- same residual targets;
- same amplitude penalty;
- same alpha selection;

but:

```text
lambda_prior = 0
```

This is the primary mechanistic comparator.

The prior claim is:

```text
PG-MT-BCR > MT-BCR-NP
```

not merely PG-MT-BCR > ordinary R0.

---

# 13. Single-task ablation

To answer the advisor's individual-vs-multi-task question, implement a
capacity-matched single-task model:

```text
ST-BCR
```

For each task separately and each modality:

```text
3 task-specific factors
0 shared factors
```

Total rank per task/modality remains 3, matching:

```text
2 shared + 1 task-specific
```

in PG-MT-BCR.

Run:

```text
ST-BCR-NP       : lambda_prior = 0
ST-PG-BCR       : matched task prior
MT-BCR-NP       : shared+specific, no prior
PG-MT-BCR       : shared+specific, matched prior
```

This gives:

```text
multi-task effect without prior:
MT-BCR-NP - ST-BCR-NP

multi-task effect with prior:
PG-MT-BCR - ST-PG-BCR

prior effect in single-task:
ST-PG-BCR - ST-BCR-NP

prior effect in multi-task:
PG-MT-BCR - MT-BCR-NP
```

Do not use the single-task models to redefine the proposed model after results.

PG-MT-BCR remains the proposed model.

---

# 14. Optimizer

Use PyTorch full-batch optimization.

Preferred:

```text
Adam
```

Settings:

```text
learning_rate = 0.01
max_steps     = 1500
min_steps     = 200
relative_training_loss_tolerance = 1e-7
patience      = 75
gradient_clip_norm = 5.0
dtype         = float64
```

Stopping is based only on TRAINING penalized loss.

Do NOT early-stop on validation performance.

Use two deterministic optimization restarts per fit.

Choose the restart with lower final TRAINING penalized objective only.

Do not choose a restart by validation/test correlation.

If CUDA float64 behavior is unstable, CPU float64 is acceptable because the
model is small.

---

# 15. Hyperparameter grid

The search space is intentionally small.

For no-prior models:

```text
lambda_amp_grid = [0.1, 1.0, 10.0]
```

For matched-prior models:

```text
lambda_amp_grid   = [0.1, 1.0, 10.0]
lambda_prior_grid = [0.01, 0.1, 1.0]
```

Residual amplitude:

```text
alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
```

Do not add rank search.

Do not tune gamma.

Do not add new regularizers after seeing results.

---

# 16. Residual-to-final prediction

For task t:

\[
\hat y_t^{\mathrm{final}}
=
\hat y_t^{\mathrm{R0}}
+
\alpha_t \hat r_t.
\]

`alpha_t=0` exactly recovers R0.

Alpha is task-specific but chosen only using training OOF predictions.

---

# 17. Development CV protocol on the 412 subjects

Use exactly:

```text
DEV_CV_SEEDS = [3535, 3636, 3737, 3838]
outer_folds  = 5
inner_folds  = 3
```

These are development splits, NOT confirmatory evidence.

The 98-subject holdout remains sealed.

For every outer development split:

## Step A — corrected R0
Generate:
- R0 OOF residual targets on outer-training;
- R0 outer-test prediction.

## Step B — inner selection
Within outer-training, use 3-fold inner CV.

For every candidate:
- refit FC/SC edge scaling on inner-train;
- generate/obtain leakage-safe baseline residual target for the inner fitting
  scope;
- train BCR only on inner-train;
- predict inner-validation residual;
- reconstruct final task prediction with each alpha.

Selection objective for a paired MT candidate:

1. compute WM and FI validation Pearson;
2. Fisher transform;
3. use mean Fisher-z;
4. tie-break by mean normalized RMSE;
5. then mean normalized MAE;
6. then smaller `alpha_WM + alpha_FI`;
7. then smaller lambda_prior;
8. then larger lambda_amp;
9. deterministic final tie-break.

For single-task candidates, optimize the corresponding task Pearson with
RMSE/MAE and smaller-alpha tie-breaks.

## Step C — outer prediction
Refit the selected BCR candidate on the complete outer-training residuals
(using true R0 OOF residuals).

Predict outer-test residuals.

Combine with the untouched R0 outer-test prediction.

Evaluate once.

---

# 18. Non-inferiority safeguard during inner selection

The proposed model should not win the joint objective by sacrificing one task.

A paired MT candidate is eligible only if, on the concatenated inner OOF
predictions:

```text
delta_r_WM >= -0.002
delta_r_FI >= -0.002
```

relative to R0.

If no prior-aware candidate passes, the prior-aware model uses:

```text
alpha_WM = 0
alpha_FI = 0
```

for that outer split.

Do not use outer-test results for this safeguard.

---

# 19. Development prior controls

Run controls only for the proposed PG-MT-BCR architecture.

## Cross-task control

For task-specific factors:
- WM factor uses FI prior;
- FI factor uses WM prior.

Shared prior remains the geometric mean.

## Shuffled control

Create one fixed deterministic permutation per task:

```text
WM shuffle seed = 8101
FI shuffle seed = 8102
```

Apply it to ROI labels.

Shared control prior is recomputed from the shuffled task priors.

## Random control

Create one fixed Uniform[0,1] 116-vector per task:

```text
WM random seed = 8103
FI random seed = 8104
```

Shared prior is the geometric mean.

Freeze these controls before development CV.

Use identical hyperparameter grids and CV logic.

Do not select the best random control.

---

# 20. Models to report

Exactly:

```text
A0  Corrected R0

B0  ST-BCR-NP
B1  ST-PG-BCR

C0  MT-BCR-NP
C1  PG-MT-BCR matched          <-- proposed

C2  PG-MT-BCR cross-task
C3  PG-MT-BCR shuffled
C4  PG-MT-BCR random
```

Do not add another prediction architecture in Phase 3A.

---

# 21. Primary development comparisons

For WM and FI separately report:

## Final model gain
```text
C1 - A0
```

## Architecture-matched prior effect
```text
C1 - C0
```

## Multi-task effect without prior
```text
C0 - B0
```

## Multi-task effect with prior
```text
C1 - B1
```

## Single-task prior effect
```text
B1 - B0
```

## Prior specificity
```text
C1 - C2
C1 - C3
C1 - C4
```

For every comparison:
- 4 seed-level deltas;
- mean;
- median;
- positive seeds / 4.

No confirmatory p-value claim from Phase 3A.

---

# 22. Biomarker extraction

For a trained modality/task model, define the coefficient matrix:

Shared contribution for task t:

\[
B_{m,t}^{\mathrm{shared}}
=
\sum_{k=1}^2
a^{\mathrm{shared}}_{m,t,k}
u^{\mathrm{shared}}_{m,k}
u^{\mathrm{shared}\top}_{m,k}.
\]

Task-specific contribution:

\[
B_{m,t}^{\mathrm{specific}}
=
a^{\mathrm{specific}}_{m,t}
u^{\mathrm{specific}}_{m,t}
u^{\mathrm{specific}\top}_{m,t}.
\]

Total residual coefficient matrix:

\[
B_{m,t} =
B_{m,t}^{\mathrm{shared}}
+
B_{m,t}^{\mathrm{specific}}.
\]

Because the diagonal of the connectome input is zero, set the coefficient
diagonal to zero for reporting.

Extract:
- full 116×116 matrices;
- 6670 upper-triangle vectors;
- shared component;
- task-specific component.

Residual correction map in the final model is:

```text
alpha_t * B_m,t
```

Do not mix this with R0 baseline coefficients unless the R0 primal map has been
independently validated.

For Phase 3A the BCR residual map itself is the biomarker object.

---

# 23. Biomarker stability diagnostics on development data

These do NOT affect model selection.

Across the 20 outer fits compute for C0 and C1:

For each task/modality:

```text
Spearman(abs(beta))
top-100 edge Jaccard
top-300 edge Jaccard
top-10 ROI Jaccard
top-20 ROI Jaccard
sign consistency
```

ROI importance:

\[
s_i = \sum_{j\ne i} |B_{ij}|.
\]

Also compare:
- shared-map stability;
- task-specific-map stability.

Report matched prior minus no-prior stability.

---

# 24. Development faithfulness diagnostic

Do NOT use this to select the model.

For each outer-development test fold:

1. rank ROIs using training-fit C1 residual maps;
2. mask incident standardized FC/SC edges for:
   - top 5 ROIs;
   - top 10 ROIs;
3. compare against:
   - bottom same-size ROIs;
   - 50 deterministic random same-size sets;
4. recompute BCR residual correction without retraining;
5. report change in FINAL prediction RMSE.

Also run the same perturbation for C0.

Main diagnostic:

```text
C1 top-vs-random degradation
vs
C0 top-vs-random degradation
```

This is development-only.

---

# 25. Development gate before holdout unlock

The 98-subject holdout may be unlocked in Phase 3B ONLY if all of the following
are satisfied.

## Gate 1 — final predictive improvement

For BOTH targets:

```text
mean(C1 - A0) >= +0.005
positive seed deltas >= 3/4
```

and at least one target:

```text
mean(C1 - A0) >= +0.010
```

## Gate 2 — prior adds value inside the same architecture

For BOTH targets:

```text
mean(C1 - C0) >= +0.003
positive seed deltas >= 3/4
```

## Gate 3 — prior specificity

For BOTH targets:

```text
mean(C1 - C3) > 0
mean(C1 - C4) > 0
```

and matched prior must not be worse than cross-task on both tasks.

## Gate 4 — no severe instability

For both tasks:
- no NaN/inf;
- optimizer convergence valid in >=95% fits;
- no outer split has catastrophic RMSE > 2× R0 RMSE.

If all four gates pass:

```text
PHASE3A_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

Otherwise:

```text
PHASE3A_DECISION: DO_NOT_TOUCH_HOLDOUT
```

Do not weaken these gates after seeing development results.

---

# 26. Automatic method freeze

If and only if Phase 3A passes, create:

```text
outputs/iclr/palf_phase3a_pg_mt_bcr/MODEL_FROZEN.json
outputs/iclr/palf_phase3a_pg_mt_bcr/MODEL_FROZEN.sha256
```

The freeze file must record:

- git HEAD;
- full uncommitted diff hash;
- Python/PyTorch versions;
- CUDA/device info;
- 412-development manifest SHA256;
- 98-holdout manifest SHA256;
- ROI prior SHA256s;
- shuffled/random control SHA256s;
- exact model equations/version;
- ranks;
- gamma;
- epsilon;
- optimizer settings;
- restart rule;
- hyperparameter grids;
- CV seeds/folds;
- model-selection objective;
- eligibility safeguard;
- alpha grid;
- preprocessing;
- biomarker ranking rule;
- holdout evaluation metrics planned for Phase 3B;
- timestamp.

After `MODEL_FROZEN.json` is written:

```text
NO Phase-3A code/config/model changes are permitted before Phase 3B.
```

If any change is made, the holdout remains locked until a new independent
holdout exists.

---

# 27. What is frozen vs what is selected during Phase 3B

The following PROCEDURE is frozen:

- all model architecture;
- all grids;
- optimizer;
- priors;
- preprocessing;
- CV selection algorithm.

Phase 3B may use the entire 412-development cohort to select numeric
hyperparameters via the frozen CV rule before fitting the final 412-subject
model.

Phase 3B may NOT:
- add a hyperparameter;
- delete a candidate based on holdout behavior;
- alter the prior;
- alter rank;
- alter optimizer;
- alter feature scaling;
- alter alpha grid;
- alter task weighting.

This distinction must be explicitly stored in `MODEL_FROZEN.json`.

---

# 28. Phase-3A outputs

Create:

```text
outputs/iclr/palf_phase3a_pg_mt_bcr/
```

with:

```text
COMPLETE
BASELINE_AUDIT.json
HOLDOUT_SEAL_REPORT.json
VALIDATION_REPORT.json
RUN_REPORT.md

development_412_manifest.txt
holdout_98_manifest.txt

split_metrics.csv
seed_metrics.csv
model_summary.csv
selection_details.csv
optimizer_diagnostics.csv

prior_controls/
coefficients/
predictions/
biomarkers/
plots/
```

If gate passes also include:

```text
MODEL_FROZEN.json
MODEL_FROZEN.sha256
READY_FOR_PHASE3B
```

If gate fails create instead:

```text
HOLDOUT_REMAINS_LOCKED
```

ZIP:

```text
outputs/iclr/palf_phase3a_pg_mt_bcr.zip
```

---

# 29. Required plots

Generate:

```text
plots/fig_phase3a_model_comparison.pdf/png
plots/fig_phase3a_seed_deltas.pdf/png
plots/fig_phase3a_prior_effect.pdf/png
plots/fig_phase3a_multitask_effect.pdf/png
plots/fig_phase3a_factor_stability.pdf/png
plots/fig_phase3a_biomarker_faithfulness.pdf/png
```

---

# 30. Tests

Add:

```text
tests/test_phase3a_pg_mt_bcr.py
```

Minimum tests:

1. holdout manifest has exactly 98 unique IDs.
2. canonical holdout SHA256 equals:
   `89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425`.
3. development manifest has exactly 412 unique IDs.
4. development/holdout intersection is empty.
5. Phase-3A loader raises on any holdout ID.
6. no Phase-3A metrics file contains a holdout ID.
7. no Phase-3A prediction/target array has 98-holdout rows.
8. R0 audit reproduces corrected baseline.
9. R0 residual target uses OOF prediction, not in-sample prediction.
10. edge standardization is train-scope only.
11. standardized edge vector reconstructs symmetric 116×116 matrix.
12. factor columns remain unit norm.
13. shared-factor columns are orthogonal within tolerance.
14. no-prior model has exactly `lambda_prior=0`.
15. matched prior model uses only frozen ROI priors.
16. cross-task prior swaps only task-specific prior assignment.
17. shuffled/random controls are deterministic and frozen.
18. alpha=0 exactly reproduces R0 prediction.
19. ST model has 3 task-specific factors and no shared factors.
20. MT model has 2 shared + 1 task-specific factor.
21. ST and MT have matched total rank per task/modality.
22. optimizer restart selected by training loss only.
23. no validation/test performance chooses restart.
24. outer validation subjects do not enter optimizer training.
25. inner selection follows the predefined joint Fisher-z objective.
26. prior-aware eligibility safeguard is applied before candidate selection.
27. BCR coefficient reconstruction reproduces direct residual prediction.
28. biomarker diagnostics do not affect hyperparameter selection.
29. MODEL_FROZEN is created only when every gate passes.
30. Phase-3A code contains no Phase-3B holdout-evaluation call.

Run targeted tests and the full suite.

---

# 31. Final OpenCode report

Print exactly:

## A. Holdout seal
- 98 count
- unique count
- SHA256
- dev/holdout intersection
- confirmation that no holdout feature/target was loaded

## B. Development cohort
- exact 412 count
- manifest SHA256

## C. Baseline audit
- WM r/RMSE
- FI r/RMSE
- PASS/FAIL

## D. Model implementation
- ranks
- optimizer
- grids
- prior hashes
- coefficient reconstruction max error

## E. Development prediction results
For WM and FI:

```text
A0 R0
B0 ST-BCR-NP
B1 ST-PG-BCR
C0 MT-BCR-NP
C1 PG-MT-BCR matched
C2 cross-task
C3 shuffled
C4 random
```

Report Pearson/RMSE/MAE.

## F. Seed-level mechanism decomposition
For each target:
- C1-A0
- C1-C0
- C0-B0
- C1-B1
- B1-B0
- C1-C2/C3/C4

Each with four seed deltas, mean, median, positive count.

## G. Optimizer stability
- converged fraction
- restart disagreement
- failed fits
- runtime

## H. Biomarker readiness
- C0 vs C1 edge-rank stability
- ROI Jaccard
- sign consistency
- top-vs-random faithfulness

## I. Gate evaluation
Print each Gate 1–4 separately as PASS/FAIL.

## J. Freeze
If all pass:
- MODEL_FROZEN path
- MODEL_FROZEN SHA256
- READY_FOR_PHASE3B

If any fail:
- HOLDOUT_REMAINS_LOCKED

## K. Tests
- targeted
- full suite
- pre-existing failures

## L. Outputs
- directory
- ZIP
- CSVs
- coefficient paths
- plots

Then exactly one:

```text
PHASE3A_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

or:

```text
PHASE3A_DECISION: DO_NOT_TOUCH_HOLDOUT
```

Finally:

```text
STATUS: PHASE3A_PG_MT_BCR_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE3A_NOT_VALID
```

and holdout MUST remain sealed.

---

# 32. Critical integrity rule

The 98 subjects are the only genuinely untouched confirmation cohort available
after extensive development on the original 412.

Do not spend them to debug code.

Do not inspect their target distribution.

Do not use them for early stopping.

Do not run R0 on them "just to check".

Do not evaluate one candidate and then change the model.

Phase 3A must produce a frozen procedure first.

Only a successful `FREEZE_AND_UNLOCK_PHASE3B` decision permits one
confirmatory holdout evaluation in a separate Phase-3B run.
