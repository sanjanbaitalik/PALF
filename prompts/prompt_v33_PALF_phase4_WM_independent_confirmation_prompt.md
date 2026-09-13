# PALF / ICLR 2027 — Phase 4
# WM-Only Frozen PS-NCR-EF Confirmation on the 98-Subject Independent Holdout
# Prediction + Biomarker Faithfulness, One-Time Evaluation

## Executive objective

This is the FINAL confirmatory experiment for the current paper.

The development evidence has already selected:

```text
Target:
Working Memory (NIH List Sorting / ListSort_Unadj)

Frozen method:
PS-NCR-EF
Prior-Selected Network-Constrained Expert Fusion

Development cohort:
the original 412 subjects

Independent confirmatory cohort:
98 previously untouched processed subjects
```

The goal is NOT to search for another model.

The goal is to answer two predeclared questions on the independent 98-subject
holdout:

1. Does the frozen WM PS-NCR-EF procedure improve prediction over the corrected
   R0 no-prior baseline?

2. Are the frozen WM biomarkers learned only from the 412 development subjects
   faithful on unseen participants, i.e. does perturbing the top-ranked
   biomarkers hurt holdout prediction more than perturbing random/bottom ROIs?

This prompt is deliberately WM-only.

Do NOT use Fluid Intelligence holdout labels in this experiment.

---

# 0. Expected runtime and mandatory measured ETA

## Historical planning estimate

On the same machine that ran Phase 2D-FIX in roughly ~50 minutes for its
development/audit workload, the expected end-to-end Phase-4 wall-clock time is:

```text
approximately 2–4 hours
```

Typical breakdown:

```text
A. integrity + R0 audit                  ~10–20 min
B. final 412-subject model selection    ~45–90 min
C. final refits + controls + freezing   ~15–30 min
D. pre-holdout tests / dry-run          ~10–20 min
E. one-time 98-subject inference        ~5–15 min
F. perturbation + 10k bootstrap         ~15–45 min
G. reports / plots / packaging          ~5–15 min
```

If execution is CPU-only or the environment is substantially slower, a
reasonable upper planning range is approximately:

```text
4–6 hours
```

These are planning estimates only.

## Mandatory machine-specific ETA

Before running the full process:

1. benchmark a representative DEVELOPMENT-ONLY Phase-4 fitting workload;
2. measure seconds per fold / expert-selection operation;
3. estimate:
   - Stage A time;
   - Stage B time;
   - Stage C time;
   - total remaining wall-clock time;
4. create:

```text
outputs/iclr/palf_phase4_wm_confirmation/RUNTIME_ESTIMATE.md
```

5. print clearly:

```text
PHASE4_ESTIMATED_TOTAL_TIME: <hours/minutes>
PHASE4_ESTIMATED_FINISH_FROM_START: <hours/minutes>
```

Update:

```text
RUNTIME_PROGRESS.json
```

after each major stage with:
- elapsed time;
- estimated remaining time;
- percent complete.

Do NOT access the 98-subject holdout during benchmarking.

---

# 1. Scientific status and scope

The prediction-method search on the 412-subject cohort is over.

Valid development evidence indicates that Working Memory is the only phenotype
with a plausible combined prediction/biomarker story.

The frozen scientific decomposition is:

```text
NCR expert fusion:
candidate source of WM predictive gain

task-semantic prior:
candidate source of biomarker stabilization / meaningful feature restriction
```

Do NOT claim that the semantic prior uniquely caused the development prediction
gain, because shuffled controls were competitive in some prediction analyses.

The confirmatory paper claim, if supported, is narrower:

> A frozen task-semantic NCR expert-fusion model improves Working-Memory
> prediction over a strong no-prior baseline and yields development-derived
> biomarkers that are faithful on previously unseen participants.

The holdout must decide this claim.

---

# 2. Independent holdout manifest

Use the existing sealed manifest:

```text
data_splits/phase3_holdout_98.txt
```

Required:

```text
n = 98
unique = 98
canonical SHA256 =
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

Development manifest:

```text
data_splits/phase3_development_412.txt
```

Required:

```text
n = 412
unique = 412
development ∩ holdout = empty
```

Before any holdout access create:

```text
HOLDOUT_PREOPEN_AUDIT.json
```

and verify all conditions.

---

# 3. WM target only

The ONLY confirmatory target is:

```text
Working Memory / ListSort_Unadj
```

Use the exact WM label definition and subject alignment used throughout the
valid corrected PALF experiments.

Do NOT load:

```text
Fluid Intelligence holdout labels
```

Do not report FI holdout performance.

If the current data loader automatically loads FI labels, refactor it BEFORE
the freeze so Phase 4 can request WM labels only.

After the model/protocol freeze, this loader may not be changed.

---

# 4. Frozen corrected R0 baseline

R0 must be exactly the validated same-solver baseline:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use the validated code in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

At minimum reuse the established R0 path using:

```text
CONDITIONS["R0"]
generate_crossfit_oof
oof.fp_oof
oof.sc_oof
search_fusion_weights
reselect_and_fit_final
```

Historical correctness audit on seeds 0–9 must reproduce:

```text
WM Pearson r ≈ 0.263515
WM RMSE      ≈ 11.292921
```

Hard tolerances:

```text
abs Pearson error <= 5e-4
abs RMSE error    <= 0.05
```

If this fails:

```text
STATUS: PHASE4_BASELINE_AUDIT_FAILED
```

and DO NOT access the 98 holdout.

---

# 5. Frozen proposed method: PS-NCR-EF

Use the VALID Phase 2D-FIX implementation.

Locate and audit the final corrected Phase-2D code/output, expected under paths
such as:

```text
src/metascfc/experiments/prior_subspace_expert_fusion.py
scripts_paper/phase2d_prior_subspace_ncr_expert_pilot.py
outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion/
```

Do not use the original buggy Phase-2D implementation.

The frozen method is:

```text
PS-NCR-EF =
Prior-Selected Network-Constrained Expert Fusion
```

with:

1. untouched corrected R0 backbone;
2. prior-selected FC expert;
3. prior-selected SC expert;
4. NCR on the selected subspace;
5. two-branch FC/SC expert fusion;
6. two-branch R0/expert final fusion.

No residual model.
No new multi-task model.
No Phase-3A BCR.
No kernel model.
No representation search.

---

# 6. Frozen Phase-2D mask families and grids

Use EXACTLY the valid Phase-2D-FIX scientific search space.

## Prior-derived mask families

### Direct top-K edge score

For edge:

```text
(i,j)
```

score:

```text
q_ij = p_i * p_j
```

Grid:

```text
K_EDGE_GRID = [100, 300, 600, 1200]
```

### Top-M ROI incident-edge family

Grid:

```text
M_ROI_GRID = [5, 10, 15]
```

Use all edges incident to the selected top-M prior ROIs.

Expected counts for AAL116:

```text
M=5  -> 565 edges
M=10 -> 1105 edges
M=15 -> 1590 edges
```

Do not add/remove mask sizes.

---

# 7. Ridge and NCR grids

Restricted Ridge expert:

```text
lambda_R =
[0.001, 0.01, 0.1, 1, 10, 100, 1000]
```

NCR:

\[
||y-X_E\beta||^2
+
\lambda_R ||\beta||^2
+
\lambda_L \beta^T L_E\beta
\]

with:

```text
lambda_L = ratio * lambda_R

ratio =
[0.0, 0.1, 0.3, 1.0]
```

Hard identity test:

```text
ratio=0
```

must equal restricted Ridge for the same:
- edge mask;
- lambda_R;
- preprocessing.

Do not alter these grids.

---

# 8. Frozen hierarchical fusion

## Expert fusion

For selected FC and SC experts:

\[
\hat y_E
=
v\hat y_{FC,E}
+
(1-v)\hat y_{SC,E}
\]

with:

```text
v ∈ [0, 0.05, 0.10, ..., 1.0]
```

selected on development OOF predictions only.

## Final fusion

\[
\hat y_{final}
=
(1-\alpha)\hat y_{R0}
+
\alpha \hat y_E
\]

with:

```text
alpha ∈ [0, 0.05, 0.10, ..., 1.0]
```

selected on development OOF predictions only.

Hard assertion:

```text
alpha=0
```

exactly reproduces R0.

Primary metric:
```text
Pearson r
```

Tie-break:
1. RMSE;
2. MAE;
3. smaller expert/final weight where still tied;
4. deterministic tuple order.

Use the already validated Phase-2D-FIX semantics if they are more specific;
do not invent a new weight-selection rule.

---

# 9. Frozen WM LLM prior

Use the exact WM ROI prior used in valid Phase 2D-FIX.

Record:
- source path;
- SHA256;
- 116 ROI values;
- ROI names/order.

No LLM call.
No prior regeneration.
No prompt changes.
No use of holdout data.

---

# 10. Frozen controls

Train/finalize these DEVELOPMENT-only control procedures before holdout access:

```text
R0
Matched PS-NCR-EF
Matched restricted-Ridge expert fusion
Cross-task PS-NCR-EF
Shuffled PS-NCR-EF
Random PS-NCR-EF
```

Shuffled/random identities/seeds must be exactly those used by valid
Phase 2D-FIX if available.

Do not generate a panel of random controls.

Do not select the best random control.

If valid Phase-2D-FIX used one fixed shuffled/random prior, reuse exactly it.

---

# 11. Final model selection on all 412 development subjects

The 98 holdout must play NO role.

We need one deterministic final model trained on all 412.

Use the SAME Phase-2D-FIX nested selection logic, now applied to the complete
412-subject development cohort.

## 11.1 Finalization CV

Use exactly:

```text
FINALIZATION_CV_SEEDS = [6161, 6262, 6363]
folds_per_seed = 5
```

These are used only to fit/freeze the final 412-subject procedure before
opening the holdout.

Do not modify them after results.

## 11.2 Cross-fitted expert predictions

For each finalization seed/fold:

1. training subset:
   - select FC prior subspace strictly inside training;
   - select SC prior subspace strictly inside training;
   - select restricted-Ridge hyperparameters;
   - select NCR hyperparameters on the same selected mask;
2. predict the held-out development fold;
3. also generate corrected R0 held-out prediction.

Concatenate the 15 OOF predictions.

Use them to select:

```text
expert fusion weight v
final fusion alpha
```

No 98-subject data.

## 11.3 Final full-412 mask/hyperparameter selection

After OOF fusion weights are frozen:

re-run the SAME modality-specific selector using all 412 development subjects
with repeated training-only CV using the same:

```text
[6161,6262,6363]
```

For FC and SC independently choose:
- mask family;
- mask size;
- lambda_R;
- NCR ratio.

Selection uses mean/pooled development CV Pearson with:
- RMSE;
- MAE;
- smaller mask;
- deterministic tie-break.

No holdout data.

Then fit:
- final FC NCR expert on all 412;
- final SC NCR expert on all 412;
- final R0 on all 412.

The already frozen `v` and `alpha` are then used.

No selection occurs after holdout is opened.

---

# 12. Final development-derived biomarker map

Biomarkers are derived ONLY from the final all-412 matched PS-NCR-EF expert.

Export validated primal coefficients.

For modality m:

```text
c_m =
alpha
*
expert_fusion_weight_m
*
beta_m_standardized
```

where:

```text
expert_fusion_weight_FC = v
expert_fusion_weight_SC = 1-v
```

Define multimodal ROI importance:

\[
I_i
=
\sum_{j\ne i}|c^{FC}_{ij}|
+
\sum_{j\ne i}|c^{SC}_{ij}|
\]

Use this EXACT score for final WM biomarker ranking.

Freeze:

```text
top5_ROIs
top10_ROIs
bottom5_ROIs
bottom10_ROIs
```

before holdout access.

Save exact:
- 0-based AAL indices;
- 1-based AAL indices;
- ROI names;
- importance scores;
- SHA256 of ranking table.

Create:

```text
WM_BIOMARKER_RANKING_FROZEN.csv
WM_BIOMARKER_RANKING_FROZEN.sha256
```

---

# 13. Freeze control biomarker rankings

Using all 412 development subjects and the frozen control procedures, also
create top-5/top-10 ROI rankings for:

```text
cross-task
shuffled
random
restricted-Ridge matched
```

These rankings are SECONDARY confirmation controls.

Do not alter them after holdout access.

---

# 14. Pre-holdout frozen artifact

Before reading ANY 98-subject FC/SC/WM labels, create:

```text
outputs/iclr/palf_phase4_wm_confirmation/
PHASE4_WM_MODEL_FROZEN.json
PHASE4_WM_MODEL_FROZEN.sha256
READY_TO_OPEN_WM_HOLDOUT
```

Freeze file must contain:

- git HEAD;
- working-tree diff hash;
- environment;
- 412 manifest SHA;
- 98 manifest SHA;
- WM prior SHA;
- control-prior SHAs;
- R0 code/config hash;
- Phase-2D-FIX code/config hash;
- finalization CV seeds;
- selected FC mask;
- selected SC mask;
- selected lambdas;
- selected NCR ratios;
- expert fusion weight v;
- final alpha;
- all training scalers;
- coefficient-file hashes;
- exact top/bottom ROI rankings;
- random perturbation seeds;
- bootstrap seed;
- primary/secondary endpoints;
- exact success rules;
- runtime estimate;
- timestamp.

After this file is created:

```text
NO code/config/model/ranking/statistical-test changes are permitted.
```

---

# 15. Mandatory pre-holdout functional validation

Before unlocking holdout run all tests on development/synthetic data.

At minimum:

1. holdout manifest exact 98/SHA.
2. development/holdout disjoint.
3. historical R0 audit passes.
4. valid Phase-2D-FIX implementation is used.
5. ratio=0 NCR equals Ridge.
6. alpha=0 exactly equals R0.
7. final all-412 coefficient reconstruction <= 1e-8.
8. final prediction reconstruction <= 1e-8.
9. top/bottom rankings generated from development coefficients only.
10. perturbation code never retrains.
11. perturbation sign convention is correct:
    `RMSE_masked - RMSE_unmasked`.
12. positive delta means masking hurts prediction.
13. bootstrap code preserves paired subject rows.
14. holdout loader can request WM labels without FI labels.
15. no holdout data was touched during any test.

Run a synthetic perturbation test where masking a known informative feature
increases error, and assert positive delta_RMSE.

---

# 16. HOLDOUT UNLOCK BARRIER

Proceed to the 98 subjects ONLY if:

```text
READY_TO_OPEN_WM_HOLDOUT exists
all pre-holdout tests pass
PHASE4_WM_MODEL_FROZEN SHA verifies
```

At this exact point record:

```text
HOLDOUT_OPEN_TIMESTAMP
```

After this timestamp:

```text
NO code changes
NO config changes
NO ranking changes
NO hyperparameter changes
NO reruns with alternate settings
```

If a technical failure occurs after holdout access:
- save the failure;
- do not modify the model based on results;
- stop and report it for external review.

---

# 17. One-time holdout inputs

Load ONLY:

```text
98-subject FC
98-subject SC
98-subject WM labels
```

Do NOT load FI labels.

Verify:
- subject ordering against manifest;
- all 98 unique;
- all WM labels finite;
- all required FC/SC features finite.

No subject exclusion after viewing outcomes.

If a subject has invalid/missing required data:
- apply only a missing-data rule already frozen before unlock;
- otherwise declare the confirmation invalid rather than silently dropping.

---

# 18. One-time holdout prediction

Using models trained on the 412:

generate predictions for:

```text
R0
Matched PS-NCR-EF
Matched restricted-Ridge expert fusion
Cross-task PS-NCR-EF
Shuffled PS-NCR-EF
Random PS-NCR-EF
```

No refitting on 98.

No recalibration on 98.

No use of holdout y except to compute final metrics.

Report:

```text
Pearson r
RMSE
MAE
```

for all models.

Primary prediction comparison:

```text
Matched PS-NCR-EF vs R0
```

NCR mechanism comparison:

```text
Matched PS-NCR-EF vs matched restricted-Ridge expert fusion
```

Prior-specificity prediction comparisons are secondary:

```text
matched vs cross
matched vs shuffled
matched vs random
```

Do NOT require semantic-prior prediction specificity for the main WM prediction
claim.

---

# 19. Primary prediction hypothesis and inference

Predeclared:

\[
H_{pred}:
r_{PS-NCR-EF} > r_{R0}.
\]

Define:

```text
delta_r =
r_matched - r_R0
```

Primary confirmatory success requires:

```text
delta_r >= +0.005
```

AND evidence that the directional improvement is positive.

Use:

```text
10,000 paired subject-level bootstrap replicates
```

resampling the 98 triplets together:

```text
(y_WM, pred_R0, pred_matched)
```

Bootstrap seed:

```text
PHASE4_BOOTSTRAP_SEED = 9101
```

Report:
- observed delta_r;
- bootstrap mean;
- two-sided 95% percentile CI;
- one-sided 95% lower confidence bound
  (equivalently the 5th percentile of bootstrap delta_r);
- fraction of bootstrap replicates with delta_r <= 0.

Prediction confirmation gate:

```text
observed delta_r >= +0.005
AND
one-sided 95% lower bound > 0
```

Also report, as sensitivity:
- Williams/Steiger dependent-correlation test if correctly implemented and
  unit-tested;
- paired bootstrap delta_RMSE;
- paired bootstrap delta_MAE.

Do not substitute a more favorable test after seeing results.

---

# 20. Holdout biomarker perturbation

Biomarker rankings are already frozen from the 412.

For a masked ROI set S:

for every FC and SC edge incident to any ROI in S, replace the holdout raw edge
value with the corresponding DEVELOPMENT-412 training mean for that edge.

Then rerun the complete fixed predictor:

```text
R0 + PS-NCR-EF
```

without retraining.

This corresponds to zeroing those edges after development-standardization.

Never fit a mean/scaler on the 98.

Define:

```text
delta_RMSE(S) =
RMSE_masked(S) - RMSE_unmasked
```

Interpretation:

```text
positive  -> masking hurts prediction -> faithful
negative  -> masking improves prediction -> unfavorable
```

Never reverse this sign.

---

# 21. Primary biomarker endpoint

Primary biomarker set:

```text
frozen matched-prior top-10 WM ROIs
```

Compute:

```text
delta_RMSE_top10
```

Generate exactly:

```text
1000 deterministic random 10-ROI sets
```

using:

```text
RANDOM_MASK_SEED = 9201
```

Every random set has exactly 10 unique ROIs from the same AAL116 universe.

Because every ROI in the complete AAL edge representation has equal graph
degree, cardinality matching is sufficient for the primary random control.

For each random set calculate:

```text
delta_RMSE_random10
```

Report:
- random mean;
- random median;
- random 95th percentile;
- empirical one-sided p-value:

```text
p =
(1 + count(random_delta >= top_delta))
/
(1 + 1000)
```

Primary biomarker confirmation requires:

```text
delta_RMSE_top10 > 0
AND
delta_RMSE_top10 > mean(random_delta_RMSE10)
AND
empirical p < 0.05
```

This is a genuinely independent faithfulness test because:
- ranking came only from 412;
- prediction model came only from 412;
- perturbation evaluated only on 98.

---

# 22. Secondary biomarker endpoints

Also report without redefining the primary endpoint:

## Top-5

```text
delta_RMSE_top5
```

versus 1000 random 5-ROI sets:

```text
RANDOM_MASK5_SEED = 9202
```

## Bottom controls

```text
delta_RMSE_bottom5
delta_RMSE_bottom10
```

Expected:
```text
top > bottom
```

but this is secondary.

## Control ranking comparison on SAME matched final predictor

Using the frozen rankings learned from the 412, perturb the matched final
predictor according to:

```text
matched top10 ranking
cross-task top10 ranking
shuffled top10 ranking
random-prior top10 ranking
restricted-Ridge top10 ranking
```

This holds the evaluated model fixed and changes ONLY the ranking being tested.

Report all delta_RMSE values.

Evidence for semantic biomarker specificity is strengthened if:

```text
matched ranking delta_RMSE
>
shuffled ranking delta_RMSE

and

matched ranking delta_RMSE
>
random-prior ranking delta_RMSE
```

These are secondary unless explicitly stated otherwise before holdout unlock.

---

# 23. Subject-bootstrap biomarker sensitivity analysis

In addition to the mask-randomization test, run:

```text
10,000 paired subject bootstrap replicates
```

for:

```text
delta_RMSE_top10 - mean_random10_delta_RMSE
```

using:

```text
BIOMARKER_BOOTSTRAP_SEED = 9301
```

For each bootstrap replicate:
- resample the 98 subjects with replacement;
- preserve all model predictions/masked predictions by subject;
- compute the contrast.

Report:
- mean;
- 95% CI;
- one-sided lower bound.

This is a sensitivity analysis.

The primary biomarker random-mask empirical test remains the frozen primary.

---

# 24. Development biomarker stability table

Do not recompute/tune based on holdout.

Include the already valid development Phase-2D-FIX stability comparison for
context.

Recompute from valid Phase-2D-FIX coefficient files if necessary, excluding
fits where the expert final weight is zero.

At minimum report for matched, cross, shuffled, random:

```text
FC abs-edge Spearman
SC abs-edge Spearman
FC top-10 ROI Jaccard
SC top-10 ROI Jaccard
```

This is DEVELOPMENT evidence, clearly labelled as such.

Do not mix it statistically with the independent holdout faithfulness test.

---

# 25. Final paper-level success logic

## Prediction-confirmed

```text
PREDICTION_CONFIRMATION: PASS
```

if:

```text
delta_r >= +0.005
AND one-sided bootstrap lower bound > 0
```

Otherwise:

```text
PREDICTION_CONFIRMATION: FAIL
```

## Biomarker-confirmed

```text
BIOMARKER_CONFIRMATION: PASS
```

if:

```text
delta_RMSE_top10 > 0
AND top10 > random10 mean
AND random-mask empirical p < 0.05
```

Otherwise:

```text
BIOMARKER_CONFIRMATION: FAIL
```

## Full WM ICLR story

Only if BOTH pass:

```text
PHASE4_DECISION: WM_PREDICTION_AND_BIOMARKER_CONFIRMED
```

If prediction passes but biomarker fails:

```text
PHASE4_DECISION: WM_PREDICTION_ONLY
```

If biomarker passes but prediction fails:

```text
PHASE4_DECISION: WM_BIOMARKER_ONLY
```

If both fail:

```text
PHASE4_DECISION: WM_CONFIRMATION_FAILED
```

Do not change the thresholds after holdout access.

---

# 26. Important claim boundaries

Even if Phase 4 fully passes, do NOT claim:

```text
LLM semantic prior uniquely causes all predictive improvement.
```

The development shuffled comparison did not establish that.

Safe claim if full Phase 4 passes:

> The frozen prior-selected NCR expert-fusion procedure improves independent
> Working-Memory prediction over a strong no-prior multimodal baseline, while
> task-semantic development-derived biomarkers demonstrate held-out predictive
> faithfulness.

Safe semantic-prior biomarker strengthening if secondary control rankings also
support it:

> The matched semantic prior yields more reproducible development biomarkers
> than shuffled/random priors, and its frozen biomarker ranking is more
> faithful on the independent holdout.

FI remains a secondary negative boundary in the manuscript using development
evidence only.

Do not evaluate FI on the 98 in this Phase.

---

# 27. Output directory

Create:

```text
outputs/iclr/palf_phase4_wm_confirmation/
```

Required before holdout:

```text
RUNTIME_ESTIMATE.md
RUNTIME_PROGRESS.json

HOLDOUT_PREOPEN_AUDIT.json
BASELINE_AUDIT.json
PHASE2D_FIX_AUDIT.json

FINALIZATION_CV_RESULTS.csv
FINAL_MODEL_SELECTION.json

WM_BIOMARKER_RANKING_FROZEN.csv
WM_BIOMARKER_RANKING_FROZEN.sha256

CONTROL_BIOMARKER_RANKINGS_FROZEN.csv

PHASE4_WM_MODEL_FROZEN.json
PHASE4_WM_MODEL_FROZEN.sha256
READY_TO_OPEN_WM_HOLDOUT
```

Required after one-time holdout:

```text
HOLDOUT_OPEN_AUDIT.json
holdout_predictions.csv
holdout_model_metrics.csv

prediction_bootstrap.csv
prediction_inference.json

biomarker_mask_results.csv
biomarker_random10_distribution.csv
biomarker_random5_distribution.csv
biomarker_inference.json

development_biomarker_stability.csv

FINAL_CONFIRMATION_REPORT.md
VALIDATION_REPORT.json
COMPLETE
```

Also:

```text
models/
coefficients/
plots/
tests/
```

ZIP:

```text
outputs/iclr/palf_phase4_wm_confirmation.zip
```

---

# 28. Required plots

Generate publication-ready:

```text
plots/fig_phase4_holdout_prediction.pdf/png
plots/fig_phase4_delta_r_bootstrap.pdf/png
plots/fig_phase4_biomarker_randomization.pdf/png
plots/fig_phase4_top_vs_bottom_perturbation.pdf/png
plots/fig_phase4_development_stability.pdf/png
```

Do not make a plot that visually implies significance not supported by the
predeclared test.

---

# 29. Tests before unlock

Add:

```text
tests/test_phase4_wm_confirmation.py
```

Minimum functional tests:

1. holdout manifest count=98.
2. holdout SHA exact.
3. dev count=412.
4. intersection empty.
5. FI holdout label path is never loaded.
6. historical R0 audit exact.
7. valid Phase-2D-FIX code path only.
8. old buggy Phase-2D path rejected.
9. FC mask selected using development only.
10. SC mask selected using development only.
11. finalization CV uses only 412.
12. ratio=0 equals restricted Ridge.
13. alpha=0 equals R0.
14. final coefficient reconstruction <=1e-8.
15. final prediction reconstruction <=1e-8.
16. top ROI ranking uses final 412 coefficients only.
17. top ranking is frozen before holdout access.
18. random-mask lists are frozen/deterministic.
19. perturbation replaces edges with DEVELOPMENT means.
20. perturbation never retrains.
21. positive synthetic delta_RMSE correctly interpreted as degradation.
22. negative synthetic delta_RMSE correctly interpreted as improvement.
23. bootstrap samples paired triplets by subject.
24. holdout metrics are computed only after model-freeze hash exists.
25. no model/config write occurs after HOLDOUT_OPEN_TIMESTAMP.
26. no 98 label affects mask/hyperparameter/fusion selection.
27. only WM labels are used from holdout.
28. 1000 random top-10 comparator sets exactly.
29. 1000 random top-5 comparator sets exactly.
30. final decisions use the frozen thresholds exactly.

All must pass before holdout access except tests that explicitly require the
post-open immutable output; those should have a pre-open equivalent and final
post-open validation.

Run full repository suite before opening the holdout.

---

# 30. Final OpenCode report

Print:

## A. Runtime
- historical planning estimate
- measured machine-specific ETA
- actual total runtime
- per-stage runtime

## B. Pre-open integrity
- 412/98 counts
- holdout SHA
- intersection
- FI holdout label not loaded
- R0 audit
- valid Phase-2D-FIX audit

## C. Final frozen 412 model
- selected FC mask family/size
- selected SC mask family/size
- lambda_R values
- NCR ratios
- expert fusion v
- final alpha
- standalone expert development OOF r
- final development OOF r
- model-freeze SHA

## D. Frozen WM biomarker ranking
- top 5
- top 10
- coefficient/ranking SHA

## E. Holdout prediction
For:
- R0
- matched Ridge expert
- matched PS-NCR-EF
- cross
- shuffled
- random

Report Pearson/RMSE/MAE.

## F. Primary prediction inference
- observed delta_r
- 10k bootstrap CI
- one-sided lower bound
- bootstrap fraction <=0
- sensitivity test

## G. Biomarker faithfulness
- unmasked RMSE
- top5 delta_RMSE
- top10 delta_RMSE
- bottom5/bottom10
- random5 distribution
- random10 distribution
- empirical p-values
- bootstrap sensitivity

## H. Ranking controls
On the SAME frozen matched final predictor:
- matched top10 ranking delta_RMSE
- cross top10
- shuffled top10
- random-prior top10
- Ridge top10

## I. Development stability context
Clearly labelled DEVELOPMENT ONLY.

## J. Confirmation decisions
Print:

```text
PREDICTION_CONFIRMATION: PASS/FAIL
BIOMARKER_CONFIRMATION: PASS/FAIL
```

Then exactly one:

```text
PHASE4_DECISION: WM_PREDICTION_AND_BIOMARKER_CONFIRMED
```

```text
PHASE4_DECISION: WM_PREDICTION_ONLY
```

```text
PHASE4_DECISION: WM_BIOMARKER_ONLY
```

or:

```text
PHASE4_DECISION: WM_CONFIRMATION_FAILED
```

Finally:

```text
STATUS: PHASE4_WM_CONFIRMATION_COMPLETE
```

---

# 31. Irreversibility rule

This is a one-time independent confirmation.

Once:

```text
HOLDOUT_OPEN_TIMESTAMP
```

exists, the 98-subject outcome is no longer an untouched holdout.

Therefore after opening it:

- no model changes;
- no task changes;
- no new mask size;
- no new lambda;
- no new fusion grid;
- no new prior;
- no new primary biomarker endpoint;
- no switching from top10 to top5 because one looks better;
- no rerunning with a different seed to improve the result.

All results must be reported exactly as obtained.

A failed confirmation is a scientific result and must not trigger another
adaptive model search on these same 98 subjects.
