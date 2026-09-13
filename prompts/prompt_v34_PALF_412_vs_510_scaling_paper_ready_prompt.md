# PALF / ICLR 2027 — 412 vs 510 Sample-Size Scaling Study
# Separate exploratory analysis using the combined 510-subject pool
# Prediction + Biomarker Stability/Faithfulness + Detailed Ablations
# DO NOT overwrite or reinterpret the previous independent-holdout results

## Executive objective

Run a NEW, SEPARATE post-hoc sample-size scaling experiment comparing:

```text
REGIME A: original 412-subject development cohort only

REGIME B: same original 412-subject development cohort
          + the 98 previously held-out subjects as ADDITIONAL TRAINING DATA
          = 510-subject combined pool
```

The 98-subject cohort has already been opened in Phase 4.

Therefore:

```text
THIS 510 ANALYSIS IS NOT INDEPENDENT CONFIRMATION.
```

It is an exploratory/sample-size-scaling analysis.

Do not overwrite, delete, relabel, or conceal the previous Phase-4 result.

Keep every output under a NEW directory:

```text
outputs/iclr/palf_412_vs_510_scaling/
```

The purpose is to answer:

1. Does adding 98 additional processed subjects improve prediction on the
   SAME original 412 evaluation subjects?

2. Does adding those subjects improve biomarker stability and cross-fold
   faithfulness?

3. Does increased sample size change the relative advantage of:
   - R0;
   - PALF;
   - prior-selected Ridge expert fusion;
   - prior-selected NCR expert fusion;
   - matched/cross/shuffled/random prior controls?

4. If the 510-training regime is even nominally better than 412-only on a
   predefined comparison, generate a complete PAPER-READY scaling-results
   package, while labelling the result correctly as exploratory/post-hoc.

---

# 1. Scientific integrity / claim boundary

The 98 subjects are no longer an untouched holdout.

Forbidden claims:

```text
"independent validation"
"external validation"
"held-out confirmation"
"prospective confirmation"
```

for this 510 analysis.

Allowed phrasing:

```text
post-hoc combined-cohort analysis
sample-size scaling study
training-set expansion analysis
exploratory 510-subject analysis
```

Do not claim that a higher 510 result invalidates the previous 98-subject
negative confirmation.

A possible interpretation, if supported, is narrower:

> Increasing the training sample from the 412-only regime to a 510-subject
> combined training pool improves cross-validated performance on the same
> original evaluation subjects, suggesting sample-size sensitivity.

Do not silently report only the 510 number.

Every paper-ready table/plot MUST show:
- 412-only result;
- 510-training result;
- their difference;
- uncertainty / consistency.

---

# 2. Core design: paired evaluation on the SAME original 412 subjects

This is the PRIMARY scaling comparison.

We want an apples-to-apples test where the evaluation subjects are IDENTICAL.

Let:

```text
D412 = original 412 subjects
D98  = the 98 additional subjects
```

Use only the original 412 subjects as outer-test subjects.

For each outer fold of D412:

```text
T412 = original-412 training portion
V412 = original-412 held-out test portion
```

Run two training regimes on the SAME `V412`.

## Regime A — 412-only

Train/tune on:

```text
T_A = T412
```

Evaluate on:

```text
V412
```

## Regime B — +98 training augmentation

Train/tune on:

```text
T_B = T412 ∪ D98
```

Evaluate on the SAME:

```text
V412
```

Thus every original 412 subject receives:
- an OOF prediction from the 412-only regime;
- an OOF prediction from the +98 regime;

on exactly the same outer fold.

This is the PRIMARY comparison.

It isolates the effect of adding 98 training subjects far better than comparing
two unrelated CV scores on different evaluation populations.

---

# 3. No leakage from V412

For BOTH regimes:

- all standardization fit only on current training scope;
- all hyperparameters selected only inside current training scope;
- all fusion weights selected only from training OOF predictions;
- all masks selected from training data/prior only;
- no `V412` labels enter selection.

For Regime B, D98 are ordinary training subjects.

They may participate in:
- inner CV;
- hyperparameter selection;
- final training.

They are NEVER part of the primary outer evaluation.

---

# 4. Optional secondary 510-wide CV

After the primary paired experiment is complete, run a SECONDARY descriptive
5-fold nested CV over all 510 subjects.

Label this:

```text
SECONDARY_FULL510_CV
```

This provides a descriptive absolute-performance estimate on the combined
population.

Do NOT use the full-510 CV to choose models for the primary paired result.

Do NOT compare its raw Pearson directly with historical 412 CV and call the
difference causal, because evaluation populations differ.

Primary sample-size conclusions come from the paired D412 test subjects.

---

# 5. Exact subjects / manifests

Use and save:

```text
data_splits/scaling_D412.txt
data_splits/scaling_D98.txt
data_splits/scaling_D510.txt
```

Hard assertions:

```text
len(D412) = 412
len(D98)  = 98
len(D510) = 510

D412 ∩ D98 = empty
D510 = D412 ∪ D98
```

Save SHA256 for all three manifests.

The historical 98 canonical SHA must remain:

```text
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

---

# 6. Targets

Run BOTH:

```text
Working Memory (WM)
Fluid Intelligence (FI)
```

using the exact target definitions and subject ordering used in the corrected
project pipeline.

Because this is now a combined exploratory analysis, both target labels may be
used.

Record missingness.

Do not drop subjects based on observed prediction performance.

---

# 7. Correct baseline R0

Use ONLY the validated corrected same-solver baseline:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

Historical audit must reproduce approximately:

```text
WM r    = 0.263515
WM RMSE = 11.292921

FI r    = 0.370917
FI RMSE = 4.566689
```

Hard tolerance:

```text
Pearson <= 5e-4
RMSE    <= 0.05
```

If audit fails, stop.

Never substitute concatenated Ridge.

---

# 8. Models / ablations to run

Use the SAME model definitions under Regime A and Regime B.

## A0 — R0

Corrected same-solver no-prior baseline.

## A1 — Full PALF

Use the valid corrected same-solver PALF:
- matched WM/FI ROI prior;
- same generalized FC solver;
- same SC branch;
- same nested fusion;
- prior anisotropy + network penalty.

## A2 — PALF anisotropy-only

Same solver, matched prior, no line-graph term.

## A3 — PALF network-only

Same solver, identity anisotropy, matched line-graph prior.

These recover the core PALF ablation.

---

## B0 — Matched prior-selected Ridge Expert Fusion

Valid Phase-2D-FIX expert family with:

```text
NCR ratio = 0
```

Same mask/grid/fusion logic.

## B1 — Matched PS-NCR-EF

Valid Phase-2D-FIX matched prior-selected NCR expert fusion.

This is the main WM expert model.

## B2 — Cross-task PS-NCR-EF

Same architecture/capacity, cross-task prior.

## B3 — Shuffled PS-NCR-EF

Same architecture/capacity, frozen shuffled prior.

## B4 — Random PS-NCR-EF

Same architecture/capacity, frozen random prior.

Use the exact corrected Phase-2D-FIX implementation and prior/control
definitions.

Do not use the buggy original Phase 2D.

---

# 9. Fresh scaling-study CV seeds

Use exactly:

```text
SCALING_OUTER_SEEDS = [7171, 7272, 7373, 7474, 7575]
outer_folds = 5
```

For each seed:
- create outer folds ONLY over D412 for the PRIMARY experiment;
- use the same D412 folds for Regime A and Regime B.

Use:

```text
inner_folds = 3
```

inside each training regime.

Do not change seeds after results.

---

# 10. Primary paired sample-size comparison

For every:
- target;
- model;
- outer seed;
- outer fold;

store:

```text
subject IDs in V412
pred_412only
pred_plus98
y_true
```

Concatenate OOF predictions over D412.

Report for each model/target:

```text
Pearson_412
Pearson_plus98
delta_r = plus98 - 412

RMSE_412
RMSE_plus98
delta_RMSE = plus98 - 412

MAE_412
MAE_plus98
delta_MAE = plus98 - 412
```

At seed level:
- 5 seed means;
- median;
- positive seeds / 5.

---

# 11. Uncertainty for 412 vs +98

Because both regimes predict the SAME D412 subjects, use paired resampling.

For each target/model:

```text
10,000 paired subject bootstrap replicates
```

resampling tuples:

```text
(y, pred_412only, pred_plus98)
```

Seed:

```text
SCALING_BOOTSTRAP_SEED = 9401
```

Report:
- observed delta_r;
- bootstrap mean;
- 95% percentile CI (2.5,97.5);
- fraction delta_r <= 0;
- delta_RMSE CI;
- delta_MAE CI.

Also report Fisher-z difference descriptively.

Do NOT call nominal positive difference significant unless CI supports it.

---

# 12. Improvement tiers

Use these labels.

## NOMINAL_SCALE_GAIN

If:

```text
delta_r > 0
```

even if tiny.

This satisfies the user's requested trigger for generating paper-ready assets.

But paper-ready text MUST call it:

```text
nominal exploratory improvement
```

unless stronger criteria pass.

## CONSISTENT_SCALE_GAIN

If:

```text
delta_r >= +0.005
AND positive seed deltas >= 4/5
```

## ROBUST_SCALE_GAIN

If:

```text
delta_r >= +0.005
AND bootstrap 95% CI lower bound > 0
```

Never use "significant" for NOMINAL_SCALE_GAIN alone.

---

# 13. Method-effect analysis within each sample-size regime

A larger sample can improve all models.

Therefore also report, separately for 412-only and +98:

## PALF effect

```text
A1 - A0
A2 - A0
A3 - A0
```

## NCR effect

```text
B1 - B0
```

## Prior specificity

```text
B1 - B2
B1 - B3
B1 - B4
```

## Expert-vs-R0

```text
B1 - A0
```

This prevents a sample-size gain from being misrepresented as a method gain.

---

# 14. Interaction: does more data help the proposed method disproportionately?

For each target define:

```text
interaction_B1_R0 =
(B1_plus98 - A0_plus98)
-
(B1_412 - A0_412)
```

Similarly:

```text
interaction_PALF_R0 =
(A1_plus98 - A0_plus98)
-
(A1_412 - A0_412)
```

Positive interaction suggests the method benefits disproportionately from the
larger training set.

Use paired seed-level reporting.

This is exploratory.

---

# 15. Learning-curve control on the SAME outer test subjects

For B1 and A0 only, add a training-size learning curve.

For each primary outer fold:

Full augmented training pool:

```text
T_B = T412 ∪ D98
```

Evaluate on fixed `V412`.

Train on nested subsets of T_B of sizes approximately:

```text
n_train = [250, 300, 350, len(T412), len(T_B)]
```

If `len(T412)` varies slightly by fold, record exact sizes.

Subset construction:
- deterministic;
- stratification not required for continuous target, but preserve subject IDs;
- use 3 fixed subset seeds:

```text
[9511, 9512, 9513]
```

For each n:
- select/tune model only inside that subset;
- evaluate on same V412.

Plot Pearson versus training n.

This directly tests whether apparent gain follows sample size rather than cohort
identity.

Do not use the learning curve to retune the model.

---

# 16. Biomarker extraction

For B0/B1/B2/B3/B4, export valid primal expert coefficients.

For PALF, export coefficients only if validated primal reconstruction passes.

For every outer fit, compute:
- FC edge map;
- SC edge map;
- multimodal ROI importance.

For B1:

```text
ROI score =
sum_j |alpha * v_FC * beta_FC_ij|
+
sum_j |alpha * (1-v_FC) * beta_SC_ij|
```

If final alpha=0:
- mark biomarker `ABSTAINED`;
- exclude that fit from biomarker stability calculations.

Do not treat an all-zero map's tie order as stability.

---

# 17. Biomarker stability: 412 vs +98

For each target/model/regime compute across valid outer fits:

```text
FC abs-edge Spearman
SC abs-edge Spearman

FC top-100 edge Jaccard
SC top-100 edge Jaccard

FC top-10 ROI Jaccard
SC top-10 ROI Jaccard

multimodal top-10 ROI Jaccard
multimodal top-20 ROI Jaccard

sign consistency
```

Report:

```text
stability_plus98 - stability_412
```

The main question:

> Does additional training data improve reproducibility of the learned
> biomarkers?

---

# 18. Cross-fold biomarker faithfulness

This is EXPLORATORY cross-validation faithfulness, not independent holdout
confirmation.

For every primary D412 outer fold:

1. derive biomarker ranking from TRAINING data only;
2. evaluate perturbation on V412;
3. perturb:
   - top 5 ROIs;
   - top 10 ROIs;
   - bottom 5;
   - bottom 10;
   - 100 deterministic random 5-ROI sets;
   - 100 deterministic random 10-ROI sets;
4. do not retrain.

For Regime A ranking/model:
- training uses T412.

For Regime B ranking/model:
- training uses T412 ∪ D98.

Evaluation is SAME V412.

Replace masked edges with the corresponding CURRENT TRAINING-REGIME mean:
- Regime A: T412 mean;
- Regime B: (T412 ∪ D98) mean.

Define:

```text
delta_RMSE =
RMSE_masked - RMSE_unmasked
```

Positive = faithful.

For each model/target/regime report:

```text
top10 delta_RMSE
random10 mean
top10 - random10
bottom10
```

Then compare:

```text
(top10-random10)_plus98
-
(top10-random10)_412
```

This tests whether more data improves biomarker faithfulness.

---

# 19. Biomarker paper-ready success categories

## NOMINAL_BIOMARKER_STABILITY_GAIN

If any primary stability metric improves numerically under +98.

## CONSISTENT_BIOMARKER_GAIN

For a target/model if:
- top10 ROI Jaccard improves;
- edge Spearman improves;
- top10-minus-random faithfulness improves.

## ROBUST_BIOMARKER_GAIN

If:
- above three improve;
- at least 4/5 seed-level faithfulness contrasts are positive.

Do not label a mere stability increase as "faithfulness".

---

# 20. Detailed ablations required

Create a dedicated:

```text
ABLATION_RESULTS.md
```

and CSVs covering:

## A. Sample-size
412-only vs +98.

## B. PALF components
R0 / anisotropy / network / full.

## C. NCR contribution
matched Ridge expert vs matched NCR expert.

## D. Prior identity
matched / cross / shuffled / random.

## E. Modalities
FC expert only / SC expert only / both
for B1.

## F. Fusion
expert alone / R0 alone / fused.

## G. Mask family
direct top-K vs ROI-incident.

## H. Training-size learning curve
250/300/350/T412/T412+98.

For every ablation report:
- Pearson;
- RMSE;
- MAE;
- seed consistency;
- runtime.

Do not add new hyperparameter values.

---

# 21. Runtime / ETA requirements

Before full execution:

1. benchmark:
   - one R0 fold;
   - one PALF fold;
   - one Phase-2D-FIX expert fold;
2. estimate total workload.

Create:

```text
RUNTIME_ESTIMATE.md
RUNTIME_PROGRESS.json
```

Print:

```text
SCALING_STUDY_ESTIMATED_TOTAL_TIME: <h:mm>
SCALING_STUDY_ESTIMATED_FINISH_FROM_START: <h:mm>
```

Planning estimate before benchmark:

```text
~4–8 hours
```

depending on cache reuse and hardware.

Update after each stage:

```text
Stage 0 integrity/audit
Stage 1 primary paired prediction
Stage 2 controls/ablations
Stage 3 learning curves
Stage 4 biomarker stability
Stage 5 cross-fold faithfulness
Stage 6 plots/tables/paper package
```

Report:
- elapsed;
- remaining ETA;
- percent complete.

At final report include actual per-stage times.

---

# 22. Separate output package

Create only:

```text
outputs/iclr/palf_412_vs_510_scaling/
```

Never overwrite:
- Phase 2 outputs;
- Phase 3 outputs;
- Phase 4 outputs;
- deadline-rescue outputs.

Required:

```text
COMPLETE

STUDY_STATUS.md
MANIFEST_AUDIT.json
BASELINE_AUDIT.json

RUNTIME_ESTIMATE.md
RUNTIME_PROGRESS.json
RUNTIME_FINAL.json

primary_subject_predictions.csv
primary_seed_metrics.csv
primary_model_metrics.csv
paired_bootstrap_results.csv

full510_secondary_metrics.csv

method_effects.csv
sample_size_interactions.csv
learning_curve.csv

biomarker_stability.csv
biomarker_faithfulness.csv

ABLATION_RESULTS.md
ablation_results.csv

plots/
paper_ready/
supplementary/
models/
coefficients/
tests/
```

ZIP:

```text
outputs/iclr/palf_412_vs_510_scaling.zip
```

---

# 23. Paper-ready trigger

If ANY target/model satisfies:

```text
NOMINAL_SCALE_GAIN
```

i.e.

```text
Pearson_plus98 > Pearson_412
```

then automatically generate the COMPLETE paper-ready package below.

IMPORTANT:

The package may emphasize the positive scaling result, but it MUST:
- show the 412 comparator;
- show delta;
- show uncertainty;
- label the analysis exploratory/post-hoc;
- never call the 510 result independent validation.

Do NOT silently omit comparisons where +98 was worse.

---

# 24. Paper-ready numerical package

Create:

```text
paper_ready/PAPER_READY_RESULTS.md
paper_ready/PAPER_READY_RESULTS.tex
paper_ready/PAPER_READY_TABLES.tex
paper_ready/PAPER_READY_CAPTIONS.md
paper_ready/PAPER_READY_CLAIMS.md
paper_ready/PAPER_READY_LIMITATIONS.md
```

For every positive nominal scaling result provide publication-ready prose:

```text
"Under the paired training-expansion protocol, increasing the training pool
from T412 to T412+98 changed Pearson r from X to Y
(Δr=..., paired bootstrap 95% CI ...)."
```

Use:
- exact n;
- exact metrics;
- exact CI.

Never write:
```text
"significantly improved"
```
unless robust criteria support it.

---

# 25. Main paper figures

If paper-ready trigger fires, create publication-quality PDF + PNG + SVG
where possible.

## Figure 1 — Main illustrative method/data-scaling diagram

File:

```text
paper_ready/fig_main_scaling_framework.pdf
paper_ready/fig_main_scaling_framework.png
paper_ready/fig_main_scaling_framework.svg
```

Illustrate:

```text
Original 412
   |
   +--> identical outer test folds V412
   |
   +--> Regime A training: T412
   |
98 additional subjects
   |
   +--> Regime B training: T412 + 98

Both predict SAME V412
       |
       v
R0 / PALF / PS-NCR-EF
       |
       +--> prediction
       +--> biomarker stability
       +--> cross-fold faithfulness
```

The diagram must visually make the paired evaluation design obvious.

Use a clean ICLR-style vector illustration.

No decorative clutter.

---

## Figure 2 — Main 412 vs +98 prediction figure

For WM and FI:
- 412-only;
- +98;
- delta;
- seed-level dots/lines;
- CI.

Prefer paired lines by seed.

---

## Figure 3 — Method effect across sample size

Show:
- R0;
- full PALF;
- matched Ridge EF;
- matched NCR EF;

under both sample-size regimes.

This figure answers:
```text
does more data improve the method specifically,
or merely all models?
```

---

## Figure 4 — Learning curve

Pearson vs effective training n for:
- R0;
- matched PS-NCR-EF;

WM and FI.

---

## Figure 5 — Biomarker stability scaling

412 vs +98:
- edge Spearman;
- top10 ROI Jaccard;
- sign consistency.

---

## Figure 6 — Cross-fold faithfulness scaling

412 vs +98:
- top10 delta_RMSE;
- random10 mean;
- top10-random contrast.

Remember:
positive delta_RMSE = degradation = faithful.

---

# 26. Supplementary illustrative diagrams

If paper-ready trigger fires, create:

## Supplementary Figure S1 — Full nested-CV data-flow diagram

Show:
- outer D412 fold;
- Regime A;
- Regime B;
- inner CV;
- no leakage;
- same V412.

## Supplementary Figure S2 — PALF ablation diagram

Show:
```text
R0
+ anisotropy
+ network
+ both
```

and where prior enters.

## Supplementary Figure S3 — PS-NCR-EF architecture diagram

Show:
```text
WM/FI prior
-> selected FC/SC subspace
-> restricted Ridge / NCR
-> FC-SC expert fusion
-> fusion with R0
```

## Supplementary Figure S4 — Biomarker extraction/faithfulness diagram

Show:
```text
training fold
-> coefficient map
-> ROI ranking
-> top/random/bottom masks
-> same held-out V412
-> delta_RMSE
```

## Supplementary Figure S5 — Sample-size learning-curve schematic

Show increasing training n and fixed evaluation fold.

## Supplementary Figure S6 — Prior-control diagram

Matched / cross / shuffled / random under identical architecture.

Save each as:
```text
pdf
png
svg
```

where supported.

---

# 27. Paper-ready tables

Create:

## Table 1 — Primary paired 412 vs +98

For both tasks/models:
- 412 r/RMSE/MAE;
- +98 r/RMSE/MAE;
- delta;
- CI;
- positive seeds.

## Table 2 — Method effect at each sample size

R0 / PALF / Ridge EF / NCR EF.

## Table 3 — Detailed PALF ablation

R0 / anisotropy / network / full.

## Table 4 — Prior controls

matched / cross / shuffled / random.

## Table 5 — Biomarker stability

412 vs +98.

## Table 6 — Cross-fold faithfulness

top/random/bottom.

## Table S1 — hyperparameters selected by fold

## Table S2 — runtime by model/regime

## Table S3 — learning-curve results

## Table S4 — full secondary 510-wide CV

---

# 28. Paper-ready wording modes

Create three wording tiers.

## Tier A — nominal only

If:
```text
delta_r > 0
```
but CI includes 0.

Use:
```text
"numerically higher"
"nominal improvement"
"exploratory scaling trend"
```

## Tier B — consistent

If:
```text
delta_r >= .005
4/5 seeds positive
```

Use:
```text
"consistent improvement across development splits"
```

## Tier C — robust

If paired bootstrap CI lower bound >0.

Use:
```text
"robust paired improvement in the exploratory scaling analysis"
```

Still do NOT call it independent validation.

---

# 29. Main-paper result block

If nominal trigger fires, generate a self-contained LaTeX section:

```text
paper_ready/section_sample_size_scaling.tex
```

Recommended subsections:

```text
\subsection{Does additional training data improve prediction?}
\subsection{Does the prior-aware advantage scale with sample size?}
\subsection{Biomarker stability under training-set expansion}
\subsection{Cross-fold biomarker faithfulness}
```

Include:
- table refs;
- figure refs;
- exact numbers;
- claim-safe wording.

---

# 30. Supplementary material package

Create:

```text
supplementary/SUPPLEMENTARY_METHODS.tex
supplementary/SUPPLEMENTARY_ABLATIONS.tex
supplementary/SUPPLEMENTARY_RESULTS.tex
supplementary/SUPPLEMENTARY_CAPTIONS.md
```

Include:
- manifests;
- exact CV seeds;
- exact fold construction;
- all hyperparameter grids;
- all selected configs;
- per-fold metrics;
- full ablations;
- learning curves;
- runtime;
- negative results within this scaling study.

Do not omit negative 412/+98 comparisons from the supplement.

---

# 31. Optional paper-ready combined-cohort narrative

If 510 is better ONLY in absolute R0 performance but not proposed-vs-R0:

Safe narrative:

> Additional subjects improve predictive estimation, but the prior-aware method
> does not gain a clear advantage over the strong no-prior baseline.

If 510 also increases the method-vs-R0 advantage:

Safe narrative:

> The prior-aware advantage grows with training-set size in the exploratory
> scaling analysis.

If biomarker stability improves but faithfulness does not:

Safe narrative:

> Larger training sets improve reproducibility, but not necessarily
> faithfulness.

If both prediction and faithfulness improve:

Safe narrative:

> The combined cohort suggests that both predictive utility and biomarker
> faithfulness are sample-size-sensitive.

Do not say:
```text
"the previous negative holdout was caused by small sample"
```
unless this analysis directly supports only a sample-size sensitivity
interpretation, and even then write:
```text
"consistent with sample-size sensitivity"
```
not causal certainty.

---

# 32. Tests

Add:

```text
tests/test_412_vs_510_scaling.py
```

Minimum functional tests:

1. D412 count=412.
2. D98 count=98.
3. D510 count=510.
4. sets disjoint/union exact.
5. historical 98 SHA exact.
6. primary outer test subjects come ONLY from D412.
7. Regime A train excludes D98.
8. Regime B train includes all D98.
9. Regime A/B use identical V412.
10. no V412 subject in either training regime.
11. inner preprocessing train-only.
12. R0 exact implementation.
13. same outer split seeds for A/B.
14. same model grids under A/B.
15. Phase-2D-FIX corrected module only.
16. buggy Phase-2D module rejected.
17. NCR ratio=0 equals Ridge.
18. alpha=0 equals R0.
19. OOF prediction arrays cover all D412 exactly once/seed.
20. paired bootstrap resamples same subject tuple.
21. learning-curve test subjects identical across n.
22. learning-curve subsets nested/deterministic.
23. biomarker ranking training-only.
24. perturbation does not retrain.
25. positive delta_RMSE sign interpreted correctly.
26. abstained maps excluded from stability.
27. paper-ready trigger activates for delta_r>0.
28. nominal wording does not say significant.
29. robust wording requires CI lower bound>0.
30. no output overwrites previous Phase 4.

Run targeted and full suite.

---

# 33. Final OpenCode report

Print:

## A. Runtime
- benchmark
- predicted ETA
- actual total
- per-stage times

## B. Manifests
- 412 / 98 / 510
- SHAs
- disjointness

## C. Baseline audit
- historical corrected R0
- PASS/FAIL

## D. Primary paired 412 vs +98 prediction

For WM and FI, each model:
- 412 r/RMSE/MAE
- +98 r/RMSE/MAE
- delta
- bootstrap CI
- positive seeds / 5
- gain tier

## E. Method-effect decomposition
- PALF-R0
- NCR-Ridge
- NCR-R0
- prior controls
for both sample sizes.

## F. Interaction
Does method advantage grow with +98?

## G. Learning curve
Full table.

## H. Biomarker stability
412 vs +98.

## I. Cross-fold faithfulness
412 vs +98.

## J. Detailed ablations
All A–H groups from Section 20.

## K. Paper-ready trigger
For each positive nominal result:
```text
PAPER_READY_TRIGGER: YES
```
and list generated assets.

If none:
```text
PAPER_READY_TRIGGER: NO
```

## L. Claim boundary
State explicitly:
```text
510 analysis is exploratory/post-hoc and not independent confirmation.
```

## M. Outputs
All paths.

Then:

```text
STATUS: SCALING_412_VS_510_COMPLETE
```

---

# 34. Final integrity rule

The purpose is to understand whether more training data changes the result.

It is acceptable to emphasize a genuine positive 510 scaling result.

It is NOT acceptable to:
- hide the 412 comparator;
- call 510 independent validation;
- erase the previous Phase-4 outcome;
- claim significance from a tiny positive delta without uncertainty;
- select only the model/task that looks best and omit the rest.

If 510 is only slightly better, generate the requested paper-ready package,
but label the result exactly as a nominal exploratory sample-size improvement.
