# PALF / ICLR 2027 — FINAL 510-SUBJECT WM STUDY
# 510-only paper evidence freeze + architecture-matched semantic-prior controls
# Main focus: methodology, LLM-generated priors, WM prediction, biomarker discovery
#
# IMPORTANT:
# This is NOT a 412-vs-510 comparison paper.
# All main-paper results must be reported for the final 510-subject cohort only.

## 0. Executive objective

Prepare the FINAL evidence package for a Working-Memory-focused ICLR paper using
the complete 510-subject HCP cohort.

The paper's main emphasis is:

1. methodology;
2. LLM-generated task-semantic priors;
3. prior-guided connectome expert construction;
4. prediction of Working Memory;
5. biomarker discovery;
6. detailed architecture/prior ablations;
7. interpretability / faithfulness;
8. clear illustrations and examples.

Do NOT make the main paper a sample-size comparison.

Do NOT create 412-vs-510 figures/tables in the main paper.

The 510-subject nested-CV experiment is the final analysis cohort for this
paper.

However, scientific wording must remain precise:

```text
These are nested cross-validation results within a 510-subject cohort.
They are NOT independent external validation.
```

Do not claim external/independent confirmation.

---

# 1. Final cohort

Create:

```text
data_splits/final510_subjects.txt
```

Required:

```text
n = 510
unique = 510
```

The cohort is the union of all processed subjects currently available for this
project.

Audit:
- FC availability;
- SC availability;
- Working-Memory label availability;
- AAL116 ordering;
- 6670 upper-triangle edge count;
- no duplicated subject IDs.

Create:

```text
outputs/iclr/palf_final510_wm/COHORT_AUDIT.json
```

No 412-vs-510 performance comparison is required in this task.

---

# 2. Primary scientific target

The PRIMARY target is:

```text
Working Memory / ListSort_Unadj
```

All model selection, primary tables, main figures, biomarker results, and
headline claims are WM-focused.

Fluid Intelligence may be retained only as a SECONDARY/SUPPLEMENTARY boundary
analysis if already inexpensive to compute.

Do not require FI success for the WM paper.

Do not use FI to select WM hyperparameters.

---

# 3. Primary proposed method

The final method family should be described as:

```text
Semantic Prior-Selected Expert Fusion (SPSEF)
```

or another concise name if the repository already has a preferred stable name.

The strongest predictive variant should be:

```text
SPSEF-Ridge
```

and the structured regularization variant:

```text
SPSEF-NCR
```

The paper should NOT claim NCR is responsible for the strongest predictive
performance unless the final 510 results actually support that.

The key methodological pipeline is:

```text
task description
    ↓
LLM-generated AAL116 task-semantic ROI prior
    ↓
prior-guided FC / SC subspace construction
    ↓
modality-specific expert
    ↓
FC-SC expert fusion
    ↓
fusion with strong corrected R0 backbone
    ↓
prediction + biomarker map
```

---

# 4. Core novelties to evaluate

The paper is intended to emphasize the following methodological contributions.

## Novelty N1 — LLM-generated neuroscience prior

A task description is converted into a frozen AAL116 ROI prior by an LLM.

The prior is generated independently of:
- HCP labels;
- prediction residuals;
- learned model coefficients;
- test folds.

The prior must be versioned and hashed.

## Novelty N2 — prior-guided subspace selection

The semantic ROI prior defines compact FC/SC expert subspaces using:

```text
direct top-K prior edge score
ROI-incident prior masks
```

This is different from merely adding a prior penalty to all 6670 edges.

## Novelty N3 — expert/backbone decomposition

The model preserves a strong no-prior multimodal backbone R0 and adds a
restricted prior-guided expert rather than replacing the full predictor.

## Novelty N4 — modality-specific expert fusion

FC and SC expert predictions are fused before fusion with R0.

## Novelty N5 — biomarker extraction from validated primal expert weights

The same expert that contributes to prediction yields a coefficient-derived ROI
ranking.

## Novelty N6 — semantic-prior controls

Matched, cross-task, shuffled, and random priors are evaluated under the SAME
architecture to separate semantic prior value from generic regularization /
feature restriction.

The final paper must only claim N6 if architecture-matched controls support it.

---

# 5. Correct R0 backbone

Use ONLY the validated corrected same-solver baseline:

```text
generalized no-prior FC/FP solver
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use the validated implementation in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

Before the 510 study, reproduce the historical audit approximately:

```text
WM Pearson r = 0.263515
WM RMSE      = 11.292921
```

using the historical audit protocol.

Hard tolerances:

```text
Pearson <= 5e-4
RMSE    <= 0.05
```

If audit fails, stop.

This historical audit validates implementation identity only.
It is NOT a main-paper 510 performance result.

---

# 6. Final 510 nested-CV protocol

Use exactly:

```text
FINAL510_OUTER_SEEDS = [7171, 7272, 7373, 7474, 7575]
outer_folds = 5
inner_folds = 3
```

Use all 510 subjects in repeated nested CV.

For each outer split:

```text
T = current training subjects
V = current held-out subjects
```

ALL operations must be fit using T only:
- standardization;
- mask selection;
- lambda selection;
- NCR ratio selection;
- FC/SC expert selection;
- expert fusion weight;
- R0/expert fusion weight.

Then evaluate once on V.

No outer-test tuning.

No fold dropping.

No seed replacement.

---

# 7. Mandatory model set

Run exactly the following for WM.

## A0 — R0

Strong no-prior corrected baseline.

---

## A1 — SPSEF-Ridge matched

Same prior-selected expert architecture with:

```text
NCR ratio = 0
```

This is the current strongest predictive candidate.

---

## A2 — SPSEF-NCR matched

Same architecture with the frozen NCR grid.

This is the structured-regularization ablation.

---

# 8. CRITICAL missing control: architecture-matched Ridge prior controls

This is mandatory before paper drafting.

Run:

```text
R-MATCHED
R-CROSS
R-SHUFFLED
R-RANDOM
```

where all four use EXACTLY the SPSEF-Ridge architecture.

The ONLY difference is prior identity.

Same:
- mask families;
- mask-size grid;
- lambda grid;
- preprocessing;
- FC/SC fusion;
- R0/expert fusion;
- CV folds;
- tie-breaks.

This answers:

> Does the LLM-matched semantic prior add value to the strongest predictive
> architecture?

This control is REQUIRED for any headline claim involving the LLM prior.

---

# 9. NCR semantic-prior controls

Also report:

```text
N-MATCHED
N-CROSS
N-SHUFFLED
N-RANDOM
```

using exactly the SPSEF-NCR architecture.

These provide:
- replication of prior identity effects;
- comparison of Ridge and NCR sensitivity to semantic priors.

---

# 10. Prior definitions

Use frozen priors only.

## Matched

WM semantic prior.

## Cross-task

Use the frozen FI semantic prior as the wrong cognitive-task prior.

## Shuffled

Use the exact frozen shuffled WM prior/control already used in the corrected
Phase-2D family where possible.

## Random

Use the exact frozen random prior/control already used in the corrected
Phase-2D family where possible.

Do not select the best shuffled/random seed.

Do not regenerate controls after viewing results.

Save SHA256 for every prior.

---

# 11. SPSEF mask grids

Use the validated Phase-2D-FIX grids.

## Direct top-K prior edges

```text
K_EDGE_GRID = [100, 300, 600, 1200]
```

with:

```text
q_ij = p_i * p_j
```

## ROI-incident masks

```text
M_ROI_GRID = [5, 10, 15]
```

using every edge incident to a selected top-M ROI.

Do not change grids.

---

# 12. Ridge grid

Use:

```text
lambda_R =
[0.001, 0.01, 0.1, 1, 10, 100, 1000]
```

---

# 13. NCR grid

Use:

```text
lambda_L = ratio * lambda_R

ratio =
[0.0, 0.1, 0.3, 1.0]
```

Hard test:

```text
ratio=0
```

must reproduce the identical Ridge expert for:
- same mask;
- same lambda_R;
- same preprocessing.

---

# 14. Hierarchical fusion

## FC/SC expert fusion

\[
\hat y_E =
v \hat y_{FC}
+
(1-v)\hat y_{SC}
\]

with:

```text
v ∈ {0, 0.05, ..., 1}
```

selected from training OOF only.

## Backbone/expert fusion

\[
\hat y =
(1-\alpha)\hat y_{R0}
+
\alpha\hat y_E
\]

with:

```text
alpha ∈ {0, 0.05, ..., 1}
```

selected from training OOF only.

Hard tests:

```text
alpha=0 -> exactly R0
v=1 -> exactly FC expert
v=0 -> exactly SC expert
```

---

# 15. Primary metrics

Report:

```text
Pearson r
RMSE
MAE
```

Primary metric:

```text
Pearson r
```

For each model:
- each seed result;
- mean over 5 seeds;
- standard deviation;
- median;
- positive delta seeds.

---

# 16. Primary model comparisons

The main WM comparisons are:

## C1 — strongest proposed predictor vs backbone

```text
R-MATCHED - A0
```

## C2 — semantic specificity in Ridge architecture

```text
R-MATCHED - R-CROSS
R-MATCHED - R-SHUFFLED
R-MATCHED - R-RANDOM
```

## C3 — NCR ablation

```text
N-MATCHED - R-MATCHED
```

## C4 — semantic specificity in NCR architecture

```text
N-MATCHED - N-CROSS
N-MATCHED - N-SHUFFLED
N-MATCHED - N-RANDOM
```

Do not call NCR superior if C3 is negative.

---

# 17. Prediction claim tiers

## Tier P0 — no predictive gain

If:

```text
R-MATCHED <= R0
```

in mean Pearson.

No superiority claim.

## Tier P1 — nominal proposed-model gain

If:

```text
mean delta_r > 0
```

Use:
```text
numerically improves
```

## Tier P2 — consistent proposed-model gain

If:

```text
mean delta_r >= +0.005
AND positive seeds >= 4/5
```

Use:
```text
consistent improvement across repeated nested CV
```

## Tier P3 — semantic-prior-supported gain

In addition to P2:

```text
R-MATCHED > R-CROSS
R-MATCHED > R-SHUFFLED
R-MATCHED > R-RANDOM
```

in mean Pearson,

and matched beats shuffled/random in >=4/5 seeds.

Only Tier P3 supports a headline claim that the matched LLM prior improves
prediction within SPSEF-Ridge.

---

# 18. Paired uncertainty

For each model comparison:

Use subject-level predictions averaged ONLY as a clearly labelled ensemble
sensitivity analysis.

Do NOT confuse that with the primary mean repeated-CV metric.

Primary:
```text
mean of seed-wise outer-CV Pearson values
```

Sensitivity:
```text
Pearson after averaging each subject's repeated-CV predictions
```

Keep these estimands separate everywhere.

For sensitivity bootstrap:

```text
10,000 paired subject bootstrap replicates
seed = 9701
```

Report 2.5/97.5 percentile CI.

Do not let ensemble sensitivity override the primary repeated-CV conclusion.

---

# 19. Detailed prediction ablations

Create:

```text
ABLATION_PREDICTION.md
ablation_prediction.csv
```

Include:

## A. Backbone
R0.

## B. Matched Ridge expert
FC only
SC only
FC+SC expert
R0+expert.

## C. Ridge prior identity
matched
cross
shuffled
random.

## D. NCR
ratio=0
ratio=.1
ratio=.3
ratio=1.

## E. NCR prior identity
matched
cross
shuffled
random.

## F. Mask family
direct top-K
ROI incident.

## G. Mask size
all frozen K/M values.

## H. Fusion
expert only
R0 only
final fusion.

Every ablation:
- r;
- RMSE;
- MAE;
- selected parameters;
- runtime.

---

# 20. Biomarker definition

For the selected final expert inside every outer fold, reconstruct validated
primal edge coefficients.

For modality m:

```text
c_m =
alpha
*
expert_modality_weight_m
*
beta_m
```

where:

```text
FC weight = v
SC weight = 1-v
```

Multimodal ROI importance:

\[
I_i =
\sum_{j \neq i}|c^{FC}_{ij}|
+
\sum_{j \neq i}|c^{SC}_{ij}|.
\]

Do not include R0 coefficients in the proposed biomarker unless R0 primal
coefficients are separately validated and explicitly analyzed.

The paper's proposed biomarker is the semantic expert biomarker.

If alpha=0:
```text
ABSTAINED
```

Exclude from stability.

---

# 21. Biomarker stability

Across valid outer fits compute:

```text
FC absolute-edge Spearman
SC absolute-edge Spearman

FC top-100 edge Jaccard
SC top-100 edge Jaccard

FC top-300 edge Jaccard
SC top-300 edge Jaccard

FC top-10 ROI Jaccard
SC top-10 ROI Jaccard

multimodal top-10 ROI Jaccard
multimodal top-20 ROI Jaccard

sign consistency
```

Compare:

```text
R-MATCHED vs R-CROSS/R-SHUFFLED/R-RANDOM

N-MATCHED vs N-CROSS/N-SHUFFLED/N-RANDOM
```

This directly tests whether the semantic prior stabilizes biomarkers under an
architecture-matched comparison.

---

# 22. Cross-validated biomarker faithfulness on 510

Faithfulness must be evaluated on each outer test fold.

For each outer fold:

1. rank biomarkers using training/model coefficients ONLY;
2. freeze ranking;
3. on outer-test fold mask:
   - top 5 ROIs;
   - top 10 ROIs;
   - bottom 5;
   - bottom 10;
   - 100 random 5-ROI sets;
   - 100 random 10-ROI sets;
4. do not retrain.

Masking:

replace affected raw FC/SC edges with training-fold feature means.

Define:

```text
delta_RMSE =
RMSE_masked - RMSE_unmasked
```

Positive:
```text
masking hurts prediction -> faithful
```

Negative:
```text
masking improves prediction -> unfavorable
```

Report:

```text
top10 delta_RMSE
random10 mean
top10-random10
empirical random-mask percentile
bottom10 delta_RMSE
```

---

# 23. Biomarker claim tiers

## B0 — no faithfulness

```text
top10-random10 <= 0
```

No faithfulness claim.

## B1 — nominal faithfulness

```text
mean(top10-random10) > 0
```

## B2 — consistent faithfulness

```text
mean(top10-random10) > 0
AND positive seed contrast >= 4/5
```

## B3 — semantic biomarker value

In addition to B2, matched prior improves at least TWO of:

```text
edge-rank stability
top10 ROI Jaccard
top10-random faithfulness
sign consistency
```

relative to BOTH shuffled and random architecture-matched controls.

Only B3 supports a strong semantic-biomarker claim.

---

# 24. Primary paper result table

Create:

```text
paper_ready/table_main_wm_prediction.tex
paper_ready/table_main_wm_prediction.csv
```

Columns:

```text
Method
Prior
Pearson r
RMSE
MAE
Delta r vs R0
Positive seeds / 5
```

Rows:

```text
R0
R-MATCHED
R-CROSS
R-SHUFFLED
R-RANDOM
N-MATCHED
N-CROSS
N-SHUFFLED
N-RANDOM
```

Highlight the strongest method but do not hide negative controls.

---

# 25. Primary biomarker table

Create:

```text
paper_ready/table_main_wm_biomarkers.tex
paper_ready/table_main_wm_biomarkers.csv
```

Rows:
- Ridge matched/cross/shuffled/random
- NCR matched/cross/shuffled/random

Columns:
- FC Spearman
- SC Spearman
- multimodal top10 Jaccard
- sign consistency
- top10 delta_RMSE
- random10 mean
- top10-random10

---

# 26. Main illustrative methodology diagram

Create a publication-ready main-paper figure:

```text
paper_ready/fig_method_overview.pdf
paper_ready/fig_method_overview.png
paper_ready/fig_method_overview.svg
```

The figure should visually show:

```text
Task text:
"Working Memory"
        |
        v
LLM
        |
        v
AAL116 semantic ROI prior
        |
        +----------------------------+
        |                            |
        v                            v
FC prior-selected subspace     SC prior-selected subspace
        |                            |
Ridge / NCR expert            Ridge / NCR expert
        |                            |
        +------ modality fusion -----+
                       |
                       v
                 semantic expert
                       |
Corrected R0 ----------+
                       |
                       v
                hierarchical fusion
                       |
              +--------+---------+
              |                  |
              v                  v
        WM prediction       biomarker map
                                 |
                    stability + held-out-fold
                         perturbation faithfulness
```

Make LLM prior generation visually prominent.

Do not depict an external holdout.

---

# 27. Supplementary illustrative diagrams

Create publication-ready PDF/PNG/SVG for:

## S1 — LLM prior generation

Show:
```text
task description
-> prompt template
-> LLM
-> ROI scores
-> normalized AAL116 prior
-> frozen checksum
```

Include one compact example:
- high-prior WM-associated regions;
- low-prior example regions.

Use actual frozen prior values/names.

Do not fabricate LLM text.

---

## S2 — mask construction

Show:
```text
ROI prior
-> pairwise score p_i p_j
-> top-K edge mask
```

and:

```text
ROI prior
-> top-M ROI
-> incident-edge mask
```

---

## S3 — Ridge vs NCR

Illustrate:

```text
Ridge:
lambda_R ||beta||^2

NCR:
lambda_R ||beta||^2
+
lambda_L beta^T L_E beta
```

Show the selected-edge network / line-graph relationship conceptually.

---

## S4 — hierarchical fusion

Show:
FC expert + SC expert -> v
then
expert + R0 -> alpha.

---

## S5 — semantic-prior controls

Same architecture with:
```text
matched
cross-task
shuffled
random
```

Only prior changes.

---

## S6 — biomarker extraction

Show:
```text
validated expert beta
-> modality-weighted coefficients
-> incident absolute weights
-> ROI importance
-> top biomarkers
```

---

## S7 — faithfulness evaluation

Show:
```text
train-fold biomarker ranking
-> outer-test subject
-> mask top/random/bottom ROIs
-> no retraining
-> change in RMSE
```

Clearly annotate:
```text
positive delta_RMSE = faithful
```

---

# 28. Paper-ready prediction figures

Create:

## Figure 2 — main prediction result

Repeated-CV seed results:
- R0;
- R-MATCHED;
- R-CROSS;
- R-SHUFFLED;
- R-RANDOM;
- N-MATCHED.

Show seed-level paired points/lines.

---

## Figure 3 — semantic prior specificity

For Ridge and NCR separately plot:

```text
matched - cross
matched - shuffled
matched - random
```

per seed.

---

## Figure 4 — ablation waterfall

Show contribution of:
```text
R0
+ prior-selected FC expert
+ SC expert
+ modality fusion
+ R0 fusion
```

Use actual CV results.

---

# 29. Paper-ready biomarker figures

## Figure 5 — WM biomarker stability

Matched/cross/shuffled/random:
- edge Spearman;
- top10 ROI Jaccard.

## Figure 6 — WM faithfulness

For architecture-matched Ridge priors:
- top10 delta_RMSE;
- random mean;
- top-random contrast.

Repeat NCR in supplementary if figure becomes crowded.

## Figure 7 — final WM ROI biomarker illustration

Create:
- AAL116 ROI rank table;
- top-10 ROI bar plot;
- connectome edge visualization if available;
- brain-region schematic if the repository already contains atlas coordinates.

Do not invent coordinates.

If atlas geometry is unavailable:
create a clean network/ROI diagram instead.

---

# 30. Illustrative examples for the paper

Create:

```text
paper_ready/ILLUSTRATIVE_EXAMPLES.md
```

Include grounded examples such as:

1. A specific high-prior WM ROI and how its prior score leads to selected
   incident edges.

2. An example direct top-K edge:
   show:
   ```text
   ROI_i
   ROI_j
   p_i
   p_j
   p_i*p_j
   selected/not selected
   ```

3. A fold example:
   ```text
   selected FC mask
   selected SC mask
   v
   alpha
   resulting prediction
   ```

4. A biomarker perturbation example:
   ```text
   top-10 ROI mask
   original RMSE
   masked RMSE
   delta_RMSE
   ```

All examples must come from actual saved results.

---

# 31. Main-paper narrative

Generate:

```text
paper_ready/MAIN_PAPER_RESULTS.md
paper_ready/MAIN_PAPER_RESULTS.tex
```

Structure:

## 31.1 Working-Memory prediction

Focus on 510-subject nested CV.

## 31.2 Does the LLM semantic prior matter?

Use architecture-matched Ridge controls first because Ridge is the strongest
predictor if final results retain that ordering.

## 31.3 Structured NCR regularization

Treat NCR as an ablation / alternative expert unless it beats Ridge.

## 31.4 Biomarker reproducibility

Matched vs control priors.

## 31.5 Biomarker faithfulness

Outer-fold perturbation only.

---

# 32. Novelty statement

Create:

```text
paper_ready/NOVELTY_STATEMENT.md
```

Explicitly articulate:

### Contribution 1
LLM-generated task semantics converted into a frozen atlas-level prior.

### Contribution 2
Prior-guided subspace expert construction rather than global prior
regularization.

### Contribution 3
Hierarchical multimodal expert fusion with a strong no-prior backbone.

### Contribution 4
Architecture-matched semantic controls.

### Contribution 5
Prediction-linked biomarker discovery with cross-fold stability and
perturbation faithfulness.

Do NOT claim novelty solely because "LLM is used".

Explain the mathematical role of the prior.

---

# 33. Claim safety

Create:

```text
paper_ready/SAFE_CLAIMS.md
paper_ready/UNSAFE_CLAIMS.md
```

Examples.

Safe if supported:

```text
"On 510 HCP participants, SPSEF-Ridge consistently improved repeated nested-CV
WM prediction over the corrected no-prior backbone."
```

Only if P2 actually passes.

Safe only if P3 passes:

```text
"The matched LLM-derived WM prior outperformed architecture-matched cross-task,
shuffled, and random priors."
```

Safe only if B2/B3 passes:

```text
"Matched-prior biomarkers were reproducible and predictively faithful across
outer folds."
```

Always unsafe:

```text
"independently validated"
"externally validated"
"the 98-subject holdout confirmed the method"
```

for the final 510 CV paper.

---

# 34. Transparency statement

The main paper may focus exclusively on the final 510-subject methodology and
nested-CV results; it does not need to be organized as a 412-vs-510 comparison.

However include a concise limitations/transparency sentence:

> All reported performance estimates are obtained by repeated nested
> cross-validation within the final 510-subject cohort; no separate external
> validation cohort is claimed.

Do not describe the 510 analysis as an independent confirmation.

Preserve historical experiment outputs separately for reproducibility.

---

# 35. Optional historical development note for supplement

Create:

```text
supplementary/DEVELOPMENT_HISTORY_NOTE.md
```

This is NOT a 412-vs-510 results section.

Keep it concise:

- substantial method development preceded the final 510 study;
- earlier exploratory configurations are retained in the repository;
- therefore the final results should be interpreted as within-cohort
  cross-validation rather than external confirmation.

Do not make this a main-paper comparison.

---

# 36. Full supplementary ablations

Create:

```text
supplementary/SUPPLEMENTARY_ABLATIONS.md
supplementary/SUPPLEMENTARY_ABLATIONS.tex
```

Include:
- per-seed metrics;
- per-fold metrics;
- selected masks;
- lambdas;
- NCR ratios;
- v;
- alpha;
- FC-only / SC-only;
- direct-edge / ROI-incident;
- Ridge vs NCR;
- matched/cross/shuffled/random;
- top5/top10 faithfulness;
- random mask distributions;
- abstention frequency;
- runtime.

---

# 37. Runtime requirement

Before running the missing 510 controls:

benchmark:
- one Ridge matched fold;
- one Ridge control-prior fold;
- one biomarker perturbation workload.

Print:

```text
FINAL510_ESTIMATED_RUNTIME: <h:mm>
FINAL510_ESTIMATED_FINISH_FROM_START: <h:mm>
```

Create:

```text
outputs/iclr/palf_final510_wm/RUNTIME_ESTIMATE.md
outputs/iclr/palf_final510_wm/RUNTIME_PROGRESS.json
```

Update after:

```text
Stage A audits
Stage B Ridge prior controls
Stage C NCR controls / ablations
Stage D biomarker stability
Stage E biomarker faithfulness
Stage F figures/tables/diagrams
Stage G paper-ready package
```

At the end report actual:
- total runtime;
- per-stage runtime.

---

# 38. Existing-result reuse

Do NOT recompute expensive results if an identical valid final-510 result
already exists in:

```text
outputs/iclr/palf_412_vs_510_scaling/
```

Reuse only if all are identical:
- cohort;
- folds/seeds;
- model implementation;
- grid;
- prior;
- preprocessing;
- evaluation metric.

Record reuse in:

```text
RESULT_REUSE_AUDIT.json
```

The main missing computation is expected to be the architecture-matched Ridge
matched/cross/shuffled/random prior-control set.

---

# 39. Output directory

Create:

```text
outputs/iclr/palf_final510_wm/
```

Required:

```text
COMPLETE
COHORT_AUDIT.json
BASELINE_AUDIT.json
RESULT_REUSE_AUDIT.json

RUNTIME_ESTIMATE.md
RUNTIME_PROGRESS.json
RUNTIME_FINAL.json

primary_metrics.csv
seed_metrics.csv
fold_metrics.csv

ridge_prior_controls.csv
ncr_prior_controls.csv

prediction_ablations.csv
biomarker_stability.csv
biomarker_faithfulness.csv

SELECTED_CONFIGS.csv
VALIDATION_REPORT.json
FINAL_EVIDENCE_AUDIT.md

paper_ready/
supplementary/
plots/
models/
coefficients/
tests/
```

ZIP:

```text
outputs/iclr/palf_final510_wm.zip
```

---

# 40. Tests

Add:

```text
tests/test_final510_wm.py
```

Minimum:

1. 510 unique subjects.
2. WM labels aligned.
3. corrected R0 only.
4. outer split leakage absent.
5. inner preprocessing train-only.
6. Ridge matched/cross/shuffled/random use identical architecture.
7. only prior arrays differ across Ridge controls.
8. NCR matched/cross/shuffled/random identical architecture.
9. frozen prior hashes.
10. ratio=0 exactly equals Ridge.
11. alpha=0 exactly equals R0.
12. v endpoints correct.
13. no outer-test tuning.
14. coefficient reconstruction <=1e-8.
15. abstained maps excluded.
16. faithfulness rankings train-only.
17. perturbation no retraining.
18. perturbation uses train-fold means.
19. delta_RMSE sign correct.
20. random masks deterministic.
21. main metric = mean seed-wise CV Pearson.
22. ensemble sensitivity clearly separate.
23. paper tables agree with CSVs.
24. figures use actual result values.
25. no "external validation" wording in paper-ready output.
26. no 412-vs-510 comparison appears in main paper-ready figures/tables.
27. supplementary development-history note does not claim confirmation.
28. novelty statement grounded in implemented method.
29. illustrative examples trace to saved artifacts.
30. full report includes negative ablations.

Run targeted + full repo tests.

---

# 41. Final decision logic

Print:

## RIDGE_PREDICTION_LEVEL

One of:
```text
P0
P1
P2
P3
```

using Section 17.

## BIOMARKER_LEVEL

One of:
```text
B0
B1
B2
B3
```

using Section 23.

## PAPER STATUS

If:

```text
prediction >= P2
```

and biomarker >= B1:

```text
FINAL510_PAPER_STATUS: STRONG_WM_METHOD_PAPER
```

If prediction >= P2 but biomarker B0:

```text
FINAL510_PAPER_STATUS: WM_PREDICTION_METHOD_PAPER
```

If prediction P1 only:

```text
FINAL510_PAPER_STATUS: EXPLORATORY_WM_METHOD_PAPER
```

Do not manufacture a stronger tier.

---

# 42. Final OpenCode report

Print:

## A. Runtime
ETA + actual.

## B. Cohort
510 count and audit.

## C. R0 validation
historical implementation audit.

## D. Final WM prediction

For:
- R0
- R-MATCHED
- R-CROSS
- R-SHUFFLED
- R-RANDOM
- N-MATCHED
- N-CROSS
- N-SHUFFLED
- N-RANDOM

Report:
- r
- RMSE
- MAE
- delta vs R0
- positive seeds.

## E. Semantic prior evidence

Architecture-matched Ridge:
- matched-cross
- matched-shuffled
- matched-random.

Architecture-matched NCR:
same.

## F. Ablations
all detailed ablations.

## G. Biomarker stability
all prior controls.

## H. Faithfulness
top/random/bottom.

## I. Top WM biomarkers
names + importance.

## J. Novelty
final concise list.

## K. Paper figures
list all generated main/supplementary diagrams.

## L. Claim levels
P0–P3 / B0–B3.

## M. Paper status

Print one exact status from Section 41.

Finally:

```text
STATUS: FINAL510_WM_EVIDENCE_FREEZE_COMPLETE
```

---

# 43. Final integrity instruction

The goal is to prepare a 510-subject methodology paper, not a sample-size
comparison paper.

It is acceptable for the main paper to discuss only the final 510 nested-CV
study.

It is NOT acceptable to:
- call the 510 result independent confirmation;
- change models after seeing the final CV results;
- choose a different random prior because it performs worse;
- hide architecture-matched controls;
- claim the LLM prior improves prediction unless the matched Ridge/NCR controls
  support that;
- call NCR the key predictive novelty if Ridge performs better.

Focus the final narrative on:
- the LLM-generated semantic prior;
- prior-selected multimodal experts;
- hierarchical fusion;
- architecture-matched prior controls;
- WM prediction;
- biomarker stability and faithfulness;
- detailed interpretable ablations.
