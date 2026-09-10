# PALF ICLR 2027 — Phase 2E: LLM Network-Interaction Structure–Function Coupling NCR Pilot

## Executive decision

Phase 2D-FIX is now a valid negative result for the current ROI-activation prior
used on raw FC/SC edges.

Do NOT continue tuning:
- global PALF anisotropy,
- line-graph penalties on the same raw-edge prior,
- adaptive rho/tau,
- differential stacking,
- multi-penalty raw-edge Ridge,
- prior-selected raw FC/SC edge experts.

The corrected Phase-2D-FIX result shows:

```text
Working Memory:
matched PS-NCR-EF improves the strong baseline,
but matched ≈ shuffled, so the gain is not demonstrably semantic.

Fluid Intelligence:
matched prior is worse than baseline,
while shuffled/random controls can perform as well or better.
```

The important code-level diagnosis is that the CURRENT LLM prior generator asks:

> how strongly is each AAL region implicated in the cognitive domain?

That is an activation/relevance prior.

It does NOT ask:

> which STRUCTURE–FUNCTION CONNECTIVITY INTERACTIONS are expected to explain
> individual differences in cognitive performance?

All previous prediction methods attempted to lift an ROI-activation prior into
connectome edges.

Phase 2E changes the *prior representation and predictive feature family*,
not merely the optimizer.

The new hypothesis is:

> An LLM prior may be more useful when it specifies task-relevant
> network-to-network structure–function interactions, and when those priors act
> on nonlinear SC–FC coupling features that the current linear late-fusion
> baseline cannot represent.

Working name:

```text
LI-SFC-NCR
LLM Interaction-Prior Structure–Function Coupling
Network-Constrained Regression
```

The strong corrected R0 late-fusion model remains untouched and is the floor.

This is a DEVELOPMENT PILOT only.

Do not edit the manuscript.
Do not run final seeds.
Do not claim significance from four development seeds.

---

# 1. Baseline must remain the corrected R0 model

Reuse the validated implementation in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

The baseline is exactly:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC late fusion
```

Before Phase 2E, run the same correctness audit used in Phase 2D-FIX.

Required frozen means:

```text
WM Pearson ≈ 0.263515
WM RMSE    ≈ 11.292921

FI Pearson ≈ 0.370917
FI RMSE    ≈ 4.566689
```

Required tolerance:

```text
abs Pearson error <= 5e-4
abs RMSE error    <= 0.05
```

If this fails:

```text
STATUS: PHASE2E_BASELINE_AUDIT_FAILED
```

and stop.

---

# 2. Audit the current LLM prior generator

Before implementing the new prior, create:

```text
outputs/iclr/palf_phase2e_li_sfc_ncr/
CURRENT_PRIOR_AUDIT.md
```

Document from:

```text
scripts/46_generate_llm_priors.py
```

that the current prompt is ROI relevance / activation oriented.

Record:
- exact model tag used by the frozen priors;
- temperature;
- seed;
- prompt hashes if available;
- WM/FI prior Pearson correlation;
- WM/FI top-10 overlap;
- WM/FI top-20 overlap.

Do not modify the existing frozen priors.

They remain an ablation later.

---

# 3. Do not use the old prototype anatomical-module mapping

The repository contains:

```text
inputs/atlases/AAL116_coarse_modules.csv
scripts/00_create_aal116_coarse_modules.py
```

The script explicitly states that this is a pragmatic prototype and not a
rigorous AAL-to-Yeo overlap mapping.

Do NOT use this file for the primary Phase-2E prior.

Also do NOT use any fallback that invents/draws random atlas coordinates.

The existing FIP code may be inspected for edge-index conventions and external
prior engineering ideas, but do not reuse:
- dummy/random-coordinate fallback,
- prototype module labels as scientific ground truth.

---

# 4. Build a rigorous AAL116 → functional-system mapping

Use an external canonical atlas independent of HCP labels/prediction data.

Preferred:

```text
Yeo 7-network atlas
```

for cortical AAL ROIs.

Procedure:

1. Load the real AAL116 NIfTI already in the repository.
2. Fetch/load the Yeo-7 atlas using Nilearn or an equivalent canonical source.
3. Resample Yeo labels to the AAL image grid using nearest-neighbor
   interpolation.
4. For every cortical AAL ROI, calculate voxel overlap with the seven Yeo
   networks.
5. Assign the ROI to the network with maximum nonzero overlap.
6. Record:
   - voxel count of AAL ROI;
   - overlap count for every Yeo network;
   - winning network;
   - winning overlap fraction.

For AAL subcortical regions, use one explicit category:

```text
SUBCORTICAL
```

For AAL cerebellar/vermis regions, use:

```text
CEREBELLAR
```

These categories are determined from canonical AAL labels, not HCP data.

Final system set therefore contains up to:

```text
7 Yeo cortical networks
+ SUBCORTICAL
+ CEREBELLAR
= 9 systems
```

Do not silently assign a cortical ROI with zero Yeo overlap.

If any cortical ROI has zero valid overlap:
- record it;
- attempt a documented nearest-labelled-voxel or atlas-resolution fix;
- if unresolved, stop before HCP evaluation.

Save:

```text
priors/system_mapping/AAL116_to_Yeo7_plus_subcortical_cerebellar.csv
priors/system_mapping/mapping_qc.json
```

Do not use HCP FC/SC or cognitive labels to construct this mapping.

---

# 5. Generate NEW LLM network-interaction priors

The new LLM prior is not an ROI activation prior.

It scores SYSTEM PAIRS according to their expected STRUCTURE–FUNCTION
INTERACTION relevance for individual differences in cognitive performance.

Use the same local LLM family used for the final frozen prior where possible:

```text
qwen3.8:27b
```

Before execution, verify the exact available Ollama tag and record it.

Use:

```text
temperature = 0.2
generation_seeds = [31, 37, 43, 47, 53]
```

Do not alter these after generation.

Generate one prior for:

```text
Working Memory / NIH List Sorting
```

and one for:

```text
Fluid Intelligence / Penn Matrix Reasoning
```

The two prompts MUST be contrastive.

---

# 6. Exact LLM prompt semantics

The prompt should explain that:

- the data are resting-state functional connectivity and diffusion-derived
  structural connectivity;
- the goal is prediction of BETWEEN-SUBJECT cognitive differences;
- scores must refer to the importance of the CONNECTIVITY / interaction between
  two systems, not regional activation;
- high scores mean that individual differences in the structural-functional
  relationship between those systems are expected to be informative for the
  target;
- generic cognitive involvement is insufficient for a high score;
- WM must be contrasted against abstract reasoning/fluid intelligence;
- FI must be contrasted against short-term maintenance/working-memory-specific
  processing.

For every unordered system pair, including within-system pairs, return:

```text
system_a
system_b
sf_coupling_relevance : float [0,1]
reason_short          : <= 25 words
```

Do NOT ask for coefficient direction.

Do NOT ask the LLM to use HCP findings or our experimental results.

Do NOT expose any:
- HCP labels,
- current model coefficients,
- current residuals,
- Phase-2D results,
- prior-control results.

This must be external semantic prior construction.

---

# 7. Prior self-consistency and freeze

For each task and each system pair, aggregate the five generation seeds using:

```text
median score
```

Save:
- median;
- mean;
- SD;
- min;
- max.

Then normalize the median vector to `[0,1]`.

Create:

```text
priors/llm_network_interaction/working_memory_pair_prior.csv
priors/llm_network_interaction/fluid_intelligence_pair_prior.csv
priors/llm_network_interaction/generation_provenance.json
```

Also save every raw LLM response and prompt SHA256.

Before ANY HCP prediction is run, create:

```text
priors/llm_network_interaction/FROZEN
```

and a SHA256 manifest.

After this point, the prior files are read-only.

---

# 8. Prior-quality diagnostics before HCP evaluation

Compute:

```text
WM/FI pair-prior Pearson
WM/FI pair-prior Spearman
top-5 pair overlap
top-10 pair overlap
```

Also compare the new pair priors to the old node-product priors aggregated by
the same systems.

This diagnostic is descriptive only.

Do not reject or regenerate the new prior because it "looks bad".

Once frozen, use it.

---

# 9. New nonlinear structure–function coupling features

This is the crucial predictive change.

The corrected R0 baseline is linear in FC and SC predictions and does NOT
contain edgewise FC×SC interaction features.

For every train/validation/test fitting scope:

1. Standardize FC edges using statistics fit on the current TRAINING subjects.
2. Standardize SC edges independently using current TRAINING subjects.
3. For each subject and edge:

\[
h_e = z^{FC}_e z^{SC}_e.
\]

This is an elementwise SC–FC interaction feature.

Do NOT standardize FC/SC globally before the outer/inner splits.

Then aggregate `h_e` by canonical system pair.

For system pair `g=(a,b)` with edge set `E_g`:

\[
c_g =
\frac{1}{|E_g|}
\sum_{e \in E_g} h_e.
\]

This yields at most 45 nonlinear subject-level coupling features.

Call the resulting matrix:

```text
X_sfc_pair
```

Important:
- `X_sfc_pair` must be recomputed with fold-local FC/SC standardization;
- it may NOT be materialized once globally and reused across CV splits.

These interaction features are outside the linear function class of the
existing separate FC/SC late-fusion predictor.

---

# 10. Sanity-check the new feature family

Before using the LLM prior, create a NO-PRIOR coupling expert using all valid
system-pair coupling features.

Call:

```text
SFC Ridge expert — all pairs
```

Use ordinary Ridge.

Search:

```text
alpha_sfc_grid = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]
```

Generate fully cross-fitted OOF predictions on the outer-training set.

Fuse with the corrected R0 baseline:

\[
\hat y =
(1-\eta)\hat y_{R0}
+
\eta\hat y_{SFC}
\]

with:

```text
eta in [0, 0.05, ..., 1]
```

selected ONLY on outer-training OOF predictions.

This no-prior SFC expert separates:

```text
benefit of adding nonlinear structure–function features
```

from:

```text
benefit of the LLM prior.
```

---

# 11. LLM prior-selected SFC expert

Rank the system-pair coupling features by the frozen LLM pair prior.

Search:

```text
TOP_PAIR_GRID = [5, 10, 15, 25]
```

For each candidate:
- select the top pair features;
- fit Ridge using the same alpha grid;
- evaluate in inner CV.

Call:

```text
LLM SFC Ridge expert
```

This is the hard-selection prior ablation.

---

# 12. Proposed NCR model on pair-coupling features

The proposed model is:

```text
LI-SFC-NCR
```

Use the SAME selected pair-coupling feature subset as the LLM SFC Ridge expert.

Features are nodes in a small "pair graph".

For two selected system-pair features:

```text
(a,b)
(c,d)
```

connect them if they share one canonical system.

Example:

```text
(FPN, DMN)
(FPN, DAN)
```

are adjacent because both contain FPN.

Construct a symmetric pair-feature Laplacian.

The LLM prior enters the pair graph through edge weights:

\[
w_{uv} =
\sqrt{q_u q_v}
\]

for adjacent pair-features `u,v`, where `q` is the frozen task pair prior.

Use the symmetric-normalized Laplacian.

Fit:

\[
\hat\beta
=
\arg\min_\beta
\left\{
\|y-X_{SFC}\beta\|^2
+
\lambda_R\|\beta\|^2
+
\lambda_L\beta^T L_{LLM}\beta
\right\}.
\]

Parameterize:

```text
lambda_L = ratio * lambda_R
```

with:

```text
NCR_RATIO_GRID = [0.0, 0.1, 0.3, 1.0]
```

`ratio=0` MUST reproduce the LLM-selected SFC Ridge expert exactly for the
same mask/lambda.

This is the proposed NCR method.

---

# 13. Architecture-matched prior controls

Run the identical selected-coupling NCR architecture with:

```text
matched
cross_task
shuffled
random
```

pair priors.

## Shuffled
Permute the frozen system-pair scores across pair labels using fixed external
control seed:

```text
shuffled_seed = 7301
```

Use the same shuffled vector for all CV seeds/folds of a task.

## Random
Draw one fixed random `[0,1]` vector over system pairs:

```text
random_seed = 7303
```

Freeze it before HCP evaluation.

Do NOT draw a new random prior per outer seed.

## Cross-task
Use FI pair prior for WM and WM pair prior for FI.

All controls:
- use identical candidate cardinalities;
- use identical grids;
- use identical fitting/fusion logic.

---

# 14. Models in the pilot

Evaluate exactly:

```text
A0: Corrected R0 late fusion

A1: R0 + all-pair SFC Ridge expert
    (new nonlinear feature benefit, no LLM prior)

A2: R0 + matched LLM-selected SFC Ridge expert
    (prior-selected subspace, no NCR)

A3: R0 + matched LI-SFC-NCR expert
    (proposed)

A4: R0 + cross-task LI-SFC-NCR

A5: R0 + shuffled LI-SFC-NCR

A6: R0 + random LI-SFC-NCR
```

Do not add more models after seeing test results.

---

# 15. Fully nested evaluation

Use the validated split hierarchy from:

```text
palf_crossfit_ablation.py
```

For every outer split:

## Baseline
Generate R0 OOF and final test prediction exactly as already validated.

## SFC expert OOF
For every fusion held-out fold:

1. fit FC and SC edge scalers on analysis subjects only;
2. build `X_sfc_pair` for analysis and held-out subjects;
3. inner-select:
   - top-pair cardinality if applicable;
   - Ridge lambda;
   - NCR ratio if applicable;
4. refit expert on analysis;
5. predict held-out fusion fold.

Concatenate OOF expert predictions.

## Final baseline/expert fusion
Select eta from:

```text
[0,0.05,...,1]
```

using R0 OOF + SFC expert OOF.

Then:
- reselect/final-fit SFC expert on all outer training;
- predict outer test;
- combine using already-selected eta;
- evaluate outer test exactly once.

No outer-test label may enter:
- FC/SC standardization;
- interaction-feature construction parameters;
- pair selection;
- Ridge/NCR selection;
- eta selection.

---

# 16. Fresh development seeds

Do not use any prior development seeds.

Use exactly:

```text
PHASE2E_DEV_SEEDS = [2727, 2828, 2929, 3030]
```

with five outer folds.

Seeds 0–9 are audit only.

Do not modify this set after seeing results.

---

# 17. Individual-task first

Phase 2E primary pilot is individual-task.

Do NOT add shared/contrast multi-task prediction yet.

The advisor's individual-vs-multi-task question will become a Phase-2E.1
ablation only if A3 passes.

This avoids spending time on another multi-task mechanism before proving that
the new connectivity prior/feature family has predictive value.

---

# 18. Exact prediction comparisons

For both WM and FI report:

```text
A0 baseline
A1 no-prior SFC
A2 LLM-selected SFC Ridge
A3 LI-SFC-NCR
A4 cross-task
A5 shuffled
A6 random
```

Report:
- Pearson;
- RMSE;
- MAE.

Primary prior prediction effect:

```text
A3 - A1
```

because A1 contains the same nonlinear SFC information without LLM guidance.

Headline improvement over current model:

```text
A3 - A0
```

NCR contribution:

```text
A3 - A2
```

Prior specificity:

```text
A3 - A4
A3 - A5
A3 - A6
```

---

# 19. Biomarker-ready coefficient export

The SFC model is low-dimensional and linear in explicit nonlinear coupling
features.

For every outer final fit save:

```text
selected system-pair names
selected system-pair indices
beta_pair_standardized
beta_pair_original
intercept
lambda_R
lambda_L
eta
```

Then map each pair coefficient back to the AAL edges belonging to that
system pair for visualization:

\[
\beta^{edge}_e =
\frac{\beta_{pair(e)}}{|E_{pair(e)}|}
\]

for interpretation only.

Do NOT pretend this edge-distributed vector was independently fitted at the
edge level.

Also create ROI saliency:

\[
s_i =
\sum_{g \ni system(i)} |\beta_g|.
\]

Record the exact mapping rule.

---

# 20. Lightweight stability diagnostics

Without using stability for model selection, compute across the 20 final
outer fits:

- Spearman of absolute system-pair coefficients;
- top-5 pair Jaccard;
- top-10 pair Jaccard;
- sign consistency;
- top-10 ROI/system importance consistency.

Compare:
- A1 no-prior SFC;
- A3 matched LI-SFC-NCR;
- shuffled/random controls.

Call these development diagnostics, not final biomarker claims.

---

# 21. Held-out coupling-feature faithfulness pilot

For A3 only, on every untouched outer-test fold:

1. rank system-pair coupling features from the training-fit coefficients;
2. mask top:
   - 3 pairs;
   - 5 pairs;
   - 10 pairs;
3. recompute expert and final prediction without retraining;
4. compare against:
   - bottom same-size pairs;
   - 50 fixed random same-size pair sets.

Report:

```text
delta_RMSE_top
delta_RMSE_bottom
delta_RMSE_random_mean
```

Do not use these metrics for hyperparameter selection.

This gives an early biomarker-faithfulness signal.

---

# 22. Validation gates

The run is valid only if:

## Prior
- real LLM pair prior generated before HCP evaluation;
- all prompt/raw-response hashes saved;
- FROZEN marker exists;
- no HCP input used in prior generation;
- no old coarse-module prototype used as the primary mapping.

## Mapping
- every AAL ROI has valid documented system assignment;
- cortical Yeo mapping comes from atlas overlap;
- no random/dummy coordinates.

## Features
- SFC features recomputed using training-fitted FC/SC edge scalers;
- no global FC/SC standardization;
- all features finite.

## Baseline
- R0 audit passes exact current values.

## Model
- A0 unchanged;
- eta=0 exactly recovers A0;
- NCR ratio=0 reproduces A2 Ridge expert for same mask/lambda.

## Evaluation
- fresh seeds only for decision;
- no outer-test leakage;
- fixed shuffled/random controls.

---

# 23. Predefined decision

## STRONG_GO

Print:

```text
PHASE2E_DECISION: STRONG_GO
```

if BOTH targets satisfy:

```text
A3 - A0 mean Pearson >= +0.003
positive seeds >= 3/4
```

AND BOTH satisfy:

```text
A3 - A1 > 0
```

with positive seeds >= 3/4,

AND matched A3 is numerically better than BOTH shuffled and random on both
tasks.

At least one target must have:

```text
A3 - A0 >= +0.005
```

---

## PROMISING_GO

Print:

```text
PHASE2E_DECISION: PROMISING_GO
```

if one target meets STRONG-style prediction improvement and the other has:

```text
A3 - A0 >= 0
A3 - A1 >= 0
```

with no matched-vs-control reversal.

---

## SFC_ONLY_GO

Print:

```text
PHASE2E_DECISION: SFC_ONLY_GO
```

if A1 improves prediction but A3 does not beat A1.

This means nonlinear structure–function coupling helps, but the LLM prior does
not.

Do not claim LLM prior success.

---

## NO_GO

Print:

```text
PHASE2E_DECISION: NO_GO
```

if:
- A3 fails against A0 on both targets;
- or A3 fails against A1;
- or matched does not outperform controls.

---

# 24. Tests

Add:

```text
tests/test_li_sfc_ncr.py
```

At minimum:

1. AAL/Yeo overlap mapping is deterministic.
2. no cortical ROI has unresolved system assignment.
3. no random/dummy coordinate fallback exists in Phase-2E path.
4. network-pair list contains all valid unordered pairs exactly once.
5. pair-prior CSV covers every pair.
6. raw LLM responses and prompt hashes exist.
7. HCP arrays are not imported/called by prior-generation module.
8. SFC feature computation fits edge scalers only on training indices.
9. direct synthetic FC×SC pair summary is numerically correct.
10. A1 uses all valid SFC pair features and no LLM score.
11. matched/shuffled/random/cross-task have same pair cardinality grid.
12. random/shuffled priors are fixed across CV seeds.
13. eta=0 reproduces R0 exactly.
14. NCR ratio=0 equals pair-subspace Ridge.
15. pair-feature Laplacian is symmetric PSD.
16. outer-test target never enters feature/prior/model selection.
17. coefficient reconstruction reproduces expert prediction.
18. fresh development seeds are exact.
19. prior FROZEN marker predates prediction outputs.
20. previous result directories remain untouched.

Run targeted tests and full suite.

---

# 25. Output structure

Create:

```text
outputs/iclr/palf_phase2e_li_sfc_ncr/
```

with:

```text
COMPLETE
BASELINE_AUDIT.json
CURRENT_PRIOR_AUDIT.md
VALIDATION_REPORT.json
RUN_REPORT.md

priors/
  system_mapping/
  llm_network_interaction/

split_metrics.csv
seed_metrics.csv
model_summary.csv
prior_control_summary.csv
stability_diagnostics.csv
faithfulness_pilot.csv

coefficients/
predictions/
plots/
```

Zip:

```text
outputs/iclr/palf_phase2e_li_sfc_ncr.zip
```

---

# 26. Required plots

Generate:

```text
plots/fig_phase2e_model_comparison.pdf/png
plots/fig_phase2e_seed_deltas.pdf/png
plots/fig_phase2e_pair_prior_heatmaps.pdf/png
plots/fig_phase2e_pair_coefficients.pdf/png
plots/fig_phase2e_faithfulness.pdf/png
```

Heatmaps should show the frozen WM and FI LLM network-pair priors.

---

# 27. Final OpenCode report

Print:

## A. Baseline audit
- WM r/RMSE
- FI r/RMSE
- PASS/FAIL

## B. Current-prior diagnosis
- old prior prompt type
- old WM/FI correlation
- old overlap

## C. System mapping
- atlas source
- Yeo source/version
- number of cortical/subcortical/cerebellar ROIs
- unresolved ROIs
- PASS/FAIL

## D. New LLM prior
- exact model tag
- prompt hashes
- five generation seeds
- WM/FI pair-prior correlation
- top-5/top-10 overlap
- prior freeze checksum

## E. Prediction results
For each target:

```text
A0 R0
A1 all-pair SFC Ridge
A2 matched LLM SFC Ridge
A3 LI-SFC-NCR
A4 cross-task
A5 shuffled
A6 random
```

Pearson/RMSE/MAE.

## F. Paired development deltas
For each target:
- A3-A0 four seed deltas / mean / positive count
- A3-A1 four seed deltas / mean / positive count
- A3-A2 NCR contribution
- A3-A4/A5/A6 specificity

## G. Fusion
- eta mean/median
- eta=0 fraction
- expert standalone r
- baseline/expert OOF correlation

## H. Biomarker readiness
- coefficient stability
- top-pair Jaccard
- sign consistency
- faithfulness top vs random/bottom

## I. Tests
- targeted
- full suite
- pre-existing failures

## J. Outputs
- output directory
- ZIP
- prior files
- CSVs
- coefficients
- plots

Then exactly one:

```text
PHASE2E_DECISION: STRONG_GO
```

```text
PHASE2E_DECISION: PROMISING_GO
```

```text
PHASE2E_DECISION: SFC_ONLY_GO
```

or

```text
PHASE2E_DECISION: NO_GO
```

Finally:

```text
STATUS: PHASE2E_LI_SFC_NCR_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE2E_NOT_VALID
```

and do not interpret performance.

---

# 28. Stop after Phase 2E

Do not rewrite the paper or run final seeds until the Phase-2E pilot is reviewed.

If Phase 2E succeeds, the next stage will:
1. freeze LI-SFC-NCR;
2. add individual-vs-multi-task as an ablation;
3. run a final confirmatory evaluation;
4. run full biomarker stability/specificity/faithfulness.

If Phase 2E fails, do not introduce another prediction architecture on the
same 412-subject cohort.
