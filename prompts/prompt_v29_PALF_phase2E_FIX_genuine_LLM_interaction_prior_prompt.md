# PALF ICLR 2027 — Phase 2E-FIX: Genuine LLM Interaction Prior + Correct Nested SFC Evaluation

## Executive status

The existing Phase-2E result MUST be marked:

```text
INVALID_FOR_INTENDED_LLM_INTERACTION_PRIOR_HYPOTHESIS
```

Do not use its `NO_GO` as evidence that a genuinely generated LLM
system-interaction prior fails.

The corrected R0 baseline audit is valid, but the Phase-2E prior and expert
evaluation contain several protocol/implementation failures described below.

This corrective run does **not** introduce a new architecture. It executes the
original Phase-2E LI-SFC-NCR hypothesis correctly.

If this corrected run is a valid NO_GO, stop prediction-method development on
this 412-subject cohort.

---

# 0. Mandatory forensic audit of the old Phase-2E run

Create:

```text
outputs/iclr/palf_phase2e_fix_li_sfc_ncr/
OLD_PHASE2E_FORENSIC_AUDIT.md
```

and verify each item from the current code/output.

## F1 — The "new LLM pair prior" was not actually an LLM pair prior

The old output file:

```text
CURRENT_PRIOR_AUDIT.md
```

states that the pair prior was derived by:

```text
q_ab = mean(p_i * p_j)
```

from the existing ROI prior.

The old pair-prior CSVs also have:

```text
n_seeds = 0
std_score = 0
```

for every system pair.

No raw five-seed LLM response files are present in the output bundle.

Therefore the old A3/A4/A5/A6 results did not test the intended newly generated
LLM structure–function interaction prior.

Hard assertion for the corrected run:

```text
every pair must have n_seeds == 5
```

No fallback to ROI aggregation is allowed.

---

## F2 — Provenance was inconsistent with the actual prior

The old provenance JSON lists:

```text
model: qwen3.8:27b
seeds: [31,37,43,47,53]
```

while the pair CSV records `n_seeds=0`.

The corrected pipeline must never write generation provenance unless the
corresponding raw generations exist and validate.

---

## F3 — Ollama seed and temperature were not actually passed

The current `_call_ollama()` constructs a Python payload containing:

```text
temperature
seed
```

but then calls:

```text
ollama run <model>
```

using only the prompt string. The payload is unused.

Replace this with an explicit Ollama local REST API call:

```text
POST http://127.0.0.1:11434/api/generate
```

with JSON containing:

```json
{
  "model": "<verified-model-tag>",
  "prompt": "...",
  "stream": false,
  "options": {
    "temperature": 0.2,
    "seed": <seed>
  }
}
```

Use `requests` or the Python standard library.

If the API/model is unavailable, STOP.

Do not silently replace the prior with ROI-derived values.

---

## F4 — Yeo-7 semantic label IDs are wrong

The current code hard-codes:

```text
1 VIS
2 SM
3 DATN
4 LIMBIC
5 FPN
6 DMN
7 VAS
```

This is not the canonical semantic ordering of the Yeo 7-network estimate.

Use:

```text
1 VIS
2 SM
3 DAN
4 VAN
5 LIMBIC
6 FPN
7 DMN
```

where:
- DAN = Dorsal Attention Network
- VAN = Ventral Attention / Salience Network
- LIMBIC = Limbic
- FPN = Frontoparietal Control
- DMN = Default Mode

The old mapping therefore semantically mislabeled IDs 4–7.

Regenerate the mapping from scratch in the NEW output directory.

Do not reuse the old mapping CSV.

Also update all system descriptions so that VAN and LIMBIC are not conflated.

Prefer reading atlas LUT/metadata where available and record the exact Yeo atlas
source/version and Nilearn version.

Explicitly request:

```python
fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
```

where supported.

---

## F5 — A2 was not LLM-selected

In the old code:

```python
top_k_val = best_params_top_k(model_type) if model_type != "A2" else None
```

so A2 used all valid pairs.

This explains why old A1 and A2 predictions are exactly identical for every
split.

The corrected A2 must be:

```text
LLM-selected pair-subspace Ridge
```

and must inner-select:

```text
TOP_PAIR_GRID = [5, 10, 15, 25]
```

It must never use all 45 pairs.

Add a test:

```text
A2 selected pair count ∈ {5,10,15,25}
```

and an audit proving A1/A2 are distinct model definitions.

---

## F6 — A3/A4/A5/A6 did not search TOP_PAIR_GRID

The old helper effectively hard-codes `top_k=10`.

The corrected prior-aware arms must search:

```text
[5,10,15,25]
```

inside training data only.

Delete `best_params_top_k()`.

---

## F7 — SFC feature construction leaks across fusion/inner folds

The old runner builds `X_sfc_train` once using FC/SC means and standard
deviations fit on the entire OUTER-training set, and then reuses that matrix for
fusion-fold OOF and inner-CV selection.

For OOF model/fusion selection this allows held-out training subjects to affect
the nonlinear features via FC/SC scaling.

Correct rule:

For EVERY inner/fusion split:
1. take raw FC/SC edge matrices;
2. fit FC edge mean/std on that split's analysis/train subjects only;
3. fit SC edge mean/std on that split's analysis/train subjects only;
4. transform analysis and held-out subjects;
5. form `z_FC * z_SC`;
6. aggregate to system pairs.

Never precompute one outer-training SFC matrix and use it inside nested CV.

For the final outer-test fit only:
- fit FC/SC scalers on full outer training;
- transform outer test;
- build train/test SFC pair features.

---

## F8 — Control priors are not properly task-specific

The old code creates one:

```text
shuffled_pair_prior.csv
random_pair_prior.csv
```

from the WM prior and uses those same files for both tasks.

Correct controls:

```text
working_memory_shuffled_pair_prior.csv
fluid_intelligence_shuffled_pair_prior.csv

working_memory_random_pair_prior.csv
fluid_intelligence_random_pair_prior.csv
```

For shuffled:
- apply a deterministic permutation separately to each task's own matched
  score vector;
- preserve that task's score distribution.

Use:

```text
WM shuffled seed = 7301
FI shuffled seed = 7302
```

For random use fixed independent vectors:

```text
WM random seed = 7303
FI random seed = 7304
```

Freeze all controls before HCP evaluation.

---

# 1. New clean output directory

Create:

```text
outputs/iclr/palf_phase2e_fix_li_sfc_ncr/
```

Never read pair priors or mapping files from:

```text
outputs/iclr/palf_phase2e_li_sfc_ncr/
```

except for the forensic audit.

Required structure:

```text
OLD_PHASE2E_FORENSIC_AUDIT.md
BASELINE_AUDIT.json
VALIDATION_REPORT.json
RUN_REPORT.md
COMPLETE

priors/
  system_mapping/
  llm_network_interaction/

split_metrics.csv
seed_metrics.csv
model_summary.csv
prior_control_summary.csv
selection_details.csv

coefficients/
predictions/
plots/
```

Zip:

```text
outputs/iclr/palf_phase2e_fix_li_sfc_ncr.zip
```

---

# 2. Correct system list

Use exactly:

```text
VIS
SM
DAN
VAN
LIMBIC
FPN
DMN
SUBCORTICAL
CEREBELLAR
```

Descriptions:

```text
VIS:
visual network

SM:
somatomotor network

DAN:
dorsal attention network

VAN:
ventral attention / salience network

LIMBIC:
limbic network

FPN:
frontoparietal control network

DMN:
default mode network

SUBCORTICAL:
thalamus, basal ganglia, amygdala/hippocampal and other explicitly classified
non-Yeo structures

CEREBELLAR:
cerebellum/vermis
```

Do not describe VAN as LIMBIC or vice versa.

The 9-system unordered pair count must be exactly:

```text
45
```

---

# 3. Correct AAL116 → Yeo7 mapping

Rebuild from scratch using atlas overlap.

For cortical ROIs:
- resample Yeo7 to AAL116 image grid with nearest-neighbor interpolation;
- determine voxel overlap with each canonical Yeo network;
- assign maximum-overlap network;
- save all overlap counts and fractions.

For non-Yeo regions:
- use explicit documented SUBCORTICAL/CEREBELLAR rules.

Save:

```text
AAL116_to_Yeo7_plus_subcortical_cerebellar.csv
mapping_qc.json
```

QC must include:
- Nilearn version;
- atlas file path/hash;
- label/LUT used;
- system counts;
- number unresolved;
- minimum cortical winning-overlap fraction.

If any cortical ROI is unresolved, stop unless a documented deterministic atlas
resolution fix is applied.

---

# 4. Generate a genuinely new LLM interaction prior

Do NOT derive these scores from the old ROI priors.

Do NOT read HCP FC, SC, targets, coefficients, residuals, or prediction results
in the prior-generation module.

Before generating:
1. query the local Ollama model list;
2. verify the intended Qwen model tag;
3. record the exact model digest/tag.

Preferred model:

```text
qwen3.8:27b
```

If unavailable, STOP and report available tags.

Do not silently switch models.

Use:

```text
temperature = 0.2
seeds = [31,37,43,47,53]
```

via Ollama REST options.

---

# 5. LLM prior prompt must be external and connectivity-specific

Do NOT mention:
- HCP;
- sample size 412;
- our current baseline;
- current prediction numbers;
- current prior results;
- biomarkers already observed.

The LLM should receive only general neuroscientific task/context information.

For WM:

```text
Target:
individual differences in working-memory capacity / maintenance and manipulation.

Question:
which large-scale brain-system STRUCTURE–FUNCTION COUPLING relationships are
expected to be most informative for between-person variation in this target,
relative to fluid reasoning?
```

For FI:

```text
Target:
individual differences in fluid reasoning / novel problem solving and rule
induction.

Question:
which large-scale brain-system STRUCTURE–FUNCTION COUPLING relationships are
expected to be most informative for between-person variation in this target,
relative to working-memory-specific maintenance?
```

For every one of the 45 unordered pairs return:

```text
system_a
system_b
sf_coupling_relevance ∈ [0,1]
reason_short <= 25 words
```

No coefficient direction.

No subject-specific prediction.

---

# 6. Generation validity is a hard gate

For EACH generation seed:

- response must parse;
- exactly 45 unique valid unordered pairs must be present;
- no pair missing;
- no duplicate;
- every score finite in [0,1].

Retry the SAME seed up to 3 times if parsing/coverage fails.

If it still fails:

```text
STATUS: PHASE2E_FIX_PRIOR_GENERATION_FAILED
```

and stop.

Never fill missing scores with zero.

Never use ROI prior aggregation as fallback.

Save complete raw responses:

```text
working_memory_raw_seed31.json
...
fluid_intelligence_raw_seed53.json
```

not previews.

Aggregate each pair across exactly five generations:

```text
median
mean
std
min
max
n_seeds
```

Hard assert:

```text
n_seeds == 5
```

for every row.

---

# 7. Prior provenance and freeze

Save:

```text
working_memory_generation_provenance.json
fluid_intelligence_generation_provenance.json
```

containing:
- exact model tag/digest;
- Ollama version if available;
- temperature;
- exact five seeds;
- complete prompt SHA256;
- raw response SHA256 per seed;
- 45/45 validation status;
- generation timestamp.

Then generate controls.

Then create:

```text
FROZEN
SHA256SUMS.txt
```

BEFORE loading any HCP predictive arrays.

The predictive pilot must refuse to run if:
- FROZEN is absent;
- any prior/control file checksum differs.

---

# 8. Correct control priors

Create task-specific:

```text
working_memory_shuffled_pair_prior.csv
fluid_intelligence_shuffled_pair_prior.csv
working_memory_random_pair_prior.csv
fluid_intelligence_random_pair_prior.csv
```

Cross-task means:
- WM uses genuine FI matched prior;
- FI uses genuine WM matched prior.

Shuffled means:
- same task's 45 values permuted over pair labels.

Random means:
- independent fixed Uniform[0,1] vector for the task.

No control may be regenerated across CV seeds.

---

# 9. Prior diagnostics before HCP evaluation

After freeze, report:

```text
WM-FI Pearson
WM-FI Spearman
top-5 overlap
top-10 overlap
```

Also list:
- WM top 10 pairs;
- FI top 10 pairs.

Do not regenerate the LLM prior based on these diagnostics.

A high correlation is an experimental outcome, not a reason to retry.

---

# 10. Keep the original Phase-2E nonlinear feature family

Do not introduce a new SFC feature formula in this correction.

For every fitting scope:

\[
z^{FC}_{se}
=
\frac{x^{FC}_{se}-\mu^{FC}_{e,\text{train}}}
{\sigma^{FC}_{e,\text{train}}},
\]

\[
z^{SC}_{se}
=
\frac{x^{SC}_{se}-\mu^{SC}_{e,\text{train}}}
{\sigma^{SC}_{e,\text{train}}},
\]

\[
h_{se}=z^{FC}_{se}z^{SC}_{se}.
\]

For system pair g:

\[
c_{sg}
=
\frac{1}{|E_g|}
\sum_{e\in E_g} h_{se}.
\]

There are 45 pair features.

The only correction is that the FC/SC edge scalers must be trained at the
CURRENT nested training scope.

---

# 11. Refactor nested feature generation

Do not pass a globally/outer-precomputed `X_sfc` into expert selection.

Implement something like:

```python
build_sfc_features_from_raw(
    X_fc_raw,
    X_sc_raw,
    analysis_idx,
    validation_idx,
    edge_pair_idx,
)
```

which:
- fits edge scalers on `analysis_idx`;
- returns pair features for analysis and validation.

For inner CV inside a fusion-analysis subset:
- call this helper with inner-train and inner-val.

For fusion OOF:
- after selecting parameters using inner CV, call it with fusion-analysis and
  fusion-validation.

For final outer fit:
- call with full outer-train and outer-test.

Add an index-spy unit test proving that held-out rows never enter scaler fits.

---

# 12. Correct A1

A1:

```text
Corrected R0 + ALL 45 pair SFC Ridge expert
```

No LLM prior.

Inner-select Ridge alpha:

```text
[0.001,0.01,0.1,1,10,100,1000]
```

Generate proper fusion-OOF expert predictions.

Select:

```text
eta ∈ [0,0.05,...,1]
```

using only outer-training OOF predictions.

A1 measures whether the nonlinear SFC feature family helps by itself.

---

# 13. Correct A2

A2:

```text
Corrected R0 + matched LLM-selected SFC Ridge expert
```

For every training scope jointly search:

```text
top_k ∈ [5,10,15,25]
alpha ∈ [0.001,0.01,0.1,1,10,100,1000]
```

using inner validation Pearson, then RMSE/MAE tie-break.

A2 must never use all 45 pairs.

Save selected top_k for every OOF/final fit.

---

# 14. Correct A3 NCR comparison

A3 is the proposed:

```text
LI-SFC-NCR
```

For a fair NCR-vs-Ridge comparison:

1. choose the matched LLM pair subspace/cardinality exactly as for A2 within
   the same training scope;
2. hold that selected pair subset fixed;
3. tune:

```text
alpha
ratio ∈ [0,0.1,0.3,1.0]
```

where:

```text
lambda_L = ratio * alpha
```

on the same inner folds.

Hard test:

```text
ratio=0 with same alpha/subset == Ridge
```

A3 is then fused with corrected R0 using the same eta grid.

---

# 15. Correct A4/A5/A6

For each control prior:
- use its own ranking;
- search the SAME top_k grid;
- tune the SAME alpha/ratio grids;
- use identical nested procedures.

Models:

```text
A4 cross-task LI-SFC-NCR
A5 task-specific shuffled LI-SFC-NCR
A6 task-specific random LI-SFC-NCR
```

Do not share WM shuffled scores with FI.

---

# 16. Corrected baseline

Reuse only the validated R0 code path.

Audit seeds 0–9:

```text
WM r ≈ 0.263515
FI r ≈ 0.370917
WM RMSE ≈ 11.292921
FI RMSE ≈ 4.566689
```

Baseline prediction is computed once per split and reused by A1–A6.

---

# 17. Fresh corrective development seeds

The old Phase-2E seeds have already been inspected.

Do not reuse:

```text
2727,2828,2929,3030
```

for the corrected decision.

Use exactly:

```text
PHASE2E_FIX_DEV_SEEDS = [3131,3232,3333,3434]
```

with five outer folds.

Seeds 0–9 are correctness-audit only.

Do not change these seeds after seeing results.

---

# 18. Required outputs and diagnostics

`split_metrics.csv` must include:

```text
task
seed
fold
model
pearson
rmse
mae
eta

expert_pearson
expert_rmse
expert_mae

selected_top_k
selected_alpha
selected_ratio

prior_type
```

`selection_details.csv` must record every:
- fusion fold;
- inner candidate;
- selected top_k;
- selected alpha;
- selected NCR ratio.

This allows independent audit.

---

# 19. Hard assertions based on old bugs

Before the full corrected run, add automated assertions:

### Prior
```text
pair CSV n_seeds == 5 for all 45 rows
raw responses count == 5 per task
provenance n_seeds_collected == 5
```

### Mapping
```text
Yeo ID 4 == VAN
Yeo ID 5 == LIMBIC
Yeo ID 6 == FPN
Yeo ID 7 == DMN
```

### A2
```text
selected_top_k ∈ {5,10,15,25}
```

### A1/A2
The definitions must differ:
```text
A1 candidate count = 45 pairs
A2 candidate count <=25
```

### Cross-fitting
For every nested fold, scaler fit indices are a strict subset excluding held-out
indices.

### Controls
Task-specific shuffled files exist and preserve each task's matched score
multiset.

---

# 20. Do not over-interpret eta=0

Report for every model/task:

```text
eta=0 count
eta mean
eta median
```

If A3 frequently selects eta=0, say that the outer-training procedure abstains
from the prior expert.

Do not convert this into evidence that the LLM prior is harmful unless
architecture-matched comparisons support that claim.

---

# 21. Primary pilot comparisons

For each target compute seed-level:

## New nonlinear feature benefit
```text
A1 - A0
```

## LLM selection benefit without NCR
```text
A2 - A1
```

## NCR contribution
```text
A3 - A2
```

## Final proposed improvement
```text
A3 - A0
```

## Prior specificity
```text
A3 - A4
A3 - A5
A3 - A6
```

For each:
- 4 seed deltas;
- mean;
- median;
- positive seeds / 4.

No final significance claims from 4 development seeds.

---

# 22. Corrected decision rule

## STRONG_GO

```text
PHASE2E_FIX_DECISION: STRONG_GO
```

only if both tasks satisfy:

```text
A3-A0 >= +0.003
positive seeds >= 3/4
```

AND:

```text
A3-A1 > 0
positive seeds >= 3/4
```

AND matched A3 beats both shuffled and random for both tasks.

At least one target must have:

```text
A3-A0 >= +0.005
```

## PROMISING_GO

```text
PHASE2E_FIX_DECISION: PROMISING_GO
```

if one target meets strong-style improvement and the other is nonnegative,
with no matched-vs-control reversal.

## SFC_ONLY_GO

```text
PHASE2E_FIX_DECISION: SFC_ONLY_GO
```

if A1 improves but A3 does not credibly improve over A1.

## NO_GO

```text
PHASE2E_FIX_DECISION: NO_GO
```

only if ALL validity gates pass and the corrected prior-aware experiment fails
the above rules.

---

# 23. Tests

Add/update tests to catch every discovered failure.

Minimum:

1. canonical Yeo ID semantic mapping test.
2. exactly 45 pair names.
3. pair-generation module has no HCP data dependency.
4. REST API payload actually contains temperature and seed.
5. five valid raw generations per task.
6. every pair row has `n_seeds=5`.
7. missing/invalid LLM generation raises error; no ROI-prior fallback.
8. FROZEN/checksum validation.
9. task-specific shuffled controls preserve corresponding task value multiset.
10. raw-edge FC scaler uses nested analysis indices only.
11. raw-edge SC scaler uses nested analysis indices only.
12. SFC interaction synthetic numerical test.
13. A1 always uses all valid pairs.
14. A2 never uses all 45 pairs.
15. A2 searches all four top_k candidates.
16. A3 uses A2-selected matched subset before NCR tuning.
17. NCR ratio=0 reproduces Ridge for same subset/alpha.
18. A4/A5/A6 search same grids as A3.
19. eta=0 exactly reproduces corrected R0.
20. baseline audit reproduces frozen R0 values.
21. fresh corrected seeds are exactly 3131/3232/3333/3434.
22. old Phase-2E output remains unchanged.

Run targeted and full suite.

---

# 24. Final report

Print:

## A. Old Phase-2E forensic audit
Explicit PASS confirmation for F1–F8.

## B. Baseline audit
WM/FI Pearson and RMSE.

## C. Correct atlas mapping
- Yeo version
- label ordering
- system counts
- number of remapped/mislabeled old ROIs
- unresolved count

## D. Genuine LLM prior
- exact model tag/digest
- temperature
- five seeds
- 45/45 pairs per seed
- `n_seeds=5` for all rows
- raw response files
- prompt SHA256
- FROZEN/checksum
- WM/FI prior Pearson/Spearman
- top-5/top-10 overlap

## E. Corrected prediction results
A0–A6 Pearson/RMSE/MAE for WM and FI.

## F. Mechanism decomposition
A1-A0
A2-A1
A3-A2
A3-A0

with four seed deltas and positive seed counts.

## G. Prior specificity
A3-A4
A3-A5
A3-A6.

## H. Expert use
- eta distributions
- expert standalone Pearson
- selected top_k frequencies
- selected alpha/ratio frequencies

## I. Tests
targeted and full suite.

## J. Output bundle
paths.

Then exactly one:

```text
PHASE2E_FIX_DECISION: STRONG_GO
```

```text
PHASE2E_FIX_DECISION: PROMISING_GO
```

```text
PHASE2E_FIX_DECISION: SFC_ONLY_GO
```

or

```text
PHASE2E_FIX_DECISION: NO_GO
```

Finally:

```text
STATUS: PHASE2E_FIX_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE2E_FIX_NOT_VALID
```

and do not interpret prediction performance.

---

# 25. Stop condition

If the corrected Phase-2E-FIX is a valid `NO_GO`, stop searching for another
prediction architecture on this same 412-subject development corpus.

At that point:
- preserve the corrected negative result;
- retain the scientifically supported prior-specificity/biomarker analyses;
- discuss the evidence honestly with the advisor;
- only pursue a new headline accuracy claim with genuinely new subjects,
  genuinely new prior information, or a separately justified future study.
