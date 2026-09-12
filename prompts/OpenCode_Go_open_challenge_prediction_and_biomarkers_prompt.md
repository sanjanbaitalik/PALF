# OpenCode Go — Open Research Challenge
# Find ONE genuinely new method that improves BOTH cognitive prediction and biomarker discovery
# on the 412-subject development cohort, while keeping the 98-subject holdout sealed

## Role

Act as a senior machine-learning researcher, neuroimaging methodologist, and code auditor.

You have full access to the PALF repository and all existing development outputs.

Your job is **not** to blindly continue the previous method family.
Your job is to inspect the evidence, reason from first principles, and decide
whether there is ONE scientifically defensible, genuinely distinct method that
has a plausible chance to improve BOTH:

1. out-of-sample cognitive prediction, and
2. connectome biomarker quality.

You may use an unusual method if justified.

You must preserve scientific integrity. The goal is not to manufacture a win.

---

# 1. The sealed confirmatory holdout is untouchable

There are 98 genuinely new processed subjects reserved for later confirmation.

Canonical holdout manifest SHA256:

```text
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

Hard rules:

- DO NOT load holdout FC.
- DO NOT load holdout SC.
- DO NOT load holdout labels.
- DO NOT inspect holdout target distributions.
- DO NOT run R0 on holdout.
- DO NOT compute any holdout metric.
- DO NOT use holdout for debugging.
- DO NOT use holdout for model choice.
- DO NOT unlock holdout in this task.

The 98 subjects may only be used in a later, separate confirmatory run after a
model/procedure is completely frozen.

If any holdout subject is accessed:

```text
STATUS: INVALID_HOLDOUT_BREACH
```

and stop.

---

# 2. Development cohort

Use ONLY the frozen original 412-subject cohort.

Hard assertions:

```text
n_development = 412
n_holdout = 98
development ∩ holdout = empty
```

All model development, nested CV, biomarker diagnostics, and ablations occur on
the 412 only.

The 412-subject cohort has been used extensively for method development, so all
results on it are DEVELOPMENT evidence only.

---

# 3. Exact strong baseline that must not be weakened

The current corrected same-solver baseline is R0:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use the validated implementation in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

The historical audit must reproduce approximately:

```text
Working Memory:
Pearson r = 0.263515
RMSE      = 11.292921

Fluid Intelligence:
Pearson r = 0.370917
RMSE      = 4.566689
```

Hard audit tolerances:

```text
Pearson abs error <= 5e-4
RMSE abs error    <= 0.05
```

If the audit fails, stop.

Never replace R0 with ordinary Ridge.
Never weaken its search space.
Never change its folds to favor the new method.

---

# 4. What has already been tried

Before proposing anything, inspect the repository and verify the relevant
implementations/results yourself.

The following mechanism families have already been explored and should NOT be
repackaged under a new name unless you can demonstrate a mathematically
substantial difference.

## A. Global prior penalties on raw FC edges

Tried:
- anisotropic shrinkage;
- network/line-graph penalty;
- full PALF;
- adaptive prior strength.

Corrected same-solver result:
the matched prior did not beat R0 on both tasks.

## B. Differential/meta stacking

A residual/difference stack was tested and failed.

## C. Joint prior-aware Ridge / grouped penalties

Prior-aware group/multi-penalty Ridge failed.

## D. Multi-task residual NCR / shared-contrast residual modeling

Multiple residual/shared-contrast variants were explored.
No reliable final gain over R0 was established.

## E. Prior-selected raw-edge FC/SC experts

Low-dimensional prior-selected Ridge/NCR experts were tested.
Architecture effects existed, but matched semantic prior specificity did not
hold reliably.

## F. Nonlinear SC×FC system-pair coupling

A genuine LLM system-interaction prior and SC×FC coupling features were tested.
The SFC feature family itself did not beat R0 and matched prior did not give a
robust gain.

## G. Low-rank multi-task bilinear connectome regression

PG-MT-BCR with shared/task-specific factors was implemented correctly in the
final FIX2 run.

Valid FIX2 development outcome:

```text
A0 corrected R0:
WM r ≈ 0.2632
FI r ≈ 0.3822

C1 PG-MT-BCR matched:
WM r ≈ 0.2410
FI r ≈ 0.3730
```

Matched PG-MT-BCR did not beat R0.
Matched semantic prior specificity also failed.

Do not simply re-run PG-MT-BCR with different ranks/seeds and call it new.

---

# 5. Current scientific evidence

The evidence suggests:

1. R0 is difficult to beat.
2. LLM priors can alter the learned representation.
3. Matched-vs-wrong-prior effects sometimes appear, especially for WM.
4. But matched semantic priors have not consistently added independent
   predictive information beyond a strong no-prior model.
5. Random/shuffled/cross-task controls often match or beat the matched prior.
6. A useful next method therefore must create a genuinely new source of
   generalizable signal rather than another way of reweighting the same raw
   features.

Treat this as a diagnosis, not a conclusion you are forced to accept.

---

# 6. Freedom: what you ARE allowed to invent

You are deliberately given methodological freedom.

You may propose a method involving, for example:

- a new connectome representation;
- spectral or graph-frequency bases;
- multi-view latent-variable models;
- reduced-rank regression;
- multi-kernel learning;
- Bayesian hierarchical models;
- sparse latent factors;
- supervised/unsupervised graph embeddings;
- connectivity motifs;
- graph wavelets;
- diffusion/spectral kernels;
- network topology features;
- low-dimensional manifold representations;
- covariance-aware shrinkage;
- canonical-correlation / PLS-style latent variables;
- task-related multi-view factorization;
- calibrated ensembling of a genuinely distinct predictor;
- external neuroscience priors;
- LLM-generated priors in a representation not previously tested;
- a model where the LLM prior affects basis construction, hyperpriors,
  initialization, or feature grouping rather than raw edges.

You are NOT limited to NCR.

However:

- do not use a large deep model merely because it is fashionable;
- n=412 is small;
- every extra degree of freedom must be justified;
- if a nonlinear model is used, you must provide a rigorous biomarker
  attribution/faithfulness strategy;
- if you regenerate any LLM/external prior, it must be frozen BEFORE seeing
  predictive results from that prior;
- the prior-generation module may not consume HCP labels, residuals, model
  coefficients, or previous prediction outcomes.

---

# 7. What "genuinely new" means

Before coding, explicitly compare your proposed model to every family in
Section 4.

You must answer:

```text
What mathematical/statistical object is new?
What information can this model represent that R0 cannot?
What information can it represent that PALF/Phase2C/2D/2E/3A cannot?
Why should this improve generalization rather than merely add capacity?
Why is the idea plausible at n=412?
```

If the answer is only:
- "different lambda";
- "different rank";
- "different top-k";
- "different seed";
- "another stacking weight";
- "another residual learner on the same representation";

then reject your own proposal and do not run it.

---

# 8. Mandatory PRE-CODING proposal lock

Before implementing the final method, create:

```text
outputs/iclr/openchallenge/MODEL_PROPOSAL_LOCK.md
```

It must contain:

## A. Diagnosis
A concise evidence-based diagnosis of why previous methods failed.

## B. Proposed method
Exact mathematical formulation.

## C. Novelty map
A table comparing it against:
- R0;
- PALF;
- residual NCR;
- prior-selected experts;
- SFC coupling;
- PG-MT-BCR.

## D. Predictive mechanism
Why the model should add signal beyond R0.

## E. Biomarker mechanism
Why biomarkers from this model should be more:
- stable;
- task-specific;
- faithful.

## F. LLM prior role
Exactly how the prior enters, or a justified statement that the final method
does not use the LLM prior.

If the method does not use the LLM prior, explain why that is scientifically
preferable and how it affects the paper claim.

## G. Architecture-matched no-prior control
A comparator with the SAME architecture/capacity but without semantic prior
information.

## H. Frozen hyperparameter grid
Small and explicit.

## I. Frozen development seeds
Use exactly:

```text
OPENCHALLENGE_DEV_SEEDS = [4747, 4848, 4949, 5050]
outer_folds = 5
inner_folds = 3
```

## J. Predeclared prediction and biomarker gates
Use the gates in this prompt.

## K. Compute estimate
Expected runtime/memory.

After writing `MODEL_PROPOSAL_LOCK.md`, compute its SHA256 and save:

```text
MODEL_PROPOSAL_LOCK.sha256
```

After this point:

```text
DO NOT change the scientific method or hyperparameter grid based on outer-CV
results.
```

Bug fixes are allowed only if they restore the locked intended computation.

---

# 9. You may reject the challenge before coding

If, after inspecting the repo and evidence, you believe no genuinely new method
has a scientifically plausible chance, you may stop with:

```text
OPENCHALLENGE_DECISION: NO_METHOD_JUSTIFIED
```

and explain why.

That is preferable to running a cosmetic variant.

---

# 10. Development evaluation protocol

Use the 412-subject development cohort only.

Use:

```text
seeds = [4747, 4848, 4949, 5050]
5 outer folds
3 inner folds
```

Requirements:

- exact same outer splits for R0, proposed, no-prior control, and prior controls;
- all preprocessing fit on training scope only;
- all supervised feature selection fit on training scope only;
- no outer-test tuning;
- no fold dropping;
- no seed replacement;
- no cherry-picking.

Primary metric:

```text
Pearson r
```

Also report:
- RMSE;
- MAE.

Use seed-level means over the 5 outer folds for the development gates.

Do not present development p-values as confirmatory evidence.

---

# 11. Architecture-matched prior control is mandatory

If the proposed model uses the LLM/semantic prior, define:

```text
P0 = same model, same capacity, no semantic prior
P1 = matched semantic prior
```

The only allowed difference must be semantic prior information.

Also test:

```text
cross-task
shuffled
random
```

using the exact same model and search space.

A claim of "prior improvement" requires:

```text
P1 > P0
```

and matched prior must outperform placebo controls.

If the proposed method does not use a semantic prior, state clearly that the
paper direction has changed; do not fabricate a prior-based claim.

---

# 12. Prediction gate

A method is prediction-successful only if ALL hold:

## Gate P1 — beats R0

For BOTH WM and FI:

```text
mean(Proposed - R0) >= +0.005 Pearson
positive seed deltas >= 3/4
```

and at least one task:

```text
mean(Proposed - R0) >= +0.010
```

## Gate P2 — semantic prior value

If semantic prior is used, for BOTH tasks:

```text
mean(P1 - P0) >= +0.003
positive seed deltas >= 3/4
```

## Gate P3 — specificity

If semantic prior is used:

For BOTH tasks:

```text
matched > shuffled
matched > random
```

in mean development Pearson.

Matched must not lose to cross-task on both tasks.

No loosening of these thresholds after results.

---

# 13. Biomarker requirements

Prediction alone is NOT enough.

The final method must produce a biomarker object that is scientifically
well-defined.

For a linear/latent-linear model:
- export true coefficients/loadings;
- validate coefficient-to-prediction reconstruction.

For a nonlinear model:
- define the attribution method before outer results;
- attribution must be deterministic for fixed model/data;
- validate attribution using held-out perturbation.

Do not use "correlation with the prior" as biomarker validation.

That is circular.

---

# 14. Biomarker stability

Across the 20 outer-development fits, evaluate as applicable:

```text
edge/loadings absolute-rank Spearman
top-100 edge Jaccard
top-300 edge Jaccard
top-10 ROI Jaccard
top-20 ROI Jaccard
sign consistency
```

For latent/non-edge representations, define the exact mapping to edge/ROI
importance in `MODEL_PROPOSAL_LOCK.md`.

Important:

- if a model abstains and contributes zero to final prediction, mark the
  biomarker as `ABSTAINED`;
- do not treat an all-zero map's deterministic tie order as biomarker stability.

Compare proposed vs architecture-matched no-prior model.

---

# 15. Held-out biomarker faithfulness on DEVELOPMENT outer folds

Biomarker ranking is computed from outer-training data/model only.

On each development outer-test fold:

1. identify top biomarkers;
2. perturb/mask them WITHOUT retraining;
3. compare with bottom and deterministic random same-size sets.

At minimum test:

```text
top 5 ROIs
top 10 ROIs
```

or equivalent feature sizes.

Define:

```text
delta_RMSE =
RMSE_masked - RMSE_unmasked
```

Therefore:

```text
positive delta_RMSE = masking hurts prediction = evidence of faithfulness
negative delta_RMSE = masking improves prediction = unfavorable
```

Do not reverse the sign interpretation.

Use at least 50 deterministic random same-size perturbations per outer split.

Report:

```text
top - random_mean
top - bottom
```

---

# 16. Biomarker task specificity

For WM biomarkers:
- perturb WM-selected biomarkers in WM and FI predictions.

For FI biomarkers:
- perturb FI-selected biomarkers in FI and WM predictions.

A task-specific biomarker story is supported when, on average:

```text
WM-selected perturbation hurts WM more than FI
FI-selected perturbation hurts FI more than WM
```

where scientifically applicable.

---

# 17. Biomarker specificity to semantic prior

If the model uses an LLM prior, compare biomarker quality under:

```text
matched
no-prior
shuffled
random
cross-task
```

The matched prior should not merely make maps smoother.

It should improve downstream stability/faithfulness/task-specificity relative
to controls.

---

# 18. Biomarker success gate

The proposed method must satisfy ALL:

## Gate B1 — held-out faithfulness

For BOTH tasks:

```text
mean(top10 delta_RMSE - random10 delta_RMSE) > 0
```

and positive seed-level differences in at least:

```text
3/4 seeds
```

## Gate B2 — architecture-matched biomarker advantage

Against the architecture-matched no-prior model, the proposed model must show
a positive mean improvement on at least TWO of:

```text
absolute-rank stability
top-10 ROI Jaccard
held-out top-vs-random faithfulness
task-specific perturbation contrast
```

for EACH task.

Do not require all metrics to improve.

## Gate B3 — no circularity

Biomarker validation cannot use:
- prior alignment alone;
- training-set deletion only;
- outer-test data for ranking.

---

# 19. Overall success gate

The method succeeds only if:

```text
Prediction Gates P1-P3 PASS
AND
Biomarker Gates B1-B3 PASS
AND
all validity/leakage tests PASS
```

If the method does not use a semantic prior, omit P2/P3 but explicitly report:

```text
PRIOR_CLAIM_NOT_APPLICABLE
```

and do not call the method "LLM-prior improved".

If overall gate passes:

```text
OPENCHALLENGE_DECISION: CANDIDATE_FOR_FREEZE
```

If prediction passes but biomarker gate fails:

```text
OPENCHALLENGE_DECISION: PREDICTION_ONLY_NO_FREEZE
```

If biomarker passes but prediction fails:

```text
OPENCHALLENGE_DECISION: BIOMARKER_ONLY_NO_FREEZE
```

If both fail:

```text
OPENCHALLENGE_DECISION: NO_GO
```

Do NOT touch the 98-subject holdout regardless of result in this prompt.

---

# 20. Scientific integrity constraints

Forbidden:

- inspecting holdout;
- outer-test hyperparameter changes;
- removing "bad" seeds/folds;
- trying many new architectures after observing outer results;
- weakening R0;
- choosing the best of many random priors;
- generating priors after seeing model residuals;
- changing primary metrics after results;
- declaring tiny numerical parity as a breakthrough;
- claiming biomarker discovery solely because coefficients align with the prior.

Allowed:

- fixing an implementation bug if it violates `MODEL_PROPOSAL_LOCK.md`;
- rerunning the same locked experiment after such a bug fix;
- stopping early if the method is clearly invalid computationally.

---

# 21. Code-audit requirements

Before trusting results, add tests for the exact bug classes previously found
in this project:

- baseline identity and calibration;
- correct FP+SC R0;
- OOF labels aligned to validation subjects;
- residual predictions in correct units;
- inner preprocessing fit only on inner training;
- final refit matches selected procedure;
- task outputs not overwritten;
- architecture-matched control differs only in prior;
- coefficient/attribution reconstruction;
- perturbation sign convention;
- abstained maps excluded from stability;
- no holdout access.

Tests must be functional.

Do NOT write tautological tests such as `condition or True`.

---

# 22. Required output directory

Create:

```text
outputs/iclr/openchallenge/
```

Required:

```text
MODEL_PROPOSAL_LOCK.md
MODEL_PROPOSAL_LOCK.sha256

BASELINE_AUDIT.json
HOLDOUT_SEAL_REPORT.json
VALIDATION_REPORT.json
RUN_REPORT.md

split_metrics.csv
seed_metrics.csv
model_comparison.csv
selection_details.csv
optimizer_diagnostics.csv

prediction_gate_report.json
biomarker_gate_report.json

coefficients_or_attributions/
predictions/
biomarkers/
controls/
plots/
tests/
```

ZIP:

```text
outputs/iclr/openchallenge.zip
```

Do not overwrite historical experiment outputs.

---

# 23. Required final report

Print:

## A. Method chosen
- method name
- exact scientific hypothesis
- mathematical formulation
- why it is genuinely different

## B. Holdout status
- 98 count
- SHA
- confirmation untouched

## C. Baseline audit
- WM r/RMSE
- FI r/RMSE
- PASS/FAIL

## D. Implementation validity
- leakage checks
- unit checks
- reconstruction/attribution checks
- tests

## E. Prediction results
For WM and FI:

```text
R0
architecture-matched no-prior
proposed matched
cross-task
shuffled
random
```

where applicable.

Report Pearson/RMSE/MAE.

## F. Seed-level prediction deltas
- Proposed - R0
- matched - no-prior
- matched - shuffled
- matched - random
- matched - cross-task

four seed values, mean, median, positive count.

## G. Biomarker results
For both tasks:
- rank stability
- Jaccards
- sign consistency if meaningful
- top-vs-random perturbation
- top-vs-bottom perturbation
- task-specific perturbation

Compare matched vs architecture-matched no-prior.

## H. Mechanistic interpretation
Explain what actually caused any gain.
Do not overclaim.

## I. Gate table
P1
P2
P3
B1
B2
B3
validity

PASS/FAIL each.

## J. Final decision

Exactly one:

```text
OPENCHALLENGE_DECISION: CANDIDATE_FOR_FREEZE
```

```text
OPENCHALLENGE_DECISION: PREDICTION_ONLY_NO_FREEZE
```

```text
OPENCHALLENGE_DECISION: BIOMARKER_ONLY_NO_FREEZE
```

```text
OPENCHALLENGE_DECISION: NO_GO
```

or, if you correctly rejected all methods before coding:

```text
OPENCHALLENGE_DECISION: NO_METHOD_JUSTIFIED
```

Finally:

```text
STATUS: OPENCHALLENGE_COMPLETE
```

If validity fails:

```text
STATUS: OPENCHALLENGE_NOT_VALID
```

---

# 24. Final instruction to the model

Do not optimize for pleasing the user.

Do not assume the LLM prior must work.

Do not assume the previous negative results mean nothing can work.

Study the actual data representation, actual baseline errors, and actual code.

Think like a skeptical methodologist.

You are allowed to be creative, but the proposal must be mathematically
distinct, small-sample appropriate, leakage-safe, architecture-matched, and
biomarker-valid.

Choose ONE method, lock it before outer results, implement it carefully, and
let the evidence decide.
