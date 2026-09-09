# PALF ICLR 2027 — Final Experimental Artifact Repair + Clean Manuscript-Freeze Bundle

You are working on the PALF repository for the ICLR 2027 manuscript.

## Repository / frozen scientific state

Authoritative GitHub commit before this repair:
`9abbb56`

The **primary 400-split experiment is scientifically frozen and MUST NOT be rerun**.

Primary protocol:
- 2 targets: Working Memory and Fluid Intelligence
- 10 seeds
- 5 outer folds
- 4 same-solver configurations
- 400 condition-specific outer splits total
- Same generalized FC solver scale for all four configurations
- Fully cross-fitted OOF fusion
- Final branch hyperparameters reselected on the full outer-training set
- Outer test used for evaluation only

Internal configuration IDs may remain in code/CSV lineage:
- R0 = same-solver no prior
- R1 = anisotropy only
- R2 = network penalty only
- R3 = full PALF

However, **do not expose R0/R1/R2/R3 in manuscript-facing figures or LaTeX tables**. Use descriptive labels only.

The currently authoritative primary output is:

`outputs/iclr/palf_crossfit_ablation_v1/`

Expected current fused means from the real-prior rerun:
- Working Memory:
  - same-solver no prior: ~0.2573
  - anisotropy only: ~0.2578
  - network penalty only: ~0.2637
  - full PALF: ~0.2625
- Fluid Intelligence:
  - same-solver no prior: ~0.3642
  - anisotropy only: ~0.3678
  - network penalty only: ~0.3703
  - full PALF: ~0.3700

Expected primary Full-PALF minus same-solver-no-prior contrasts:
- Working Memory: Δr ~ +0.0052, 95% CI approximately [-0.0044, +0.0150], raw Wilcoxon p ~ 0.4922
- Fluid Intelligence: Δr ~ +0.0059, 95% CI approximately [-0.0017, +0.0143], raw Wilcoxon p ~ 0.2324

If the loaded authoritative checkpoints do not reproduce these values within ordinary rounding tolerance, STOP and report the discrepancy. Do not silently proceed with a different run.

---

# Goal

Repair the **post-processing, prior controls, statistics, and manuscript figures only**, without rerunning the frozen 400 primary splits.

Then create a **completely separate, clean manuscript-freeze bundle** containing only the corrected current results, corrected tables, corrected figures, provenance, and validation artifacts.

The old and new artifacts MUST NOT mix.

Do **not** edit the manuscript `.tex`, supplement `.tex`, or `.bib` files in this task. We will edit the paper only after this artifact-repair stage is accepted.

---

# Critical code issues already identified

I inspected the current codebase. Address all of the following.

## 1. Prior-control refits are currently not publication-safe

Relevant files include:

- `scripts_paper/prior_control_refits.py`
- `src/metascfc/experiments/palf_crossfit_ablation.py`

### 1A. Use the CURRENT real-prior Full-PALF checkpoints only

The prior controls must be regenerated from:

- `outputs/iclr/palf_crossfit_ablation_v1/working_memory/checkpoint.pkl`
- `outputs/iclr/palf_crossfit_ablation_v1/fluid_intelligence/checkpoint.pkl`

For every target, use the **50 current Full-PALF outer splits**:
10 seeds × 5 folds = 50 splits.

For each split, run the three fixed prior swaps:
1. cross-task
2. shuffled
3. random

Therefore the correct number is:

- 50 Full-PALF splits × 3 controls = **150 refits per task**
- 2 tasks × 150 = **300 refits total**

The current `prior_control_refits.py` docstring incorrectly says “300 FC refits per task / 100 R3 splits.” Fix this.

### 1B. Fixed-swap definition

For each current Full-PALF split:

- keep the exact outer train/test indices;
- keep the Full-PALF-selected `lambda_fc`;
- keep the Full-PALF-selected `lambda_l`;
- rebuild the prior-dependent anisotropic diagonal and line-graph/network penalty from the CONTROL prior;
- refit only the prior-aware FC branch on the same outer-training subjects;
- keep the already-fitted current Full-PALF SC test prediction fixed;
- keep the current Full-PALF fusion weights fixed;
- evaluate the control prior on the same outer-test subjects.

No hyperparameter reselection is allowed in the fixed-swap controls.

### 1C. Fusion key bug / no fake fallback weights

For Full PALF the fusion dictionary uses `FP` + `SC`, not `FC` + `SC`.

In `src/metascfc/experiments/palf_crossfit_ablation.py`, `evaluate_prior_swap_fixed()` currently contains logic equivalent to:

```python
w_fc = matched_result.fusion_weights.get("FC", 0.5)
```

This is wrong for the prior-aware branch.

Fix it so Full PALF uses explicit keys:

```python
w_fp = matched_result.fusion_weights["FP"]
w_sc = matched_result.fusion_weights["SC"]
```

and the fused prediction is:

```python
fused = w_fp * ctrl_fp_test + w_sc * sc_test
```

Do not use `.get(..., 0.5)` as a silent fallback for required fusion keys anywhere in final PALF post-processing. Missing required keys must raise a clear error.

`prior_control_refits.py` already uses `FP` in its main refit function; preserve that correct behavior and add assertions.

### 1D. Matched-reference integrity gate

Before running any control refit, verify for every target/seed/fold that the “matched” reference is the current Full-PALF split loaded from the current checkpoint.

Add hard assertions that:

- 50 Full-PALF splits exist per task;
- `(seed, outer_fold)` pairs are unique and exactly cover 10 × 5;
- stored Full-PALF fused test metrics match recomputation from:
  `w_fp * fp_test_pred + w_sc * sc_test_pred`;
- stored Full-PALF fused predictions are reproduced to tight floating-point tolerance;
- the matched-reference means reproduced from these 50 current splits are ~0.2625 WM and ~0.3700 FI.

If the matched prior-control reference instead reproduces the older ~0.2628 / ~0.3689 values, STOP. That means stale artifacts are being used.

---

# 2. Fix the prior-control statistics

The current `scripts_paper/prior_control_refits.py` runs Wilcoxon directly on the 50 fold-level values and then labels them as `n_seeds=50`.

That is not the intended inferential unit.

## Required inference unit

For every:
- task
- control prior
- metric/branch

first aggregate the 5 outer folds **within each seed**.

This must yield exactly **10 paired seed summaries**.

Then compare matched vs control across these 10 paired seed summaries.

### Required direction convention

Use a single unambiguous direction everywhere:

`matched_minus_control = matched_r - control_r`

Positive values therefore mean the matched semantic prior performs better than the control prior.

Do not keep ambiguous `control - matched` signs in manuscript-facing files.

You may preserve raw control-minus-matched values only in an explicitly named diagnostic column, but all published deltas must use `matched_minus_control`.

## Required statistics

For each target × control:
- matched mean Pearson r
- control mean Pearson r
- mean matched-minus-control Δr
- median Δr
- number of positive seed differences out of 10
- exact/two-sided paired Wilcoxon statistic and raw p
- paired bootstrap 95% CI using the 10 seed-level differences, 10,000 resamples
- paired Cohen's dz
- Holm-adjusted p

Apply Holm correction across the **three prior controls within each target**, for the fused prediction family.

If you also report FP-unfused prior-control statistics, treat that as a separate Holm family from fused prediction.

Create:
- `prior_control_split_metrics.csv` — 300 split-level control rows total
- `prior_control_seed_metrics.csv` — 60 fused seed/control rows total if one row per task × control × seed, plus separate FP rows only if encoded explicitly
- `prior_control_comparisons.csv` — clearly one inferential row per task × control × metric family
- `table_prior_controls.tex`

The comparison file must explicitly contain:
- `n_seeds = 10`
- `raw_p`
- `holm_p`
- `ci_95_lower`
- `ci_95_upper`
- `cohen_dz`
- `positive_seeds`

Do not call 50 fold rows “50 seeds.”

---

# 3. Fix the PRIMARY Holm correction

Relevant file:
- `scripts_paper/postprocess_palf_ablation.py`

The current code applies Holm correction grouped by `(task, family)`.
For the primary family this gives only one primary test inside each task and therefore leaves adjusted p equal to raw p.

That is not the planned primary family.

For the two prespecified primary target tests, Holm correction must be applied **across Working Memory and Fluid Intelligence together**.

Expected values from the current frozen primary run:
- WM raw p ~ 0.4922 → Holm p ~ 0.4922
- FI raw p ~ 0.2324 → Holm p ~ 0.4648

Do not hard-code these values into the analysis. Compute them from the current seed-level data and assert that the reproduced values are consistent with these expected values to rounding tolerance.

Component contrasts are secondary/descriptive. Keep their statistical family definition explicit and do not mix them into the two-test primary family.

Produce a clean manuscript-facing:
- `primary_comparisons.csv`

with the two primary rows and both raw and Holm-adjusted p values.

---

# 4. Repair fusion-weight export

Relevant files include:

- `scripts/116_run_palf_crossfit_ablation.py`
- `scripts_paper/postprocess_palf_ablation.py`
- `scripts_paper/fix_fusion_weights.py`

The current exporters save:

```python
"w_FC": split.fusion_weights.get("FC", 0.5)
```

for every condition.

For anisotropy-only, network-only, and Full PALF the active FC-side branch is `FP`, so this incorrectly writes 0.5 placeholders.

## Required schema

Do not silently overload `w_FC`.

Export explicit fields such as:

- `task`
- `seed`
- `outer_fold`
- `condition`
- `condition_label`
- `active_fc_branch` (`FC` or `FP`)
- `w_fc` — populated only when active branch is ordinary `FC`, else NaN
- `w_fp` — populated only when active branch is prior-aware `FP`, else NaN
- `w_sc`
- `weight_sum`
- `source = "checkpoint_fusion_weights"`

For every row assert:
- exactly one of `w_fc` or `w_fp` is active;
- active FC-side weight + `w_sc` = 1 within numerical tolerance;
- no required value comes from a default 0.5 fallback.

Create a new:
- `fusion_weights_corrected.csv`

inside the clean bundle.

Do not overwrite the old `fusion_weights.csv` in `palf_crossfit_ablation_v1`.

Also correct any helper function that still reads `FC` for Full PALF.

---

# 5. Correct all current PALF figures

Relevant files include:

- `scripts_paper/postprocess_palf_ablation.py`
- `scripts_paper/make_main_results_figure.py`
- `scripts_paper/plot_fusion_weights.py`
- `scripts_paper/plot_seed_deltas.py`
- `scripts_paper/make_supplement_composites.py`
- any other final-figure builder

The existing `make_main_results_figure.py` is hard-coded to old:
- `outputs/iclr/lf1_final_10x5`
- `outputs/iclr/lf1_final_evidence_audit`
- biomarker alignment data

It must NOT be used for the final PALF paper.

The current `postprocess_palf_ablation.py` also has a fusion plotting bug: it reads:

```python
split.fusion_weights.get("FC", 0.5)
```

for Full PALF.

Fix this to the explicit `FP` weight.

## New main figure

Create a new publication-ready `fig_main_results.pdf` and `.png` from CURRENT PALF primary data only.

Use a 2×2 layout:

### Panel A
Working Memory seed-level:
`Full PALF - Same-solver no prior`
fused Pearson Δr, one value per seed.

### Panel B
Fluid Intelligence seed-level:
`Full PALF - Same-solver no prior`
fused Pearson Δr.

### Panel C
Working Memory four-configuration fused Pearson means with seed SD error bars.

Labels:
- Same-solver no prior
- Anisotropy only
- Network penalty only
- Full PALF

### Panel D
Fluid Intelligence same four-configuration plot.

Do not place R0/R1/R2/R3 on manuscript-facing axes or legends.

Do not show biomarker alignment in the main figure.

## Supplementary figures

Generate, from current authoritative sources only:

1. `supp_fig_ablation_comparison.pdf/png`
2. `supp_fig_fusion_weights.pdf/png`
   - Full-PALF `w_FP` distribution for WM and FI
   - read the true `FP` weights, never `FC` placeholders
   - show the actual mean in the plot/manifest
3. `supp_fig_prior_controls.pdf/png`
   - only after the corrected 300 fixed-swap refits finish
   - use seed-level matched-minus-control summaries or matched/control means
   - clearly indicate these are fixed-swap controls
4. `supp_fig_roi_prior_working_memory.pdf/png`
5. `supp_fig_roi_prior_fluid_intelligence.pdf/png`

Regenerate the two ROI-prior maps from the authoritative frozen prior CSVs rather than copying an old `figures_iclr` directory wholesale.

## Figure-source manifest

Create `FIGURE_MANIFEST.md` containing for every figure:
- exact source CSV/checkpoint
- generation script
- target
- plot semantics
- SHA-256 of source files where practical
- statement that no LF1 artifact was used

Add an automated check that searches the new figure scripts/manifest and new bundle for prohibited source strings:
- `lf1_final_10x5`
- `lf1_final_evidence_audit`
- `prior_aware_late_fusion_integrity_audit`

The clean bundle must fail validation if any final figure depends on those paths.

---

# 6. Do NOT reconstruct biomarkers/coefficient stability in this repair

Relevant problematic files include:

- `scripts_paper/generate_remaining_outputs.py`
- `scripts_paper/export_frozen_fp_coefficients.py`
- `scripts_paper/plot_biomarker_alignment.py`
- `scripts_paper/plot_frozen_top_edges.py`
- `scripts_paper/plot_top_edges.py`

Current PALF artifacts do not contain a validated final 6,670-dimensional primal FC coefficient vector for every fit.

The current `fp_final.alpha` is a dual / subject-space solver state, not directly the edge coefficient vector required for top-edge, ROI-alignment, or coefficient-rank-stability claims.

The current `generate_remaining_outputs.py` explicitly writes null/NaN biomarker alignment values and uses prediction correlation as a proxy for fit stability. These are not publication-ready biomarker measurements.

## Required action

For this task:

- DO NOT attempt a new biomarker/coefficient reconstruction.
- DO NOT regenerate top-edge figures.
- DO NOT regenerate ROI-prior alignment figures.
- DO NOT regenerate coefficient stability / top-k overlap claims.
- DO NOT include any old LF1 biomarker/top-edge figure in the clean bundle.

Make the final post-processing path fail loudly or mark these outputs as unsupported instead of writing misleading NaN/proxy files.

For example:
- no `biomarker_fit_metrics.csv`
- no `biomarker_seed_metrics.csv`
- no `top_edges_*.tsv`
- no `top_edges*.pdf/png`
- no coefficient-alignment figure
- no “resampling coefficient stability” figure

Create:
`UNSUPPORTED_ARTIFACTS.md`

and state clearly that:
- final primal FC coefficients were not archived in validated edge space;
- biomarker alignment / top-edge / coefficient stability are intentionally excluded from the manuscript-freeze bundle;
- no old LF1 coefficient evidence is carried forward.

Also fix misleading naming/comments where feasible:
- a file storing dual `alpha` should not be described as “final FC coefficient vectors.”
- If preserving this internal export, call it solver/dual state, not primal edge coefficients.

Do not delete historical code if it is useful for provenance; just prevent it from contaminating the final bundle.

---

# 7. Build a completely separate clean freeze bundle

Do NOT overwrite, rename, move, or delete:

- `outputs/iclr/palf_crossfit_ablation_v1/`
- `figures_iclr/`
- any LF1 outputs

Treat the current primary output directory as READ-ONLY scientific source data.

Create this NEW directory:

`outputs/iclr/palf_manuscript_freeze_9abbb56/`

Use the following structure:

```text
outputs/iclr/palf_manuscript_freeze_9abbb56/
├── README.md
├── SOURCE_PROVENANCE.json
├── VALIDATION_REPORT.json
├── MANIFEST.json
├── checksums.sha256
├── COMPLETE
│
├── results/
│   ├── primary_condition_summary.csv
│   ├── primary_seed_metrics.csv
│   ├── primary_comparisons.csv
│   ├── branch_fusion_summary.csv
│   ├── fusion_weights_corrected.csv
│   ├── prior_control_split_metrics.csv
│   ├── prior_control_seed_metrics.csv
│   └── prior_control_comparisons.csv
│
├── tables/
│   ├── table_primary_prediction.tex
│   ├── table_component_ablation.tex
│   └── table_prior_controls.tex
│
├── figures/
│   ├── fig_main_results.pdf
│   ├── fig_main_results.png
│   ├── supp_fig_ablation_comparison.pdf
│   ├── supp_fig_ablation_comparison.png
│   ├── supp_fig_fusion_weights.pdf
│   ├── supp_fig_fusion_weights.png
│   ├── supp_fig_prior_controls.pdf
│   ├── supp_fig_prior_controls.png
│   ├── supp_fig_roi_prior_working_memory.pdf
│   ├── supp_fig_roi_prior_working_memory.png
│   ├── supp_fig_roi_prior_fluid_intelligence.pdf
│   └── supp_fig_roi_prior_fluid_intelligence.png
│
├── audit/
│   ├── matched_reference_integrity.json
│   ├── fusion_weight_integrity.json
│   ├── prior_control_integrity.json
│   ├── primary_statistics_integrity.json
│   ├── no_legacy_source_audit.txt
│   └── UNSUPPORTED_ARTIFACTS.md
│
└── logs/
    ├── prior_control_refits.log
    ├── freeze_bundle_build.log
    └── tests.log
```

If a file is not meaningful, adjust the exact name minimally, but preserve this clean separation.

Also create a ZIP:

`outputs/iclr/palf_manuscript_freeze_9abbb56.zip`

containing only the clean freeze directory.

Do not include:
- checkpoints
- LF1 outputs
- old figures
- NaN biomarker files
- top-edge artifacts
- stale prior-control files

unless an artifact is explicitly required for provenance. Prefer hashes/references over copying historical data.

---

# 8. Manuscript-facing table rules

The generated LaTeX tables in the clean bundle must follow Prof. Rajapakse's formatting requirements:

- no manuscript-facing `R0`, `R1`, `R2`, `R3`, `LF1`, etc.;
- use descriptive configuration names;
- do not bold mathematical symbols;
- bold only winning numeric results;
- use `*` only for statistically significant results worth reporting;
- no significance star on the two primary Full-PALF vs no-prior tests, because they are nonsignificant;
- if prior controls survive Holm correction at p < .05, star only those appropriate prior-control numeric results and define the star in the caption/footnote.

Do not make claims of:
- significant overall PALF prediction improvement;
- component synergy.

The component result is descriptive: network penalty only has the highest mean fused Pearson r for both targets in the current primary run.

---

# 9. Validation gates

The bundle is considered COMPLETE only if all of the following pass.

## Primary-data gates
- 2 tasks
- 10 seeds/task
- 5 outer folds/seed
- 4 configurations
- 400 condition-specific primary splits represented
- current matched real-prior checkpoints used
- WM Full PALF fused mean ~0.2625
- FI Full PALF fused mean ~0.3700
- primary WM Δr ~+0.0052
- primary FI Δr ~+0.0059

## Primary-inference gates
- exactly 10 paired seed summaries per primary target
- WM raw p ~0.4922
- FI raw p ~0.2324
- two-target Holm values approximately:
  - WM 0.4922
  - FI 0.4648

## Fusion gates
- no silent 0.5 fallback for required branch weights
- R1/R2/R3 use `FP` + `SC`
- R0 uses `FC` + `SC`
- every active pair sums to 1 within tolerance
- true Full-PALF weights used in plots

## Prior-control gates
- 50 current Full-PALF matched splits per target
- 3 controls per split
- 150 refits per task
- 300 total refits
- exact same train/test indices as matched split
- exact same selected lambda_F/lambda_L as matched Full PALF
- fixed current SC test predictions
- fixed current Full-PALF fusion weights
- 5 folds aggregated within seed before inference
- exactly 10 paired seed values for Wilcoxon
- Holm across 3 controls within each target
- matched-reference fused values agree with current Full-PALF primary predictions split-by-split

## Legacy-contamination gates
The new bundle must contain no source dependency or manuscript figure source pointing to:
- `lf1_final_10x5`
- `lf1_final_evidence_audit`
- old fixed-prior swap outputs
- old biomarker audit outputs

## Biomarker gates
- no NaN biomarker table presented as evidence
- no dual alpha presented as edge-space coefficients
- no top-edge manuscript figure
- no coefficient stability manuscript claim
- `UNSUPPORTED_ARTIFACTS.md` explicitly documents exclusion

---

# 10. Tests

Add targeted tests, preferably in:

`tests/test_palf_manuscript_freeze.py`

At minimum test:

1. Full-PALF fusion requires keys `FP` and `SC`.
2. Same-solver no-prior fusion requires keys `FC` and `SC`.
3. Missing required branch weight raises instead of defaulting to 0.5.
4. Corrected fusion-weight CSV has 400 rows if exporting all conditions, with valid active-branch schema.
5. Prior-control matched reference equals current Full-PALF split predictions.
6. Prior-control raw output has 300 rows total.
7. Prior-control seed aggregation has exactly 10 seeds per task/control.
8. Prior-control inferential `n_seeds` is 10, never 50.
9. Primary Holm correction is across the two targets.
10. Final main figure source contains no LF1 path.
11. Clean bundle contains no biomarker/top-edge artifacts.
12. Clean bundle contains no prohibited legacy path strings.
13. All generated CSVs have finite values in columns that are supposed to be numerical evidence.
14. All PDF/PNG expected figures exist and are non-empty.
15. `checksums.sha256` covers all freeze-bundle files except itself if necessary.

Run the targeted tests first.

Then run the full test suite and report:
- total passing
- total failing
- whether failures are pre-existing and unrelated
- exact failing tests

Do not hide or delete unrelated failing tests merely to obtain a green count.

---

# 11. Implementation organization

Prefer a small, explicit finalization pipeline rather than continuing to overload old LF1 scripts.

It is acceptable and recommended to add a new orchestrator such as:

`scripts_paper/build_palf_manuscript_freeze.py`

This script should:

1. verify source commit/provenance;
2. validate the frozen primary output;
3. create the new clean directory;
4. regenerate current primary summaries;
5. compute corrected primary Holm statistics;
6. export corrected fusion weights;
7. run/refuse stale prior controls and execute the 300 corrected refits;
8. aggregate prior controls to seed level and compute corrected statistics;
9. generate only current valid tables;
10. generate only current valid figures;
11. write audit reports;
12. write checksums and manifest;
13. create `COMPLETE` only if every required gate passes;
14. create the final ZIP.

Make all output paths configurable, but default to:

`outputs/iclr/palf_manuscript_freeze_9abbb56/`

The build must be deterministic given the frozen primary checkpoints and frozen priors.

---

# 12. Protect the frozen primary outputs

This task is a repair/finalization task, not another primary experiment.

DO NOT:
- rerun `scripts/116_run_palf_crossfit_ablation.py --mode production`;
- alter the scientific primary checkpoints;
- overwrite `outputs/iclr/palf_crossfit_ablation_v1`;
- overwrite `figures_iclr`;
- regenerate the semantic priors;
- tune any hyperparameters;
- add new conditions;
- substitute a different target or cohort;
- edit the paper TeX/Bib files.

You MAY:
- read the current checkpoints;
- run the 300 fixed prior-swap FC refits;
- correct post-processing/statistics;
- repair exporters and plots;
- add tests;
- generate a clean new bundle.

---

# 13. Final report required from OpenCode

When finished, print a concise but complete report with:

## A. Source
- starting commit / `git rev-parse HEAD`
- whether the working tree was initially clean
- authoritative primary output path
- source run fingerprint if available

## B. Code fixes
List each changed source file and what was repaired.

## C. Primary results
For each target/configuration:
- FP Pearson r
- fused Pearson r
- fused RMSE
- fused MAE

Then print the two primary Full-PALF vs no-prior contrasts:
- mean Δr
- 95% CI
- raw p
- Holm p
- positive seeds / 10
- Cohen dz

## D. Correct fusion weights
Print Full-PALF:
- WM mean `w_FP`
- FI mean `w_FP`
- min/max
- proof that `w_FP + w_SC = 1`

## E. Corrected prior controls
For each target/control:
- matched fused r
- control fused r
- matched-minus-control Δr
- 95% CI
- raw p
- Holm p
- positive seeds / 10
- Cohen dz

Also print:
- refits per task
- total refits
- proof `n_seeds=10` for inference

## F. Excluded unsupported evidence
Explicitly state that no biomarker alignment, top-edge, or coefficient-stability result is carried into the clean bundle.

## G. Bundle
Print:
- final directory path
- ZIP path
- manifest path
- figure manifest path
- checksums path
- number of files
- validation status

## H. Tests
Print targeted and full-suite test results separately.

End with exactly one of:

`STATUS: PALF_MANUSCRIPT_FREEZE_BUNDLE_READY`

or, if any gate fails:

`STATUS: NOT_READY`

If NOT_READY, explain exactly which gate failed and do not create `COMPLETE`.

---

# Important scientific interpretation constraint

The final repaired artifacts must remain consistent with the frozen result:

- Prediction improvements are modest positive trends.
- The primary Full-PALF vs same-solver-no-prior differences are not statistically significant.
- Network penalty only has the highest mean fused Pearson correlation for both targets in the current primary ablation.
- Do not claim component synergy.
- Do not resurrect older LF1 values or old biomarker evidence.

Do the implementation, run the required prior-control refits and validation, build the isolated clean bundle, and report the final verified numbers.
