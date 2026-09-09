# PALF ICLR 2027 — Critical Same-Solver Fusion Correction + Final Freeze Bundle v2

## Context

We have reviewed the current codebase and the bundle:

- current primary source output: `outputs/iclr/palf_crossfit_ablation_v1/`
- current freeze bundle: `outputs/iclr/palf_manuscript_freeze_9abbb56/`
- current code lineage: commit `9abbb56` plus the recent local fixes

The prior-aware branch wiring and the real semantic priors are now correct. The 400 branch fits do **not** need to be rerun unless the stored checkpoint is missing the required OOF/test branch predictions.

However, a critical semantic mismatch remains in the **same-solver baseline fusion**.

---

# 1. Critical issue: R0 is not actually using the same-solver FC branch in fusion

The intended primary baseline is:

**Same-solver no-prior = generalized FC solver with `D = I`, `lambda_L = 0` + SC Ridge, followed by the same fully cross-fitted late fusion.**

In the current code:

- `generate_crossfit_oof()` computes:
  - `fc_oof`: ordinary sklearn Ridge FC diagnostic
  - `fp_oof`: generalized FC branch using the PALF/MS-A-NCR solver
- `reselect_and_fit_final()` similarly computes:
  - `fc_final`: ordinary sklearn Ridge FC
  - `fp_final`: generalized FC branch

For R0, `fp_final` is exactly the desired **same-solver no-prior** branch because:
- `D = I`
- `lambda_L = 0`
- the same generalized solver scale and machinery are used.

But `evaluate_ablation_split()` currently selects and applies fusion for R0 using:

```python
{"FC": oof.fc_oof, "SC": oof.sc_oof}
```

and later:

```python
w_fc * fc_final.test_pred + w_sc * sc_final.test_pred
```

That means the published R0 fused baseline is actually ordinary FC Ridge + SC Ridge, not the claimed same-solver generalized FC baseline.

This must be fixed before the manuscript is frozen.

---

# 2. Correct scientific behavior

For **all four primary ablation conditions**, the active FC-side branch for the main fusion must be the generalized FC branch:

- Same-solver no prior: `FP + SC`
- Anisotropy only: `FP + SC`
- Network penalty only: `FP + SC`
- Full PALF: `FP + SC`

The ordinary sklearn-Ridge `FC` branch may remain in the checkpoint as a diagnostic/reference branch only. It must not be the main fused branch for R0.

Therefore in `evaluate_ablation_split()`:

### OOF fusion selection

Replace the condition-dependent `FC+SC` / `FP+SC` logic with:

```python
fusion_weights, fusion_pearson = search_fusion_weights(
    y_train,
    {"FP": oof.fp_oof, "SC": oof.sc_oof},
    ["FP", "SC"],
)
```

for all four conditions.

### Final fused test prediction

Use:

```python
w_fp = fusion_weights["FP"]
w_sc = fusion_weights["SC"]
fused_test = w_fp * fp_final.test_pred + w_sc * sc_final.test_pred
```

for all four conditions.

### Equal-weight comparator

Use:

```python
equal_weight_test = 0.5 * fp_final.test_pred + 0.5 * sc_final.test_pred
```

for all four primary conditions.

The ordinary FC Ridge branch should remain available only as a diagnostic metric and should be explicitly labeled as such.

Do not use `.get(..., 0.5)` for any required fusion weight.

---

# 3. DO NOT rerun the 400 branch fits if the checkpoint is sufficient

The current checkpoint already stores for each R0 split:

- `fp_oof`
- `sc_oof`
- `fp_test_pred`
- `sc_test_pred`
- `train_idx`
- `test_idx`
- `fp_final`
- `sc_final`
- selected parameters

These are sufficient to correct R0 fusion without refitting either branch.

## Required derived R0 correction

For every R0 split:

1. load the original target vector `y`;
2. use `y[s.train_idx]`;
3. recompute the convex fusion weights with:

```python
search_fusion_weights(
    y[s.train_idx],
    {"FP": s.fp_oof, "SC": s.sc_oof},
    ["FP", "SC"],
)
```

4. compute corrected test fusion:

```python
fused_test = w_fp * s.fp_test_pred + w_sc * s.sc_test_pred
```

5. recompute fused Pearson r, RMSE, MAE;
6. recompute equal-weight prediction from `FP + SC`;
7. preserve all branch predictions and branch-selected hyperparameters unchanged.

### Hard gate

Do not rerun any branch/model fitting merely to change R0 fusion.

Only if the checkpoint lacks a required stored R0 OOF/test prediction should you stop and report:

`STATUS: NEEDS_R0_REFIT`

Do not silently launch the 400-split experiment.

---

# 4. Create a NEW corrected derived result directory

Do not overwrite:

- `outputs/iclr/palf_crossfit_ablation_v1/`
- `outputs/iclr/palf_manuscript_freeze_9abbb56/`

Create:

```text
outputs/iclr/palf_same_solver_fusion_corrected_v2/
```

This directory should contain a corrected derived checkpoint or equivalent canonical corrected result representation where:

- R0 fusion weights are recomputed from stored `FP+SC` OOF predictions;
- R0 fused/equal-weight predictions and metrics are corrected;
- R1/R2/R3 are numerically unchanged from the source checkpoint.

If you create a corrected checkpoint, make it a COPY in the v2 directory. Never mutate the source checkpoint.

Also write:

`R0_FUSION_CORRECTION_AUDIT.json`

containing per target/seed/fold:
- old R0 fusion keys
- old R0 fused r
- new `w_FP`
- new `w_SC`
- new R0 fused r
- whether FP branch predictions changed (`must be false`)
- whether SC branch predictions changed (`must be false`)
- whether R1/R2/R3 data changed (`must be false`)

---

# 5. Recompute the primary results from the corrected same-solver R0

After correcting R0 fusion, recompute all manuscript-facing primary quantities.

Do **not** assume the old primary contrast values remain valid:

- WM old delta ~ +0.0052
- FI old delta ~ +0.0059

Those values were based on the wrong R0 fusion and are no longer authoritative.

Recompute:

## Per condition / target
- generalized FC / FP Pearson r
- fused Pearson r
- fused RMSE
- fused MAE
- equal-weight Pearson r
- fusion weights

## Primary contrast
Full PALF vs corrected same-solver no-prior:

- 5 outer folds averaged within seed
- 10 paired seed summaries
- mean delta
- median delta
- positive seeds / 10
- exact two-sided paired Wilcoxon p
- paired 10,000-bootstrap 95% CI
- paired Cohen's dz
- Holm adjustment across the two target tests

The sign convention must be:

```text
Full PALF - Same-solver no prior
```

positive = Full PALF better.

No claim of significance unless the corrected statistics support it.

---

# 6. Prior-control results

The fixed-swap prior controls compare controls against **Full PALF R3**, so scientifically they should be unaffected by the R0 fusion correction.

Still verify this.

Hard checks:

- R3 FP predictions unchanged
- R3 SC predictions unchanged
- R3 fusion weights unchanged
- matched prior-control values unchanged to numerical tolerance
- the existing 300 refits remain valid

If all checks pass, the 300 controls may be copied/reused into the v2 freeze bundle rather than recomputed.

If any R3 value differs, stop and investigate rather than silently mixing results.

Continue to use:
- 10 seed summaries for inference
- Holm across the three controls within each target and metric family
- manuscript-facing direction = `matched - control`

---

# 7. Fix the codebase so future runs use the correct semantics

Patch all relevant current PALF code.

## `src/metascfc/experiments/palf_crossfit_ablation.py`

- all four primary conditions fuse `FP + SC`
- equal-weight uses `FP + SC`
- direct dictionary indexing for required fusion keys
- ordinary `FC` branch remains diagnostic only
- update misleading comments/docstrings that currently say R0 fuses FC+SC

## `scripts/116_run_palf_crossfit_ablation.py`

The active manuscript fusion branch is now `FP` for all four primary conditions.

Remove:

```python
fw.get("FP", fw.get("FC", 0.5))
```

and any required `.get(..., 0.5)` fallback.

Use direct `FP` and `SC` keys for primary conditions.

## `scripts_paper/postprocess_palf_ablation.py`

Ensure:
- R0 uses corrected `FP+SC` fusion;
- no figure or table uses the old R0 fused metrics;
- no required fusion-key fallback remains.

## `scripts_paper/prior_control_refits.py`

This helper is still stale even though the freeze-builder path is corrected.

Fix it too:
- direct `fusion_weights["FP"]` / `["SC"]`
- aggregate five folds within each seed before Wilcoxon inference
- `n_seeds = 10`, never 50
- use manuscript-facing `matched_minus_control`
- Holm correction across three controls within target/family

Do not leave a second public script that can regenerate statistically incorrect prior-control tables.

## `scripts_paper/fix_fusion_weights.py`

Remove silent 0.5 fallbacks and update to the all-primary-conditions `FP+SC` semantics.

---

# 8. Update tests — current tests encode the wrong R0 behavior

The current `tests/test_palf_manuscript_freeze.py` explicitly expects:

```text
R0 -> FC + SC
```

That test is scientifically wrong for the intended same-solver baseline.

Replace it.

Required tests:

1. R0 fusion requires `FP` and `SC`.
2. R1 fusion requires `FP` and `SC`.
3. R2 fusion requires `FP` and `SC`.
4. R3 fusion requires `FP` and `SC`.
5. Ordinary `FC` Ridge exists only as a diagnostic branch.
6. Missing `FP` or `SC` raises `KeyError`.
7. No `.get("FP", 0.5)`, `.get("FC", 0.5)`, or `.get("SC", 0.5)` remains in the current PALF primary/finalization path.
8. R0 correction does not modify stored FP predictions.
9. R0 correction does not modify stored SC predictions.
10. R1/R2/R3 source results are unchanged bitwise or to strict numerical tolerance.
11. Corrected R0 fusion weights satisfy `w_FP + w_SC = 1`.
12. Corrected primary inference has exactly 10 paired seed summaries per target.

Run targeted tests and then the complete suite.

---

# 9. Correct the freeze bundle presentation issues found in review

Create a NEW bundle:

```text
outputs/iclr/palf_manuscript_freeze_v2_same_solver/
```

and ZIP:

```text
outputs/iclr/palf_manuscript_freeze_v2_same_solver.zip
```

Do not overwrite the old freeze.

## 9A. Primary prediction table winner bug

The current generator incorrectly hard-codes Full PALF as bold.

Do not hard-code a condition as winner.

For each target and each reported metric, compute the actual best value and bold only the winning numeric value.

For Pearson r: larger is better.
For RMSE/MAE: smaller is better.

Never bold the condition name or mathematical symbol.

If there is a numerical tie at displayed precision, either bold all tied winning numbers or use additional precision to resolve the true winner consistently.

## 9B. Prior-control table must show Holm p

The current `table_prior_control_specificity.tex` prints:

```text
p_adj = ---
```

even though `prior_control_comparisons.csv` contains the corrected Holm values.

Generate the table directly from the comparison dataframe.

For the fused prior controls report:
- matched mean
- control mean
- matched-minus-control delta
- 95% CI
- Cohen dz
- raw p
- Holm p
- significance star only if Holm p < .05

Define `*` in the caption/footnote.

## 9C. Fusion-weight schema

Because the corrected primary design now uses generalized `FP+SC` for all four configurations, manuscript-facing fusion-weight data may consistently use:

- `w_FP`
- `w_SC`

for all four configurations.

Do not rename ordinary Ridge FC weight as FP.

After correction, R0 genuinely has FP fusion weights, so this ambiguity disappears.

## 9D. Delta direction naming

Do not publish generic `fp_delta_r` / `fused_delta_r` columns whose stored sign is `control - matched`.

Prefer explicit columns:

- `fp_matched_minus_control`
- `fused_matched_minus_control`

If retaining the opposite diagnostic direction, name it explicitly:

- `fp_control_minus_matched`
- `fused_control_minus_matched`

No ambiguous sign columns in the new freeze bundle.

## 9E. `UNSUPPORTED_ARTIFACTS.md`

The current file says `prior_control_refits.csv` is excluded and later lists it as supported.

Remove this contradiction.

## 9F. COMPLETE marker and validation

Create a `COMPLETE` marker only after all gates pass.

Include a dedicated `VALIDATION_REPORT.json`.

Do not claim “all gates pass” only inside an audit blob without a final completion marker.

---

# 10. Include machine-readable primary summaries

The current freeze bundle is missing some useful canonical source tables.

Add:

```text
data/primary_condition_summary.csv
data/primary_seed_metrics.csv
data/branch_fusion_summary.csv
data/primary_comparisons.csv
data/fusion_weights_corrected.csv
data/prior_control_comparisons.csv
data/prior_control_seed_metrics.csv
```

`primary_condition_summary.csv` must contain one row per target × descriptive condition with:
- FP mean r
- fused mean r
- fused SD across seeds
- fused RMSE
- fused MAE
- equal-weight mean r

`primary_seed_metrics.csv` must contain one row per target × condition × seed.

This makes the future paper-generation step independent of parsing LaTeX tables.

---

# 11. Figures for the final bundle

The current main figure layout is scientifically acceptable:

- A/B: seed-level Full PALF minus same-solver-no-prior primary deltas
- C/D: fixed-swap matched-minus-control prior specificity

You may keep this layout **after recomputing panels A/B using the corrected same-solver R0**.

Also regenerate:
- supplementary ablation figure from corrected R0 fused values
- supplementary Full-PALF fusion-weight figure
- supplementary prior-control figure if useful

Add the two authoritative semantic-prior maps to the clean bundle as supplementary figures:

- Working Memory prior
- Fluid Intelligence prior

Regenerate them from the frozen prior CSVs rather than copying LF1-era figures.

No biomarker/top-edge/coefficient-stability figure is allowed.

---

# 12. New validation gates

The v2 bundle is complete only if:

## Same-solver gate
- R0 active fusion branch is generalized FP, not ordinary Ridge FC.
- R0 cache has `D = I` and `lambda_L = 0`.
- R0 uses the same generalized solver family as R1/R2/R3.
- R0 OOF fusion uses stored cross-fitted `fp_oof + sc_oof`.
- R0 test fusion uses stored `fp_test_pred + sc_test_pred`.

## No-refit gate
- branch predictions for R0 are unchanged from source
- only fusion weights/fused/equal-weight metrics change
- R1/R2/R3 remain unchanged

## Primary-statistics gate
- exactly 10 seed summaries per target
- corrected primary p-values and Holm values recomputed from corrected R0

## Prior-control gate
- 300 fixed swaps remain aligned with unchanged R3
- prior-control inference remains seed-level

## Presentation gate
- true table winners are bolded dynamically
- no hard-coded Full PALF winner
- prior-control table contains actual Holm p values
- no manuscript-facing experiment IDs
- no ambiguous delta sign
- no unsupported biomarker artifacts

## Reproducibility gate
- no required 0.5 fusion fallback remains
- all targeted tests pass
- full suite result reported
- `VALIDATION_REPORT.json`
- `COMPLETE`
- checksums verify

---

# 13. Final OpenCode report

Print:

## A. Critical correction
Explicitly state:

> The previous R0 fused baseline used ordinary Ridge FC+SC and therefore was not the intended same-solver comparator. It has now been corrected to generalized no-prior FP+SC using already stored fully cross-fitted branch predictions.

## B. No-refit proof
Report:
- number of R0 splits corrected
- whether any R0 FP branch was refit
- whether any R0 SC branch was refit
- whether R1/R2/R3 changed

Expected:
- 100 R0 splits total across both tasks
- zero branch refits for this correction
- R1/R2/R3 unchanged

## C. Corrected condition results
For each target × condition:
- FP r
- fused r
- RMSE
- MAE
- equal-weight r
- mean w_FP / w_SC

## D. Corrected primary contrast
For WM and FI:
- Full PALF mean
- corrected same-solver no-prior mean
- mean delta
- median delta
- positive seeds
- 95% CI
- raw Wilcoxon p
- Holm p
- Cohen dz

## E. Prior controls
Confirm whether all 300 existing prior-control refits remain valid and unchanged.

## F. Tables
State the actual winning condition for each metric and verify that LaTeX bolding follows the data.

## G. Code cleanup
List all removed fusion 0.5 fallbacks and the repaired stale helper scripts.

## H. Tests
Targeted + full suite.

## I. New bundle
Print:
- corrected source directory
- v2 freeze directory
- v2 ZIP
- COMPLETE path
- validation report path
- checksums path

End with exactly:

`STATUS: PALF_V2_SAME_SOLVER_FREEZE_READY`

or, if any gate fails:

`STATUS: NOT_READY`

---

# Scientific interpretation

Do not preserve any previous narrative just because it was already written.

The corrected same-solver R0 fusion may change the primary effect size and p-values. Use the newly computed values only.

Do not claim:
- significant overall PALF improvement unless the corrected statistics support it;
- synergy unless the corrected component results support it.

The manuscript will be edited only after this v2 bundle is independently reviewed.
