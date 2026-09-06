# PALF ICLR 2027 — Final Publication Figure Cleanup

You are working in the CURRENT LOCAL checkout of the PALF repository.

Repository:

https://github.com/sanjanbaitalik/PALF

Do not rely on a cached remote copy. Treat the current local checkout as
authoritative.

This task modifies PAPER-PLOTTING CODE ONLY.

The scientific experiment is frozen.

DO NOT:

- retrain a model
- rerun model selection
- regenerate priors
- change seeds
- change folds
- modify any experimental CSV/PKL/NPY/NPZ
- overwrite frozen final results
- change numerical values
- change statistical procedures
- tune figure parameters based on which result looks more favorable

Only publication presentation may change.

---

# A. Inspect repository state first

Before editing anything, run:

```bash
git status
git log --oneline -15
git rev-parse HEAD
```

Record the HEAD commit in the final report.

Inspect:

```text
scripts_paper/
outputs/iclr/lf1_final_10x5/
outputs/iclr/lf1_final_evidence_audit/
figures_iclr/
```

At minimum inspect the current implementations of:

```text
scripts_paper/plot_seed_deltas.py
scripts_paper/plot_biomarker_alignment.py
scripts_paper/plot_roi_prior.py
scripts_paper/plot_fusion_weights.py
scripts_paper/plot_frozen_top_edges.py
scripts_paper/export_frozen_fp_coefficients.py
```

Do not assume their current contents from an earlier prompt.

---

# B. Protect frozen results

Before modifying plotting scripts, hash the complete frozen source files used
to generate figures.

At minimum protect:

```text
outputs/iclr/lf1_final_10x5/
outputs/iclr/lf1_final_evidence_audit/
```

Do NOT modify any file below those directories.

After figure generation verify the hashes remain identical.

If any frozen artifact changes, STOP and report failure.

---

# C. General publication styling

Apply a consistent style to all paper plots.

Requirements:

- vector PDF output
- white background
- no external plotting style package
- no seaborn
- no decorative backgrounds
- no embedded black title rectangles
- no unnecessary chart title when LaTeX caption already identifies the task
- clear axis labels
- readable at ICLR paper scale
- consistent font sizes across scripts
- thin axes
- unobtrusive grid only if already scientifically useful
- deterministic plotting
- no alteration of data values

Do not hard-code custom colors solely to make one method visually dominant.

Preserve existing scientifically meaningful color mappings where already used,
especially signed-connectivity maps and prior-score colorbars.

---

# D. Modify `plot_seed_deltas.py`

The current seed-level prediction plot connects seeds with a line.

REMOVE the line connecting seed 0 → seed 1 → ... → seed 9.

Seed IDs are independent repeated-CV seeds and must not visually imply a
trajectory.

Display:

- one marker per seed
- horizontal dashed zero line
- horizontal dotted mean-delta line
- x-axis: `Seed`
- y-axis: `PALF − no-prior Pearson Δr`

Do not connect markers.

Keep the actual seed values unchanged.

Working Memory should still reproduce approximately:

```text
mean Δr   = +0.0170
median Δr = +0.0149
positive  = 8/10
```

Fluid Intelligence should still reproduce approximately:

```text
mean Δr   = +0.0079
median Δr = +0.0105
positive  = 7/10
```

Print the numerical sanity check before saving.

Generate:

```text
figures_iclr/fig_seed_deltas_wm.pdf
figures_iclr/fig_seed_deltas_fluid.pdf
```

Do NOT silently overwrite if the calculated means no longer match the frozen
values within normal rounding tolerance.

---

# E. Review `plot_biomarker_alignment.py`

The final main-paper biomarker plots MUST use the
`lf1_final_evidence_audit` seed-level files, not the older one-row
`lf1_final_10x5/*/biomarker_metrics.csv` summaries.

Authoritative means are approximately:

Working Memory:

```text
No prior   0.0005
Matched    0.6699
Unrelated  0.4974
Shuffled  -0.1006
Random     0.0873
```

Fluid Intelligence:

```text
No prior   0.1764
Matched    0.7615
Unrelated  0.5831
Shuffled  -0.0766
Random     0.1432
```

Do not substitute the older final-run values such as 0.7130 or 0.8527.

Requirements:

- retain seed-level variability
- error bars must represent the statistic actually implemented
- if they are standard errors across ten seed summaries, document this in
  console output
- no embedded task title if unnecessary for the composite paper figure
- y-axis label:
  `Alignment with matched-task prior`
- condition labels:
  `No prior`
  `Matched`
  `Unrelated`
  `Shuffled`
  `Random`

If the labels fit cleanly, do not abbreviate them.

Generate:

```text
figures_iclr/fig_biomarker_alignment_wm.pdf
figures_iclr/fig_biomarker_alignment_fluid.pdf
```

Print the five calculated means for each task before saving.

---

# F. Create a single main-paper quantitative figure

Create:

```text
scripts_paper/make_main_results_figure.py
```

Preferred approach:

replot directly from the authoritative source CSVs using the same plotting
helpers, rather than rasterizing or screenshotting existing PDFs.

Create a publication-quality 2 × 2 figure:

```text
(a) Working Memory — prediction Δr
(b) Fluid Intelligence — prediction Δr
(c) Working Memory — prior alignment
(d) Fluid Intelligence — prior alignment
```

Requirements:

- full-width two-column ICLR figure
- balanced panel dimensions
- panel letters `(a)` `(b)` `(c)` `(d)`
- no repeated large titles
- prediction panels share comparable visual scale where reasonable
- biomarker panels share the same y-axis range
- no connected lines between prediction seeds
- preserve exact data
- vector PDF output
- also optionally save a high-resolution PNG preview

Output:

```text
figures_iclr/fig_main_results.pdf
figures_iclr/fig_main_results.png
```

The script must print all source files used.

Do not add p-values inside the figure.

Do not add significance stars.

Inferential statistics belong in the paper text/table.

---

# G. Modify `plot_roi_prior.py`

The currently generated semantic-prior figures contain an embedded title area
that appears as a black rectangle in the PDF rendering.

Remove the embedded plot title completely.

The LaTeX subfigure/caption will identify the task.

Retain:

- actual AAL116 spatial prior
- frozen prior values
- shared score range `[0,1]`
- meaningful colorbar

Use a clean colorbar label such as:

```text
Prior score
```

Do not alter prior values or atlas mapping.

Generate:

```text
figures_iclr/supp_fig_wm_prior_map.pdf
figures_iclr/supp_fig_fluid_prior_map.pdf
```

Both tasks MUST use the same 0–1 color scale.

No black title box may remain.

---

# H. Review `plot_fusion_weights.py`

The source must remain:

```text
all_split_results.pkl → lf1_weights["FP"]
```

There must be 50 actual split-level values per task.

Do NOT construct a distribution from only a summary mean.

Working Memory should reproduce:

```text
mean w_FP ≈ 0.76
```

Fluid Intelligence:

```text
mean w_FP ≈ 0.37
```

Retain:

- 0.05-bin structure if it corresponds to the frozen weight grid
- mean reference line
- x-axis:
  `Prior-aware FC fusion weight`
- y-axis:
  `Outer splits`

Remove unnecessary embedded task title if LaTeX will provide subfigure labels.

Generate:

```text
figures_iclr/supp_fig_fusion_weights_wm.pdf
figures_iclr/supp_fig_fusion_weights_fluid.pdf
```

Export/retain source-data CSVs as already implemented.

---

# I. Modify `plot_frozen_top_edges.py`

The coefficient reconstruction has already been validated for all 100 outer
fits with maximum prediction reproduction error:

```text
0.00e+00
```

DO NOT reconstruct or retrain coefficients again for this plotting task unless
the plotting script strictly requires the existing frozen coefficient export.

Use:

```text
outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/
```

as the authoritative source.

Remove the embedded title:

```text
Top FC Edges (top 20 by mean |coef|)
```

and any black title rectangle.

LaTeX will provide the task/panel title.

Keep:

```text
top-k = 20
```

for BOTH tasks.

Do not choose different k values.

Use the existing selection criterion:

```text
mean absolute coefficient across 50 frozen outer fits
```

Preserve signed edge colors.

Do not normalize the two tasks onto one common coefficient range unless doing
so is mathematically justified.

Instead, preserve task-specific coefficient scales and clearly expose the
colorbars.

Generate:

```text
figures_iclr/supp_fig_wm_top_edges.pdf
figures_iclr/supp_fig_fluid_top_edges.pdf
```

Also retain:

```text
*.top_edges.tsv
```

The plotting script should print:

```text
Top 20 edges plotted.
```

and a descriptive/causal-disclaimer caption suggestion.

---

# J. Do not overinterpret top-edge maps in code

Do not put neurobiological claims directly into the plots.

No labels such as:

```text
working-memory network
reasoning network
validated biomarker
causal connection
```

inside the figures.

Those interpretations belong in the manuscript and will be phrased
conservatively.

---

# K. Optional supplementary composite figures

Create a helper:

```text
scripts_paper/make_supplement_composites.py
```

that can generate:

```text
figures_iclr/supp_fig_priors_combined.pdf
figures_iclr/supp_fig_fusion_weights_combined.pdf
figures_iclr/supp_fig_top_edges_combined.pdf
```

Each should contain:

```text
(a) Working Memory
(b) Fluid Intelligence
```

Do not rasterize existing PDFs if the source-data plotting functions can be
reused directly.

If direct replotting is impractical, leave the individual PDFs untouched and
report that LaTeX subfigures should be used instead.

Do not compromise vector quality merely to make combined PDFs.

---

# L. Figure-source manifest

Create:

```text
figures_iclr/FIGURE_MANIFEST.md
```

For every final figure record:

- figure filename
- generating script
- exact source artifact(s)
- whether it is main-paper or supplementary
- key scientific quantity
- expected sanity-check values

Example:

```text
fig_main_results.pdf
  script: scripts_paper/make_main_results_figure.py
  source:
    working_memory/seed_metrics.csv
    fluid_intelligence/seed_metrics.csv
    lf1_final_evidence_audit/biomarker_seed_metrics_*.csv
  paper: main
```

Do not place speculative interpretation in this manifest.

---

# M. Add plotting regression tests

Add or update tests so that:

1. seed-delta source calculations reproduce frozen means
2. seed-delta plotting does not use a connected line for observations
3. biomarker plots use evidence-audit data
4. biomarker means reproduce final audited values
5. fusion plots contain exactly 50 values per task
6. fusion-weight means reproduce ~0.76 / ~0.37
7. prior maps use identical [0,1] scaling
8. top-edge plot uses exactly top 20 for both tasks
9. top-edge source is frozen reconstructed coefficients
10. plotting scripts do not modify frozen experiment artifacts

Tests should focus on data provenance and semantics rather than brittle pixel
comparisons.

---

# N. Generate everything

After code changes run the relevant plotting tests.

Then regenerate all final figures.

The expected final directory is approximately:

```text
figures_iclr/
├── fig_seed_deltas_wm.pdf
├── fig_seed_deltas_fluid.pdf
├── fig_biomarker_alignment_wm.pdf
├── fig_biomarker_alignment_fluid.pdf
├── fig_main_results.pdf
├── fig_main_results.png
├── supp_fig_wm_prior_map.pdf
├── supp_fig_fluid_prior_map.pdf
├── supp_fig_fusion_weights_wm.pdf
├── supp_fig_fusion_weights_fluid.pdf
├── supp_fig_wm_top_edges.pdf
├── supp_fig_fluid_top_edges.pdf
├── supp_fig_wm_top_edges.top_edges.tsv
├── supp_fig_fluid_top_edges.top_edges.tsv
└── FIGURE_MANIFEST.md
```

Optional combined supplementary PDFs may also be present.

---

# O. Final automated audit

At the end print a concise report:

```text
FINAL ICLR FIGURE AUDIT

Git HEAD: <sha>

Frozen experimental artifacts modified: NO

Prediction figures:
  WM mean Δr: ...
  WM positive seeds: .../10
  Fluid mean Δr: ...
  Fluid positive seeds: .../10

Biomarker alignment:
  WM matched mean: ...
  Fluid matched mean: ...

Fusion weights:
  WM n: 50
  WM mean w_FP: ...
  Fluid n: 50
  Fluid mean w_FP: ...

Semantic prior maps:
  shared range: [0,1]
  embedded titles removed: YES

Frozen FC edge maps:
  WM top-k: 20
  Fluid top-k: 20
  coefficient source: frozen reconstruction
  embedded titles removed: YES

Main results composite generated: YES

Tests: <passed>/<total>

STATUS: READY_FOR_VISUAL_REVIEW
```

Do not print READY_FOR_VISUAL_REVIEW unless every provenance and numerical
sanity check succeeds.

---

# P. Important final rule

This task is purely presentational.

No scientific result may change because of these edits.

If a plotting change causes a numerical mismatch with the frozen final
results, STOP and investigate rather than silently producing a figure.
