# Phase 1: Differential Stacking Feasibility Run Report

## Experiment Summary

- **Run directory**: `/home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase1_differential_stacking`
- **Script**: `scripts_paper/phase1_differential_stacking_feasibility.py`
- **Date**: 2026-09-09 18:09:32
- **Total splits evaluated**: 100 (50 per target x 2 targets)

## Frozen Means Verification

- **Working Memory**: R0=0.263515 (expected 0.263515) OK | R3=0.262500 (expected 0.262500) OK
- **Fluid Intelligence**: R0=0.370917 (expected 0.370917) OK | R3=0.370031 (expected 0.370031) OK

## Primary Comparison: Differential Stack vs Same-solver Baseline

| Task | Baseline $r$ | Stack $r$ | $\Delta r$ | 95% CI | Cohen's $d_z$ | $p$ | $p_{\mathrm{adj}}$ |
|---|---|---|---|---|---|---|---|
| Working Memory | 0.2635 | 0.2545 | -0.0090 | [-0.0272, +0.0034] | -0.329 | 0.9219 | 0.9219 |
| Fluid Intelligence | 0.3709 | 0.3656 | -0.0053 | [-0.0100, -0.0017] | -0.733 | 0.0098 | 0.0195$^{*}$ |

## Architecture Selection Diagnostics

### Working Memory

| Candidate | n selected | Fraction | Mean test $r$ |
|---|---|---|---|
| Baseline FP+SC | 16 | 0.32 | 0.2488 |
| Residual $+a\delta$ | 34 | 0.68 | 0.2572 |

### Fluid Intelligence

| Candidate | n selected | Fraction | Mean test $r$ |
|---|---|---|---|
| Baseline FP+SC | 18 | 0.36 | 0.3806 |
| Residual $+a\delta$ | 32 | 0.64 | 0.3572 |

## Meta-CV Reliability

- **Working Memory**: meta-test correlation=-0.3123 (p=0.0272), mean test improvement=-0.00902, fraction positive improvement=0.34
- **Fluid Intelligence**: meta-test correlation=-0.3768 (p=0.0070), mean test improvement=-0.00532, fraction positive improvement=0.30

## Files Generated

- `split_metrics.csv` - Per-split results
- `seed_metrics.csv` - Seed-level aggregated metrics
- `primary_comparisons.csv` - Paired Wilcoxon tests with Holm correction
- `architecture_selection.csv` - Candidate selection diagnostics
- `meta_cv_reliability.csv` - Meta-CV reliability diagnostic
- `outer_predictions/*.csv` - Per-subject predictions
- `tables/*.tex` - LaTeX tables
- `plots/*.pdf/png` - Figures
- `VALIDATION_REPORT.json` - Machine-readable validation
- `RUN_REPORT.md` - This report