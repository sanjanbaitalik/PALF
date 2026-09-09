# Unsupported Artifacts

This document lists data artifacts and intermediate files that are NOT part of
the clean freeze bundle and are NOT used in the manuscript figures or tables.

## Excluded from freeze

| Artifact | Reason |
|---|---|
| `prior_control_refits.csv` (raw 300 rows) | Intermediate; seed-level aggregates are canonical |
| `selection_scores/*.json` | Per-split selection detail; not reported in manuscript |
| `coefficients/*.npz` | Dual-space coefficients; not used for publication |
| `oof_predictions/*.csv` | OOF predictions; used only internally for fusion weight selection |
| `outer_predictions/*.csv` | Per-subject predictions; not reported in manuscript |
| `stability_pair_metrics.csv` | Resampling stability diagnostic; not in main paper |
| `stability_seed_metrics.csv` | Seed-level stability; not in main paper |
| `biomarker_fit_metrics.csv` | Alignment proxy; coefficient reconstruction pending |
| `backup_uniform_prior/` | Backup checkpoint; not used |
| `protocol.json` | Experiment protocol; internal only |
| `RUN_REPORT.md` | Internal run report |
| `TABLE_AND_FIGURE_MANIFEST.md` | Internal manifest |
| `diagram_readiness.json` | Internal readiness check |

## Supported artifacts in freeze

| Artifact | Used in |
|---|---|
| `data/primary_comparisons.csv` | Table 2, main text |
| `data/prior_control_comparisons.csv` | Table 3, Section 7.2 |
| `data/prior_control_refits.csv` | Raw refit data (300 rows) |
| `data/seed_level_prior_control.csv` | Seed-level aggregates (60 rows) |
| `data/fusion_weights.csv` | Table 4, fusion weight analysis |
| `figures/fig_main_2x2.pdf` | Figure 1 (main) |
| `figures/fig_main_2x2.png` | Figure 1 (main, raster) |
| `figures/fig_fusion_weights_supp.pdf` | Supplementary figure |
| `figures/fig_ablation_supp.pdf` | Supplementary figure |
| `tables/table_primary_prediction.tex` | Table 1 |
| `tables/table_primary_comparisons.tex` | Table 2 |
| `tables/table_prior_control_specificity.tex` | Table 3 |
| `tables/table_fusion_weights.tex` | Table 4 |
