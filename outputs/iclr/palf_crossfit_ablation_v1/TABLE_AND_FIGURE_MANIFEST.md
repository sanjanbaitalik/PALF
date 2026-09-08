# Table and Figure Manifest — PALF Cross-Fitted Ablation

## Run Directory
`outputs/iclr/palf_crossfit_ablation_v1/`

## Tables

| Table | Source | Script | Definition |
|---|---|---|---|
| `tables/table_primary_prediction.tex` | `paired_comparisons.csv` | `postprocess_palf_ablation.py` | R3−R0 primary contrast, Holm-corrected |
| `tables/table_component_ablation.tex` | `component_summary.csv` | `postprocess_palf_ablation.py` | R0–R3 branch/fusion metrics per task |
| `tables/table_branch_and_fusion.tex` | `branch_fusion_summary.csv` | `generate_remaining_outputs.py` | FC/SC/FP/fused/equal-weight per condition |
| `tables/table_prior_controls.tex` | `prior_control_comparisons.csv` | `prior_control_refits.py` | Fixed prior-swap predictive specificity (300 refits × 2 tasks) |
| `tables/table_resampling_stability.tex` | `stability_pair_metrics.csv` | `generate_remaining_outputs.py` | Within-seed pairwise fit stability |

## Figures

| Figure | Source | Script | Definition |
|---|---|---|---|
| `plots/ablation_comparison.pdf` | `seed_metrics.csv` | `postprocess_palf_ablation.py` | Four-condition ablation bar chart |
| `plots/fig_main_results.pdf` | `seed_metrics.csv` + `biomarker_seed_metrics.csv` | `postprocess_palf_ablation.py` | Four-panel main results |
| `plots/fusion_weights.pdf` | `fusion_weights.csv` | `postprocess_palf_ablation.py` | R3 fusion weight distributions |
| `plots/stability_working_memory.pdf` | `stability_pair_metrics.csv` | `generate_remaining_outputs.py` | WM resampling stability |
| `plots/stability_fluid_intelligence.pdf` | `stability_pair_metrics.csv` | `generate_remaining_outputs.py` | FI resampling stability |
| `plots/top_edges_*.tsv` | `coefficients/` | `postprocess_palf_ablation.py` | R3 top-20 FC-edge maps |

## Data Files

| File | Rows | Description |
|---|---|---|
| `outer_metrics.csv` | 400 | Per-split outer-test metrics |
| `seed_metrics.csv` | 80 | Per-seed condition summaries |
| `component_summary.csv` | 21 | Condition-level branch/fusion summaries |
| `paired_comparisons.csv` | 10 | All declared contrasts with Holm correction |
| `branch_fusion_summary.csv` | 2000 | Per-split branch/fusion metrics |
| `selected_hyperparameters.csv` | 400 | Final selected parameters per split |
| `fusion_weights.csv` | 50 | Per-split fusion weights |
| `stability_pair_metrics.csv` | ~200 | Within-seed pairwise fit stability |
| `stability_seed_metrics.csv` | ~80 | Seed-level stability summaries |
| `biomarker_fit_metrics.csv` | ~1000 | Per-fit alignment (pending coefficient reconstruction) |
| `oof_predictions/*.csv` | ~412×200 | Branch OOF predictions |
| `outer_predictions/*.csv` | ~412×200 | Branch/fusion outer predictions |
| `prior_control_metrics.csv` | 300 | Per-split prior control refit results (FP + fused) |
| `prior_control_comparisons.csv` | 12 | Paired comparisons: matched vs each control (Wilcoxon, bootstrap CI) |
