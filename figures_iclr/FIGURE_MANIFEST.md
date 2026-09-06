# Figure Source Manifest

## Main Paper Figures

### fig_seed_deltas_wm.pdf
- **Script**: `scripts_paper/plot_seed_deltas.py`
- **Source**: `outputs/iclr/lf1_final_10x5/working_memory/seed_metrics.csv`
- **Paper**: main
- **Key quantity**: PALF minus no-prior Pearson delta r per seed
- **Expected values**: mean Δr ≈ +0.0170, positive seeds ≈ 8/10

### fig_seed_deltas_fluid.pdf
- **Script**: `scripts_paper/plot_seed_deltas.py`
- **Source**: `outputs/iclr/lf1_final_10x5/fluid_intelligence/seed_metrics.csv`
- **Paper**: main
- **Key quantity**: PALF minus no-prior Pearson delta r per seed
- **Expected values**: mean Δr ≈ +0.0079, positive seeds ≈ 7/10

### fig_biomarker_alignment_wm.pdf
- **Script**: `scripts_paper/plot_biomarker_alignment.py`
- **Source**: `outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_working_memory.csv`
- **Paper**: main
- **Key quantity**: Matched-task prior alignment per condition
- **Expected values**: No prior ≈ 0.0005, Matched ≈ 0.6699, Unrelated ≈ 0.4974, Shuffled ≈ -0.1006, Random ≈ 0.0873

### fig_biomarker_alignment_fluid.pdf
- **Script**: `scripts_paper/plot_biomarker_alignment.py`
- **Source**: `outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_fluid_intelligence.csv`
- **Paper**: main
- **Key quantity**: Matched-task prior alignment per condition
- **Expected values**: No prior ≈ 0.1764, Matched ≈ 0.7615, Unrelated ≈ 0.5831, Shuffled ≈ -0.0766, Random ≈ 0.1432

### fig_main_results.pdf / fig_main_results.png
- **Script**: `scripts_paper/make_main_results_figure.py`
- **Source**:
  - `outputs/iclr/lf1_final_10x5/working_memory/seed_metrics.csv`
  - `outputs/iclr/lf1_final_10x5/fluid_intelligence/seed_metrics.csv`
  - `outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_working_memory.csv`
  - `outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_fluid_intelligence.csv`
- **Paper**: main
- **Key quantity**: 2×2 composite of prediction deltas and prior alignment
- **Panel (a)**: WM prediction Δr, **(b)**: FI prediction Δr, **(c)**: WM alignment, **(d)**: FI alignment

---

## Supplementary Figures

### supp_fig_wm_prior_map.pdf
- **Script**: `scripts_paper/plot_roi_prior.py`
- **Source**: `outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv`
- **Paper**: supplementary
- **Key quantity**: AAL116 ROI-level matched prior scores
- **Expected range**: [0, 1]

### supp_fig_fluid_prior_map.pdf
- **Script**: `scripts_paper/plot_roi_prior.py`
- **Source**: `outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv`
- **Paper**: supplementary
- **Key quantity**: AAL116 ROI-level matched prior scores
- **Expected range**: [0, 1]

### supp_fig_fusion_weights_wm.pdf
- **Script**: `scripts_paper/plot_fusion_weights.py`
- **Source**: `outputs/iclr/lf1_final_10x5/working_memory/all_split_results.pkl`
- **Paper**: supplementary
- **Key quantity**: Prior-aware FC fusion weight (w_FP) distribution across 50 outer splits
- **Expected values**: n=50, mean w_FP ≈ 0.76

### supp_fig_fusion_weights_fluid.pdf
- **Script**: `scripts_paper/plot_fusion_weights.py`
- **Source**: `outputs/iclr/lf1_final_10x5/fluid_intelligence/all_split_results.pkl`
- **Paper**: supplementary
- **Key quantity**: Prior-aware FC fusion weight (w_FP) distribution across 50 outer splits
- **Expected values**: n=50, mean w_FP ≈ 0.37

### supp_fig_wm_top_edges.pdf
- **Script**: `scripts_paper/plot_frozen_top_edges.py`
- **Source**: `outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv`
- **Paper**: supplementary
- **Key quantity**: Top 20 prior-aware FC edges by mean absolute coefficient
- **Expected values**: top-k=20, coefficient source = frozen reconstruction

### supp_fig_fluid_top_edges.pdf
- **Script**: `scripts_paper/plot_frozen_top_edges.py`
- **Source**: `outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/fluid_intelligence/stable_top_edges.csv`
- **Paper**: supplementary
- **Key quantity**: Top 20 prior-aware FC edges by mean absolute coefficient
- **Expected values**: top-k=20, coefficient source = frozen reconstruction

### supp_fig_fusion_weights_combined.pdf
- **Script**: `scripts_paper/make_supplement_composites.py`
- **Source**:
  - `outputs/iclr/lf1_final_10x5/working_memory/all_split_results.pkl`
  - `outputs/iclr/lf1_final_10x5/fluid_intelligence/all_split_results.pkl`
- **Paper**: supplementary
- **Key quantity**: Fusion weight distributions for both tasks

### supp_fig_top_edges_combined.pdf
- **Script**: `scripts_paper/make_supplement_composites.py`
- **Source**:
  - `outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv`
  - `outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/fluid_intelligence/stable_top_edges.csv`
- **Paper**: supplementary
- **Key quantity**: Top-20 edge connectome maps for both tasks
