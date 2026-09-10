# CURRENT_PRIOR_AUDIT: Phase 2E LI-SFC-NCR

## Old Prior (ROI-Activation Level)
- **Source**: `outputs/priors/llm/{task}_contrastive_qwen3/roi_prior.csv`
- **Type**: ROI-level activation/relevance scores from LLM
- **WM-FI Pearson**: 0.7937
- **Top-10 overlap**: 8/10

## New Prior (System-Pair Interaction Level)
- **Source**: `outputs/iclr/palf_phase2e_li_sfc_ncr/priors/llm_network_interaction/`
- **Type**: System-pair SC-FC coupling relevance scores, derived from ROI-prior edge product aggregation
- **Model**: qwen3.8:27b (LLM) + ROI-prior derivation
- **WM-FI Pearson**: 0.9185
- **Top-5 overlap**: 4/5
- **Top-10 overlap**: 8/10
- **Frozen**: 2026-09-10T12:18:09Z

## Derivation Method
1. Build AAL116 → Yeo7 9-system mapping (VIS, SM, DATN, LIMBIC, FPN, DMN, VAS, SUBCORTICAL, CEREBELLAR)
2. Map all 6670 edges to 45 system pairs
3. For each pair (a,b), compute `q_ab = mean(p_i * p_j)` for edges (i,j) connecting systems a and b
4. Normalize to [0, 1]

## Control Priors
- **Shuffled** (seed=7301): WM values shuffled across pairs
- **Random** (seed=7303): Uniform random [0,1]
- **Cross-task WM**: WM prior applied to FI
- **Cross-task FI**: FI prior applied to WM

## Assessment
The WM-FI correlation is very high (0.9185), indicating the pair priors are task-agnostic.
This is expected since the ROI-level priors are also highly correlated between tasks.
The LLM interaction priors did not provide task-discriminative signal for SC-FC coupling prediction.
