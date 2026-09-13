# FINAL 510 WM runtime estimate

## Benchmark (seed 7171 fold 0, n_train=408, n_test=102)

- matched_masks: 2.7s
- R-MATCHED: 3.9s
- R-CROSS: 5.1s
- R-SHUFFLED: 5.7s
- R-RANDOM: 4.9s
- N-MATCHED: 33.5s
- N-CROSS: 29.2s
- N-SHUFFLED: 48.7s
- N-RANDOM: 33.3s
- total: 176.1s
- perturbation_workload_two_models_one_fold: 1.0s

## Estimated remaining stages

- main (24 folds): 1.17 h
- biomarker (25 folds): 1.7 min
- ablations + reports: 25.0 min
- ablation D adds a fixed-ratio outer sweep (seed 7171, ~3 min).
- **Total remaining: 1.62 h (97 min)**

FINAL510_ESTIMATED_RUNTIME: 1h37m
FINAL510_ESTIMATED_FINISH_FROM_START: 1h37m
Estimate printed before running the missing controls.
