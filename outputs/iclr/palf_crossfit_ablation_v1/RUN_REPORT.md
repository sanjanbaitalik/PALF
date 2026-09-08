# PALF Cross-Fitted Ablation Run Report

## Experiment Summary

- **Run directory**: `/home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_crossfit_ablation_v1`
- **Configuration**: `configs/iclr/palf_crossfit_ablation.yaml`
- **WM splits completed**: 200/200
- **FI splits completed**: 200/200

## Conditions

| ID | Name | D | Network | lambda_L grid |
|---|---|---|---|---|
| R0 | Same-solver no prior | I | None | [0.0] |
| R1 | Anisotropy only | D(q;0.5) | None | [0.0] |
| R2 | Network only | I | L_p | [0.03, 0.1, 0.5, 1.0, 2.0, 5.0] |
| R3 | Full PALF | D(q;0.5) | L_p | [0.03, 0.1, 0.5, 1.0, 2.0, 5.0] |

## Key Results

### Working Memory

| Condition | FC r | SC r | FP r | Fused r | Equal-weight r |
|---|---|---|---|---|---|
| R0 | 0.2599 | 0.1320 | 0.2751 | 0.2573 | 0.2676 |
| R1 | 0.2599 | 0.1320 | 0.2751 | 0.2635 | 0.2646 |
| R2 | 0.2599 | 0.1320 | 0.2760 | 0.2628 | 0.2651 |
| R3 | 0.2599 | 0.1320 | 0.2760 | 0.2628 | 0.2651 |

### Fluid Intelligence

| Condition | FC r | SC r | FP r | Fused r | Equal-weight r |
|---|---|---|---|---|---|
| R0 | 0.1848 | 0.3571 | 0.2166 | 0.3642 | 0.3231 |
| R1 | 0.1848 | 0.3571 | 0.2166 | 0.3709 | 0.3700 |
| R2 | 0.1848 | 0.3571 | 0.2125 | 0.3689 | 0.3671 |
| R3 | 0.1848 | 0.3571 | 0.2125 | 0.3689 | 0.3671 |

## Paired Comparisons (Primary: R3 vs R0)

- **working_memory**: +0.0055 (95% CI [-0.0041, +0.0139], p=0.3223, adj_p=0.3223)
- **fluid_intelligence**: +0.0047 (95% CI [-0.0027, +0.0129], p=0.4922, adj_p=0.4922)

## Resampling Stability (within-seed pairwise fit)

### Working Memory
- **R0**: mean FC α=96.0000, mean SC α=88.0001
- **R1**: mean FC α=96.0000, mean SC α=88.0001
- **R2**: mean FC α=96.0000, mean SC α=88.0001
- **R3**: mean FC α=96.0000, mean SC α=88.0001

### Fluid Intelligence
- **R0**: mean FC α=100.0000, mean SC α=96.0000
- **R1**: mean FC α=100.0000, mean SC α=96.0000
- **R2**: mean FC α=100.0000, mean SC α=96.0000
- **R3**: mean FC α=100.0000, mean SC α=96.0000

## Scientific Questions

### 1. Were the two scientific defects repaired?
Yes. OOF predictions now exclude held-out subjects from preprocessing, selection, and fitting.
Final branch parameters are reselected on all outer-training subjects via 3-fold CV.

### 2. How does R3 compare with R0?
See results table above. R3 uses anisotropic diagonal penalty D(q;0.5) and network Laplacian L_p.

### 3. What do R1/R2 reveal?
R1 isolates anisotropy; R2 isolates the network penalty. Both are retuned independently.

### 4. Does learned fusion improve over branches and equal averaging?
Fusion weights are selected on OOF predictions within the outer-training set.

### 5-8. See tables and plots in the run directory.

## Limitations
- Reused HCP cohort with development history
- Subject-wise CV (not family-aware)
- Grid-boundary selection possible
- Fixed random/shuffled realizations
- No external validation
