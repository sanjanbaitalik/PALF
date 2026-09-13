# Scaling study status (412 vs 412+98)

- SCALING_DECISION: **NO_SAMPLE_SIZE_GAIN**
- PAPER_READY_TRIGGER: NO
- Analysis type: post-hoc combined-cohort sample-size scaling (exploratory)
- This is NOT independent or external validation.
- The 98 subjects were evaluated in Phase 4 and are not a holdout here.
- Phase 4 result (WM_CONFIRMATION_FAILED) is preserved and not reinterpreted.

## Main result

Expanding the training pool from T412 to T412+D98 did not improve prediction on the identical V412 evaluation subjects: 0/18 model/target combinations had a positive paired delta_r on seed-averaged predictions. The best observed delta was -0.0064 (WM/A0).

## Per-model decision table

```
target model    r_412  r_plus98  delta_r_observed  delta_r_ci_lo  delta_r_ci_hi  delta_r_seed_mean  positive_seeds    tier
    WM    A0 0.296699  0.290324         -0.006374      -0.054576       0.040977           0.004152               3 NO_GAIN
    WM    A1 0.294567  0.282391         -0.012176      -0.061797       0.038509          -0.003772               2 NO_GAIN
    WM    A2 0.300529  0.282835         -0.017694      -0.063253       0.027914          -0.009693               2 NO_GAIN
    WM    A3 0.300819  0.292780         -0.008038      -0.057436       0.039986           0.004094               2 NO_GAIN
    WM    B0 0.307091  0.297988         -0.009103      -0.056116       0.037839           0.005752               2 NO_GAIN
    WM    B1 0.303134  0.296756         -0.006378      -0.054311       0.041563           0.007935               2 NO_GAIN
    WM    B2 0.302477  0.291032         -0.011444      -0.058965       0.035923          -0.005514               2 NO_GAIN
    WM    B3 0.288992  0.279944         -0.009048      -0.057368       0.039774          -0.004523               2 NO_GAIN
    WM    B4 0.301642  0.281336         -0.020305      -0.070741       0.028485          -0.007056               3 NO_GAIN
    FI    A0 0.388599  0.360746         -0.027853      -0.068786       0.012344          -0.019685               0 NO_GAIN
    FI    A1 0.389474  0.355706         -0.033768      -0.076309       0.007992          -0.030104               0 NO_GAIN
    FI    A2 0.389751  0.360467         -0.029284      -0.070215       0.010312          -0.022419               0 NO_GAIN
    FI    A3 0.390634  0.361532         -0.029101      -0.071156       0.012355          -0.024625               1 NO_GAIN
    FI    B0 0.386509  0.355732         -0.030777      -0.072064       0.010327          -0.020449               1 NO_GAIN
    FI    B1 0.386721  0.357580         -0.029141      -0.069833       0.011860          -0.017316               1 NO_GAIN
    FI    B2 0.384371  0.358443         -0.025928      -0.067838       0.016405          -0.016510               2 NO_GAIN
    FI    B3 0.385323  0.356992         -0.028331      -0.068773       0.011162          -0.020422               0 NO_GAIN
    FI    B4 0.389154  0.377372         -0.011782      -0.051208       0.027681          -0.004110               3 NO_GAIN
```

## Biomarker scaling

- Cross-fold faithfulness (top10 minus random10 delta_RMSE; positive = faithful) improved from 412 to +98 for every target/model combination (see biomarker_faithfulness.csv and Figure 6).
- Coefficient stability was mixed across targets (see biomarker_stability.csv and Figure 5).

## Learning curve

- Fixed-V412 learning curves from n=250 to full T412+D98 are non-monotone and show no reliable gain (learning_curve.csv, Figure 4).

## Runtime

- Primary stage: 25/25 splits in 18,280 s.
- Learning stage: 8,355 s.
- Secondary 510 CV: 670 s.
- Biomarker stage: 13 s.
