# Scaling ablations

## A_sample_size

```
        group target variant           metric     value  positive  n
A_sample_size     WM      A0 delta_r(+98-412)  0.004152       3.0  5
A_sample_size     WM      A1 delta_r(+98-412) -0.003772       2.0  5
A_sample_size     WM      A2 delta_r(+98-412) -0.009693       2.0  5
A_sample_size     WM      A3 delta_r(+98-412)  0.004094       2.0  5
A_sample_size     WM      B0 delta_r(+98-412)  0.005752       2.0  5
A_sample_size     WM      B1 delta_r(+98-412)  0.007935       2.0  5
A_sample_size     WM      B2 delta_r(+98-412) -0.005514       2.0  5
A_sample_size     WM      B3 delta_r(+98-412) -0.004523       2.0  5
A_sample_size     WM      B4 delta_r(+98-412) -0.007056       3.0  5
A_sample_size     FI      A0 delta_r(+98-412) -0.019685       0.0  5
A_sample_size     FI      A1 delta_r(+98-412) -0.030104       0.0  5
A_sample_size     FI      A2 delta_r(+98-412) -0.022419       0.0  5
A_sample_size     FI      A3 delta_r(+98-412) -0.024625       1.0  5
A_sample_size     FI      B0 delta_r(+98-412) -0.020449       1.0  5
A_sample_size     FI      B1 delta_r(+98-412) -0.017316       1.0  5
A_sample_size     FI      B2 delta_r(+98-412) -0.016510       2.0  5
A_sample_size     FI      B3 delta_r(+98-412) -0.020422       0.0  5
A_sample_size     FI      B4 delta_r(+98-412) -0.004110       3.0  5
```

## B_PALF_components

```
            group target         variant metric     value  positive  n
B_PALF_components     WM    PALF_full-R0      A -0.003096       2.0  5
B_PALF_components     WM    PALF_full-R0      B -0.011020       2.0  5
B_PALF_components     WM   PALF_aniso-R0      A  0.005559       4.0  5
B_PALF_components     WM   PALF_aniso-R0      B -0.008286       2.0  5
B_PALF_components     WM PALF_network-R0      A  0.001274       3.0  5
B_PALF_components     WM PALF_network-R0      B  0.001217       3.0  5
B_PALF_components     WM       NCR-Ridge      A -0.004546       1.0  5
B_PALF_components     WM       NCR-Ridge      B -0.002363       1.0  5
B_PALF_components     WM       NCR-cross      A -0.002774       3.0  5
B_PALF_components     WM       NCR-cross      B  0.010675       4.0  5
B_PALF_components     WM    NCR-shuffled      A  0.009258       4.0  5
B_PALF_components     WM    NCR-shuffled      B  0.021716       5.0  5
B_PALF_components     WM      NCR-random      A  0.001788       2.0  5
B_PALF_components     WM      NCR-random      B  0.016779       5.0  5
B_PALF_components     WM          NCR-R0      A  0.001841       3.0  5
B_PALF_components     WM          NCR-R0      B  0.005624       4.0  5
B_PALF_components     FI    PALF_full-R0      A  0.000440       2.0  5
B_PALF_components     FI    PALF_full-R0      B -0.009979       1.0  5
B_PALF_components     FI   PALF_aniso-R0      A  0.001448       4.0  5
B_PALF_components     FI   PALF_aniso-R0      B -0.001286       2.0  5
B_PALF_components     FI PALF_network-R0      A  0.003036       4.0  5
B_PALF_components     FI PALF_network-R0      B -0.001904       2.0  5
B_PALF_components     FI       NCR-Ridge      A -0.000705       2.0  5
B_PALF_components     FI       NCR-Ridge      B  0.002427       3.0  5
B_PALF_components     FI       NCR-cross      A -0.000220       2.0  5
B_PALF_components     FI       NCR-cross      B -0.001027       3.0  5
B_PALF_components     FI    NCR-shuffled      A -0.003244       2.0  5
B_PALF_components     FI    NCR-shuffled      B -0.000138       2.0  5
B_PALF_components     FI      NCR-random      A -0.006038       2.0  5
B_PALF_components     FI      NCR-random      B -0.019244       0.0  5
B_PALF_components     FI          NCR-R0      A -0.006496       1.0  5
B_PALF_components     FI          NCR-R0      B -0.004128       1.0  5
```

## E_modality_F_fusion

```
              group target          variant  metric    value  positive  n
E_modality_F_fusion     WM     B1_fc_only_A pearson 0.260878       NaN 25
E_modality_F_fusion     WM     B1_sc_only_A pearson 0.247536       NaN 25
E_modality_F_fusion     WM B1_expert_only_A pearson 0.220466       NaN 25
E_modality_F_fusion     WM       B1_fused_A pearson 0.263447       NaN 25
E_modality_F_fusion     WM     B1_fc_only_B pearson 0.270388       NaN 25
E_modality_F_fusion     WM     B1_sc_only_B pearson 0.185875       NaN 25
E_modality_F_fusion     WM B1_expert_only_B pearson 0.241347       NaN 25
E_modality_F_fusion     WM       B1_fused_B pearson 0.271785       NaN 25
E_modality_F_fusion     FI     B1_fc_only_A pearson 0.339657       NaN 25
E_modality_F_fusion     FI     B1_sc_only_A pearson 0.350260       NaN 25
E_modality_F_fusion     FI B1_expert_only_A pearson 0.248914       NaN 25
E_modality_F_fusion     FI       B1_fused_A pearson 0.349110       NaN 25
E_modality_F_fusion     FI     B1_fc_only_B pearson 0.311097       NaN 25
E_modality_F_fusion     FI     B1_sc_only_B pearson 0.329257       NaN 25
E_modality_F_fusion     FI B1_expert_only_B pearson 0.278429       NaN 25
E_modality_F_fusion     FI       B1_fused_B pearson 0.331338       NaN 25
```

## G_mask_family

```
        group target        variant        metric  value  positive  n
G_mask_family     WM  direct_topk_A n_selected_fc    8.0       NaN 25
G_mask_family     WM roi_incident_A n_selected_fc   17.0       NaN 25
G_mask_family     WM  direct_topk_B n_selected_fc    8.0       NaN 25
G_mask_family     WM roi_incident_B n_selected_fc   17.0       NaN 25
G_mask_family     FI  direct_topk_A n_selected_fc    5.0       NaN 25
G_mask_family     FI roi_incident_A n_selected_fc   20.0       NaN 25
G_mask_family     FI  direct_topk_B n_selected_fc   19.0       NaN 25
G_mask_family     FI roi_incident_B n_selected_fc    6.0       NaN 25
```

## H_learning_curve

```
target model size_label  n_requested   r_mean    r_std  n_eff_mean  n_runs
    FI    A0       n250          250 0.318935 0.034565       250.0      15
    FI    A0       n300          300 0.340603 0.018321       300.0      15
    FI    A0       n350          350 0.309403 0.014837       350.0      15
    FI    A0  full_T412          329 0.356077 0.011476       329.6       5
    FI    A0    full_TB          427 0.336392 0.016720       427.6       5
    FI    B1       n250          250 0.229959 0.054832       250.0      15
    FI    B1       n300          300 0.253237 0.026124       300.0      15
    FI    B1       n350          350 0.249843 0.028107       350.0      15
    FI    B1  full_T412          329 0.349581 0.009857       329.6       5
    FI    B1    full_TB          427 0.332264 0.019508       427.6       5
    WM    A0       n250          250 0.243239 0.026068       250.0      15
    WM    A0       n300          300 0.256434 0.035259       300.0      15
    WM    A0       n350          350 0.224858 0.046940       350.0      15
    WM    A0  full_T412          329 0.263544 0.029880       329.6       5
    WM    A0    full_TB          427 0.267696 0.020775       427.6       5
    WM    B1       n250          250 0.203630 0.022530       250.0      15
    WM    B1       n300          300 0.246224 0.036559       300.0      15
    WM    B1       n350          350 0.179762 0.052013       350.0      15
    WM    B1  full_T412          329 0.265385 0.041859       329.6       5
    WM    B1    full_TB          427 0.273320 0.015023       427.6       5
```
