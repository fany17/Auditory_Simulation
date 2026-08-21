# M6A-PUBLIC-003 Receptive-Field Ablation

This report separates local kernel size, dilation schedule, parallel multiscale RF and RF/downsampling decoupling. The table is generated from the same architecture specs as the trained models.

| variant | parameters | RF ms | final resolution ms | onset_timing_error_ms | omission_detection_balanced_accuracy | rate_discrimination_balanced_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| uniform_local | 4619 | 11.0 | 1.0 | 85.6421 ± 0.8089 | 0.5381 ± 0.0139 | 0.3839 ± 0.1427 |
| exponential_growth | 4619 | 33.0 | 1.0 | 86.6782 ± 2.5891 | 0.4970 ± 0.0060 | 0.3219 ± 0.1002 |
| delayed_growth | 4619 | 13.0 | 1.0 | 86.6219 ± 2.7920 | 0.5252 ± 0.0264 | 0.3752 ± 0.1207 |
| parallel_multiscale | 4815 | 59.0 | 1.0 | 85.0804 ± 0.0857 | 0.5000 ± 0.0000 | 0.2500 ± 0.0000 |
| rf_stride_coupled | 4619 | 33.0 | 4.0 | 88.2105 ± 6.1345 | 0.4933 ± 0.0116 | 0.2993 ± 0.0777 |
| rf_dilation_decoupled | 4619 | 33.0 | 1.0 | 86.6782 ± 2.5891 | 0.4970 ± 0.0060 | 0.3219 ± 0.1002 |

| variant | parameters | RF ms | final resolution ms | unseen_rate_mae_hz | unseen_jitter_mae_ms | unseen_phase_magnitude_mae_rad |
|---|---:|---:|---:|---:|---:|---:|
| uniform_local | 4619 | 11.0 | 1.0 | 1.8611 ± 0.1605 | 2.5274 ± 0.0215 | 0.7634 ± 0.0071 |
| exponential_growth | 4619 | 33.0 | 1.0 | 2.0062 ± 0.0414 | 2.5320 ± 0.0096 | 0.7528 ± 0.0475 |
| delayed_growth | 4619 | 13.0 | 1.0 | 1.9092 ± 0.1680 | 2.5286 ± 0.0267 | 0.7688 ± 0.0140 |
| parallel_multiscale | 4815 | 59.0 | 1.0 | 2.0225 ± 0.0008 | 2.5141 ± 0.0109 | 0.7450 ± 0.0187 |
| rf_stride_coupled | 4619 | 33.0 | 4.0 | 2.0233 ± 0.0589 | 2.5338 ± 0.0037 | 0.7542 ± 0.0368 |
| rf_dilation_decoupled | 4619 | 33.0 | 1.0 | 2.0062 ± 0.0414 | 2.5320 ± 0.0096 | 0.7528 ± 0.0475 |

## Kernel-size diagnostic

| variant | parameters | RF ms | final resolution ms | onset_timing_error_ms | phase_shift_detection_balanced_accuracy |
|---|---:|---:|---:|---:|---:|
| kernel_3 | 4619 | 11.0 | 1.0 | 85.6421 ± 0.8089 | 0.5000 ± 0.0000 |
| kernel_7 | 4575 | 27.0 | 1.0 | 85.2714 ± 0.6434 | 0.5000 ± 0.0000 |
| kernel_15 | 4811 | 59.0 | 1.0 | 90.9835 ± 7.3174 | 0.5000 ± 0.0000 |
| kernel_31 | 3855 | 123.0 | 1.0 | 85.4695 ± 2.2032 | 0.5000 ± 0.0000 |

`rf_stride_coupled` and `rf_dilation_decoupled` are designed to have the same final theoretical RF, while their final output steps differ. The exact per-layer audit is authoritative. Kernel width changes use width adjustment but any residual parameter mismatch is retained as `CONFOUNDED`.

## 10/20/50 ms Supplementary Perturbation Evaluation

This additive evaluation was run after the formal results and does not replace
or overwrite them. The supplement retrained the same 15 architecture variants
because the formal run saved no checkpoints; it reused the formal train,
validation, test-seen and test-unseen generation rules, AdamW optimizer,
8-epoch budget, batch size 64, gradient clip 5.0 and seeds 11/22/33.

Perturbation definition:

- **Localization:** the same test-seen signal is translated by a positive
  global shift of exactly 10, 20 or 50 ms with zero fill. Only the onset target
  is translated by the same amount. `localization_onset_mae_ms` measures the
  shifted-onset error; `localization_shift_recovery_mae_ms` measures the error
  in recovering the shift from the model's predicted onset change.
- **Discrimination:** each magnitude is an independent mixed regular/jitter
  probe generated with `unseen_jitter_ms=[m]`. Rates and phase magnitudes are
  reset to the training ranges, so `discrimination_balanced_accuracy` isolates
  regular-versus-jitter classification.
- **Generalization:** the positive-jitter subset of the same probe is used for
  `generalization_jitter_mae_ms` and its magnitude-normalized counterpart.

The 10/20/50 ms probes are all **unseen/extrapolative**: training contains
2/4/6 ms jitter and the original unseen split contains 3/5/7 ms. None of the
three supplementary magnitudes is used for training or tuning. This is a
small synthetic perturbation benchmark, not localization or discrimination
evidence for real audio or neural recordings.

Supplement run status: `45/45` seed runs PASS. The
formal parameter-match labels are retained unchanged: early_downsample=PASS, late_downsample=PASS, uniform_local=PASS, exponential_growth=PASS, delayed_growth=PASS, parallel_multiscale=PASS, rf_stride_coupled=PASS, rf_dilation_decoupled=PASS, kernel_3=PASS, kernel_7=PASS, kernel_15=PASS, kernel_31=CONFOUNDED, event_baseline=PASS, explicit_change=PASS, ordinary_second_branch=PASS.

Traceability files: `m6a_public_003_perturbation_metrics_by_seed.csv`,
`m6a_public_003_perturbation_summary.csv`,
`m6a_public_003_perturbation_run_status.csv` and
`m6a_public_003_perturbation_manifest.json`.

### RF growth, multiscale and kernel curves

**localization_onset_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| uniform_local | 86.3510 ± 1.6195 | 87.5969 ± 2.4667 | 95.1036 ± 5.2131 |
| exponential_growth | 87.9755 ± 2.9060 | 89.8859 ± 3.3060 | 99.2501 ± 4.2599 |
| delayed_growth | 87.6237 ± 4.3522 | 89.1145 ± 5.9256 | 97.0217 ± 10.9479 |
| parallel_multiscale | 85.3168 ± 0.4985 | 86.0391 ± 0.8492 | 91.7290 ± 2.5625 |
| rf_stride_coupled | 89.6111 ± 7.7308 | 91.8673 ± 8.6547 | 101.6654 ± 12.2994 |
| rf_dilation_decoupled | 87.9755 ± 2.9060 | 89.8859 ± 3.3060 | 99.2501 ± 4.2599 |
| kernel_3 | 86.3510 ± 1.6195 | 87.5969 ± 2.4667 | 95.1036 ± 5.2131 |
| kernel_7 | 86.0325 ± 1.0136 | 87.2900 ± 1.4139 | 94.9801 ± 2.5398 |
| kernel_15 | 93.2111 ± 8.1822 | 96.0551 ± 8.9230 | 108.4186 ± 11.3189 |
| kernel_31 | 85.3194 ± 0.7955 | 85.6413 ± 0.6835 | 90.5998 ± 5.3031 |

**localization_shift_recovery_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| uniform_local | 10.0155 ± 0.0208 | 20.0288 ± 0.0466 | 50.0269 ± 0.0637 |
| exponential_growth | 10.0207 ± 0.0411 | 20.0106 ± 0.0311 | 50.0600 ± 0.1365 |
| delayed_growth | 10.0181 ± 0.0277 | 20.0300 ± 0.0523 | 50.0151 ± 0.0712 |
| parallel_multiscale | 9.9993 ± 0.0007 | 19.9993 ± 0.0007 | 49.9999 ± 0.0009 |
| rf_stride_coupled | 10.1150 ± 0.3435 | 20.0026 ± 0.0122 | 49.7878 ± 0.0731 |
| rf_dilation_decoupled | 10.0207 ± 0.0411 | 20.0106 ± 0.0311 | 50.0600 ± 0.1365 |
| kernel_3 | 10.0155 ± 0.0208 | 20.0288 ± 0.0466 | 50.0269 ± 0.0637 |
| kernel_7 | 9.9996 ± 0.0222 | 19.9907 ± 0.0219 | 49.9980 ± 0.2652 |
| kernel_15 | 10.0058 ± 0.0157 | 19.9972 ± 0.0183 | 50.0713 ± 0.0416 |
| kernel_31 | 9.9284 ± 0.0657 | 20.0535 ± 0.0814 | 50.6101 ± 0.6599 |

**discrimination_balanced_accuracy**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| uniform_local | 0.4973 ± 0.0144 | 0.4934 ± 0.0087 | 0.4978 ± 0.0058 |
| exponential_growth | 0.5041 ± 0.0238 | 0.4899 ± 0.0040 | 0.5112 ± 0.0172 |
| delayed_growth | 0.4990 ± 0.0154 | 0.4980 ± 0.0154 | 0.4965 ± 0.0030 |
| parallel_multiscale | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| rf_stride_coupled | 0.4904 ± 0.0268 | 0.4983 ± 0.0090 | 0.4925 ± 0.0118 |
| rf_dilation_decoupled | 0.5041 ± 0.0238 | 0.4899 ± 0.0040 | 0.5112 ± 0.0172 |
| kernel_3 | 0.4973 ± 0.0144 | 0.4934 ± 0.0087 | 0.4978 ± 0.0058 |
| kernel_7 | 0.4966 ± 0.0232 | 0.4962 ± 0.0079 | 0.4836 ± 0.0150 |
| kernel_15 | 0.4974 ± 0.0037 | 0.4991 ± 0.0023 | 0.5004 ± 0.0062 |
| kernel_31 | 0.4984 ± 0.0028 | 0.5074 ± 0.0128 | 0.4969 ± 0.0027 |

**generalization_jitter_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| uniform_local | 8.1471 ± 0.2865 | 18.1513 ± 0.2801 | 48.1461 ± 0.2860 |
| exponential_growth | 8.1685 ± 0.1635 | 18.1729 ± 0.1624 | 48.1718 ± 0.1621 |
| delayed_growth | 8.1738 ± 0.3233 | 18.1776 ± 0.3195 | 48.1714 ± 0.3212 |
| parallel_multiscale | 8.0271 ± 0.1271 | 18.0271 ± 0.1271 | 48.0271 ± 0.1272 |
| rf_stride_coupled | 8.1656 ± 0.0791 | 18.1759 ± 0.0670 | 48.1662 ± 0.0713 |
| rf_dilation_decoupled | 8.1685 ± 0.1635 | 18.1729 ± 0.1624 | 48.1718 ± 0.1621 |
| kernel_3 | 8.1471 ± 0.2865 | 18.1513 ± 0.2801 | 48.1461 ± 0.2860 |
| kernel_7 | 8.0116 ± 0.3010 | 18.0023 ± 0.2997 | 48.0175 ± 0.3026 |
| kernel_15 | 8.2126 ± 0.1637 | 18.2083 ± 0.1692 | 48.2123 ± 0.1710 |
| kernel_31 | 7.9994 ± 0.1130 | 17.9579 ± 0.1848 | 47.9336 ± 0.2178 |

**generalization_relative_jitter_error**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| uniform_local | 0.8147 ± 0.0286 | 0.9076 ± 0.0140 | 0.9629 ± 0.0057 |
| exponential_growth | 0.8168 ± 0.0163 | 0.9086 ± 0.0081 | 0.9634 ± 0.0032 |
| delayed_growth | 0.8174 ± 0.0323 | 0.9089 ± 0.0160 | 0.9634 ± 0.0064 |
| parallel_multiscale | 0.8027 ± 0.0127 | 0.9014 ± 0.0064 | 0.9605 ± 0.0025 |
| rf_stride_coupled | 0.8166 ± 0.0079 | 0.9088 ± 0.0033 | 0.9633 ± 0.0014 |
| rf_dilation_decoupled | 0.8168 ± 0.0163 | 0.9086 ± 0.0081 | 0.9634 ± 0.0032 |
| kernel_3 | 0.8147 ± 0.0286 | 0.9076 ± 0.0140 | 0.9629 ± 0.0057 |
| kernel_7 | 0.8012 ± 0.0301 | 0.9001 ± 0.0150 | 0.9604 ± 0.0061 |
| kernel_15 | 0.8213 ± 0.0164 | 0.9104 ± 0.0085 | 0.9642 ± 0.0034 |
| kernel_31 | 0.7999 ± 0.0113 | 0.8979 ± 0.0092 | 0.9587 ± 0.0044 |

### RF-growth and multiscale-group conclusion

No RF-growth, dilation-schedule, parallel-multiscale or RF/downsampling
decoupling variant shows a credible advantage on 10/20/50 ms localization,
discrimination or extrapolation. Uniform, exponential and delayed RF growth
and the multiscale branch did not rescue the failed displacement recovery,
chance-level discrimination or near-training-scale jitter outputs. Any apparent
pattern for `kernel_31` remains `CONFOUNDED` because its parameter match failed;
it is not evidence for a kernel-size effect.
