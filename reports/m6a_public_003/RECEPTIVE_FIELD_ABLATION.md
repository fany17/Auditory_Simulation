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
