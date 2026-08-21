# M6A-PUBLIC-003 Downsampling Ablation

E1 compares early and late stride timing with the same depth, kernel, width, total downsampling factor 4 and convolutional parameterization. Because moving stride also changes graph RF, the timing comparison is reported with an RF confound rather than treated as a pure RF-independent effect.

| variant | parameters | RF ms | final resolution ms | rate_discrimination_balanced_accuracy | omission_detection_balanced_accuracy | onset_timing_error_ms |
|---|---:|---:|---:|---:|---:|---:|
| early_downsample | 6667 | 39.0 | 4.0 | 0.4522 ± 0.0183 | 0.5409 ± 0.0045 | 86.5589 ± 3.4079 |
| late_downsample | 6667 | 23.0 | 4.0 | 0.4461 ± 0.0034 | 0.5437 ± 0.0018 | 87.2262 ± 3.0985 |

| variant | parameters | RF ms | final resolution ms | unseen_rate_mae_hz | unseen_jitter_mae_ms | unseen_phase_magnitude_mae_rad |
|---|---:|---:|---:|---:|---:|---:|
| early_downsample | 6667 | 39.0 | 4.0 | 1.8337 ± 0.0791 | 2.5354 ± 0.0018 | 0.7571 ± 0.0248 |
| late_downsample | 6667 | 23.0 | 4.0 | 1.8482 ± 0.1229 | 2.5128 ± 0.0128 | 0.7494 ± 0.0301 |

Interpretation is descriptive across three seeds. A positive difference is not a biological or causal claim; the exact RF and temporal-resolution rows are in `receptive_field_by_layer.csv`.

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

### Downsampling curves

**localization_onset_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 87.4849 ± 5.3533 | 89.0685 ± 6.7338 | 96.7472 ± 12.2416 |
| late_downsample | 88.7612 ± 5.6117 | 90.3566 ± 6.7589 | 99.6641 ± 13.0525 |

**localization_shift_recovery_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 10.5602 ± 1.1107 | 19.9972 ± 0.0070 | 50.0641 ± 1.2608 |
| late_downsample | 10.6439 ± 2.4911 | 20.0033 ± 0.0120 | 49.6433 ± 2.2536 |

**discrimination_balanced_accuracy**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 0.5008 ± 0.0056 | 0.5072 ± 0.0101 | 0.5000 ± 0.0142 |
| late_downsample | 0.4784 ± 0.0179 | 0.5080 ± 0.0046 | 0.4840 ± 0.0055 |

**generalization_jitter_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 8.2839 ± 0.1106 | 18.2832 ± 0.1044 | 48.2755 ± 0.1085 |
| late_downsample | 7.9905 ± 0.2194 | 18.0067 ± 0.2034 | 47.9876 ± 0.2215 |

**generalization_relative_jitter_error**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 0.8284 ± 0.0111 | 0.9142 ± 0.0052 | 0.9655 ± 0.0022 |
| late_downsample | 0.7991 ± 0.0219 | 0.9003 ± 0.0102 | 0.9598 ± 0.0044 |

### Downsampling-group conclusion

Neither `early_downsample` nor `late_downsample` shows a credible advantage on
the direct 10/20/50 ms question. Both have displacement-recovery error near
the imposed magnitude, discrimination near chance and jitter generalization
near magnitude minus 2 ms with larger relative error. The early-versus-late
choice therefore did not rescue localization, regular-versus-jitter
discrimination or extrapolation; raw onset MAE is not displacement recovery.
