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
