# M6A-PUBLIC-003 Explicit Change/Event Branch Ablation

The event experiment compares a single content branch, a first-difference (`Δx(t)`) branch and a parameter-matched ordinary second raw-input branch. Only the second input transformation differs between the two branch variants; both use two width-11 branch backbones and the same 1x1 fusion.

| variant | parameters | RF ms | final resolution ms | onset_timing_error_ms | omission_detection_balanced_accuracy | phase_shift_detection_balanced_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| event_baseline | 4619 | 11.0 | 1.0 | 85.6421 ± 0.8089 | 0.5381 ± 0.0139 | 0.5000 ± 0.0000 |
| explicit_change | 4867 | 11.0 | 1.0 | 85.0875 ± 0.0692 | 0.5019 ± 0.0032 | 0.5000 ± 0.0000 |
| ordinary_second_branch | 4867 | 11.0 | 1.0 | 85.0551 ± 0.0515 | 0.5117 ± 0.0203 | 0.5000 ± 0.0000 |

| variant | parameters | RF ms | final resolution ms | unseen_rate_mae_hz | unseen_jitter_mae_ms | unseen_phase_magnitude_mae_rad |
|---|---:|---:|---:|---:|---:|---:|
| event_baseline | 4619 | 11.0 | 1.0 | 1.8611 ± 0.1605 | 2.5274 ± 0.0215 | 0.7634 ± 0.0071 |
| explicit_change | 4867 | 11.0 | 1.0 | 2.0326 ± 0.0187 | 2.5141 ± 0.0033 | 0.7588 ± 0.0296 |
| ordinary_second_branch | 4867 | 11.0 | 1.0 | 1.9416 ± 0.1627 | 2.5106 ± 0.0020 | 0.7501 ± 0.0190 |

本轮未满足显式变化支路的必要证据模式：onset error 为 explicit_change 85.0875 ± 0.0692、ordinary_second_branch 85.0551 ± 0.0515；omission balanced accuracy 为 explicit_change 0.5019 ± 0.0032、ordinary_second_branch 0.5117 ± 0.0203；phase-shift detection 两者均为 chance-level。故本轮不支持 explicit_change 相对普通第二支路的稳定优势。
三 seeds 和 synthetic data 只支持工程 benchmark 表述，不建立神经 event pathway 结论。

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

### Explicit change/event branch curves

**localization_onset_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| event_baseline | 86.3510 ± 1.6195 | 87.5969 ± 2.4667 | 95.1036 ± 5.2131 |
| explicit_change | 85.3547 ± 0.3616 | 86.0920 ± 0.6096 | 91.8303 ± 1.8740 |
| ordinary_second_branch | 85.0709 ± 0.0574 | 85.5932 ± 0.0705 | 90.3798 ± 0.2027 |

**localization_shift_recovery_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| event_baseline | 10.0155 ± 0.0208 | 20.0288 ± 0.0466 | 50.0269 ± 0.0637 |
| explicit_change | 9.9974 ± 0.0069 | 19.9983 ± 0.0075 | 49.9942 ± 0.0076 |
| ordinary_second_branch | 9.9998 ± 0.0039 | 19.9970 ± 0.0078 | 50.0181 ± 0.0357 |

**discrimination_balanced_accuracy**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| event_baseline | 0.4973 ± 0.0144 | 0.4934 ± 0.0087 | 0.4978 ± 0.0058 |
| explicit_change | 0.4943 ± 0.0137 | 0.4987 ± 0.0120 | 0.5002 ± 0.0106 |
| ordinary_second_branch | 0.4880 ± 0.0131 | 0.5053 ± 0.0080 | 0.4939 ± 0.0074 |

**generalization_jitter_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| event_baseline | 8.1471 ± 0.2865 | 18.1513 ± 0.2801 | 48.1461 ± 0.2860 |
| explicit_change | 8.0339 ± 0.0610 | 18.0334 ± 0.0572 | 48.0341 ± 0.0621 |
| ordinary_second_branch | 8.0057 ± 0.0422 | 18.0033 ± 0.0369 | 48.0065 ± 0.0434 |

**generalization_relative_jitter_error**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| event_baseline | 0.8147 ± 0.0286 | 0.9076 ± 0.0140 | 0.9629 ± 0.0057 |
| explicit_change | 0.8034 ± 0.0061 | 0.9017 ± 0.0029 | 0.9607 ± 0.0012 |
| ordinary_second_branch | 0.8006 ± 0.0042 | 0.9002 ± 0.0018 | 0.9601 ± 0.0009 |

### Explicit change/event-group conclusion

No event-branch variant shows a credible advantage on the direct 10/20/50 ms
question. `explicit_change` does not outperform the parameter-matched
`ordinary_second_branch` in displacement recovery, regular-versus-jitter
discrimination or jitter-magnitude extrapolation; both remain near the same
failed pattern. The baseline, explicit-change branch and ordinary second branch
therefore do not establish a successful event-based solution.
