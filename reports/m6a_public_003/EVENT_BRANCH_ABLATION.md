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
