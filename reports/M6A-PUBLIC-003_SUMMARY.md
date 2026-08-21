# M6A-PUBLIC-003 Summary

状态：`READY_FOR_REVIEW`。这是 Agent B 的执行交付，不是 `ACCEPT` 或 `HUMAN_ACCEPTED`。

- Runs recorded: `45`; PASS: `45`; failed/diverged/other: `0`.
- Scope: exactly the three approved groups; no pretrained model, model download, patient/STN data, cochlear frontend, fast/slow branch or Mamba gating.
- Primary evidence: `m6a_public_003_metrics_by_seed.csv`, `m6a_public_003_metrics_summary.csv`, `m6a_public_003_model_parameters.csv`, `receptive_field_by_layer.csv` and generated figures.

## Engineering PASS criteria

- Parameter/RF smoke completed before formal training.
- Every formal variant was attempted under the same optimizer, epoch count, batch size, split and three seeds.
- RF recurrence and output frame step are computed from the exact model specification.
- Negative, failed and parameter-confounded states are retained in structured outputs.

## Evidence boundary

A performance difference in this synthetic benchmark is engineering evidence about these small model graphs and task distributions. It is not evidence of brain-region correspondence, auditory-system homology, patient/STN applicability, causality in neural tissue or clinical utility. Theoretical RF is not effective RF, and three seeds do not estimate biological variability.

## Open scientific questions

- Whether the observed task effects persist across real audio perturbations and public neural recordings remains unresolved.
- Whether effective rather than theoretical RF tracks the performance differences remains unresolved.
- Whether explicit change information adds stable value after a broader family of matched controls remains unresolved.

下一步必须经过 Agent A/人工审核，不由本 worker 自动启动。

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
The explicit magnitude-axis figure is
`reports/figures/m6a_public_003_performance_vs_perturbation_magnitude.{png,svg,pdf}`.
Each magnitude-specific probe array is generated once and reused across all
15 structures and three seeds; no magnitude-specific tuning is performed.

### Readout across all structures

**localization_onset_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 87.4849 ± 5.3533 | 89.0685 ± 6.7338 | 96.7472 ± 12.2416 |
| late_downsample | 88.7612 ± 5.6117 | 90.3566 ± 6.7589 | 99.6641 ± 13.0525 |
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
| event_baseline | 86.3510 ± 1.6195 | 87.5969 ± 2.4667 | 95.1036 ± 5.2131 |
| explicit_change | 85.3547 ± 0.3616 | 86.0920 ± 0.6096 | 91.8303 ± 1.8740 |
| ordinary_second_branch | 85.0709 ± 0.0574 | 85.5932 ± 0.0705 | 90.3798 ± 0.2027 |

**discrimination_balanced_accuracy**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 0.5008 ± 0.0056 | 0.5072 ± 0.0101 | 0.5000 ± 0.0142 |
| late_downsample | 0.4784 ± 0.0179 | 0.5080 ± 0.0046 | 0.4840 ± 0.0055 |
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
| event_baseline | 0.4973 ± 0.0144 | 0.4934 ± 0.0087 | 0.4978 ± 0.0058 |
| explicit_change | 0.4943 ± 0.0137 | 0.4987 ± 0.0120 | 0.5002 ± 0.0106 |
| ordinary_second_branch | 0.4880 ± 0.0131 | 0.5053 ± 0.0080 | 0.4939 ± 0.0074 |

**generalization_jitter_mae_ms**

| variant | 10 ms | 20 ms | 50 ms |
|---|---:|---:|---:|
| early_downsample | 8.2839 ± 0.1106 | 18.2832 ± 0.1044 | 48.2755 ± 0.1085 |
| late_downsample | 7.9905 ± 0.2194 | 18.0067 ± 0.2034 | 47.9876 ± 0.2215 |
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
| event_baseline | 8.1471 ± 0.2865 | 18.1513 ± 0.2801 | 48.1461 ± 0.2860 |
| explicit_change | 8.0339 ± 0.0610 | 18.0334 ± 0.0572 | 48.0341 ± 0.0621 |
| ordinary_second_branch | 8.0057 ± 0.0422 | 18.0033 ± 0.0369 | 48.0065 ± 0.0434 |

### Result boundary

The new curves answer whether this small synthetic model family preserves
onset localization, regular-versus-jitter discrimination and jitter-magnitude
extrapolation at 10/20/50 ms. They do not show that a structure localizes a
real-world perturbation, distinguish causal event types, or generalize to
neural data. The existing formal negative findings remain unchanged, including
chance-level phase-shift detection and the absence of a stable explicit-change
advantage over the parameter-matched ordinary second branch.
