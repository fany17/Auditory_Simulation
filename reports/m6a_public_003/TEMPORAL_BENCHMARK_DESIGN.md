# M6A-PUBLIC-003 Temporal Benchmark Design

## 状态与图形契约

- 状态：`READY_FOR_REVIEW`（Agent B 产物，未写 ACCEPT/HUMAN_ACCEPTED）。
- 核心工程命题：在同一合成时序任务、统一优化和分组匹配下，结构变量改变快速事件的可定位性、区分性或未见参数泛化时，才构成结构 benchmark 证据。
- 图形类型：quantitative grid；主证据为任务性能与 RF/temporal-resolution 曲线，控制证据为参数匹配、representation distance、event-triggered response 和 autocorrelation。
- 后端：Python/Matplotlib；导出 PNG、SVG、PDF，所有图由同一 Python 脚本生成。
- Reviewer risks：3 seeds 不是生物学重复；theoretical RF 不等价于 effective RF；synthetic signal 不等价于声音、神经记录或听觉生理；参数超出 ±10% 的组标为 `CONFOUNDED`。

## Data and split

- Sequence length: `512 samples`, `1.0 ms/sample`; input is a synthetic 1D temporal signal, not patient or STN data.
- Train/validation/test-seen/test-unseen sizes: `1024/256/512/512`.
- Train and validation use the seen parameter regime. Test-seen uses an independent generation seed with the same ranges. Test-unseen uses held-out rates, jitter magnitudes and phase magnitudes.
- No adjacent-window, clip, story, subject or patient records are read; each sample is generated from an independent seed stream and split arrays are reused by every architecture.

## Tasks and metrics

- Rate discrimination: 4/6/8/10 Hz seen classes; balanced accuracy and rate MAE.
- Regular vs jitter: binary balanced accuracy and jitter-magnitude MAE.
- Omission detection: binary balanced accuracy.
- Phase-shift detection: binary balanced accuracy and phase-magnitude MAE.
- Onset/event timing: absolute error in milliseconds.
- Unseen generalization: held-out rate, jitter and phase magnitude MAE.
- Representation checks without additional fitting: no-fit latent distance after a 10-ms shift, event-triggered response, temporal lag-1 autocorrelation.

## Matched training contract

- Optimizer: AdamW, learning rate `0.002`, weight decay `0.0001`; epochs `8`; batch `64`; gradient clip `5.0`.
- Seeds: `[11, 22, 33]`. No per-architecture hyperparameter search or early stopping; the final epoch is evaluated for every run.
- Failure policy: retain non-finite loss, divergence, timeout and failed seed rows; never replace a failed run with a best run.

## RF convention

For a layer with kernel `k`, dilation `d`, stride `s`, previous jump `j` and RF `r`: `r_new = r + (k-1)d j`, `j_new = j s`. RF is reported as the number of inclusive input samples and `RF_ms = RF_samples × 1 ms`; output temporal resolution is the jump, not the RF width. Theoretical RF describes graph connectivity and is not an effective influence estimate learned from data.

## Reproducibility artifacts

- Code: `scripts/m6a_public_003_temporal_benchmark.py`.
- Configuration: `configs/m6a_public_003.json`.
- Raw model checkpoints and large training tensors are not saved; the report bundle contains only lightweight structured results, logs and figures.

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

## Boundary and interpretation

The localization shift and jitter magnitude are different probes. A model can
recover the global onset translation without identifying the periodic jitter
cause, and balanced accuracy does not establish magnitude estimation. The
three seeds quantify repeatability for this generator/model pair only; they do
not estimate biological or participant variability. A flat or chance-level
curve is retained as a negative result, and any failed/non-PASS seed remains
in the status CSV and manifest.
