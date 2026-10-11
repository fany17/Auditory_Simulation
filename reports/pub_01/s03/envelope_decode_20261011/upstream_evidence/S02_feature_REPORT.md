# PUB-01-S02 特征比较：envelope/onset、log-mel 与冻结 wav2vec2

运行状态：`PASS_WITH_LIMITATION`；模式：`full`；创建时间：`2026-09-24T18:10:07+00:00`。

## 结果先行

| 特征变体 | matched 被试 | matched test mean r | matched test mean R² | +13 s test mean r | +13 s test mean R² | Δr (shifted−matched) |
|---|---:|---:|---:|---:|---:|---:|
| `envelope_onset` | 83 | 0.02224 | 0.00039 | 0.00020 | -0.00107 | -0.02204 |
| `log_mel_pca` | 83 | 0.01936 | -0.00307 | -0.00088 | -0.01546 | -0.02024 |
| `wav2vec2_selected_by_matched_validation` | 83 | 0.02502 | -0.00153 | -0.00044 | -0.01140 | -0.02546 |
| `wav2vec2_projected` | 3 | 0.01373 | -0.00314 | -0.00142 | -0.01529 | -0.01515 |
| `wav2vec2_transformer_04` | 18 | 0.02506 | -0.00093 | -0.00034 | -0.00879 | -0.02540 |
| `wav2vec2_transformer_08` | 62 | 0.02555 | -0.00163 | -0.00042 | -0.01197 | -0.02597 |
| `wav2vec2_transformer_12` | 0 | NA | NA | NA | NA | NA |

`wav2vec2_selected_by_matched_validation` 是按每个被试选层后汇总的 83-subject 配对结果；下面各 `wav2vec2_*` 行只覆盖选择该层的被试子集，不能直接当作完整层间 test 比较。wav2vec2 的 test 只对每个被试在 matched validation 上选出的一个层计算；未选层只保留 validation 指标，避免用 test 选层。HuBERT 因缺少已批准的 model identity/cache 只登记为阻断分支，不影响其余条件。

## 时间基准与采样率

- 原始声音：2203 上 `/home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_format_qc_20260919_v1/expanded_archives` 内按 story label 读取 `.npz`，核验为 48000 Hz、单通道；raw 坐标是 `source_sample / 48000`。
- 模型声音：每个固定 20 s passage 独立从 48 kHz crop，再以 `resample_poly(up=1, down=3)` 得到 16000 Hz；wav2vec2 passage 内独立做 float32 z-score，禁止跨 passage 统计、padding 和 attention mask。
- log-mel：16000 Hz native frame（400-sample window、320-sample hop、80 HTK mel bins）约 50 Hz；wav2vec2 由卷积 receptive-field center 与累计 stride 给出 native frame time，二者分别保留 native grid。
- EEG 与 publisher envelope：均按发布 derivative 的 64 Hz `k/64` 目标网格，使用 `min(index.samples, EEG time, envelope time)` common overlap；publisher envelope 仅作为与 EEG 共享 story-origin 的 derivative，不据此声明硬件触发或物理播放同步。
- 所有深层/native feature 只在各自 native timestamps 与目标 `k/64` 时间之间做无外推线性插值；没有把 48 kHz、16 kHz、50 Hz 和 64 Hz 的采样整数直接混用。

## 公平比较与拟合冻结

- 沿用已完成声学首轮的 83 subjects / 659 records / 同一 subject 内 canonical source-family split；不重算、不改写 split。固定 20 s 非重叠 passage，所有 feature variants 共用同一 record/window 范围与 common valid mask。
- lag 为 `[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]` s；+13 s 错配为 target-grid feature index `+832` 预测原 EEG 时间，不循环、不跨 record、不跨 window。
- log-mel 和 wav2vec2 的原始维度标准化、PCA（20 components）及 score standardization 只用 train；EEG 与最终 lag-design 也只用 train；alpha 只由 validation mean channel Pearson r 选择。
- wav2vec2 候选层在 full run 前冻结为 projected、transformer_04、transformer_08、transformer_12；每个被试只由 matched validation 选层，并把该层用于 matched 与 shifted test。

## 资源与边界

- 资源估算状态：`PASS`；预计 cache：43.00 GiB；实际 feature cache 只留在 2203。
- 本次未下载模型、未回传 raw audio/EEG/cache/weights；Windows 仅保存轻量 CSV/JSON/PNG/Markdown。
- 结果是实际 test 描述性比较，不是显著性检验，也不把阴性或错配结果改写为新的任务分支。

## 产物

- `metrics_per_subject.csv`：逐被试、逐 feature variant、逐 matched/shifted 条件指标。
- `metrics_per_record.csv`：逐 test record 指标。
- `validation_layer_selection.csv`：wav2vec2 层选择只使用 matched validation 的证据。
- `feature_timing_manifest.csv`：raw/model/native/target 时间基准与 rates。
- `paired_mismatch_deltas.csv`：同一被试/变体的 shifted−matched 配对差值。
- `selected_wav2vec2_summary.csv`：按被试 validation 选层后的 83-subject wav2vec2 总体汇总。
- `paired_test_performance.png`、`paired_mismatch_deltas.png`、`wav2vec2_layer_selection.png`、`prediction_example.png`：静态图。
