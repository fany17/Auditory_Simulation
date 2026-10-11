# PUB-01-S02 首轮声学 ridge/TRF 基线

运行状态：`PASS`；模式：`full`。这是固定首轮范围的完整运行。

## 结果先行

| 条件 | 已拟合被试 | test 样本 | 被试宏平均 mean-channel r | 被试宏平均 mean-channel R² |
|---|---:|---:|---:|---:|
| matched | 83 | 9714767 | 0.0224 | 0.0004 |
| +13 s non-circular shift | 83 | 9576367 | 0.0001 | -0.0004 |

错配相对 matched 的被试宏平均差值：`Δr = -0.0223`，`ΔR² = -0.0008`。这些是实际 test 结果，不是显著性检验，也不构成新被试泛化结论。

## 口径与冻结配置

- 输入：SparrKULee 发布方 64 Hz preprocessed EEG（Y）与对应 envelope（X）；X 为 envelope feature-axis 均值及其正的一阶差分。共同起点和 `min(index.samples, EEG time, envelope time)` 是发布方 common-overlap 坐标约定，不是独立测得的硬件/物理播放同步。
- 设计：lag 秒为 `[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`，共 18 个设计列；训练集逐设计列、逐通道做 train-only 均值/标准差；ridge alpha 候选为 `[0.01, 0.1, 1.0, 10.0, 100.0]`，只用 validation 的平均通道 Pearson r 选择。
- 评估：被试内跨 `canonical source family`；每个被试按 family 字典序，`index % 5 = 0` 为 test、1 为 val、2–4 为 train。test 未参与 alpha 选择。
- 对照：同一记录内 feature index `+13.0 s` 预测原 EEG 时间点；不循环、不跨记录、不跨 split 补边界，使用完全相同的拟合/评估流程。

## 拟合前分组修订

S01 只冻结了 raw label 可用性，未冻结 alias-family。拟合前根据已有 SparrKULee 来源命名/库存和 2203 上针对性 envelope 检查，把 `audiobook_N`、其 `_part`、`_shifted`、`_artefact` 变体合为 `audiobook_N`；`podcast_N` 保留为各自 family。这是来源 family 分组，不是人工语义标签；不能把 family 名称解释成已经完成的故事内容标注。未能按规则映射的标签会在首轮前排除并写入 `skipped_records.csv`，本次实际排除数为 `0`。

针对性 family QC（只检查冻结规则涉及的已知变体，不是新全库比对）：`audiobook_1_1` vs `audiobook_1_1_shifted`：status=PASS，shape=29275x1/29275x1，envelope Pearson r=0.9860，max abs diff=0.7049。 `audiobook_1_2` vs `audiobook_1_2_shifted`：status=PASS，shape=26796x1/26796x1，envelope Pearson r=0.9529，max abs diff=1.3007。 `audiobook_1` vs `audiobook_1_artefact`：status=PASS，shape=59958x1/19223x1，envelope Pearson r=1.0000，max abs diff=0.1355。

## 一个实际输入实例

`input_example.csv` 和 `split_manifest.csv` 提供逐记录清单。首个 test 输入实例为：

- `source_recording`：`sub-001_ses-shortstories01_task-listeningActive_run-04_eeg.bdf.gz`；subject：`sub-001`；raw story：`audiobook_1`；source family：`audiobook_1`。
- EEG shape（time × channels）：`60096x64`；envelope shape（time × feature）：`59958x1`；fs：`64.0 Hz`；common samples：`59958`。
- matched 设计/目标 shape：`X=[59830, 18]`、`Y=[59830, 64]`；channel 数：`64`；此记录去向：`test`。训练/验证/test 的 family 去向见 `split_manifest.csv`。

## QC 与真实性边界

- 索引候选：`666`；header/path 可用：`659`；实际参与本模式：`659`；跳过/不可用记录：`7`。
- 索引中大尾差记录共 `26` 条，其中 `25` 条满足至少 3 个 source family 的 train/val/test 构造并进入本轮，`1` 条因所属被试只有 2 个 source family 而登记在 `skipped_records.csv`；本轮没有按旧 3 秒 advisory 删除，也没有把大尾差自动作为失败。进入本轮的记录按各自 common-overlap 评估。
- 这轮使用发布方 derivative；不读 restricted raw、stimulation.tsv，不声明 raw trigger、硬件时钟或物理播放延迟。没有额外下载模型，也没有使用 ds004703 参与本轮拟合。
- 已知需区分的 QC 问题：`不能读/路径或 shape 失败`、`非有限样本被逐时间点丢弃`、`平直通道导致该通道 r 不可定义`，以及仍未知的 `physical_sync`。未知同步信息不是已发现的信号错误；本轮不为填平 UNKNOWN 而改规则。

## 解释

matched 如果高于 +13 s 对照，只能作为“在发布方 common-overlap 坐标下，声学特征到 EEG 的时间对应性更符合该基线”的证据；它仍不能升级为硬件同步证据。若 matched 与错配接近或为阴性，这也是有效结果，不据此反复换 split、窗口、模型或 negative sampling。

## 复现与产物

- [subject_performance_comparison.png](subject_performance_comparison.png)
- [performance_distribution.png](performance_distribution.png)
- [split_coverage.png](split_coverage.png)
- [prediction_example.png](prediction_example.png)
- [metrics_per_subject.csv](metrics_per_subject.csv)
- [metrics_per_record.csv](metrics_per_record.csv)
- [alpha_selection.csv](alpha_selection.csv)
- [sample_manifest.csv](sample_manifest.csv)
- [split_manifest.csv](split_manifest.csv)
- [story_family_map.csv](story_family_map.csv)
- [story_family_mapping_qc.csv](story_family_mapping_qc.csv)
- [input_example.csv](input_example.csv)
- [run_summary.json](run_summary.json)
- [configs/pub_01_s02_acoustic_baseline.json](configs/pub_01_s02_acoustic_baseline.json)
- [pub_01_s02_acoustic_baseline.py](pub_01_s02_acoustic_baseline.py)

完整服务器命令、输入路径和固定参数见 [SAMPLE_PARAMETERS_COMMANDS.md](SAMPLE_PARAMETERS_COMMANDS.md)。
