# M6A-PUBLIC-001 G3 single-recording alignment candidate

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 状态 | `G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY` |
| 协调审核 | `ACCEPT`（2026-08-13） |
| scoped config | `configs/m6a_g3_single_recording_candidate.json` |
| 唯一机器报告 | `reports/g3_single_recording_candidate_20260813.json` |
| 远端数组 | `/home/fanyu/auditory_simulation_m6a/outputs/g3_single_recording_candidate_20260813T081355Z` |
| 完整性策略 | `NON_HASH_AUDIT` |

## 选择与真实读取范围

机器规则只使用 G2 metadata，不读取波形质量或结果：先要求 EDF header 可读、header/TSV channel count 一致、eligible names 全在 EDF、sampling/reference gate 通过，再按 eligible channel count 最小、recording ID 字典序选择；随后按 start time、sample ID 选择最早 eligible train passage。

- recording：`sub-SD012_ses-02_task-PassiveListen`，512 Hz，36 个 eligible channels；
- passage：`sub-SD012_ses-02_task-PassiveListen__seg-004`；stimulus `s4002b-ex01`；
- audio：`stimuli/excerpts/Block 1/s4002b-ex01_normed.wav`；
- passage window：`[126.519807944, 161.15130454263945) s`；
- EDF 分段读取：samples `[64219, 83069)`，即 `[125.427734375, 162.244140625) s`，共 18,850 samples；左右各只加入冻结的 559-sample 有限支持；
- `preload=false`，只请求上述 36 channels；其他 recording/segment 读取数为 0，原始波形未保存。

G2 sidecar 声明 recording duration 为 3552.75 s，G2 EDF header 的最后样本时间为 2599.748046875 s，MNE 实际 `n_times/fs` 为 2599.75 s。该既有差异被显式保留；本次读取终点 162.244140625 s 同时位于两条时间轴范围内，不影响 scoped passage，但不得据此声称完整 recording duration 已重新协调冻结。

## 表征与统一时间网格

所有声音表征均仅从该独立 passage 生成。44.1 kHz mono 音频重采样为 554,104 个 16 kHz samples；模型缓存只读，`download_allowed=false`、`local_files_only=true`、`trust_remote_code=false`、`weights_only=true`。

该 G3 运行发生在 passage-wise `Wav2Vec2FeatureExtractor` normalization 契约冻结之前。其 wav2vec2 raw-input representation 仅保留工程 shape/time 证据，状态为 `MUST_NOT_REUSE_FOR_G4_SCIENTIFIC_BASELINE`；本轮不重算 G3，任何 G4 科学执行必须使用后续独立审核的 preprocessing implementation 与新生成的正式 G4 表征。

| 表征 | native shape | aligned shape |
|---|---:|---:|
| amplitude envelope | `1731 x 1` | `1732 x 1` |
| raw log-mel | `1731 x 80` | `1732 x 80` |
| wav2vec2 projected + 12 layers | `1731 x 13 x 768` | `13 x 1732 x 768` |
| six-subband neural power | 18,850 source samples/channel | `1732 x 36 x 6` |
| pre-transform smoke log-power | 18,850 source samples/channel | `1732 x 36 x 6` |

统一网格为 recording-origin `k/50 s`，共 1,732 frames，范围 126.52–161.14 s。仅使用不外推线性插值；audio、log-mel、wav2vec2 与 neural 完整支持交集为 1,622 common valid frames，范围 127.62–160.04 s。所有 tensor finite，时间戳严格递增且共用同一 frame-time 数组。

有限 FIR 平滑后有 160 个微小负 power 值，最大绝对值 `1.34212234967172e-14`，低于冻结 synthetic 数值容差 `1e-12`。scoped config 在运行前已固定 smoke 公式 `ln(max(power, 0)+1e-30)`；超出该容差将 fail closed。该 smoke log-power 不是正式 train-only target transform，未拟合或冻结 epsilon/center/scale，也不得复用于 baseline。

## 内部格式基准

真实 aligned shape 仅在 2203 比较 NPY-per-tensor 与 NPZ-compressed，均禁止 object array/allow-pickle，未新增依赖。

| 格式 | 字节数 | 写入 | 全读 | 单 wav2vec2 层 | 单神经电极 |
|---|---:|---:|---:|---:|---:|
| NPY-per-tensor | 72,739,700 | 0.5050 s | 0.0579 s | 0.0021 s | 0.0004 s |
| NPZ-compressed | 67,287,978 | 2.9471 s | 0.4002 s | 0.3456 s | 0.0081 s |

provisional internal selection 为 `NPY_PER_TENSOR`：支持 mmap、直接 layer/electrode slice 和逐 tensor 原子失败恢复。该选择只适用于当前内部高维数组，不是已冻结的 M6A→M6B exchange payload contract。

## Fail-closed gate 返工

- `model_runtime` 现精确要求不执行仓库自定义代码、`weights_only=true`、state dict tensor-only、模型处于 eval、无可训练参数，并要求 base encoder 为 0 missing / 0 mismatch / 0 error，unexpected keys 只能是预声明的 7 个预训练头；
- `audio_read`、EDF 分段读取、native/aligned shape、frame-time 公式、逐输入 valid count 与 7 个 tensor 的路径/dtype/shape/字节非空/remote-only/object=false 均改为精确契约；
- 2203 对既有 7 个 NPY 做了只读 header 与全数组回读：文件数 7、active partial 0、无 unexpected NPY、全部 finite、mask true count 1,622、frame time 严格递增；未重新读取 EDF、音频或运行模型；
- benchmark 现要求两种格式字节数非空、所有 timing finite 且非负，并精确核对 NPY/NPZ 的 allow-pickle、mmap、direct-slice 与原子写入语义；选择理由与四项预声明标准不可事后漂移；
- `reports/g3_single_recording_candidate_pre_numeric_tolerance_v1.json` 的正式定位为 `SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE`；它只保留历史来源，不是当前候选机器证据。

## 验证、声称边界与停止点

- 2203 专用环境最终复核：`122 passed, 247 subtests`；Ruff PASS；mypy 37 files PASS；主 config 与 neural method gate PASS；accepted final-embargo 定向测试 `13 passed, 36 subtests`；G3 gate 13/13 checks PASS，G3 定向测试 `13 passed, 77 subtests`；
- `real_neural_waveform_read_scope=ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_PLUS_FROZEN_FINITE_SUPPORT_ONLY`；
- `formal_baseline_run=false`、`scientific_result_claimed=false`、`exchange_candidate_created=false`、`formal_train_only_transform_fitted=false`；
- 未计算 ridge、Pearson r、R2、null、显著性或层优劣；不能声称任何神经表征、模型层优势、脑区结论或泛化结果；
- 首次通过输出未显式机器记录既有 duration 差异与负 power 容差，保留为上述 superseded provenance；当前唯一候选为 `reports/g3_single_recording_candidate_20260813.json`；
- G3 协调接受后只解除“可起草 G4 协议”的依赖；它没有授权 G4 数据执行、其他 recording/subject 扩展或 exchange candidate。

协调已独立复核并接受本候选。该接受仅证明单 passage 工程时间轴、shape、有界读取和数组可读性，不构成 neural encoding、层优劣、泛化、显著性或其他科学结果。后续 G4 仍须独立冻结协议并获得执行授权。
