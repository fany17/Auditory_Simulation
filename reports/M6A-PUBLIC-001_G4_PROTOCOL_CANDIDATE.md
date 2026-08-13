# M6A-PUBLIC-001 G4 protocol candidate

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 当前状态 | `G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW` |
| 原协议历史状态 | `G4_PROTOCOL_COORDINATOR_ACCEPTED` |
| 协议配置 | `configs/m6a_g4_protocol_candidate.json` |
| schema | `schemas/m6a_g4_protocol_candidate.schema.json` |
| 原协议 accepted 报告 | `reports/g4_protocol_candidate_20260813_v3.json` |
| 当前修订候选报告 | `reports/g4_protocol_amendment_candidate_20260813_v2.json` |
| 当前 2203 日志 | `/home/fanyu/auditory_simulation_m6a/logs/g4_protocol_amendment_candidate_20260813_v2.json` |
| 完整性边界 | `NON_HASH_AUDIT`；不提供密码学完整性声称 |

## 依赖、范围与停止点

G3 已由协调者接受为 `G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY`。G4 原协议也已由协调者接受；本文件原位保留该历史，同时把新发现的 wav2vec2 preprocessing 输入契约作为独立实质性修订候选，不能静默继承 accepted 状态。G4 范围继续固定为同一受试者、同一 `sub-SD012_ses-02_task-PassiveListen` recording：24 train、8 validation、8 test passages，排除 `ses-01`，不允许按波形、质量或结果选择。

原协议 accepted 机器报告 `reports/g4_protocol_candidate_20260813_v3.json` 中的 `protocol_config` 是可变路径；该路径现已承载 amendment。机器边界已明确：原接受范围不包含 `WAV2VEC2_PREPROCESSING_INPUT_CONTRACT`，不得仅凭该历史报告宣称当前 amendment accepted。协议修订唯一当前候选是 `reports/g4_protocol_amendment_candidate_20260813_v2.json`；无版本旧报告已标记 `SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE` 并指向 v2。

本轮仅使用轻量 split/config 与 synthetic matrix 验证协议。`g4_execution_authorized=false`、`new_real_edf_read=false`、`new_real_audio_read=false`、`new_feature_extraction_run=false`、`ridge_run=false`、`null_run=false`、`metric_run=false`。未执行 G4，也未建立 exchange candidate。

## Wav2vec2 preprocessing 修订

- `preprocessor_config.json` 机器冻结为 `feature_size=1`、`sampling_rate=16000`、`padding_value=0.0`、`do_normalize=true`、`return_attention_mask=false`、`padding_side=right`；
- 每个 passage 独立以 float32 mean、population variance (`ddof=0`) 和 epsilon `1e-7` 计算 `(x-mean)/sqrt(var+epsilon)`；不得跨 passage 或 split 共享统计量，常数或 non-finite waveform fail closed；
- 输入为单 passage、16 kHz mono、无 padding；`return_attention_mask=false` 时不创建或传入 attention mask；
- 自有实现须与本地 `Wav2Vec2FeatureExtractor` 在 `rtol=atol=1e-7` 内等价；warm-up 与 longest synthetic input 实测最大绝对差均为 0；
- G3 既有 raw-input wav2vec2 representation 只保留工程 shape/time 证据，状态为 `MUST_NOT_REUSE_FOR_G4_SCIENTIFIC_BASELINE`。不重算 G3；未来真实 G4 必须复用本修订接受后的同一实现并重新生成 G4 表征。

2203 无代理优先探测清华 TUNA，目标路径返回 HTTP 404；随后只回退到单一 `https://hf-mirror.com`，返回 HTTP 200，正文为 159 bytes 且语义字段匹配。首次 hf-mirror 请求返回 HTTP 403 的失败报告被保留，有限重试成功。缓存仍为 remote-only，未写入网络正文，模型下载保持关闭。第三方镜像、mutable `main` 与 no-hash 边界不提供密码学完整性或不可变 provenance。

## 时间、target 与 feature transform

- feature X 使用 recording-origin 50 Hz `k/50 s` grid；11 个 lag 固定为 0–0.5 s、步长 0.05 s。target 在精确 `t+lag` 上从有限滤波后的 native neural power 做两点线性采样，不把 0.05 s 默默取整为 50 Hz frame，也不外推或跨 passage/split。
- 所有 lag 使用同一 common complete-support X frame 集。target transform 只拟合一次：对 24 个 train passages、11 个 lag 的 exact target time 做 recording-origin 10 ms tick 去重，再按 channel×subband 拟合 epsilon、log、mean 与 population standard deviation (`ddof=0`)；同一组参数应用全部 lag 与 validation/test。
- envelope 与 wav2vec2 各自只用 train 统计量。log-mel 固定 80→20 维 PCA；train samples、numeric rank 均须至少 20，前 20 个 singular value 必须 finite，且第 20 个必须高于冻结的 float64 rank tolerance。失败时该 variant 为 `NOT_ESTIMABLE`，禁止自动减维。

## Ridge 计算与 test 边界

Ridge 固定为 float64 thin-SVD closed form、无 intercept、6 个预声明 alpha。每个 feature variant、每个 fit partition 最多做一次 decomposition，并以 multi-target 路径复用于全部 11 lag×36 electrodes×6 alpha；15 个 variants 的 train 与 train+validation decomposition 上限分别为 15，总上限 30。synthetic 测试证明复用路径与 direct ridge 在 `rtol/atol=1e-10` 内一致。

Alpha 只由 validation 选择；`1e-12` tie 内取最小 alpha。锁定后允许用 train+validation 重拟合 ridge，但所有 transform/PCA/target 参数仍只来自 train。test 只评估一次，不用于 layer/lag/alpha 选择。

## Null、metric 与校正

- primary smoke null 是 test stimulus derangement：8 个 test passages 形成一个预声明 stratum，完整枚举 14,833 个无 fixed-point bijection，再以 NumPy PCG64 seed `20260813` 无放回均匀抽取 20。每个 permutation 的同一 mapping 跨所有 feature/layer/lag/electrode cell 与两个 correction family 共用。
- Null 不重跑模型，也不重拟合 transform、PCA、ridge、alpha 或 neural target；只把锁定的 donor test prediction 映射到 target passage neural target。每对 passage 内将 donor prediction 按 normalized complete-support phase 线性映射到 target common frames，禁止先跨 passage 拼接再 time warp。
- Observed 与 null 均先逐 target passage 计算 Pearson r/R2，再由 8 个 passage 等权聚合；任一 passage 不可估计时，该 cell 为 `NOT_ESTIMABLE`。常数、样本不足或 non-finite 均 fail closed。
- wav2vec2 的 passage 内 Transformer 具有全局上下文，因此 circular shift 不适用；acoustic circular shift 只保留为 secondary mechanical diagnostic，不得替代 primary null。
- 两个固定 family 分别为 wav2vec2 `36×13×11=5,148` cells 与 acoustic `36×2×11=792` cells。one-sided max Pearson statistic 使用 observed 与全部 20 null 均可估计 cell 的固定交集；交集为空即失败。smoke p 仅按 `(1 + count(max_null >= observed))/21` 检查机械流程，不提供稳定显著性结论。

## 资源门禁

协议静态上界为 40 passages、1,532.45596371882 s、76,623 frames、3,151,044,252 core tensor bytes、预计新增不超过 20 GB、累计预计 GPU 不超过 4 h。这些不是实跑资源证据。既有单次 smoke 硬边界保持不变：每个 GPU invocation 必须严格低于 2 h，计算必须拆成可恢复阶段，并在达到 2 h 前 checkpoint/停报。

当前资源/runtime preflight 状态仅为 `G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW`。只读空间盘点与 synthetic longest-passage canary 已完成，但这不等于执行授权。实际 free bytes 为 978,024,435,712；data+cache+预计新增 20 GB 为 34,553,621,929 bytes。保守估算单次 invocation 上界 150 s、40 passages 合计 1.6666666666666665 GPU h，仍须每 passage 独立、成功后原子 checkpoint。任一门禁漂移即停止，不得启动 G4。

## 验证与证据边界

- 原协议 accepted 报告的 9/9 required checks 为 true。当前 preprocessing 修订机器报告同样为 9/9 true，但状态严格保持 `G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW`；
- current-candidate 治理只允许 v2 协议修订报告为当前候选；superseded 报告的状态、`current_candidate=false` 与 `superseded_by` 均由主 config gate 复核；
- 旧 `reports/g4_protocol_candidate_20260813.json` 与 `reports/g4_protocol_candidate_20260813_v2.json` 均已机器标记为 `SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE`；v2 因错误放宽 main config 的单次 2 h 硬边界而被替代，它不是当前候选；
- 一次文档同步把任务书副本平铺到 2203 snapshot 的 `doc/` 根；核对绝对路径后已将该副本移动到 `/home/fanyu/auditory_simulation_m6a/logs/provenance/g4_protocol_sync_layout_20260813/M6A-PUBLIC-001.md`，正式路径 `code_snapshot/doc/tasks/M6A-PUBLIC-001.md` 已更新。未删除文件，数据目录未受影响；
- 2203 专用环境最终验证：`147 passed, 349 subtests`；Ruff 对完整 `src scripts tests` PASS；mypy 对完整 46 个文件 PASS；主 config gate PASS；当前协议修订报告 9/9、preflight 报告 15/15 required checks 为 true；
- 当前没有 ridge、Pearson r、R2、null、max-statistic、显著性、层优劣或神经表征结果。`region_summary=NOT_ESTIMABLE`，不得从 contact 名称推断脑区。

本协议修订候选与资源/runtime preflight 候选均等待协调审核；在两者获协调接受并取得新的 scoped execution authorization 前停止。
