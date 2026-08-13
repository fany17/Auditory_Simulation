# M6A-PUBLIC-001 G4 resource and runtime preflight candidate

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 候选状态 | `G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW` |
| 协调结论 | `G4_RESOURCE_AND_RUNTIME_PREFLIGHT_COORDINATOR_ACCEPTED`（2026-08-13） |
| 协议依赖 | `G4_PROTOCOL_AMENDMENT_COORDINATOR_ACCEPTED` |
| 配置 | `configs/m6a_g4_resource_runtime_preflight_candidate.json` |
| schema | `schemas/m6a_g4_resource_runtime_preflight_candidate.schema.json` |
| 当前机器报告 | `reports/g4_resource_runtime_preflight_candidate_20260813_v3.json` |
| 2203 当前日志 | `/home/fanyu/auditory_simulation_m6a/logs/g4_resource_runtime_preflight_candidate_20260813_v3.json` |
| 完整性边界 | `NON_HASH_AUDIT`；不提供密码学完整性声称 |

## 范围与状态

本节点只执行 2203 空间只读盘点、轻量 manifest 最长 passage 身份选择、preprocessor 语义审计，以及确定性 synthetic mono waveform 的 passage-global wav2vec2 canary。未读取新真实 EDF/audio，未提取真实特征，未运行 ridge、null 或指标；`g4_execution_authorized=false`、`scientific_result_claimed=false`、`exchange_candidate_created=false`。

已接受原 G4 协议的历史 provenance 保持不变。passage-wise wav2vec2 preprocessing 修订与本 preflight 候选均已由协调独立复核并接受；候选报告仍保留候选生成时的机器状态。二者只解除正式执行管线候选的依赖，不得直接作为真实 G4 执行授权。

## 镜像与 preprocessor 语义

2203 直接无代理探测清华 TUNA 目标路径，HTTP 状态为 404；随后只使用单一回退 endpoint `https://hf-mirror.com`，HTTP 状态为 200，正文 159 bytes。机器核对字段为：

- `feature_size=1`；
- `sampling_rate=16000`；
- `padding_value=0.0`；
- `do_normalize=true`；
- `return_attention_mask=false`；
- `padding_side=right`。

缓存文件位于 `/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813/preprocessor_config.json`，159 bytes，时间戳 `2026-08-13T07:15:07.342413+00:00`。本次审计未改写缓存，也未持久化网络正文。首次回退请求返回 HTTP 403 的失败报告保留为 `/home/fanyu/auditory_simulation_m6a/logs/wav2vec2_preprocessor_mirror_audit_20260813.json`；有限重试成功报告为 `/home/fanyu/auditory_simulation_m6a/logs/wav2vec2_preprocessor_mirror_audit_20260813_v2.json`。

冻结算法按 passage 独立计算 float32 mean 与 population variance (`ddof=0`)，输出 `(x-mean)/sqrt(var+1e-7)`；常数和 non-finite 输入 fail closed，不使用训练统计量，不跨 passage/split 共享统计量。输入为 16 kHz mono、无 padding；由于 `return_attention_mask=false`，不创建也不传入 attention mask。自有实现与本地 `Wav2Vec2FeatureExtractor` 在 warm-up 与 longest synthetic input 上最大绝对差均为 0。

第三方镜像、mutable `main` 与 no-hash 政策不能提供密码学完整性或不可变 provenance。模型缓存状态保持 `SEMANTICALLY_VALIDATED_REMOTE_ONLY`，`download_allowed=false`。

## 只读空间盘点

| 类别 | bytes | 文件数 |
|---|---:|---:|
| data | 14,173,350,514 | 377 |
| cache | 380,271,415 | 4 |
| outputs | 280,055,356 | 16 |
| log + logs | 6,569,171,507 | 387 |
| code snapshot | 120,929,017 | 236 |
| 选定类别合计 | 21,523,777,809 | — |
| project root 合计 | 21,525,428,164 | 1,058 |
| 未分类差额 | 1,650,355 | — |
| 实际可用空间 | 978,024,435,712 | — |
| data + cache + 预计新增 20 GB | 34,553,621,929 | — |

实际 free bytes 大于 500 GB；data+cache+预计新增 20 GB 严格小于 500 GB。盘点只使用路径、字节数、文件数与时间戳。

## Synthetic longest-passage canary

最长 passage 由已接受轻量 manifest 按 `MAX_AUDIO_DURATION_SECONDS_THEN_SAMPLE_ID_LEXICOGRAPHIC` 选择：`sub-SD012_ses-02_task-PassiveListen__seg-028`，duration 77.08981859410432 s。输入长度严格为 `ceil(77.08981859410432 * 16000)=1,233,438` samples；输入为内存中确定性 finite mono waveform，没有真实文件路径。

- 模型：`facebook/wav2vec2-base@main`，`transformers 5.14.1`；
- 安全加载：`local_files_only=true`、`trust_remote_code=false`、`weights_only=true`、tensor-only、eval、inference mode、requires-grad 参数数 0；
- warm-up：1 s / 16,000 samples，0.19773226405959576 s；
- longest forward：batch 1，3,854 frames，projected + 12 Transformer layers 共 13 个 `[1,3854,768]` finite tensors；
- attention scope：只在该 synthetic passage 内全局，不使用 chunk/window 近似；
- wall time：0.07231187901925296 s；
- CUDA peak allocated/reserved：1,660,685,824 / 1,962,934,272 bytes；
- OOM：false。

第一次机器报告因 gate 错把有限负 post-normalization mean 判为 non-finite 而 `FAIL`，保留为 `reports/g4_resource_runtime_preflight_candidate_20260813.json`。修正 gate 后只重验既有证据，未重跑模型或重新读取真实数据；当前 v3 报告 15/15 required checks 为 true。

唯一当前 preflight 候选是 `reports/g4_resource_runtime_preflight_candidate_20260813_v3.json`。v2 已机器标记为 `SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE` 并指向 v3；无版本原始报告继续保持 `FAIL`，均未删除。首次 hf-mirror HTTP 403 只保留 provenance，不再是规范性 PASS 前提；正式门禁只依赖当前 TUNA 404、当前 hf-mirror 200、159-byte 严格语义、缓存未写与无代理证据。

## 运行上界与可恢复性

估算不假设 global-attention runtime 与音频长度线性。机器化保守上界为：每次进程模型加载 60 s、最长 forward 60 s、checkpoint/process overhead 30 s，故单 passage invocation 上界 150 s（0.041666666666666664 GPU h），40 passages 总上界 1.6666666666666665 GPU h。正式设计仍要求每次只处理一个 passage、成功后原子写 checkpoint、仅跳过已验证 final checkpoint、失败 partial 保留，并在单次 2 h 硬边界前停止。

## 验证与停止点

- 当前机器报告 15/15 required checks 为 true；
- 2203 专用环境完整验证为 `147 passed, 349 subtests`；Ruff 对完整 `src scripts tests` PASS；mypy 对完整 46 个文件 PASS；主 config gate PASS；协议修订 9/9、preflight 15/15 required checks 为 true；
- G3 raw-input representation 仅保留工程 shape/time 证据，状态为 `MUST_NOT_REUSE_FOR_G4_SCIENTIFIC_BASELINE`，本轮未重算 G3；
- 本 preflight 只支持“资源与 synthetic 运行 shape 候选可行”，不证明真实特征、神经 target、ridge、null、指标或科学结果；
- 当前停止于协调审核。不得执行 G4、扩大 recording/subject、建立 exchange candidate 或启用无范围 neural extraction。
