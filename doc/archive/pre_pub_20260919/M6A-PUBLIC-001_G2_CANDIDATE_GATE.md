# M6A-PUBLIC-001 G2 candidate 提交门禁

| 字段 | 内容 |
|---|---|
| 状态 | `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE` |
| 当前是否为 G2 candidate | `YES` |
| 完整性策略 | `NON_HASH_AUDIT` |
| 后续模型/神经/baseline | `NOT_AUTHORIZED_BEFORE_LATER_GATES` |

只有下列条件全部成立，才允许把 G2 标记为 candidate 并提交协调审核。

## 1. 官方 inventory 精确闭环

- inventory 取得方式：`scripts/public_s3_inventory.py` 调用 OpenNeuro public S3 `ListObjectsV2`；
- source：`https://s3.amazonaws.com/openneuro.org`；
- 取得时间：`2026-08-11T11:14:08.907738+00:00`；
- expected：377 paths、14,173,350,514 bytes；
- 必须同时满足：final path 377/377、total bytes 完全一致、active partial=0、missing/unexpected/byte_mismatch 全为空；
- 证据字段仅为相对路径、字节数、时间戳、数量与可读性，不使用哈希或校验和。

原始取得记录：`reports/ds004703_s3_inventory_summary.json` 与 `reports/ds004703_s3_inventory.csv`。

## 2. 11/11 recording 头与时间轴

每个 recording 必须：

- EDF header 可读；
- EDF sampling rate 与 iEEG sidecar 一致；
- events 最大 `onset + duration` 不越 EDF 时间轴；
- channels.tsv 中 analysis-eligible SEEG/ECOG 名称全部存在于 EDF；
- EDF 与 channels.tsv 总通道数差异只作 warning，但必须列出 TSV-only/EDF-only 名称、type、status，不得只报告总数。

## 3. 必要 metadata 与边界

- 11 sidecars、11 channels TSV、11 events TSV、11 audio-offset JSON 全部可读；
- dataset DOI/version、CC0 声明和 README 更严格的非商业/禁止再识别边界闭环；
- PyBIDS validated layout 可建立；
- 319 行 preliminary primary split guard 仍为 PASS；
- Catalan 继续排除，当前 generalization 仍只限 within-subject unseen-stimulus/block。

## 4. Candidate 输出

G2 candidate 只包含轻量报告、inventory、recording metadata、split/guard 与失败边界，不包含原始数据。单一机器入口为 `scripts/g2_promotion_gate.py`；它只组合固定版本的 dataset audit、neural metadata audit、split guard 与主 config，不重复读取数据。缺输入、陈旧状态、非有限数、字段缺失或任一 required check 非 true 均 fail closed。该历史机器入口输出为 `G2_CANDIDATE_AWAITING_COORDINATOR_REVIEW`，从未自称 G2 PASS。协调者于 2026-08-13 独立核对后给出 `G2_CANDIDATE_REVIEW=ACCEPT`；当前推进状态因此原位更新为 `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE`，但这不表示整条 M6A PASS/FROZEN。

## 5. G2 之后仍独立阻塞的门禁

- `ANATOMY_MAPPING_NOT_READY`：不阻塞电极级 method smoke，但 `region_summary=NOT_ESTIMABLE`；
- 逐 recording `iEEGReference` 必须记录并在 11 个 sidecar 中一致为 `scalp electrode, not included with data`；首轮只能使用 `AS_RECORDED_SCALP_REFERENCE`；
- neural target 方法已冻结为 `METHOD_FROZEN_AWAITING_EXECUTION_GATES`，但 `neural_extraction_allowed=false`；这不等于 G3、整条 M6A 或 exchange contract 已冻结；
- G2 协调验收后，audio cross-split context、2.0 s final embargo 与 baseline-final split guard 也已于 2026-08-13 独立获协调接受；该后续接受不改变本文件的 G2 provenance，也不等于 G3/M6A/scientific PASS；
- revised exchange DRAFT 虽已接受用于 candidate 准备，但真实 method/runtime/canary 尚无，consumer cross-test 仍 `NOT_RUN`。

## 6. 当前候选证据

- Range 重试：仅补齐 2 个缺失区间，复用 350 个已有 chunk，失败 0；三个 EDF 均以官方字节数装配，staging 与 dataset active partial 均为 0；
- dataset audit：377/377 paths、14,173,350,514 bytes，missing/unexpected/byte mismatch 均为 0，11/11 EDF header 可读；
- neural metadata audit：11/11 sidecar/channels/events/audio-offset、sampling/timeline/eligible channel、reference 与 PyBIDS 门禁通过；analysis-eligible=1346，C-prefix exclusion=727；
- split guard：319 行，train/validation/test=223/48/48；block 01/02/05/06=train、03=validation、04=test；English=319、Catalan=0；
- promotion gate：全部 required checks 为 true，`g2_pass_claimed=false`，`candidate_contains_raw_data=false`；首次过度约束 C-prefix 唯一性的失败证据保留于 `reports/ds004703_g2_promotion_gate_failed_channel_identity_v1.json`；
- 2203 专用环境：91 tests + 133 subtests、Ruff、mypy（29 source files）、主 config gate、method gate 与 formal-src direct-convolution scan 全部通过。
