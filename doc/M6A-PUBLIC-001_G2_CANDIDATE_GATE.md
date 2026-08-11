# M6A-PUBLIC-001 G2 candidate 提交门禁

| 字段 | 内容 |
|---|---|
| 状态 | `PENDING_3_EDF_DOWNLOADS` |
| 当前是否为 G2 candidate | `NO` |
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

G2 candidate 只包含轻量报告、inventory、recording metadata、split/guard 与失败边界，不包含原始数据。状态只能是 `CANDIDATE_AWAITING_COORDINATOR_REVIEW`，不能自称 G2 PASS。

## 5. G2 之后仍独立阻塞的门禁

- `ANATOMY_MAPPING_NOT_READY`：不阻塞电极级 method smoke，但 `region_summary=NOT_ESTIMABLE`；
- `NEURAL_TARGET_REDESIGN_REQUIRED_BEFORE_G3`：target/filter 未协调冻结，`neural_extraction_allowed=false`；
- final embargo 的 filter edge 与 wav2vec2 context 未实测，`baseline_final=false`；
- revised exchange DRAFT 虽已接受用于 candidate 准备，但真实 method/runtime/canary 尚无，consumer cross-test 仍 `NOT_RUN`。
