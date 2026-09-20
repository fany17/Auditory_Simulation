# SUBTASK — PUB-01-S02 — Feature Bank and Leak-safe Baseline

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S02` |
| Parent Task | `PUB-01` |
| Title | `Feature Bank and Leak-safe Baseline` |
| Status | `BACKLOG` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `PUB-01-S01` + `PUB-01-S06` |
| Inputs | accepted dataset manifest/split keys, approved frozen audio models |
| Deliverables | interpretable feature bank, frozen split manifest, ridge/TRF baseline |
| Acceptance | features reproducible, split leak-safe, baseline runs without test-driven selection |
| Validation | feature canary + split audit + baseline reproduction |
| Go/Hold/Fail | PASS unlocks S03; leakage or unreproducible features → REWORK/HOLD |
| Files Allowed to Change | public configs/src/scripts/reports + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-09-15` |

## 1. Objective
建立可解释声学/时间特征、冻结 SSL layer bank 和 leak-safe baseline。

## 2. Scope
至少包含 envelope/onset/log-mel or cochleagram、适用的 F0/spectral-flux/timing variables、wav2vec2/HuBERT layerwise frozen features，以及 ridge/TRF baseline。

## 3. Inputs / Preconditions
S01 对相应 dataset 为 PASS；S06 下载与初步清洗已验收；group split keys 已冻结。

## 4. Execution Steps
1. 冻结 feature extraction spec/version。
2. 生成可重建 acoustic/timing features。
3. 提取批准的 frozen SSL layers。
4. 固化 train/val/test grouped split。
5. 运行 ridge/TRF baseline 与 null。
6. 输出 canary/shape/timing/lag checks。

## 5. Controls / Failure Modes
所有 normalization/PCA/layer choice 只在 train fold；禁止相邻时间泄漏和 test-driven layer selection。

## 6. Deliverables
feature manifest、split manifest、baseline metrics/null、canary、known-failure report。

## 7. Acceptance Criteria
特征可复现；时间轴可解释；split 审计通过；baseline 与 null 完整；失败 layer/recording 保留。

## 8. Validation Procedure
Technical: canary/shape/timing/split tests。Acceptance: leak-safe baseline checklist。Parent consistency: 不增加未批准 backbone，不接触患者数据。

## 9. Completion Record
`NOT_STARTED`
