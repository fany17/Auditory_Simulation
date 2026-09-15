# SUBTASK — PUB-01-S03 — Public EEG Representation Benchmark

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | `Public EEG Representation Benchmark` |
| Status | `BACKLOG` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `PUB-01-S02` |
| Inputs | frozen features/splits/baselines from S02 |
| Deliverables | held-out EEG encoding/retrieval benchmarks + neural encoder comparison |
| Acceptance | within-subject and held-out-subject evidence reported with predefined nulls and failures |
| Validation | rerun selected folds + leakage audit + metric consistency |
| Go/Hold/Fail | complete regardless of effect sign; output feeds S04/S05 |
| Files Allowed to Change | public model/config/report paths + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-09-15` |

## 1. Objective
在公开 EEG 上建立可审计 auditory-neural representation benchmark，而不是只追求最高准确率。

## 2. Scope
至少比较 interpretable baseline、simple temporal neural model、channel-agnostic neural encoder；alignment 同时保留 encoding 与 contrastive/retrieval 视角。

## 3. Inputs / Preconditions
S02 PASS；split、feature、lag policy 已冻结。

## 4. Execution Steps
1. 固化 model conditions 和 seeds。
2. 运行 within-subject held-out clip/story。
3. 运行 held-out subject/session。
4. 运行 encoding/null 和 contrastive/retrieval/null。
5. 汇总 representation family/layer 稳定性与失败分布。
6. 输出 data-efficiency 基线，为 S04 transfer 做参照。

## 5. Controls / Failure Modes
禁止 test leakage、recording-ID shortcut、只报告 pooled result、只保留最佳 seed。

## 6. Deliverables
benchmark manifest、metrics tables、retrieval/encoding results、subject distribution、negative/failure table、review report。

## 7. Acceptance Criteria
所有预定义 split/metric/null 完成；held-out subject 结果完整；失败/阴性不删；结论限定在 public EEG。

## 8. Validation Procedure
Technical: rerun subset + metric/unit checks。Acceptance: predefined conditions completeness。Parent consistency: 不把 EEG 成功外推成 intracranial/STN。

## 9. Completion Record
`NOT_STARTED`
