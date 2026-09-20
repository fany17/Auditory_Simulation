# Auditory_Simulation Task Registry

执行规范：`doc/TASK_EXECUTION_STANDARD.md`  
文档索引：`doc/DOCUMENT_INDEX.md`  
Pipeline：`AuditoryReading/RESEARCH_PIPELINE.md`

`TASKS.md` 是本仓库**父任务**状态 source of truth。subtask 状态由各父任务自己的 `SUBTASKS.md` 管理。

## Current / future tasks

| Task ID | Title | Track | Class | Status | Blocking | Patient Data | Parent Task Spec | Subtask Registry | Primary Output |
|---|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | ACTIVE | NO for direct STN READ | FORBIDDEN | `doc/tasks/PUB-01/TASK.md` | `doc/tasks/PUB-01/SUBTASKS.md` | `ART-AUDREP-v1` candidate + public benchmark |

## Historical provenance

| Historical ID | Status | Role |
|---|---|---|
| `M6A-PUBLIC-001` | HISTORICAL_REFERENCE | preliminary public alignment checkpoint |
| `M6A-PUBLIC-002` | COMPLETED | frozen pretrained model/reference bank |
| `M6A-PUBLIC-003` | REVIEW | temporal-architecture perturbation evidence; review/close only |

这些历史 ID 不再作为 current/future task 命名来源。

## PUB-01 current subtask entry

执行 `PUB-01` 时必须先读：

1. `doc/tasks/PUB-01/TASK.md`
2. `doc/tasks/PUB-01/SUBTASKS.md`
3. 其中最早满足依赖的 READY subtask

2026-09-20：`PUB-01-S06 — Server Dataset Download and Initial Cleaning` 已 COMPLETED（获准公开范围，PASS_WITH_LIMITATION），本次数据准备goal已验收。当前无ACTIVE执行子任务；S01完整模型准入仍REVIEW，S02–S05保持BACKLOG。详见 `reports/pub_01/COORDINATOR_REVIEW.md`；父任务PUB-01整体尚未完成。

不允许跳过 S01 直接进入模型训练。

## Hard boundary

- 患者/STN 数据不得进入本仓库；
- 不读取患者派生 embedding；
- 不使用 STN 结果选择 public model/layer；
- 不继续堆 pretrained model 名单；
- downstream handoff 必须是冻结、版本化 artifact。
