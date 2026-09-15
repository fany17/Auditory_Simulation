# Auditory_Simulation Document Index

更新时间：`2026-09-15`

## A. Authoritative current documents

读取顺序：

1. `../README.md`
2. `../AGENTS.md`
3. Program `AuditoryReading/AGENTS.md`（可见时）
4. `PROJECT_BOUNDARY.md`
5. `DOCUMENT_INDEX.md`
6. `TASK_EXECUTION_STANDARD.md`
7. `TASKS.md`
8. `CURRENT_TASK.md`
9. parent `tasks/<TASK-ID>/TASK.md`
10. parent `tasks/<TASK-ID>/SUBTASKS.md`
11. 指定 `tasks/<TASK-ID>/subtasks/<SUBTASK-ID>_<TITLE>.md`

`TASKS.md` 是父任务状态 source of truth；每个父任务的 `SUBTASKS.md` 是其 subtask 状态 source of truth。

## B. Current executable task tree

```text
tasks/PUB-01/
  TASK.md
  SUBTASKS.md
  subtasks/
    PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md
    PUB-01-S02_FEATURE_BANK_BASELINE.md
    PUB-01-S03_PUBLIC_EEG_BENCHMARK.md
    PUB-01-S04_INTRACRANIAL_TRANSFER.md
    PUB-01-S05_ARTIFACT_INTEGRATION.md
```

当前 first executable subtask：`PUB-01-S01`。

## C. Templates / execution governance

- `TASK_TEMPLATE.md`：父任务模板；
- `SUBTASK_TEMPLATE.md`：subtask 模板；
- `TASK_EXECUTION_STANDARD.md`：本地执行标准；
- Program 全局标准：`AuditoryReading/AGENTS.md`。

## D. Historical milestone task specs

保留 provenance，不自动执行：

- `tasks/M6A-PUBLIC-001.md` — historical preliminary public alignment；
- `tasks/M6A-PUBLIC-002.md` — completed pretrained baseline；
- `tasks/M6A-PUBLIC-003.md` — review-only temporal perturbation evidence。

旧 `M6A-*` 名称不再是 current/future 命名来源，也不继续编号。

## E. Background / interface / reports

`01_听觉时变信息处理项目总纲.md`、`PROJECT_CHARTER.md`、historical exchange drafts、reports/experiment packages 均用于背景或 provenance，不决定 current task/subtask status。

只有 PUB-01 完成 integration review 并生成新的 `ART-AUDREP-v1` candidate 后，才进入患者侧 `INT-01` consumer validation。

## F. Code / config / schema assets

- `configs/`：配置；
- `schemas/`：artifact/manifest schema；
- `src/`、`scripts/`：代码；
- `tests/`：验证；
- `environment/`：环境；
- 大型模型、数据、cache 和训练输出留在计算环境，不进入 Git。

## G. New-document rule

- 新 parent task → `doc/tasks/<TASK-ID>/TASK.md`；
- parent subtask registry → `doc/tasks/<TASK-ID>/SUBTASKS.md`；
- 新 subtask → `doc/tasks/<TASK-ID>/subtasks/<TASK-ID>-S##_<TITLE>.md`；
- Task ID 使用统一 `PUB-##`；subtask 只用 `<TASK-ID>-S##`；
- 禁止 sub-subtask；
- report 内 future work 不自动创建 task/subtask；
- 活文档由 Git 历史版本化。
