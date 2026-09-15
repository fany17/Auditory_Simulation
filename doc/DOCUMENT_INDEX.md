# Auditory_Simulation Document Index

更新时间：`2026-09-15`

## A. Authoritative current documents

读取顺序：

1. `../README.md`
2. `PROJECT_BOUNDARY.md`
3. `DOCUMENT_INDEX.md`
4. `TASK_EXECUTION_STANDARD.md`
5. `TASKS.md`
6. `CURRENT_TASK.md`
7. 当前 `tasks/<TASK-ID>.md`

`TASKS.md` 是 Local Task 状态 source of truth。

## B. Current executable task specs

- `tasks/M6A-PUBLIC-004.md` — Program `R1`，公共 auditory-neural representation；当前 `READY`。

## C. Historical task specs

保留 provenance，不自动执行：

- `tasks/M6A-PUBLIC-001.md` — historical preliminary public alignment；
- `tasks/M6A-PUBLIC-002.md` — completed pretrained baseline；
- `tasks/M6A-PUBLIC-003.md` — completed execution / current review node。

其状态以 `TASKS.md` 为准，而不是原 task 文件中的旧状态文字。

## D. Project background / long-form design

以下用于理解历史设计，不决定当前执行优先级：

- `01_听觉时变信息处理项目总纲.md`
- `PROJECT_CHARTER.md`
- `PROJECT_BOUNDARY.md`（其中边界规则仍 authoritative）

如果这些长文与当前 Program 结构冲突，以：

1. `PROJECT_BOUNDARY.md` 的数据/仓库边界；
2. `TASKS.md` 的 task 状态；
3. `AuditoryReading/PROGRAM_RESEARCH_GRAPH.md` 的 Program dependency

为准。

## E. Historical/interface drafts

包括：

- `M6A-PUBLIC-001_G2_CANDIDATE_GATE.md`
- `M6A-PUBLIC-001_NEURAL_TARGET_METHOD_FREEZE_CANDIDATE.md`
- `M6A-PUBLIC-001_NEURAL_TARGET_REDESIGN_DRAFT.md`
- `M6A_TO_M6B_EXCHANGE_CONTRACT_DRAFT.md`

这些记录 M6A/M6B 接口历史，不等于当前 accepted artifact contract。

只有 `M6A-PUBLIC-004` 真正生成 candidate 后，才按当前 schema/consumer 规则进入新一轮 cross-test。

## F. Reports

`reports/` 存放执行结果、结构化指标、图与审计报告。

规则：

- report 不等于 task；
- report 内的 future work 不会自动创建 task；
- negative results 必须保留；
- `M6A-PUBLIC-003` 结果不能因为进入 004 而覆盖或删除。

## G. Code/config/schema assets

- `configs/`：任务配置；
- `schemas/`：artifact/manifest schema；
- `src/`、`scripts/`：执行代码；
- `tests/`：验证；
- `environment/`：环境；
- 2203：大型模型、数据、cache 和训练输出，不进入 Git。

## H. New-document rule

今后：

- 新 task spec → `doc/tasks/<TASK-ID>.md`；
- 新 project-wide policy → `doc/`；
- 新结果 → `reports/`；
- 不再把 task、report、draft contract 混在一个新文件名中；
- 不建立日期后缀的“总图 v2/v3”，活文档由 Git 历史版本化。