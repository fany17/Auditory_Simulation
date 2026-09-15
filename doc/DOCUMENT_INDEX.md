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
7. `tasks/PUBLIC-REPRESENTATION.md`

`TASKS.md` 是 current/future task 状态 source of truth。

## B. Current executable task specs

- `tasks/PUBLIC-REPRESENTATION.md` — public auditory-neural representation；`READY`。

## C. Historical milestone task specs

保留 provenance，不自动执行：

- `tasks/M6A-PUBLIC-001.md` — historical preliminary public alignment；
- `tasks/M6A-PUBLIC-002.md` — completed pretrained baseline；
- `tasks/M6A-PUBLIC-003.md` — completed execution / review-only evidence。

这些旧 `M6A-*` 名称不再是 current/future pipeline 命名来源，不再新增 `M6A-PUBLIC-004/005/...`。

## D. Project background / long-form design

以下用于理解历史设计，不决定当前执行优先级：

- `01_听觉时变信息处理项目总纲.md`
- `PROJECT_CHARTER.md`
- historical reports / experiment packages

冲突时，以 current `PROJECT_BOUNDARY.md`、`TASKS.md` 和全局 `AuditoryReading/RESEARCH_PIPELINE.md` 为准。

## E. Historical/interface drafts

包括旧 public→patient contract / candidate / redesign drafts。

它们记录历史接口设计，不等于当前 accepted artifact contract，也不允许凭文件名恢复旧 milestone pipeline。

只有 `PUBLIC-REPRESENTATION` 生成新的 `ART-AUDREP-v1` candidate 后，才进入当前 consumer validation。

## F. Reports

`reports/` 存放执行结果、结构化指标、图与审计报告。

规则：

- report 不等于 task；
- report 内 future work 不自动创建 task；
- negative results 必须保留；
- 历史 perturbation 结果不能因为新任务而覆盖或删除。

## G. Code / config / schema assets

- `configs/`：配置；
- `schemas/`：artifact/manifest schema；
- `src/`、`scripts/`：代码；
- `tests/`：验证；
- `environment/`：环境；
- 计算服务器：大型模型、数据、cache 和训练输出，不进入 Git。

## H. New-document rule

今后：

- 新 task spec → `doc/tasks/<SEMANTIC-TASK-ID>.md`；
- 新 policy → `doc/`；
- 新结果 → `reports/`；
- live task ID 禁止 `M#`、顺序号、日期、状态词；
- 不把 task、report、draft contract 混成一个新文件名；
- 活文档由 Git 历史版本化。