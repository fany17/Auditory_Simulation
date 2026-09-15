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
7. `tasks/PUB-01_PUBLIC_REPRESENTATION.md`

`TASKS.md` 是 current/future task 状态 source of truth。

## B. Current executable task spec

- `tasks/PUB-01_PUBLIC_REPRESENTATION.md` — `PUB-01`，Public Auditory–Neural Representation；`READY`。

## C. Historical milestone task specs

保留 provenance，不自动执行：

- `tasks/M6A-PUBLIC-001.md` — historical preliminary public alignment；
- `tasks/M6A-PUBLIC-002.md` — completed pretrained baseline；
- `tasks/M6A-PUBLIC-003.md` — review-only temporal perturbation evidence。

旧 `M6A-*` 名称不再是 current/future 命名来源，也不继续编号。

## D. Background / interface / reports

`01_听觉时变信息处理项目总纲.md`、`PROJECT_CHARTER.md`、historical exchange drafts、reports/experiment packages 均用于背景或 provenance，不决定 current task status。

只有 PUB-01 生成新的 `ART-AUDREP-v1` candidate 后，才进入患者侧 `INT-01` consumer validation。

## E. Code / config / schema assets

- `configs/`：配置；
- `schemas/`：artifact/manifest schema；
- `src/`、`scripts/`：代码；
- `tests/`：验证；
- `environment/`：环境；
- 大型模型、数据、cache 和训练输出留在计算环境，不进入 Git。

## F. New-document rule

- 新 task spec → `doc/tasks/<TASK-ID>_<SEMANTIC-SLUG>.md`；
- Task ID 使用统一 `PUB-##` 编号，不再使用 M6A 或第二套 semantic ID；
- 新 policy → `doc/`；
- 新结果 → `reports/`；
- report 内 future work 不自动创建 task；
- 活文档由 Git 历史版本化。
