# Auditory_Simulation Document Index

更新时间：`2026-10-13`

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

## B. Current executable task tree（2026-10-23）

最新入口：CODEX_START_HERE.md → tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md；仅任务书与五原型合成检查，真实实验未授权启动。代码src/m6a_public/continuity/，报告reports/pub_01/s03/continuity/prototype01/REPORT.md。旧SS02/03/05完成并冻结，PR#3/#4已合并。

### 历史任务树快照（2026-10-13；仅追溯）

- Parent：`tasks/PUB-01/TASK.md`；唯一子任务登记：`tasks/PUB-01/SUBTASKS.md`。
- 科学支线与依赖：`tasks/PUB-01/RESEARCH_BRANCHES.md`。
- 竞品/协议核对：`tasks/PUB-01/BENCHMARK_REGISTRY.md`。
- **Codex 当前执行入口**：`CODEX_START_HERE.md` → `tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`（READY）。
- A 线：S03 / SparrKULee EEG direct decoding；SS01历史 completed，SS02 READY，SS03/SS04 BACKLOG。
- B 线：S04 / ds004703 intracranial baseline and transfer；SS01/SS02 BACKLOG。
- S01/S02 completed，S05 artifact integration BACKLOG。

此索引只指路，实际执行状态始终按 `TASKS.md` / `CURRENT_TASK.md` / `tasks/PUB-01/SUBTASKS.md`。

## C. Templates / execution governance

- `TASK_TEMPLATE.md`：父任务模板；
- `SUBTASK_TEMPLATE.md`：subtask 模板；
- `TASK_EXECUTION_STANDARD.md`：本地执行标准；
- Program 全局标准：`AuditoryReading/AGENTS.md`。

## D. Historical milestone task specs

已按用户要求归档，保留原文和 provenance，不自动执行：

- `archive/pre_pub_20260919/tasks/M6A-PUBLIC-001.md` — historical preliminary public alignment；
- `archive/pre_pub_20260919/tasks/M6A-PUBLIC-002.md` — completed pretrained baseline；
- `archive/pre_pub_20260919/tasks/M6A-PUBLIC-003.md` — review-only temporal perturbation evidence。

归档入口与旧路径映射：[archive/README.md](archive/README.md)。

旧 `M6A-*` 名称不再是 current/future 命名来源，也不继续编号。

## E. Background / interface / reports

旧总纲、`PROJECT_CHARTER.md`、M6A candidate/rework/exchange 草案均位于 `archive/pre_pub_20260919/`；reports/experiment packages 仍在原报告目录。以上材料只用于背景或 provenance，不决定 current task/subtask status。

只有 PUB-01 完成 integration review 并生成新的 `ART-AUDREP-v1` candidate 后，才进入患者侧 `INT-01` consumer validation。

## F. Code / config / schema assets

- `configs/`：配置；
- `schemas/`：artifact/manifest schema；
- `src/`、`scripts/`：代码；
- `tests/`：验证；
- `environment/`：环境；
- 大型模型、数据、cache 和训练输出留在计算环境，不进入 Git。
- 所有数据下载、解压、读取核验与计算统一在 `server2203`，具体边界见 `../README.md` 和 `../environment/README.md`。
- `../test/README.md`：历史材料/演示与自动测试的区分。
- `../reports/README.md`：报告、历史交付包与 PUB 结果的导航。

## G. New-document rule

- 新 parent task → `doc/tasks/<TASK-ID>/TASK.md`；
- parent subtask registry → `doc/tasks/<TASK-ID>/SUBTASKS.md`；
- 新 subtask → `doc/tasks/<TASK-ID>/subtasks/<TASK-ID>-S##_<TITLE>.md`；
- Task ID 使用统一 `PUB-##`；subtask 只用 `<TASK-ID>-S##`；
- 已有节点内追加工作使用 S##_SS##；
- report 内 future work 不自动创建 task/subtask；
- 活文档由 Git 历史版本化。

## 追加工作与节点层级（2026-09-24 用户规则）

- S## 表示父任务下的主节点；归属于已有节点的追加工作必须登记为该节点的 SS##，例如 PUB-01-S01_SS01，不得仅因追加时间较晚占用新的主节点编号。
- SS## 在所属 S## 内从 01 独立编号；先查已有 owner，再决定新增。小修改留在现有完成记录；重复冻结流程仍使用 run，不创建 SS##。
- 主任务登记仍集中在父任务 SUBTASKS.md：主节点表登记 S##，其下按 S## 分区登记 SS##，包含 Parent Node、依赖、状态、任务书和验收。不得创建第二份竞争登记表。
- 子节点任务书路径：subtasks/<TASK-ID>-S##/<TASK-ID>-S##_SS##_<TITLE>.md；主节点任务书仍在 subtasks/<TASK-ID>-S##_<TITLE>.md。复用 SUBTASK_TEMPLATE，补 Parent Node 字段。
- SS## 继承所属节点的范围、权限、验收边界；完成 SS## 不等于完成 S##。依赖可表达为“节点前置准入 → SS## → 节点最终验收”，不得写成互相等待的循环。
- 只有超出已有节点职责的独立主阶段才分配新的 S##；不得因整理文档自动新增研究任务。更深层工作先写步骤，不自动递归建任务。
- 编号迁移必须记录旧 ID、新 ID、任务书路径和原有产物位置。已发布旧编号不复用为另一任务；当前登记移除旧节点，历史报告/脚本/服务器路径保留别名说明，不批量搬动数据。
