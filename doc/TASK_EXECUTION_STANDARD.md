# Auditory_Simulation Task Execution Standard

版本：`parent-subtask-v3.0 / 2026-09-15`  
Program governance：`AuditoryReading/AGENTS.md`  
Program task standard：`AuditoryReading/TASK_EXECUTION_STANDARD.md`

本文件只补充公共执行仓库规则；与 Program 规则冲突时，以数据边界 `PROJECT_BOUNDARY.md` 和 Program `AGENTS.md` 为准。

## 1. Parent task identity

Current/future public parent tasks 只使用 `PUB-##`。

目录：

```text
doc/tasks/<TASK-ID>/
  TASK.md
  SUBTASKS.md
  subtasks/
    <TASK-ID>-S01_<TITLE>.md
```

历史 `M6A-PUBLIC-001..003` 按 2026-09-19 用户指令归档至 `archive/pre_pub_20260919/tasks/`，保留原文，不进入当前 PUB 任务目录结构。

## 2. Subtask identity

父任务内只使用 `<TASK-ID>-S##`，如 `PUB-01-S01`。

- 每个 parent 的 S## 独立从 01 开始；
- 编号不复用；
- 已有节点内追加工作使用 S##_SS##；
- 独立 deliverable/acceptance/blocker 才建 S##；
- 小实现细节写入当前 subtask Completion Record。

## 3. Source of truth

- parent status：`doc/TASKS.md`；
- current execution entry：`doc/CURRENT_TASK.md`；
- parent spec：`doc/tasks/<TASK-ID>/TASK.md`；
- subtask status：`doc/tasks/<TASK-ID>/SUBTASKS.md`；
- subtask spec：`doc/tasks/<TASK-ID>/subtasks/...`。

## 4. Required templates

- parent：`doc/TASK_TEMPLATE.md`；
- subtask：`doc/SUBTASK_TEMPLATE.md`。

## 5. Public hard boundaries

患者/STN data、patient embedding、private clinical metadata 永远 `FORBIDDEN`。任何 S## 不得放宽。

public model/layer/hyperparameter 只能依据 public data；患者结果不得反向影响 PUB artifact。

所有数据集下载、续传、解压、读取核验、预处理、派生数据生成，以及权重下载、特征提取、训练/评估和缓存均在 `server2203` 完成。本地只保留轻量代码、配置、清单、汇总结果与报告，不下载或回传数据载荷/权重/特征张量。远端不可用时记录阻塞，不回退本地执行。先核验现有 `/home/fanyu/auditory_simulation_m6a` 资产，避免重复下载或覆盖历史结果。

## 6. Pre-execution gate

subtask ACTIVE 前必须确认：ID/registry/filename 一致、parent 正确、upstream 满足、license/data access 合规、acceptance/validation 明确、split/leakage 规则冻结、允许修改范围明确。

## 7. Validation

每个 S## 必须经过：

1. Technical validation；
2. Acceptance validation：`PASS / PASS_WITH_LIMITATION / REWORK / HOLD / NO_GO`；
3. Parent consistency validation。

父任务完成还需要 Integration Review。

## 8. Negative evidence

positive/no/negative transfer 均可验收。不得因阴性自动新增模型/窗口/split/negative-sampling S##。

## 9. Cross-repo artifact

公共→患者只通过冻结、版本化 artifact，如 `ART-AUDREP-v1`；记录 source task/subtask/commit/config、schema/runtime/canary、benchmark、known failures、license/provenance。

## 10. Stop rule

完成当前 S## 后更新 Completion Record 和 `SUBTASKS.md`，必要时更新 parent/TASKS/CURRENT_TASK，然后停止；不自动启动 sibling 或 downstream。

## 追加工作与节点层级（2026-09-24 用户规则）

- S## 表示父任务下的主节点；归属于已有节点的追加工作必须登记为该节点的 SS##，例如 PUB-01-S01_SS01，不得仅因追加时间较晚占用新的主节点编号。
- SS## 在所属 S## 内从 01 独立编号；先查已有 owner，再决定新增。小修改留在现有完成记录；重复冻结流程仍使用 run，不创建 SS##。
- 主任务登记仍集中在父任务 SUBTASKS.md：主节点表登记 S##，其下按 S## 分区登记 SS##，包含 Parent Node、依赖、状态、任务书和验收。不得创建第二份竞争登记表。
- 子节点任务书路径：subtasks/<TASK-ID>-S##/<TASK-ID>-S##_SS##_<TITLE>.md；主节点任务书仍在 subtasks/<TASK-ID>-S##_<TITLE>.md。复用 SUBTASK_TEMPLATE，补 Parent Node 字段。
- SS## 继承所属节点的范围、权限、验收边界；完成 SS## 不等于完成 S##。依赖可表达为“节点前置准入 → SS## → 节点最终验收”，不得写成互相等待的循环。
- 只有超出已有节点职责的独立主阶段才分配新的 S##；不得因整理文档自动新增研究任务。更深层工作先写步骤，不自动递归建任务。
- 编号迁移必须记录旧 ID、新 ID、任务书路径和原有产物位置。已发布旧编号不复用为另一任务；当前登记移除旧节点，历史报告/脚本/服务器路径保留别名说明，不批量搬动数据。
