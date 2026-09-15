# AGENTS.md — Auditory_Simulation Local Agent Rules

| Field | Value |
|---|---|
| Status | `ACTIVE` |
| Effective date | `2026-09-15` |
| Applies to | `Auditory_Simulation/` |
| Program governance | `AuditoryReading/AGENTS.md` |

本文件只补充 `Auditory_Simulation` 的本地执行规则。全项目 task/subtask 编号、目录、创建、校验、完成、amendment 和 handoff 规则以 `AuditoryReading/AGENTS.md` 为权威。

若当前 workspace 无法读取 sibling `AuditoryReading`，必须至少遵守本文件、`doc/TASK_EXECUTION_STANDARD.md`、`doc/TASK_TEMPLATE.md`、`doc/SUBTASK_TEMPLATE.md`；不得凭旧 M6A 文档恢复历史执行逻辑。

## 1. Public-project hard boundary

本仓库只负责：

- `PUB-##` 公共数据/公共方法任务；
- 公开声音模型；
- 公开 EEG/intracranial neural data；
- 通用 auditory-neural representation；
- transfer/few-shot benchmark；
- 冻结 public artifact。

严格禁止患者/STN raw/intermediate/derivative、患者派生 embedding、私有临床 metadata、使用患者结果反向选择 public model/layer/hyperparameter，以及患者 PINS/DBS/SEEG 刺激分析。

大型数据、模型权重、cache、训练输出留在批准的计算环境；Git 只保存轻量代码、配置、结构化结果、task/subtask 记录和报告。

## 2. Fixed read order

每次执行按顺序读取：

1. 用户当前明确指令；
2. Program `AuditoryReading/AGENTS.md`（可见时）；
3. 本 `AGENTS.md`；
4. `doc/PROJECT_BOUNDARY.md`；
5. `doc/DOCUMENT_INDEX.md`；
6. `doc/TASK_EXECUTION_STANDARD.md`；
7. `doc/TASKS.md`；
8. `doc/CURRENT_TASK.md`；
9. 父任务 `doc/tasks/<TASK-ID>/TASK.md`；
10. `doc/tasks/<TASK-ID>/SUBTASKS.md`；
11. 指定 subtask；
12. spec 列出的数据/config/artifact/report。

## 3. Current/future task layout

```text
doc/tasks/<TASK-ID>/
  TASK.md
  SUBTASKS.md
  subtasks/
    <TASK-ID>-S01_<TITLE>.md
```

历史 `M6A-PUBLIC-001/002/003` 原文件保留 provenance，不自动迁移或执行。

## 4. Subtask rule

- 父任务定义 question/scope/acceptance；subtask 不得改变。
- `PUB-01-S##` 只表示 `PUB-01` 内部执行单元。
- 一次 execution 只执行一个 primary subtask。
- 新 detail 若有独立 deliverable/acceptance/blocker，才新建 S##。
- 普通小修改直接记录在当前 subtask Completion Record。
- 禁止 sub-subtask。
- 新科学问题必须申请新 `PUB-##`，不能藏在 S## 中。

## 5. Subtask pre-execution gate

开始前必须确认：ID/filename/`SUBTASKS.md` 一致；parent 正确；patient data 仍 `FORBIDDEN`；upstream 满足；license/data access 合规；deliverable/acceptance/validation 已写；split/leakage 规则冻结；允许修改文件范围明确。

任何一项失败，不进入 `ACTIVE`。

## 6. Validation

subtask 完成必须经过：

1. Technical validation：输出/schema/test/reproduction；
2. Acceptance validation：`PASS / PASS_WITH_LIMITATION / REWORK / HOLD / NO_GO`；
3. Parent consistency：确认不改变 PUB parent scope、claim 和 data boundary。

Executor 不能只凭“跑通”自行完成；至少做一次独立 review pass。

父 `PUB-##` 完成还必须独立做 Integration Review。

## 7. Negative evidence

positive / no transfer / negative transfer 均是合法结果。

禁止因为阴性自动改 split、换窗口、加新 backbone、改 negative sampling，或新建“再试一个”的 S##。新增分支必须回到 parent scope 与 acceptance 判断。

## 8. Cross-repo handoff

公共→患者只通过冻结、版本化 artifact，例如 `ART-AUDREP-v1`。

必须记录 source task/subtask/commit/config、schema、runtime/transform、canary、benchmark、known failures、license/provenance。

患者 consumer 不静默修改 producer artifact；失败返回 rework decision。

## 9. Git / destructive operations

- 不擅自切换/重置/强制推送/改写历史；
- 不覆盖用户修改或 frozen evidence；
- 删除 current 临时文件前确认新 canonical 路径已建立且可读；
- 历史 M6A provenance 不为整洁而批量改名；
- 若仓库既有规则禁止 checksum/hash 审计，继续遵守，不主动新增此类完整性流程。

## 10. Stop rule

完成一个指定 subtask 后：更新其 Completion Record、parent `SUBTASKS.md`，必要时更新 parent `TASK.md` / owner `TASKS.md` / `CURRENT_TASK.md`，然后停止。

不得自动启动 sibling subtask、下一个 parent task 或患者侧任务。
