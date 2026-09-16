# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前父任务是 `PUB-01 — Public Auditory–Neural Representation`。

本项目不负责患者 STN 实验、临床采集、DBS/PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## Current entry

1. Program `AuditoryReading/AGENTS.md`（可见时）
2. `AGENTS.md`
3. `doc/PROJECT_BOUNDARY.md`
4. `doc/DOCUMENT_INDEX.md`
5. `doc/TASK_EXECUTION_STANDARD.md`
6. `doc/TASKS.md`
7. `doc/CURRENT_TASK.md`
8. `doc/tasks/PUB-01/TASK.md`
9. `doc/tasks/PUB-01/SUBTASKS.md`
10. 当前指定 subtask

## Current task tree — PUB-01

```text
PUB-01  Public Auditory–Neural Representation
│
├─ PUB-01-S01  Dataset / License / Timing Audit          READY
├─ PUB-01-S02  Feature Bank and Leak-safe Baseline       BACKLOG
├─ PUB-01-S03  Public EEG Representation Benchmark       BACKLOG
├─ PUB-01-S04  Public Intracranial Transfer              BACKLOG
└─ PUB-01-S05  Artifact Candidate and Integration Review BACKLOG
```

当前只执行 `PUB-01-S01`。subtask 只有在上游和三级验证满足后才能进入 READY/ACTIVE。

PUB-01 是 patient-side READ 的 optional accelerator/comparator，不是硬前置。positive/no/negative transfer 均为合法结果。

## Task / subtask governance

父任务定义 scientific question、scope、Acceptance 和 Go/No-go；subtask 只拆分可独立验收的执行 work package。

新增具体细节时：

- 小实现细节 → 当前 subtask Completion Record；
- 独立 work package → 新增 `<TASK-ID>-S##`；
- 改父任务 question/endpoint/permission/Acceptance → parent Amendment Record；
- 超出 parent scope → 新建新的 `PUB-##` 主任务。

subtask 必须经过 technical / acceptance / parent-consistency 三级验证。

## Historical milestones

旧 `M6A-PUBLIC-001/002/003` 文件只保留 provenance，不再定义 current/future pipeline，也不继续创建新的 `M6A-*` task。

## Repository split

- `Auditory_Simulation`：`PUB-##` 公共方法；
- `AuditoryReading`：全项目 pipeline、Program `AGENTS.md`、证据、教材和治理规范；
- `STN_Decoding_Encoding`：`READ-## / INT-## / WRITE-## / SEEG-## / SYS-##`。

公共项目与患者项目只通过冻结、版本化 artifact 衔接；患者数据不得进入本仓库。
