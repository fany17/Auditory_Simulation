# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前任务是 `PUB-01 — Public Auditory–Neural Representation`。

本项目不负责患者 STN 实验、临床采集、DBS/PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## Current entry

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/DOCUMENT_INDEX.md`
4. `doc/TASK_EXECUTION_STANDARD.md`
5. `doc/TASKS.md`
6. `doc/CURRENT_TASK.md`
7. `doc/tasks/PUB-01_PUBLIC_REPRESENTATION.md`

## Current task — PUB-01

Title：`Public Auditory–Neural Representation`  
Track：`PUBLIC_METHOD`  
Status：`READY`

```text
SparrKULee EEG
→ public auditory-neural representation
→ public intracranial transfer
→ few-shot / negative-transfer evaluation
→ ART-AUDREP-v1 candidate
```

PUB-01 是 patient-side READ 的 optional accelerator/comparator，不是硬前置。positive/no/negative transfer 均为合法结果。

## Historical milestones

旧 `M6A-PUBLIC-001/002/003` 文件只保留 provenance，不再定义 current/future pipeline，也不继续创建新的 `M6A-*` task。

## Repository split

- `Auditory_Simulation`：`PUB-##` 公共方法；
- `AuditoryReading`：全项目 pipeline、证据、教材、综述和治理规范；
- `STN_Decoding_Encoding`：`READ-## / INT-## / WRITE-## / SEEG-## / SYS-##`。

公共项目与患者项目只通过冻结、版本化 artifact 衔接；患者数据不得进入本仓库。

## Numbering rules

- 每个 namespace 独立从 `01` 开始；
- 新编号不是旧 `M#` 的续号；
- 一个 task 只有一个 canonical ID；
- Title 单独表达语义；
- 不再混用 `R0/W0/S0`、`M6A/M6B` 或第二套纯语义 ID；
- `doc/TASKS.md` 是 current/future 状态 source of truth。
