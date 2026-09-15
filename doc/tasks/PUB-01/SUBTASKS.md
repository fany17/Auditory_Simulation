# PUB-01 Subtask Registry

Parent task: `PUB-01 — Public Auditory–Neural Representation`

| Subtask ID | Title | Status | Blocking for Parent | Upstream | Task Spec | Validation |
|---|---|---|---|---|---|---|
| `PUB-01-S01` | Dataset / License / Timing Audit | READY | YES | NONE | `subtasks/PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md` | technical + acceptance + parent consistency |
| `PUB-01-S02` | Feature Bank and Leak-safe Baseline | BACKLOG | YES | S01 | `subtasks/PUB-01-S02_FEATURE_BANK_BASELINE.md` | same |
| `PUB-01-S03` | Public EEG Representation Benchmark | BACKLOG | YES | S02 | `subtasks/PUB-01-S03_PUBLIC_EEG_BENCHMARK.md` | same |
| `PUB-01-S04` | Public Intracranial Transfer | BACKLOG | YES | S03 | `subtasks/PUB-01-S04_INTRACRANIAL_TRANSFER.md` | same |
| `PUB-01-S05` | Artifact Candidate and Integration Review | BACKLOG | YES | S01–S04 | `subtasks/PUB-01-S05_ARTIFACT_INTEGRATION.md` | same + parent integration review |

## Rules

- 本表是 `PUB-01` 内部 subtask 状态唯一真值。
- 一次只执行一个 primary subtask。
- 新的独立 work package 使用下一个未使用编号 `PUB-01-S06` 起。
- 不因 negative result 自动新增 S##；新增必须符合 parent scope。
- 已分配编号即使 CANCELLED 也不复用。
