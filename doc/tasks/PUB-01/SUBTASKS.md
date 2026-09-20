# PUB-01 Subtask Registry

Parent task: `PUB-01 — Public Auditory–Neural Representation`

| Subtask ID | Title | Status | Blocking for Parent | Upstream | Task Spec | Validation |
|---|---|---|---|---|---|---|
| `PUB-01-S01` | Dataset / License / Timing Audit | REVIEW | YES | NONE | `subtasks/PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md` | partial license/access gate reviewed; full pairing/timing acceptance pending |
| `PUB-01-S02` | Feature Bank and Leak-safe Baseline | BACKLOG | YES | S01 + S06 | `subtasks/PUB-01-S02_FEATURE_BANK_BASELINE.md` | same |
| `PUB-01-S03` | Public EEG Representation Benchmark | BACKLOG | YES | S02 | `subtasks/PUB-01-S03_PUBLIC_EEG_BENCHMARK.md` | same |
| `PUB-01-S04` | Public Intracranial Transfer | BACKLOG | YES | S03 | `subtasks/PUB-01-S04_INTRACRANIAL_TRANSFER.md` | same |
| `PUB-01-S05` | Artifact Candidate and Integration Review | BACKLOG | YES | S01–S04 | `subtasks/PUB-01-S05_ARTIFACT_INTEGRATION.md` | same + parent integration review |
| `PUB-01-S06` | Server Dataset Download and Initial Cleaning | COMPLETED | YES | S01 license/identity gate per dataset; full audit before benchmark use | `subtasks/PUB-01-S06_DOWNLOAD_INITIAL_CLEANING.md` | 2026-09-20 independent acceptance PASS_WITH_LIMITATION for authorized public scope; full inventory/readability + virtual cleaning verified; access/source/scientific exclusions retained |

2026-09-19 当前用户 goal：协调者持续跟进 S01 审计与 S06 数据准备至实际产物可验收。S06 READY 表示可先做服务器空间/来源盘点；具体下载必须先通过相应数据集许可与访问核验。一次只执行一个 primary subtask，切换时记录交接。

2026-09-20：上述数据准备goal已通过最终独立验收，见 `reports/pub_01/COORDINATOR_REVIEW.md`。当前无ACTIVE执行子任务；S01模型准入仍REVIEW，S02–S05保持BACKLOG。

2026-09-20 S06执行交付REVIEW：SparrKULee4142公开对象/135967030049 bytes、660 BDF全解码、617公开header核验；196受限未请求、1坏源排除。ds377对象完成格式/静态范围盘点，319虚拟片段；666 published EEG虚拟视图验证通过。最终文件`reports/pub_01/s06/`，26 tests与最终结构化验证通过。S01 benchmark admission仍HOLD，S02保持BACKLOG。DONE由协调者最终独立验收决定。

## Rules

- 本表是 `PUB-01` 内部 subtask 状态唯一真值。
- 一次只执行一个 primary subtask。
- 新的独立 work package 使用下一个未使用编号 `PUB-01-S07` 起。
- 不因 negative result 自动新增 S##；新增必须符合 parent scope。
- 已分配编号即使 CANCELLED 也不复用。
