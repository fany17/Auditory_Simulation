# PUB-01 Subtask Registry

Parent task: `PUB-01 — Public Auditory–Neural Representation`

| Subtask ID | Title | Status | Blocking for Parent | Upstream | Task Spec | Validation |
|---|---|---|---|---|---|---|
| `PUB-01-S01` | Dataset preparation and initial exploration | COMPLETED | YES | NONE | `subtasks/PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md` | 用户阶段验收PASS_WITH_LIMITATION；剩余清洗/分组问题随具体分析处理 |
| `PUB-01-S02` | Feature Bank and Leak-safe Baseline | COMPLETED | YES | S01阶段完成 | `subtasks/PUB-01-S02_FEATURE_BANK_BASELINE.md` | 2026-10-11核验a063及2203既有交付：83 subjects/659 records，acoustic/log-mel/frozen wav2vec2 encoding PASS_WITH_LIMITATION；HuBERT HOLD |
| `PUB-01-S03` | Public EEG Representation Benchmark | ACTIVE | YES | S02已交公开分支 | `subtasks/PUB-01-S03_PUBLIC_EEG_BENCHMARK.md` | SS01初步包络解码已COMPLETED / PASS_WITH_LIMITATION；完整S03 neural/retrieval/held-out-subject未完成 |
| `PUB-01-S04` | Public Intracranial Transfer | BACKLOG | YES | S03 | `subtasks/PUB-01-S04_INTRACRANIAL_TRANSFER.md` | same |
| `PUB-01-S05` | Artifact Candidate and Integration Review | BACKLOG | YES | S01–S04 | `subtasks/PUB-01-S05_ARTIFACT_INTEGRATION.md` | same + parent integration review |


## S01 内部子节点

| Child ID | Parent Node | Title | Status | Upstream | Task Spec | Acceptance |
|---|---|---|---|---|---|---|
| PUB-01-S01_SS01 | PUB-01-S01 | Server Dataset Download and Initial Cleaning | COMPLETED | S01按数据集通过许可/访问门 | subtasks/PUB-01-S01/PUB-01-S01_SS01_DOWNLOAD_INITIAL_CLEANING.md | 2026-09-20 PASS_WITH_LIMITATION；下载、格式QC与虚拟筛选验收通过 |

| PUB-01-S01_SS02 | PUB-01-S01 | 数据探索与问题定位 | COMPLETED | SS01 | subtasks/PUB-01-S01/PUB-01-S01_SS02_BENCHMARK_ADMISSION.md | 已交实际信号图和概览；用户接受阶段完成，未知项不阻止后续分析 |

执行依赖：S01许可/访问准入 → S01_SS01数据准备 → S01_SS02配对/时序与实验准入收尾 → S01最终验收 → S02 → S03 → S04 → S05。
S01及SS01/SS02已阶段完成；S02已有交付经2026-10-11只读核验。S03_SS01本轮已完成并停止；S03整体ACTIVE，当前无运行中的本轮计算；S04/S05保持BACKLOG。

## S03 内部子节点

| Child ID | Parent Node | Title | Status | Upstream | Task Spec | Acceptance |
|---|---|---|---|---|---|---|
| PUB-01-S03_SS01 | PUB-01-S03 | 初步EEG→声音包络解码 | COMPLETED | S02阶段已交付的公开EEG与冻结split | subtasks/PUB-01-S03/PUB-01-S03_SS01_ENVELOPE_DECODING.md | 2026-10-11 PASS_WITH_LIMITATION：10名93条，TRF test r=0.1819/R²=0.0178，null r=−0.0033；同支持与rerun/独立review PASS，完整S03未完成 |

## 编号迁移（2026-09-24）

原 PUB-01-S06 误将 S01 内部追加工作登记成主节点，现纠正为 PUB-01-S01_SS01。旧任务书仅保留迁移链接，S06退役，不复用；当前没有需要规划的新S06。未来已有节点内追加工作使用各自SS##，仅真正独立的新主阶段另行分配未占用S##。

历史 reports/pub_01/s06/、服务器 pub_01/s06/、pub_01_s06_* 脚本与清单中的旧ID保留，均映射到S01_SS01；不改写历史数据或重跑验收。最终证据见 reports/pub_01/COORDINATOR_REVIEW.md。

## Rules

- 本表统一登记PUB-01的S##及SS##，状态以本表为准。
- 同一执行链只运行一个明确的当前节点；依赖按前置准入与最终验收分开描述。
- 小修改写已有完成记录，冻结流程重复执行使用run。
- 子节点继承父节点权限和科学边界；不得因阴性结果自动扩展任务。
2026-09-24执行修订：SS02探索性绘图在SS01已具备数据后直接开始；上述S01最终验收依赖仅约束正式benchmark，不阻止当前描述性分析。

当前状态以本表和最新用户阶段验收为准，早期REVIEW/HOLD说明保留为历史。
