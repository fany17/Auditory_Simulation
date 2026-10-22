# Auditory_Simulation Parent Task Registry

更新时间：`2026-10-19`；父任务唯一状态源。子节点请查 `doc/tasks/PUB-01/SUBTASKS.md`；路线图 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`。

| Task ID | Title | Track | Class | Status | Blocking for STN direct READ | Patient Data | Parent Task Spec | Subtask Registry | Output |
|---|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | **`ACTIVE`** | NO | FORBIDDEN | `doc/tasks/PUB-01/TASK.md` | `doc/tasks/PUB-01/SUBTASKS.md` | public EEG benchmark + public sEEG benchmark/transfer + eventual `ART-AUDREP-v1` candidate |

## 最新执行决策（2026-10-23）

PUB-01仍ACTIVE。SS06获得用户明确的“任务书+基本框架+跑通”授权；仅合成工程验证，不加载真实数据、不训练。五框架文档/实现/审阅交付后停止，未来实验须另行明确启动。SS02/SS03/SS05冻结，PR#3/#4已合并，main为60c3404；S03/SS04/S04/父S05状态不被本原型工程PASS替代。当前入口见CODEX_START_HERE.md及SUBTASKS.md。

## 历史执行决策（2026-10-19）

- `PUB-01` 仍 `ACTIVE`。研究者明确分工：**协调者制定/审阅任务，Codex负责具体实现和全部实验执行**。未经研究者要求协调者不自行运行模拟或EEG任务。
- SS05已 **COMPLETED / PASS_WITH_LIMITATION**：Codex完成原合成审查/15tests、真实pilot、新train norm、15run训练与105条件测试。新内部holdout21人/47record/7family的五组r约.091–.092，六预设区间均跨0，未检出复杂记忆额外收益；同agent separate review和轻量文件复核范围已说明。有效阴性限定本轮数据/目标/模型/2000步，不代表历史记忆普遍无用。
- SS01/SS02/SS03冻结，旧test不供SS05调参。用户最新授权已合并PR#3到main@8d21d0a；PR#4 base main且仍Draft。SS04/S04/父S05 BACKLOG；主目录CNN/TRF/stash不动，不自行合并#4。
- 运行/交付与患者数据禁止边界详见 `doc/CODEX_START_HERE.md`、`doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`。

## Current scientific branches

2026-10-19：SS05真实内部holdout阴性结果与报告已交，当前停止。见reports/pub_01/s03/predictive_memory/codex_run01/REPORT.md；SS01/SS02/SS03科学结果冻结，SS04/S04/父S05 BACKLOG，无自动下一READY节点。PR#3按新授权已合并8d21d0a，PR#4仍Draft不自行合并。


2026-10-19：SS03 **COMPLETED / PASS_WITH_LIMITATION**，真实公开fold4四组×三seed各25000步，EEG-only直接包络multiscale seen/heldout r=.10305/.11380（加权.10663），同最大RF long .06411/.06979，middle .09297/.10413。预设配对差值family95%区间均>0，限当前四原型/数据/预算，不证明生理时标或超过SS02。报告/同agent separate review在`reports/pub_01/s03/architecture/`；PR#3 Draft，当前停止。SS04/S04/S05 BACKLOG，未冻结artifact。


2026-10-15最新：S03_SS02 **COMPLETED / PASS_WITH_LIMITATION**。用户授权可用公开子集下，HDPS76人Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%、EEG-only包络加权r=.1593；三运行实际test及独立review完成。原完整论文条件仍未复现。SS02历史冻结，SS03已获授权进行原型与下一步实验；PUB-01仍ACTIVE。

- **A / `PUB-01-S03` SparrKULee EEG auditory direct decoding**：SS01/SS02/SS03完成有限制；SS03结果与强基线参考边界见reports/pub_01/s03/architecture/REPORT.md。S03仍ACTIVE，SS04保持BACKLOG，等待研究者指定；不因当前阳性自动扩大模型或任务。
- **B / `PUB-01-S04` ds004703 public intracranial**：BACKLOG，先独立 baseline（不等 A 线），再 EEG→sEEG 公共数据迁移（才依赖 A 线冻模型）。
- S01 数据准备/S02 正向声学特征→EEG encoding：COMPLETED / PASS_WITH_LIMITATION；S05 整合：BACKLOG。

下一步 Codex 入口：`doc/CODEX_START_HERE.md`。没有任何新模型训练被本次文档更新自动授权或执行。

## Historical provenance

| Historical ID | Status | Role |
|---|---|---|
| `M6A-PUBLIC-001` | HISTORICAL_REFERENCE | preliminary public alignment checkpoint |
| `M6A-PUBLIC-002` | COMPLETED | frozen pretrained model/reference bank |
| `M6A-PUBLIC-003` | REVIEW | temporal-architecture perturbation evidence; review/close only |

历史编号不再作为新任务标识。公开数据和模型计算保留在 `server2203`；患者原始信号/派生 embedding 绝不进入本仓库。公共→患者只有冻结且版本化的 artifact，转移无收益不阻塞直接 STN READ。
