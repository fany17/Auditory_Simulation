# Auditory_Simulation Parent Task Registry

更新时间：`2026-10-19`；父任务唯一状态源。子节点请查 `doc/tasks/PUB-01/SUBTASKS.md`；路线图 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`。

| Task ID | Title | Track | Class | Status | Blocking for STN direct READ | Patient Data | Parent Task Spec | Subtask Registry | Output |
|---|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | **`ACTIVE`** | NO | FORBIDDEN | `doc/tasks/PUB-01/TASK.md` | `doc/tasks/PUB-01/SUBTASKS.md` | public EEG benchmark + public sEEG benchmark/transfer + eventual `ART-AUDREP-v1` candidate |

## 最新执行决策（2026-10-19）

- `PUB-01` 仍 `ACTIVE`。研究者明确分工：**协调者制定/审阅任务，Codex负责具体实现和全部实验执行**。未经研究者要求协调者不自行运行模拟或EEG任务。
- 当前唯一 `READY_FOR_CODEX` 子节点为 **`PUB-01-S03_SS05`**：研究跨窗口历史记忆、选择性关联读写和预测误差更新。此前由协调者添加的代码/合成实验仅作待Codex独立复现的原型记录；**未完成2203真实公开EEG训练/测试，不声称SS05实验成功**。
- SS01/SS02/SS03冻结，不能用已打开的SS03 test为SS05改架构。SS04、公开颅内S04、公共artifact S05仍BACKLOG。当前工作分支 `work/pub-01-s03-ss05-predictive-memory` / PR#4 Draft，`main`与旧CNN/TRF工作目录保留。
- 运行/交付与患者数据禁止边界详见 `doc/CODEX_START_HERE.md`、`doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`。

## Current scientific branches

2026-10-19：研究者授权独立 `PUB-01-S03_SS05` 历史记忆结构探索，当前只完成合成实验与10项本地测试；2203公开EEG正式评估尚未执行。SS03既有结果依旧冻结不变，SS04/S04/S05保持BACKLOG；新任务完整边界见子任务登记与 `reports/pub_01/s03/predictive_memory/`。


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
