# Auditory_Simulation Parent Task Registry

更新时间：`2026-10-15`；父任务唯一状态源。子节点请查 `doc/tasks/PUB-01/SUBTASKS.md`；路线图 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`。

| Task ID | Title | Track | Class | Status | Blocking for STN direct READ | Patient Data | Parent Task Spec | Subtask Registry | Output |
|---|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | **`ACTIVE`** | NO | FORBIDDEN | `doc/tasks/PUB-01/TASK.md` | `doc/tasks/PUB-01/SUBTASKS.md` | public EEG benchmark + public sEEG benchmark/transfer + eventual `ART-AUDREP-v1` candidate |

## Current scientific branches

2026-10-15最新：S03_SS02 **COMPLETED / PASS_WITH_LIMITATION**。用户授权可用公开子集下，HDPS76人Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%、EEG-only包络加权r=.1593；三运行实际test及独立review完成。原完整论文条件仍未复现。停止SS02，不自动开始新架构或其他支线；PUB-01仍ACTIVE。

- **A / `PUB-01-S03` SparrKULee EEG auditory direct decoding**：SS01/SS02完成有限制；强基线实际结果、误差/对照/协议边界见reports/pub_01/s03/strong_baselines/REPRO_REPORT.md。后续架构须研究者指定；SS03/SS04保持BACKLOG。10-13完整资源HOLD作为历史保留于任务书，不覆盖10-15真实可用条件结果。
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
