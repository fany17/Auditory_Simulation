# Auditory_Simulation Parent Task Registry

更新时间：`2026-10-13`；父任务唯一状态源。子节点请查 `doc/tasks/PUB-01/SUBTASKS.md`；路线图 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`。

| Task ID | Title | Track | Class | Status | Blocking for STN direct READ | Patient Data | Parent Task Spec | Subtask Registry | Output |
|---|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | **`ACTIVE`** | NO | FORBIDDEN | `doc/tasks/PUB-01/TASK.md` | `doc/tasks/PUB-01/SUBTASKS.md` | public EEG benchmark + public sEEG benchmark/transfer + eventual `ART-AUDREP-v1` candidate |

## Current scientific branches

- **A / `PUB-01-S03` SparrKULee EEG auditory direct decoding**：已完成 S03_SS01 初步 envelope/TRF；下一执行 `PUB-01-S03_SS02 READY`，Codex 强基线复现；后续由研究者决定新模型架构，再验证 direct reconstruction 和 2AFC/5AFC/100+ 检索。
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
