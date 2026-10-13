# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前父任务是 `PUB-01 — Public Auditory–Neural Representation`。

本项目不负责患者 STN 实验、临床采集、DBS/PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## 数据与计算位置：统一使用 server2203

**所有数据集的下载、断点续传、解压、读取核验、预处理及派生数据生成，都在 2203 服务器（SSH 别名 `server2203`）完成。不得先下载到本地电脑再上传。** 模型权重下载、特征提取、训练、评估及相关缓存也统一留在 2203。

- 现有服务器项目根：`/home/fanyu/auditory_simulation_m6a`；先盘点已有数据、权重和环境，再按任务建立 PUB 子目录，避免重复下载或覆盖历史结果。
- Windows 本地仓库只保存轻量代码、配置、数据来源/许可/路径清单、结构化汇总结果、任务记录和报告；不存放数据集载荷、音频/神经信号、权重、特征张量或下载缓存。可在本地查阅官方文档和许可网页。
- 从服务器回传仅限上述轻量产物。服务器连接、磁盘或访问权限不满足时记录阻塞，不回退到本地下载或处理数据。
- 代码按需复制到服务器；本地仍为唯一 Git 工作区，不在服务器初始化第二个 Git 仓库。环境说明见 [environment/README.md](environment/README.md)。

## Repository navigation

| 路径 | 用途 |
|---|---|
| `doc/tasks/PUB-01/` | 当前父任务、子任务与验收记录 |
| `doc/` | 项目边界、任务治理及历史背景；入口见 [文档索引](doc/DOCUMENT_INDEX.md) |
| `src/`、`scripts/`、`configs/`、`schemas/` | 实现、执行脚本、配置与数据契约 |
| `tests/` | 自动测试；由 `pyproject.toml` 指定为 pytest 测试目录 |
| `test/` | 历史研究尝试、材料与演示包，见 [用途说明](test/README.md) |
| `reports/` | 轻量结果与报告，见 [报告导航](reports/README.md) |
| `environment/` | 2203 环境定义和版本记录，不存放权重或数据 |

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

## Current task tree — PUB-01（2026-10-13）

当前两条支线及依赖详见 [RESEARCH_BRANCHES.md](doc/tasks/PUB-01/RESEARCH_BRANCHES.md)，状态以 [SUBTASKS.md](doc/tasks/PUB-01/SUBTASKS.md) 为准。

```text
PUB-01  Public Auditory–Neural Representation                    ACTIVE
├─ S01  数据准备/质量审计/探索                                    COMPLETED
├─ S02  声音特征→EEG encoding baseline                           COMPLETED
├─ S03  A线：SparrKULee EEG听觉直接解码                           ACTIVE
│  ├─ SS01  EEG→声音包络线性TRF                                   COMPLETED
│  ├─ SS02  Codex强基线复现                                      READY ← NEXT
│  ├─ SS03  架构假设与直接解码                                    BACKLOG
│  └─ SS04  直接重建 + 2AFC/5AFC/百余候选检索                    BACKLOG
├─ S04  B线：ds004703公开颅内独立基线与后续transfer              BACKLOG
│  ├─ SS01  独立sEEG→声音baseline                                BACKLOG
│  └─ SS02  EEG→sEEG的少数据迁移                                 BACKLOG
└─ S05  公开artifact整合                                        BACKLOG
```

**Codex 复现第一入口：[doc/CODEX_START_HERE.md](doc/CODEX_START_HERE.md)**。目标是先复现 HDPS 与 Transformer-CLIP-EEG 等强基线，明确不同候选规模/数据协议的实际成绩；不因 60%/80% 冲刺目标改动测试难度。**新模型必须在不输入候选语音的条件下直接预测连续声音表示**；候选识别只是同一输出的下游评价。此安排仅更新任务文档，未执行新实验。

ds004703 的独立模型测试可以与 EEG 主线并列推进，但目前保留 BACKLOG；只有 EEG→公开颅内迁移需要来自 S03 的冻结输出。患者/STN 数据不进入公共项目。

## Task / subtask governance

父任务定义 scientific question、scope、Acceptance 和 Go/No-go；subtask 只拆分可独立验收的执行 work package。

新增具体细节时：

- 小实现细节 → 当前 subtask Completion Record；
- 独立 work package → 新增 `<TASK-ID>-S##`；
- 改父任务 question/endpoint/permission/Acceptance → parent Amendment Record；
- 超出 parent scope → 新建新的 `PUB-##` 主任务。

subtask 必须经过 technical / acceptance / parent-consistency 三级验证。

## Historical milestones

旧 `M6A-PUBLIC-001/002/003` 任务书、项目总纲、charter 和候选草案已归档到 [doc/archive/](doc/archive/README.md)，只保留 provenance，不再定义 current/future pipeline，也不继续创建新的 `M6A-*` task。

## Repository split

- `Auditory_Simulation`：`PUB-##` 公共方法；
- `AuditoryReading`：全项目 pipeline、Program `AGENTS.md`、证据、教材和治理规范；
- `STN_Decoding_Encoding`：`READ-## / INT-## / WRITE-## / SEEG-## / SYS-##`。

公共项目与患者项目只通过冻结、版本化 artifact 衔接；患者数据不得进入本仓库。
