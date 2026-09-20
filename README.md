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

## Current task tree — PUB-01

```text
PUB-01  Public Auditory–Neural Representation
│
├─ PUB-01-S01  Dataset / License / Timing Audit          REVIEW
├─ PUB-01-S06  Server Dataset Download / Initial Cleaning COMPLETED
├─ PUB-01-S02  Feature Bank and Leak-safe Baseline       BACKLOG
├─ PUB-01-S03  Public EEG Representation Benchmark       BACKLOG
├─ PUB-01-S04  Public Intracranial Transfer              BACKLOG
└─ PUB-01-S05  Artifact Candidate and Integration Review BACKLOG
```

2026-09-20，`PUB-01-S06` 获准公开范围下载与初步清洗已通过独立验收（PASS_WITH_LIMITATION）。SparrKULee公开4,142对象已齐，ds004703复用377对象并完成盘点；受限对象及坏源文件明确列出。见 [最终验收](reports/pub_01/COORDINATOR_REVIEW.md)。编号不重排，依赖仍为 S01 → S06 → S02 → S03 → S04 → S05；完整S01模型准入仍需审阅。

本次数据下载、初步清洗和文档归档目标已完成，所有载荷与处理留在2203。初步清洗使用已实际验证的虚拟索引/公共区间视图，不代表新增滤波信号或模型准入通过；S02仍为BACKLOG，未启动训练。

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

旧 `M6A-PUBLIC-001/002/003` 任务书、项目总纲、charter 和候选草案已归档到 [doc/archive/](doc/archive/README.md)，只保留 provenance，不再定义 current/future pipeline，也不继续创建新的 `M6A-*` task。

## Repository split

- `Auditory_Simulation`：`PUB-##` 公共方法；
- `AuditoryReading`：全项目 pipeline、Program `AGENTS.md`、证据、教材和治理规范；
- `STN_Decoding_Encoding`：`READ-## / INT-## / WRITE-## / SEEG-## / SYS-##`。

公共项目与患者项目只通过冻结、版本化 artifact 衔接；患者数据不得进入本仓库。
