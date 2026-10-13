# Codex 启动入口 — PUB-01 SparrKULee 强基线复现

**研究者明确的第一步是复现，不是设计新模型或追分。**

## 权威执行顺序

1. 当前研究者指令；
2. sibling `AuditoryReading/AGENTS.md`（可见时）与本仓库 `AGENTS.md`；
3. `doc/PROJECT_BOUNDARY.md`、`doc/TASK_EXECUTION_STANDARD.md`；
4. `doc/TASKS.md`、`doc/CURRENT_TASK.md`；
5. `doc/tasks/PUB-01/TASK.md`、`doc/tasks/PUB-01/SUBTASKS.md`；
6. `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`；
7. **唯一当前可执行节点** `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`；
8. `doc/tasks/PUB-01/BENCHMARK_REGISTRY.md`；
9. 真实现存 S02/S03_SS01 数据/代码/config/reports。

## 具体指令

首先只读盘点当前 Windows Git 工作区（未提交修改、分支、已有资产），并只读检查 `server2203:/home/fanyu/auditory_simulation_m6a` 上现有公开 SparrKULee 数据、依赖与可复用的声音模型；**不覆盖、删除或自行修复任何既有产物**。

然后逐项目核对 HDPS、Transformer-CLIP-EEG、2023/2024 Challenge 的原始代码和协议；固定官方 protocol 与可运行条件，先做轻量 smoke，再执行 `S03_SS02` 允许的实际复现。遇到作者权重、非公开数据、许可、采样率/时间轴、GPU/内存/服务器连接不可用，留下可复现阻塞，不能通过换测试协议自称复现成功。

本次必须交付 `PROTOCOL_MATRIX.md`、`REPRO_REPORT.md`、`RESULTS.csv`、`REVIEW.md`，即使部分失败也要交。逐个结果标明 `OFFICIAL_PROTOCOL` / `INTERNAL_PROTOCOL`、实际 test split、候选数和是否含语言模型。若无法完成两套，给出各自的 `HOLD` 原因与继续所需条件；**不得自行启动新模型结构或 S04**。

Git 仅在获准 Windows 工作区维护代码/配置/轻量报告；服务器只作数据/训练/评估，不初始化另一个 git。完成后停止并报告。不得把执行日志写成已证明 ICML 新颖性或已达到 SOTA。
