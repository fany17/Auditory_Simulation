> **Codex任务已整理为 [GitHub Issue #5](https://github.com/fany17/Auditory_Simulation/issues/5)**。从Issue #5进入，工作分支 `work/pub-01-s03-ss05-predictive-memory`。**任务书已就绪，尚未通过本聊天启动Codex进程。**

# Codex 启动入口 — Auditory_Simulation（2026-10-19最新）

**唯一当前执行任务：`PUB-01-S03_SS05`（跨窗口历史记忆与选择性信息更新），状态 `READY_FOR_CODEX`。** 研究者已授权 Codex 继续执行。**协调助手只更新/复核任务书，不运行 Python、不做实验、不代替 Codex 执行。**

## 立即读取（按顺序）

1. `doc/TASKS.md` → `doc/tasks/PUB-01/TASK.md` → `doc/tasks/PUB-01/SUBTASKS.md`：父任务与唯一状态源。
2. `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`：**唯一完整执行任务书及验收规则**。
3. `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md`：公开 EEG 连续历史/split/lag/跨记录 reset；仅作为实验契约，若与 SS05 完整任务书冲突，以后者为准。
4. `reports/pub_01/s03/predictive_memory/REPORT.md`、`SYNTHETIC_RESULTS.json` 以及对应 `scripts/pub_01_s03_ss05_*.py`：**以前由协调助手编写的原型/声称结果，仅作待独立审查资料，不能当作 Codex 已复现的事实**。
5. 冻结历史：SS03 `reports/pub_01/s03/architecture/REPORT.md`，SS02 `reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`；只读，绝不根据已打开的test结果调新模型。

## 直接行动（Codex自己完成）

- 在 Windows 上采用**独立worktree**进入 `work/pub-01-s03-ss05-predictive-memory` / PR #4；先检查工作树，保留本地主目录未提交的 CNN/TRF 和 stash。不要 checkout/reset/force-push/merge `main` 或 PR#3。
- **先独立审计现有源码及合成实验结果**，保留可复现与不可复现项；检查新状态模块与标准GRU、窗口因果性、状态跨记录 reset；不能以已有报告 PASS 代替自己的执行。
- 若 server2203 SSH 可访问，首先只读确认公开数据、subject/story/source_family 和可建立的**新内部分组**。既有 SS03 test_seen/test_heldout 已被打开，**不得再次用于SS05模型选择和确认性研究**。
- 再完成真实 EEG 连续窗口 loader、真实数据train/validation pilot；在触及新的预留 heldout 之前冻结 protocol、3+ seeds、预算与 lag。符合准入才一次评估。若不具备可用新test或服务器不可访问，分别标 `HOLD_NEW_TEST`/`HOLD_RUNTIME`；不要虚构结果。
- **由Codex提交并push** 自有代码、配置及轻量报告到同一SS05分支；新运行结果存server2203公开目录，Git不收原始EEG/音频/权重。每阶段更新 `reports/pub_01/s03/predictive_memory/codex_run01/EXECUTION_LOG.md`，合成与EEG结果严格分列。

## 停止条件

完成SS05限定验收后停止汇报；SS04、S04、父任务S05、GPT重排与患者实验不自动启动。`Auditory_Simulation` 只处理公开数据与通用表征（M6A）；`STN_Decoding_Encoding` 只处理患者实验与STN特异适配（M6B）；仅冻结版本化artifact可单向衔接。

---

## 历史入口（归档，不再作为执行授权）

# Codex 启动入口 — PUB-01 SparrKULee 强基线复现

**研究者明确的第一步是复现，不是设计新模型或追分。**

2026-10-15最终交付：SS02 **COMPLETED / PASS_WITH_LIMITATION**。HDPS76人100Hz/M作者split，Top1/5/10=12.03/30.45/42.10%；Transformer原fold4、70人训练/69seen/13heldout，2AFC=83.53/80.08%（加权82.38%），冻结encoder原Conv32直接包络r=.1688/.1403（加权.1593）。各自一次test及独立review通过；完整论文cohort/隐藏test/ensemble仍未复现。见`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。本轮停止，无自动下一READY节点，不启SS03/S04。

2026-10-13追加执行已完成原raw100Hz、70story Mel及独立论文功能模型batch256工程smoke；602条产物对应作者647对中的600对就绪，必要47条raw仍受限，正式训练/论文测试NOT_RUN。Transformer对应冻结EEGcheckpoint失效、官方测试载荷未齐。SS02为 **WAITING_EXTERNAL / HOLD**，缺口与恢复位置见`reports/pub_01/s03/strong_baselines/RESOURCE_RECOVERY.md`；已启动作业全部结束，无后台替代训练。用户只要同论文对照，不跑缩减cohort/64Hz适配分数交付，不转SS03/S04。

## 权威执行顺序

1. 当前研究者指令；
2. sibling `AuditoryReading/AGENTS.md`（可见时）与本仓库 `AGENTS.md`；
3. `doc/PROJECT_BOUNDARY.md`、`doc/TASK_EXECUTION_STANDARD.md`；
4. `doc/TASKS.md`、`doc/CURRENT_TASK.md`；
5. `doc/tasks/PUB-01/TASK.md`、`doc/tasks/PUB-01/SUBTASKS.md`；
6. `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`；
7. **当前SS03 ACTIVE执行节点** `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`；**SS02历史复现节点** `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`只读；
8. `doc/tasks/PUB-01/BENCHMARK_REGISTRY.md`；
9. 真实现存 S02/S03_SS01 数据/代码/config/reports。

## 具体指令

首先只读盘点当前 Windows Git 工作区（未提交修改、分支、已有资产），并只读检查 `server2203:/home/fanyu/auditory_simulation_m6a` 上现有公开 SparrKULee 数据、依赖与可复用的声音模型；**不覆盖、删除或自行修复任何既有产物**。

然后逐项目核对 HDPS、Transformer-CLIP-EEG、2023/2024 Challenge 的原始代码和协议；固定官方 protocol 与可运行条件，先做轻量 smoke，再执行 `S03_SS02` 允许的实际复现。遇到作者权重、非公开数据、许可、采样率/时间轴、GPU/内存/服务器连接不可用，留下可复现阻塞，不能通过换测试协议自称复现成功。

本次必须交付 `PROTOCOL_MATRIX.md`、`REPRO_REPORT.md`、`RESULTS.csv`、`REVIEW.md`，即使部分失败也要交。逐个结果标明 `OFFICIAL_PROTOCOL` / `INTERNAL_PROTOCOL`、实际 test split、候选数和是否含语言模型。若无法完成两套，给出各自的 `HOLD` 原因与继续所需条件；**不得自行启动新模型结构或 S04**。

Git 仅在获准 Windows 工作区维护代码/配置/轻量报告；服务器只作数据/训练/评估，不初始化另一个 git。完成后停止并报告。不得把执行日志写成已证明 ICML 新颖性或已达到 SOTA。
