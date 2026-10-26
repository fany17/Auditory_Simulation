# 当前执行：SS06 run02 强基线规模匹配（2026-10-26）

用户已授权“起码训练规模一致，或者之前框架在我们的规模下完成”。SS06重新ACTIVE，唯一run为eeg_run02_matched_baseline，执行SS06_MATCHED_BASELINE_EXECUTION.md；先C/O/P最小矩阵、真实工程pilot，协调核对共同协议/预算后正式训练/统一评估。此授权覆盖旧STOP，不重复申请启动许可。

同SS06 worktree/分支/PR#6 Draft，所有计算仅2203；只用run01固定249/63/41和新train统计，禁旧SS02 EEG/对比/head权重。audit已看，新增结果POST-HOC INTERNAL EXPLORATORY。run01全部冻结不改，不重跑39模型或自动扩五框架预算。进展见continuity/eeg_run02_matched_baseline/EXECUTION_LOG.md。旧CNN/TRF/stash/工作树不动，无下载/患者/远端Git；不合并#6或启下游。

---

# 历史交付：SS06 run01 COMPLETED / PASS_WITH_LIMITATION（2026-10-23）

39次完整record监督训练（各3262步）、3次正式SSL（各3511段）及一次前瞻INTERNAL/EXPLORATORY审计完成。M2最高点估计r=.128164，但未稳健胜GRU；八primary只有M1multi−matched long、M3finetune−frozen、M4adaptive−uniformfixed区间为正。预训练相对从头GRU、M5额外收益和内容特异历史收益未建立，不宣称外部确认/时延机制/SOTA。

入口：reports/pub_01/s03/continuity/eeg_run01/REPORT.md、REVIEW.md、FORMAL_PROTOCOL.json、243行RESULTS、四图及逐seed/逐record/逐人证据。后置汇总误差0，222个原型CPU相关max差1.55e-7；协调独立SUMMARY重算max差5.55e-17。完整与精简交付包12项检查均PASS。执行/冻结source commit ea5a1b5，最终报告commit见PR#6。

**当前STOP，无自动下一READY。** SS06完成不等于S03/PUB完成；SS04/S04/父S05/GPT/artifact不启动。#6仍Draft，不自行合并。旧SS02/03/05、主目录CNN/TRF/stash和其他worktree不改；载荷/权重只在2203。下方启动指令是已执行历史，不重新运行。

---

# 历史启动：SS06 真实公开 EEG（2026-10-23）

用户最新明确“启动吧”。**PUB-01-S03_SS06 ACTIVE / REAL_EEG_EXECUTION**。此前仅原型/禁止训练/完成后停止仅描述prototype01历史；当前可以读取既有合法公开EEG，完成真实loader、train/val pilot、协议冻结、训练及合格评估，无需重复询问启动许可。

首先读取 `doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_REAL_EEG_EXECUTION.md` 与 SS06 五框架任务书。执行会话继续使用 `codex/pub-01-s03-ss06-continuity` / PR #6 独立worktree；模型仍按既有指定6.1-sol/high。代码/轻量报告在Windows；数据和训练只在2203新eeg_run01目录。

只使用旧公开资产，禁止SS03/SS05已看test调参；可从原SS05 train按元数据冻结本轮前瞻内部审计划分，诚实注明历史训练暴露，不能当完全独立确认。缺独立test不阻断合法train/val探索。优先启动M1/local真实pilot，然后五框架和既定对照；不要在smoke或准备清单停工。

代码和轻量报告推送PR#6分支，不自行合并；SS02/03/05冻结，不启动SS04/S04/父S05。问题和结果向协调会话汇报。当前真实运行状态见 `reports/pub_01/s03/continuity/eeg_run01/EXECUTION_LOG.md`（由执行者按实际进展创建，未启动的阶段不能写已完成）。

---

## prototype01 历史入口（由上方最新授权取代）
# Codex 启动入口 — SS06 连续信息原型（2026-10-23）

本轮仅 PUB-01-S03_SS06：用户已授权五种框架任务书、基本实现和合成工程测试。**真实数据加载、预训练、pilot、参数优化与正式评估均未启动，也不自动启动。** 工程审阅与交付已完成：SS06 COMPLETED/PASS_WITH_LIMITATION，当前停止。

1. 读取最新用户指令、AGENTS.md 与父任务登记。
2. 读取 doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md（五模型结构/训练目标/对照/验收）。
3. 接口契约为同目录 SS06_IMPLEMENTATION_CONTRACT.md；运行证据/审阅入口为 reports/pub_01/s03/continuity/prototype01/REPORT.md。
4. 代码 src/m6a_public/continuity/，配置 configs/pub_01_s03_ss06_prototype_20261023.json。所有数值测试在2203；Windows仅代码轻量报告。当前无真实loader或训练命令。
5. 独立 worktree 分支 codex/pub-01-s03-ss06-continuity；保护主目录CNN/TRF、stash和旧worktree。main起点60c3404，PR#3/#4已合并；SS02/03/05结论只读，不重跑旧test。

当前 split/声学lag/实际公平预算均未冻结，不能从工程PASS推断EEG性能改善。S03/PUB仍ACTIVE；SS04、S04和父S05不启动。

---

## 历史启动入口（以下均为快照，不构成当前授权）
> **Issue#5已实际完成（2026-10-19）**：SS05 COMPLETED / PASS_WITH_LIMITATION。五组×三seed正式训练及105个历史条件结果已交；新内部holdout21人/47record/7family的r约.091–.092，关联/历史残差额外收益的六预设区间均跨0，为有效阴性。新train-only norm、后置same-agent review和实测图表见reports/pub_01/s03/predictive_memory/codex_run01/REPORT.md。当前停止，PR#4仍Draft，不合并、不启动SS04/S04/父S05。

# Codex 启动入口 — Auditory_Simulation（2026-10-19最新）

**本轮完成任务：`PUB-01-S03_SS05`，有效阴性/完成有限制。** PR#3已按用户新授权由协调者合并main@8d21d0a，SS05在PR#4独立分支交付、base main。当前无自动下一READY节点，不因阴性追加模型；报告和核验不称第三方科学签署。

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
