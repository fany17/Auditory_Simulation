# Auditory_Simulation Current Workboard

更新时间：2026-10-19（Codex 分工纠偏）。父任务状态由 `doc/TASKS.md` 定义，子任务状态由 `doc/tasks/PUB-01/SUBTASKS.md` 定义。**以下当前区是唯一执行路由，其下为历史记录。**

## 当前正在交接给 Codex

- **任务：`PUB-01-S03_SS05`，状态 `READY_FOR_CODEX`（尚无新的Codex运行证明）。** 研究者要求从 SS03 的多尺度感受野，转向跨窗口历史记忆、关联检索及预测误差更新的机制检验。
- **执行归属**：Codex 写代码、运行 tests/合成验证/2203公开EEG、复核与提交；协调助手只维护总任务和科学验收。旧协调助手合成结果属于待核验初稿，不能标成Codex通过。
- **工作位置**：`work/pub-01-s03-ss05-predictive-memory`（PR #4 Draft，叠在 PR #3 冻结 SS03 分支之上）；`main`/原本地CNN/TRF/stash只读。
- **立刻执行**：阅读 `doc/CODEX_START_HERE.md` 和 `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`。先独立源码+合成审查，再用2203真实公开资源核对分组/时轴，创建仅来自先前训练池的新 prospective internal heldout（若可行），进行新的train/validation，不使用已经打开的SS03测试集调参。
- **当下不可声称**：新架构相对GRU/SS02/SS03已优于强基线、预测误差模块有效、公开EEG复现完成、已有SS05新的heldout分数。当前SS05 **真实公开EEG训练/正式测试 NOT_RUN**。
- **其他状态**：SS01/SS02/SS03已完成有限制并冻结；SS04 BACKLOG；S04公开颅内支线BACKLOG；S05版本化artifact整合BACKLOG；公共项目与患者/STN项目单向版本化衔接，禁止患者数据进入本项目。

## 历史执行/状态快照（仅供追溯）

## 现在要做什么（Codex 第一入口）

**2026-10-19新授权 SS05：ACTIVE / SYNTHETIC_PROBE_COMPLETED；正式公开EEG TRAIN/TEST NOT_RUN。** 已写PI-MRAM（预测误差＋多速率状态＋关联delta记忆）及同窗口GRU控制、10项本地CPU单元测试与两套合成实验。延迟两键干扰160步：full/assoc_no_pred 100%，GRU62.5%；GRU600步也100%，不能声称不可替代的算法能力；预测误差独立贡献尚不成立。代码和结果仅在PR#4独立工作分支。下一步按 `PUB-01-S03_SS05_PREDICTIVE_MEMORY.md` 与 `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md`，先做公开source-family和新内部测试划分可行性审计，SS03旧测试集不可再用于SS05选模型。**不自动启动SS04/S04/S05，不产生artifact。**


2026-10-19交付：`PUB-01-S03_SS03` **COMPLETED / PASS_WITH_LIMITATION**。真实输入对齐/CPU-GPU smoke/100步pilot→测试前预算冻结→4×3正式训练及每模型一次test→同agent separate review已完成。multiscale seen/heldout r=.10305/.11380；long .06411/.06979；middle .09297/.10413。当前四原型/固定数据预算下的H1获支持，不是生理机制或超过SS02。报告`reports/pub_01/s03/architecture/REPORT.md`。

当前STOP：PR#3保留Draft，工作分支work/pub-01-s03-ss03-multiscale/独立managed worktree；主目录未提交CNN/TRF和stash未动。S03/PUB仍ACTIVE，SS04/S04/S05保持BACKLOG，未产生ART-AUDREP-v1。后续须研究者指定，无自动下一READY节点。下方10-15为SS02历史交付。



2026-10-15交付：用户授权“缺几个就缺几个”后，两套指定强基线及Transformer直接包络readout均完成真实训练/验证/一次测试与独立review。SS02 **COMPLETED / PASS_WITH_LIMITATION**；当前执行停止，没有自动下一READY节点。报告`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。

- Parent：`PUB-01 — Public Auditory–Neural Representation`，`ACTIVE`。
- 当前优先支线：**A / SparrKULee / EEG 听觉直接解码**，归属 `PUB-01-S03`，`ACTIVE`。
- **本轮完成节点**：`PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction`，**COMPLETED / PASS_WITH_LIMITATION**。
- 执行入口：`doc/CODEX_START_HERE.md` → `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`。
- 任务：按冻结的可执行公开条件复现HDPS与Transformer-CLIP-EEG，明确原论文条件差异；当前HDPS仅M，Transformer仅单模型及其原方法包络readout，不启动新架构/GPT重排序。
- 实际结果：HDPS 76人、100Hz/M、作者split及valid索引，Top1/5/10=12.03/30.45/42.10%，shuffle Top1=0.53%。Transformer作者fold4、70人训练、69seen/13heldout，64Hz/XLSR19；2AFC=83.53/80.08%，加权82.38%；冻结encoder原Conv32直接包络r=.1688/.1403，加权.1593，错配近0。三运行均review PASS_WITH_LIMITATION；官方完整cohort/hidden test/ensemble未复现。
- `S03_SS03` 架构与 `S03_SS04` 统一验证均 `BACKLOG`，需要前一环节审阅后再决定。

## 两条数据支线

| 数据与职责 | 当前状态 | 阻塞关系 |
|---|---|---|
| A / SparrKULee EEG decoding / `S03` | ACTIVE；SS02 COMPLETED / PASS_WITH_LIMITATION | 两套公开可用条件基线已交；SS03已授权原型启动，实验待运行 |
| B / ds004703 public intracranial / `S04` | BACKLOG；SS01/SS02 尚未执行 | 自身 sEEG baseline 与 A 线可独立；**仅 EEG→sEEG 迁移需 A 线 frozen 模型** |
| `S05` artifact integration | BACKLOG | 两条支线的结果与失败/许可验收后评估 |

参见 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md` 与 `BENCHMARK_REGISTRY.md`。

## 已经完成且不重跑的证据

- `S01`：公开 SparrKULee / ds004703 数据可用范围下载、审计、初步虚拟清洗、图形化探索已阶段验收；历史 ds004703 319 候选片段**不等于正式模型输入合法性**。
- `S02`：83 名参与者、659 条记录 acoustic/log-mel/frozen wav2vec2 → EEG encoding，完成有限制；HuBERT HOLD。
- `S03_SS01`：10 人、93 条记录 64 Hz EEG→声音包络 backward TRF，test r≈0.1819、R²≈0.0178、错配 r≈−0.0033；2026-10-11 PASS_WITH_LIMITATION，证据 `reports/pub_01/s03/envelope_decode_20261011/REPORT.md`。

## 硬边界 / Stop

- 所有数据、权重和实际计算 **server2203**；本地 Windows Git 只存轻量代码、配置和汇总；先只读盘点已有工作，不覆盖用户未提交内容。
- 不读取任何患者/STN 数据与 embedding；公共模型只通过冻结、版本化 `ART-AUDREP-v1` artifact 向患者侧 `INT-01` 单向交付。当前尚未冻结。
- 当前两套方法的可用条件复现不等于完整论文数值一致性；完成SS02三级审阅后停止，不自启SS03/S04或GPT新模型。
