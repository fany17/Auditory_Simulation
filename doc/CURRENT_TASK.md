# Auditory_Simulation Current Workboard

更新时间：`2026-10-19`。父任务状态来源：`doc/TASKS.md`；子节点状态来源：`doc/tasks/PUB-01/SUBTASKS.md`。

## 现在要做什么（Codex 第一入口）

2026-10-19 当前研究者授权**直接继续**：`PUB-01-S03_SS03`现为 **ACTIVE / PROTOTYPE**，首项研究假设“匹配参数量的多时间尺度卷积，能否超越同最大感受野单尺度卷积改善EEG→连续声音包络预测”。模型原型 `scripts/pub_01_s03_temporal_scales.py`，4种配置与冻结约束见 `configs/pub_01_s03_ss03_multiscale_20261019.json`；操作手册 `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`。**本提交只含任务协议与源码/单元测试，无2203新训练/科研数值，尚未通过执行验收。**

Codex下一步：在**独立worktree**中使用SS03分支，不动本地未提交CNN/TRF；只读核验2203 SS02冻结输入/对齐/分组后，执行CPU/GPU smoke，按同一64Hz公开fold4和包络读出窗口实现直接预测数据加载；train/val 100步pilot，冻结正式预算/种子，之后再做多尺度与三单尺度同预算训练。**绝不在完成前提前打开test结果**。SS04/S04/S05不得启动。



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
