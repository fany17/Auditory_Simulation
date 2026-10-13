# Auditory_Simulation Current Workboard

更新时间：`2026-10-13`。父任务状态来源：`doc/TASKS.md`；子节点状态来源：`doc/tasks/PUB-01/SUBTASKS.md`。

## 现在要做什么（Codex 第一入口）

- Parent：`PUB-01 — Public Auditory–Neural Representation`，`ACTIVE`。
- 当前优先支线：**A / SparrKULee / EEG 听觉直接解码**，归属 `PUB-01-S03`，`ACTIVE`。
- **唯一当前 READY 节点**：`PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction`。
- 执行入口：`doc/CODEX_START_HERE.md` → `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`。
- 任务：先核对并复现 HDPS 等百余候选强基线、Transformer-CLIP-EEG 二选一，以及审查官方挑战协议；**不启动架构研发和 GPT**。
- `S03_SS03` 架构与 `S03_SS04` 统一验证均 `BACKLOG`，需要前一环节审阅后再决定。

## 两条数据支线

| 数据与职责 | 当前状态 | 阻塞关系 |
|---|---|---|
| A / SparrKULee EEG decoding / `S03` | ACTIVE；SS02 READY | 完成强基线后再由研究者指定架构 |
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
- 本次仅更新规划/任务文档，没有开启新训练。Codex 完成 `SS02` 后先独立验收并停止；不自启 `SS03 / S04 / GPT`。
