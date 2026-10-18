# SUBTASK — PUB-01-S03 — SparrKULee EEG Direct Auditory Decoding

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | SparrKULee EEG Direct Auditory Decoding & Representation Benchmark |
| Status | `ACTIVE`（SS01/SS02已交；SS02可用公开条件两基线及包络readout PASS_WITH_LIMITATION；SS03多尺度结构原型已授权/未正式训练） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `PUB-01-S01` / `PUB-01-S02` 完成有限制阶段交付 |
| Inputs | SparrKULee 已授权公开 EEG /声音、S02 frozen acoustic feature bank/split、既有 S03_SS01 结果、外部基线原论文/代码 |
| Deliverables | 官方强基线复现；无需候选输入的 EEG→声音特征解码架构；direct reconstruction + 2AFC/5AFC/large-pool retrieval 分协议验证；跨被试/故事、消融与 negative evidence |
| Acceptance | 冻结输出含直接声音特征；与强基线在可比较协议下公平对照；结果保留不同候选池规模、被试/刺激拆分、错误/阴性和时间信息局限 |
| Validation | 复现 code/config/metric、test leakage / negative pools、模型无候选输入 canary、独立 review + parent consistency |
| Go/Hold/Fail | 官方代码或数据不可用时记录 HOLD/内部协议，不编造 SOTA；模型无收益保留阴性并重新审议架构 |
| Subnodes | `SUBTASKS.md` 中 S03_SS01..SS04 |
| Files Allowed to Change | public EEG model/scripts/configs/tests/reports、`doc/tasks/PUB-01/**`，历史 frozen 文件只读 |
| Last Updated | 2026-10-19 |

## 1. Scientific question

能否设计一个结构简洁、无需候选语音输入的**EEG 听觉直接解码器**，从连续头皮脑电输出可对齐的声音表征，并在完全公平的公开基准下同时支撑直接重建和不同规模的语音片段识别？

近期目标是追求强基线与 SOTA 水平，但**成绩不是算法 novelty 本身**；需要明确新算子相对已有强模型实际解决什么信号/时间信息瓶颈及其消融证据。

## 2. Scope / definitions

- `Direct output`：EEG→一维包络、Mel 频谱或获准的冻结声音表示。候选答案/音频 ID 不得进入主 decoder 的推理输入；Mel/embedding 重建不自动等于可听语音或语言文本生成。
- `Retrieval evaluation`：对已冻结 direct output 与待检索真实声音的相同特征计算相似度；依原 benchmark 定义做 2AFC、5AFC、100+ 候选 Top-1/5/10。
- `Strong baseline`：允许复现那些本身使用候选语音或对比式编码器的前人方法，但要明确它们与新 decoder 的输出差别；任何“直接解码” claim 不借用 candidate-only baseline。
- `Generalization`：被试内新故事、完全未见被试及校准数据效率均为独立 protocol。官方 paper 排名、内部统一冻结 split 严格分表。

不添加患者/STN 数据；不启动 GPT rerank、不额外下载无批准的模型/数据；不凭模型名预设 SaShiMi/TimeMixer++ 具有收益。

## 3. Subnode order

1. `S03_SS01` — 2026-10-11 已完成：10 人/93 条 backward TRF EEG→声音包络，test r≈0.1819 / R²≈0.0178、13 秒错配 null r≈−0.0033（单被试内跨音源条件，有限制，不是语音识别）。
2. `S03_SS02` — **COMPLETED / PASS_WITH_LIMITATION**（2026-10-15）：按用户事先授权的可用公开条件，HDPS76人原100Hz/M Top1/5/10=12.03/30.45/42.10%；Transformer70训练/69seen/13heldout，2AFC加权82.38%、冻结EEG编码器原Conv32包络加权r=.1593。两套方法及直接endpoint均有真实test/独立review；原完整论文条件未复现，差异与失败见`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。当前停止，不自动研发下一架构。
3. `S03_SS03` — **ACTIVE / PROTOTYPE（2026-10-19）**：研究者选定多尺度时间感受野单假设，已创建参数匹配的EEG→包络原型与测试，正式2203数据训练/验证/测试NOT_RUN。按 `PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`执行。
4. `S03_SS04` — **BACKLOG**：固定一份主 decoder，在直接特征重建、二选一、五选一、百余候选检索和跨人/跨内容条件下统一验证，严格区分不同任务指标。

当前只批准 **SS03** 在独立worktree/2203上继续原型和实验；不自动启动SS04/S04。Codex入口：`doc/CODEX_START_HERE.md`。

## 4. Scientific controls and acceptance

- 原论文数据处理/采样率/候选集/划分不一致时分别报告，不把已有 64 Hz 包络 TRF 结果拿去与 100 Hz 多候选比赛直接比较。
- 所有模型选择及 feature/layer/参数/候选挖掘策略必须冻结或只在 train/val 中决定，不得在 test 中选择更有利条件。
- 同等数据/算力/训练预算比较，保留不同种子、逐受试者成绩、错配/替换候选及先验基线。
- GPT 后验语言校准只可能在独立后续批准节点执行，必须 EEG-shuffle 和 GPT-only 控制；不是 SS02 复现的组成。
- 无法发表某官方 SOTA 声明时，也可以用透明、公平的内部协议完成方法研究；不可冒称官方刷新纪录。

## 5. Completion and evidence

`IN_PROGRESS`。SS01/SS02科学结果各自冻结且状态不变，SS03在2026-10-19进入**ACTIVE / PROTOTYPE**。代码/协议/测试仅作为可执行第一步，不代表任何SS03真实训练结果或科学提升。S03 parent继续ACTIVE，SS04 BACKLOG；没有ART-AUDREP-v1。旧CNN/TRF本地未提交改动应保持原状。
