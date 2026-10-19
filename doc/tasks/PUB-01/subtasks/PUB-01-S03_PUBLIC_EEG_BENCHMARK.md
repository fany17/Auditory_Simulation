# SUBTASK — PUB-01-S03 — SparrKULee EEG Direct Auditory Decoding

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | SparrKULee EEG Direct Auditory Decoding & Representation Benchmark |
| Status | `ACTIVE`（SS01/SS02/SS03已交有限制；SS03四原型×三seed实际结果/review已完成；SS05为READY_FOR_CODEX，由Codex独立复核既有合成原型并执行新的公开EEG实验；SS04 BACKLOG） |
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

## 当前S03任务优先级（2026-10-19）

- SS01/SS02/SS03保留 `COMPLETED / PASS_WITH_LIMITATION`，其测试与报告只读。
- 研究者已明确授权 `SS05` 历史联系/记忆结构研究，但**要求Codex，而非协调助手，执行源码审计、仿真及2203训练**；既有合成报告标待Codex核验。Codex目前尚未真实接手，不可写 `CODEX_RUNNING`。
- `SS05` 先验证关联记忆与预测误差的独立贡献，再在新冻结公共EEG分组上报告结果。SS04多种候选评测仍需另行验收/授权。

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
3. `S03_SS03` — **COMPLETED / PASS_WITH_LIMITATION（2026-10-19）**：参数匹配四原型×三seed各25000步，直接包络multiscale seen/heldout r=.10305/.11380，高于long/middle，预设配对区间均>0。仅支持当前数据预算下结构比较，非生理机制/强基线超越；报告`reports/pub_01/s03/architecture/REPORT.md`。
4. `S03_SS05` — **ACTIVE / SYNTHETIC_PROBE_COMPLETED**：新历史记忆探索，10项本地测试及合成数据对照完成；不将合成成功称作真实EEG或算法原创证明，正式公开数据处理待执行。
5. `S03_SS04` — **BACKLOG**：固定一份主 decoder，在直接特征重建、二选一、五选一、百余候选检索和跨人/跨内容条件下统一验证，严格区分不同任务指标。

当前SS03执行已停止，PR#3保持Draft，不自动启动SS04/S04。后续需研究者指定；Codex入口：`doc/CODEX_START_HERE.md`。

## 4. Scientific controls and acceptance

- 原论文数据处理/采样率/候选集/划分不一致时分别报告，不把已有 64 Hz 包络 TRF 结果拿去与 100 Hz 多候选比赛直接比较。
- 所有模型选择及 feature/layer/参数/候选挖掘策略必须冻结或只在 train/val 中决定，不得在 test 中选择更有利条件。
- 同等数据/算力/训练预算比较，保留不同种子、逐受试者成绩、错配/替换候选及先验基线。
- GPT 后验语言校准只可能在独立后续批准节点执行，必须 EEG-shuffle 和 GPT-only 控制；不是 SS02 复现的组成。
- 无法发表某官方 SOTA 声明时，也可以用透明、公平的内部协议完成方法研究；不可冒称官方刷新纪录。

## 5. Completion and evidence

`IN_PROGRESS`（S03 parent）。SS01/SS02科学结果冻结不变，SS03在2026-10-19真实完成有限制，Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS；同agent separate review，不称第三方。S03/PUB仍ACTIVE，SS04 BACKLOG，没有ART-AUDREP-v1。旧CNN/TRF未提交修改与stash保持原状；当前停止。
