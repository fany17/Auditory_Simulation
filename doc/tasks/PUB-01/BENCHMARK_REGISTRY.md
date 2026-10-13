# PUB-01 — EEG 听觉解码竞品与任务基准表

状态：`CANDIDATE_PROTOCOL_AUDIT`（2026-10-13）。**入口仓库存在已核实；模型是否能复现、论文报告的结果数值/测试定义还必须由 `S03_SS02` 在执行中验证。**

| Baseline / 代码仓库 | 首要研究任务 | 本项目角色 | 首先要核对的不可比因素 | 当前状态 |
|---|---|---|---|---|
| HDPS — https://github.com/bobwangPKU/HDPS | 语音片段多候选检索 | SparrKULee 百余候选强基线优先 | 预处理 fs、片段时长、受试者 train/test、候选构造、是否有 GPT-2 特征与候选依赖 | NOT_REPRODUCED |
| Transformer-CLIP-EEG — https://github.com/exporl/transformer-clip-eeg | 脑电—声音 match/mismatch；论文实际任务须复核 | 二选一强基线优先 | 输入/输出、数据集/split、窗口长度、负例难度、指标宏平均 | NOT_REPRODUCED |
| Auditory EEG Challenge 2023 — https://github.com/exporl/auditory-eeg-challenge-2023-code | 按官方任务定义检查二选一等 | 官方二选一评估协议候选 | 原 challenge 数据与 SparrKULee 是否相同、盲测标签和评测脚本可得性 | PROTOCOL_AUDIT_PENDING |
| Auditory EEG Challenge 2024 — https://github.com/exporl/auditory-eeg-challenge-2024-code | 按官方任务定义检查多候选匹配 | 官方五选一评估协议候选 | 官方数据/候选集、是否隐藏测试、是否能在相同划分下比较 | PROTOCOL_AUDIT_PENDING |
| S03_SS01 backward TRF | EEG→一维声音包络 | 已完成的可解释直接解码基线 | 64 Hz derivative、10 人 93 条、被试内跨 source-family；与上述 paper protocol 不同 | COMPLETED / PASS_WITH_LIMITATION |
| S02 acoustic/log-mel/wav2vec2 ridge/TRF | 声音特征→EEG encoding | 参考声音 feature bank 和正向编码对照 | 方向相反，不能作为 EEG decoder 的性能结果 | COMPLETED / PASS_WITH_LIMITATION |

## 统一 benchmark 的最低要求

**模型任务**：EEG→连续时间的声音特征（无候选输入）；可以在固定推理结果上计算听觉特征重建、二选一、五选一及百余候选相似度检索。候选材料只在**后验评价**出现，且模型的训练目标应分别标记 regression/contrastive/reconstruction 及其权重。单独训练候选匹配模型仍是有效强 baseline，但不能冒充“直接解码”。

- `DIRECT`：目标/预测表示、时间轴和分辨率、held-out metric（Pearson r、R²、声学/谱域合理指标）、错配/均值/先验对照。
- `2AFC/5AFC`：按照对应 challenge/论文的候选集构造和同任务准确率；同时记录错配类型和每被试统计。
- `LARGE_POOL`：报告每个 test unit 的候选数/候选身份、Top-1/Top-5/Top-10，通常还应分池规模作敏感性分析；不要把固定故事身份泄漏进模型。
- `GENERALIZATION`：被试内新故事（subject-specific）、严格未见被试（cross-subject）分开统计；不能用同一故事在训练与测试重复导致过度乐观。
- `PRIOR`：任何 GPT/语言模型或预训练声音模型参与，必须明确其作用；GPT 重排序只可在后续另批预注册条件下做，必须有 LM-only/prior-only 及 EEG-shuffle 对照。
- `SOTA`：只有在相同官方数据、测试分组、候选集、指标和公开证据下，才可声称对应 protocol 新纪录；内部冻结测试不等于官方挑战纪录。

研究目标可以雄心勃勃，但 60%（百余候选 Top-1）和 GPT 后 80% 目前只能作为待检验挑战目标，不能倒逼拆分或挑选有利候选来制造结果。

## 评价顺序

先完成 `S03_SS02` 论文协议与可复现 baseline；**再由研究者**在 `S03_SS03` 指定新架构与训练目标；最后于 `S03_SS04` 独立验证 direct + retrieval。ds004703 基线见 `S04`，与本表的公开 EEG 基准不能混作同一排行榜。
