# PUB-01 — EEG 听觉解码竞品与任务基准表

2026-10-15交付：SS02 COMPLETED / PASS_WITH_LIMITATION。HDPS原100Hz/M、76人599对、test12776query：Top1/5/10=12.03/30.45/42.10%，shuffle Top1=.53%；Transformer原架构/public fold4/XLSR19、70训练/69seen/13heldout：2AFC=83.53/80.08%（加权82.38%）、EEG-only原Conv32包络r=.1688/.1403（加权.1593）。三运行独立review通过，完整论文条件未复现。M论文数字仍未核实，不与MWG20%比差；100–235候选原本存在固定公开索引。下述10-13等待记录为历史。

状态：`PROTOCOL_AUDITED / REPRODUCTION_HOLD`（2026-10-13追加）。原raw100Hz/M70story与独立HDPS论文功能模型batch256工程smoke已完成；原split600/647对就绪，必要47条raw受限。**A+B同条件正式训练/测试NOT_RUN，不能判断一致性**。Transformer EEGcheckpoint失效且official test未齐；XLSR原声音权重已取得。恢复清单/三级review见`reports/pub_01/s03/strong_baselines/`；无76人/64Hz替代分数，SS02 WAITING_EXTERNAL / HOLD，无后台作业。

| Baseline / 代码仓库 | 首要研究任务 | 本项目角色 | 首先要核对的不可比因素 | 当前状态 |
|---|---|---|---|---|
| HDPS — https://github.com/bobwangPKU/HDPS | 语音片段多候选检索 | SparrKULee 强基线 | 100Hz/3s；作者split；候选=subject phase全部有效片段，非固定100；本轮M-only不含GPT2 | 可用76人功能复现COMPLETED/PASS_WITH_LIMITATION；Top1=12.03%；完整cohort/MWG仍NOT_RUN |
| Transformer-CLIP-EEG — https://github.com/exporl/transformer-clip-eeg | contrastive representation + 二选一/envelope回归 | 二选一及直接包络强基线 | 64Hz/3s；fold4；70train/69seen/13heldout；Dutch XLSR19；derived longrecord，不同于官方packet/ensemble | 单模型功能复现COMPLETED/PASS_WITH_LIMITATION；加权2AFC82.38%、包络r=.1593；完整official条件NOT_RUN |
| Auditory EEG Challenge 2023 — https://github.com/exporl/auditory-eeg-challenge-2023-code | 3s二选一 + 30s envelope | 官方二选一评估协议 | GPL3+；84人官方test，标签脚本公开；现有普通记录不替代candidate/test包 | PROTOCOL_AUDITED_WITH_LIMITATION；模型NOT_RUN / HOLD |
| Auditory EEG Challenge 2024 — https://github.com/exporl/auditory-eeg-challenge-2024-code | 5s五选一 + 30s Mel10 | 官方五选一评估协议 | GPL3+；20新被试官方mapping；任务2对齐revision与提交轴方向需核验 | PROTOCOL_AUDITED_WITH_LIMITATION；模型NOT_RUN / HOLD |
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
