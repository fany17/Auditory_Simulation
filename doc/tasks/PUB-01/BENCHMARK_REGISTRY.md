# PUB-01 — EEG 听觉解码竞品与任务基准表

2026-11-01 SS07 D10：旧内部36项全部完成，PFT*/Ponly/C/Ccap/W/F加权r=.168945/.170374/.171197/.173348/.173393/.174284，三个主要内部配对各组/seed全正；C−Ponly held全负、W−Ccap mixed和F窗口退化完整保留。306source/600inputs/36聚合/72cached、全部rawFT桥及全文三图实际review有限制通过，finetuned_internal_audit/run01 REPORT/REVIEW/HTML归档完成。旧test仍POST-HOC INTERNAL EXPLORATORY，无新fit/λ/DTU或独立确认；D11仅原TRAIN/VAL慢观测状态最小PLAN/源码资产盘点，未新增方法/数据授权，goal ACTIVE。

2026-11-01 SS07 D9：六个匹配物理目标解完成，Ccap/W/F VALr=.182340/.182474/.183874，F−W三seed正均+.001400；F单窗MSE352.068/r3s.003435、巨大bias在原Hann近相消，固定融合点增益和严重窗口表示退化并存。全TRAIN独立梯度/目标/真残差、全VAL及12缓存重建、全文四图实际review有限制通过，physical_objective/run01/REPORT/REVIEW与HTML已归档；未建立通用稳健、历史或独立确认，goal ACTIVE。D10仅固定6角色×3seed旧seen/held事后内部审计的PLAN/资产/预算准备，不能据旧test调参，DTU关闭。

2026-11-01 SS07 D8：强FT epoch4冻结encoder/head上的27固定岭解完成，P_FT*/Ponly/C/Ccap三seed VAL r=.174141/.177842/.180541/.182340，Ccap两个primary均三seed正。全部FT历史loss/raw桥差0；实际27系数/30聚合/18缓存重建和全文三PNG独立review有限制通过，完整finetuned_readout_transfer/run01图文表/REVIEW与HTML归档齐备。新增TRAIN岭/标定/搜索成本真实，重复known-person VAL不称独立泛化或纯loss因果；普通ridge与平方不是方法创新，无历史输入，goal ACTIVE。D9仅精确连续监督PLAN准备，不打开旧test/DTU、不改D8λ。

2026-11-01 SS07 D7：冻结current读出的36项旧内部评分完成、独立36聚合/72缓存重建及全文三图审阅通过，完整internal_generalization/run01/REPORT.md与REVIEW.md/HTML归档齐备。P*/P-only/Ccap原2:1加权r=.160248/.164642/.168943；Ccap对前两者每seed/组均正。同三seed历史完整P_FT=.168944726，与Ccap点均值几乎相同，配对方向混合，不能宣称超越/等效/非劣；额外预算与边界策略不同。旧测试POST-HOC INTERNAL EXPLORATORY，无独立确认或连续历史收益，goal ACTIVE。下一步仅准备P_FT TRAIN/VAL最小读出诊断及资源PLAN，不用旧test排名调参。

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
