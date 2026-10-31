# SS06 数据谱系与连续坐标

2026-10-23。此轮是 **prospective INTERNAL / EXPLORATORY source-family audit**。所有来源曾用于SS03和SS05训练/项目开发，没有新的外部独立测试数据；不使用旧已查看validation/test来调SS06。原SparrKULee内部研究授权/公开资产范围不扩张，无新下载、患者/STN或旧checkpoint输入。

SS03旧train426条 → SS05按family划分后原train353条/23family → SS06仅以元数据、排序family和numpy RNG20261023一次置换：前5audit、次5validation、其余13train。记录、subject、story与S02真实source_family映射逐一核对。split在新统计拟合、SS06模型拟合及audit载荷读取前写入；不按分数选组。

| Phase | Families | Records | Subjects | 初始60s热身后的完整60s块 |
|---|---:|---:|---:|---:|
| train | 13 | 249 | 70 | 3262 |
| validation | 5 | 63 | 45 | 980 |
| audit_holdout | 5 | 41 | 36 | 538 |

audit families：podcast_12、audiobook_4、podcast_5、podcast_13、podcast_8。validation：audiobook_14、podcast_29、audiobook_3、podcast_27、podcast_9。family/story/record/EEG路径/envelope路径跨phase无交集；人可重叠，不能称新被试泛化。仅5个audit family，分组推断不确定性很大。

64Hz、64 EEG通道。固定lag8sample：新对齐x[s]=rawEEG[s+8]，y[s]=envelope[s]；EEG晚于目标125ms。这来自已存在train/validation依据，不在SS06或audit寻峰，不证明生理时延。physical_sync仍UNKNOWN，发布方预处理历史不在本轮重新验证。

只在新249train的真实共同样本上重拟合64channel EEG及scalar envelope mean/std，保存为2203新文件newtrain_EEG_target_normalization.npz，所有phase只用该文件。旧统计不覆盖，M3 teacher/SSL也只用新train EEG。目标不能用测试段真实mean/std反变换。JSON契约记录fit IDs与样本数，npz、原始信号和cache不回传本地。

每record首3840sample做60s热身，之后不重叠3840sample逐块评分；不满60s尾段统一排除，所有模型/条件同一坐标。正式训练整条record按时间顺序推进；batch1，每60s完整串接输出一次Pearson+.1MSE和optimizer.step，随后detach state保留数值；换record/subject/session/split/gap才reset，不跨不连续区间接状态。由于每块后参数更新，后续携带的是上一更新前数值状态，属常规在线TBPTT，不保留跨optimizer计算图；梯度跨度60s不等于记忆值仅60s。

pilot只用record前两块；正式协议显式改为全部3262块、共同一轮/同seed记录及块顺序。初始SPLIT_PROVENANCE.json的training_stream文字为pilot初案，FORMAL_PROTOCOL.stream_amendment是拟合前修订；冻结split成员/字节身份不变。M3正式预训练另覆盖3511个完整60s输入段，每段fresh-state，保留teacher版本一致；没有实现跨段未来latent教师目标，也不声称已实现无限记忆。

主指标：全538评分块的60s r/MSE→record等权→person等权，seed先在record内平均后做配对统计。历史干预另用每record**第一个**评分块：共同60s前缀、同60s当前EEG/目标；state-only清零/错配保留真实local encoder缓存，全前缀干预另列。donor同phase但异record/person/family，固定规则，不看目标选事件。该历史端点与全记录主端点不同，不混表。

在全部39训练、3正式预训练、完整val/共同样本顺序/实际checkpoint内容及身份核查通过前，不开放audit。正式source/config/数据契约身份只针对本轮轻量源码与新统计文件，不对原始库做全量hash审计。
