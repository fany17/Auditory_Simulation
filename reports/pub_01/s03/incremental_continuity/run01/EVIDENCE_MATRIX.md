# SS07 投稿证据矩阵（结果前版本，2026-11-01）

当前为研究执行与缺口索引，**不是投稿就绪、性能胜出或科学验收报告**。以实际服务器产物为权威；下列待完成项不能由工程测试、候选验证集改善或单个seed替代。所有当前实验仍按已有冻结协议执行，本表不新增实验、指标、选择规则、成功阈值或审批门。

| 要回答的问题 | 已有直接证据 | 尚待的实际证据/权威产物 | 当前判断与外推边界 |
|---|---|---|---|
| 原强模型是否被完整复跑？ | 五个native预训练与head均完成；预训练按pat15，head均250上限；BASELINE_BUDGET.csv与BASELINE_CURVES.png | 最终manifest核验各自checkpoint、epoch、同seed来源 | 原可用数据/资源协议的完整复跑成立；不是官方全训练人数、500epoch或ensemble结果，不称head收敛 |
| 是否有强于仅冻结readout的控制？ | 五个原encoder+head微调均完成5epoch/19725更新；真实incumbent在更新前核验；FINETUNE_BUDGET.csv与曲线 | 统一原weighted60s/person测试与DTU同输入适配结果；final_summary中的P_FT比较 | 强控制已实际训练，测试是否改善尚未证明；全部FT最佳在预算末轮，不称已收敛或最优微调 |
| 连续历史是否超过当前EEG充分利用后的原强基线？ | 五类有界残差及屏蔽恒等/梯度/状态检查通过；18×3筛选仍进行 | 全54筛选→初始15full→前三seed验证选定primary→五seed最终配对；INTERNAL_MAIN与EXTERNAL_DTU_MAIN | 未有完整候选或实际主测试差值；不能从不完整screen排名选择主张 |
| 增益能否由容量、当前条件映射、输出历史/平滑解释？ | current/static active容量控制、output_prior、EMA、FT已预设；EMA原范围最终选tau0，保留重复保守对照 | 全部相同seed/预算匹配控制的frozen结果、active参数和COSTS；预设M6/7差值与区间 | 未证明额外历史贡献；registered参数不等于影响预测的active参数；不把共享encoder成本藏入head计时 |
| 历史是否有内容与时间特异贡献？ | 非重叠双parity历史支持检查通过；reset/content_blind/shuffle/older/两类错配的数学与固定metadata计划完成 | 六干预全部五seed输出；各自eligible keys上true配对的INTERNAL_HISTORY_DIAGNOSTICS | 工程时间支持不证明历史有用；shuffle可含未来观察、older同时改recency/parity，均仅离线分布诊断，不能直接称机制或因果证据 |
| 共享刺激与受试者依赖是否被如实处理？ | 原2/3seen+1/3heldout、跨seed/person共用抽样、共享family Exp敏感性、per-family/LOFO覆盖规则均冻结 | INTERNAL_MAIN、逐人/逐record/story/family CSV、LOFO与敏感性；同key不丢seed核验 | 原Sparr测试均已暴露，仅POST-HOC INTERNAL EXPLORATORY；近似bootstrap非有限样本覆盖保证，Exp区间不是校准CI |
| 外部泛化是否成立？ | DTU18人1080trial可读性、唯一当前3s适配、全保留与源许可/白名单记录完成；尚未做实际NN评分 | 固定全部主模型×五seed×1080trial的EXTERNAL_DTU_MAIN；独立raw/metrics review | 可检验该外部数据的固定适配迁移，不证明刺激/房间独立泛化；不得改适配、排除S11或据外部成绩选择模型 |
| 是否排除了指标/模型角色/批次重放错误？ | 独立数学37核对、完整synthetic caller77482核对、真实P/M2/FT1024合法VAL重放通过；复核source2f699c2及8source/3proof经协调审阅冻结 | 实际全部评估结束后的RAW_INPUT_REVIEW与METRICS_REVIEW，以及INDEPENDENT_REVIEW_COMPLETED；协调读取实际轻量结果 | 现有PASS为工程/合法VAL证据；实际review仍待执行。raw仅固定3条Sparr+6条DTU×5seed×全部main；复用冻结NN forward，不冒充全量独立神经数学 |
| 是否有足以支持投稿的方法新意与证据强度？ | SCIENTIFIC_DESIGN_REVIEW及RELATED_WORK_COMPARISON逐项核验9份一手来源与实际代码；明确output_prior非DECAF复现、M3非CPC、M4非物理延迟、M5无Kalman校准等边界 | 上述完整效果/控制/泛化结果、实际失败范围、完整竞品同协议直接比较的缺口、独立科学接受判断及投稿缺口决策 | 新意尚未成立。小幅单数据集r改善不等于ICML/NeurIPS证据，也不因技术PASS、某seed阳性或论文风格包装而完成goal |

指标必须与结论范围一致：3s验证准则用于训练选择；主结论用原60s有效块→人→seen/heldout原权重；外部以trial attended r→人→五seed。验证曲线不替代上述测试终点，辅助2AFC仅描述。原内部主M6/7、五family探索M5、历史诊断和外部比较各自解释，不主张跨所有表的联合FWER。

实际数据和计算继续仅在2203，Windows/Git仅保存代码、协议、轻量结构化汇总、静态图和记录；不回传raw、权重、预测数组或交互载荷。SS07和持续goal均ACTIVE，父任务及其他支线不因此启动或完成。
