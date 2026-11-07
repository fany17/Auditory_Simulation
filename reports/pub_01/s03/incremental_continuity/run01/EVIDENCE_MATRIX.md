# SS07 round01 实际证据矩阵（2026-11-01）

预设实验、实际评分与独立复核已完成。当前交付PASS_WITH_LIMITATION，超越原强基线、机制和投稿充分性未建立，SS07及持续goal仍ACTIVE。完整数值及九图解释见[REPORT](REPORT.md)，三层验收见[REVIEW](REVIEW.md)。下方结果前版本保留为历史，待办不再代表当前状态。

| 问题 | 实际证据 | 当前判断与限制 |
|---|---|---|
| 强基线与联合训练控制 | 五native完整两阶段、五FT各5epoch完成；实际最终权重VAL桥与评分复核通过 | 原可用条件复跑成立；FT内部点值最高、DTU低于P，不称跨域稳定改善/全局收敛 |
| 历史是否超越P及匹配控制 | 五seed M4−P内部−.000621、DTU+.000149，M7近似区间均跨0；current对比也跨0 | 未建立历史额外贡献、等价或非劣；内部static capacity/静态对齐/FT相对M4的优势区间不支持M4 |
| 历史机制 | 六干预实际齐备，各自同key配对；仅content-blind未校正95%为正 | 不构成六项校正机制证据。错family覆盖119records/1597blocks，其余144/2044；shuffle可含未来，older改变recency/parity |
| 共享来源与外部 | 17来源family，逐人/record CSV、LOFO与Exp敏感性齐备；DTU18人1080trial五seed八main完成 | family_only均缺一stratum，原终点null，不重赋权；Exp非CI；旧内部测试探索性，DTU无目标域拟合但不证明刺激/房间独立 |
| 实际独立复核 | raw360项最大波形差1.82e−12；metric v2共583129检查、165373内部块、43200外部trial配对；综合V2 PASS | 代表性raw复放复用冻结NN forward，不是全量独立神经数学/history干预raw NN重放。R²缩放门在测试前合法VAL修订 |
| 新意、初始化与科学目标 | 九份一手来源比较；14/15 best3s低P，M4/31415的3s与60s排序不一致；五FT测试前VAL改善保留 | 所有候选从同随机head联合训练，未检验已训练P head接历史；失败机制未分离，新意/稳健增益/投稿充分性未建立 |

全部106成本条目含54screen+5pretrain+5head+5FT+37唯一full；五方法family与17来源family不可混用。各比较族分别解释，不主张跨表联合FWER。当前只拟定下一步train/val草案，无新fit；旧内部测试和已暴露DTU不重新命名为新的确认性证据。患者/STN FORBIDDEN，信号/权重/预测只在2203。

---

## 历史：结果前投稿证据矩阵

当前为研究执行与缺口索引，**不是投稿就绪、性能胜出或科学验收报告**。以实际服务器产物为权威；下列待完成项不能由工程测试、候选验证集改善或单个seed替代。所有当前实验仍按已有冻结协议执行，本表不新增实验、指标、选择规则、成功阈值或审批门。

| 要回答的问题 | 已有直接证据 | 尚待的实际证据/权威产物 | 当前判断与外推边界 |
|---|---|---|---|
| 原强模型是否被完整复跑？ | 五个native预训练与head均完成；预训练按pat15，head均250上限；最终manifest已核验各自checkpoint/epoch/同seed来源 | 统一评分和实际raw/metric复核 | 原可用数据/资源协议的完整复跑成立；不是官方全训练人数、500epoch或ensemble结果，不称head收敛 |
| 是否有强于仅冻结readout的控制？ | 五个原encoder+head微调均完成5epoch/19725更新；真实incumbent在更新前核验；FINETUNE_BUDGET.csv与曲线 | 统一原weighted60s/person测试与DTU同输入适配结果；final_summary中的P_FT比较 | 强控制已实际训练，测试是否改善尚未证明；全部FT最佳在预算末轮，不称已收敛或最优微调 |
| 连续历史能否在相同当前输入权限下超过完成预设预算的原强基线？ | 全54筛选及37full完成；原三seed VAL规则冻结M4，固定五seed/四控制全部训练完成；47 checkpoint角色及实际最终权重10VAL桥已核验 | 统一推理已启动；INTERNAL_MAIN与EXTERNAL_DTU_MAIN及实际复核 | 初始五family三seed完整VAL均值/M4逐seed低P的证据保留；五seed M4与current均值接近且低于其他三控制，仅验证描述，不作测试胜负或重选 |
| 增益能否由容量、当前条件映射、输出历史/平滑解释？ | 五seed current/static/output_prior/static_alignment和FT均训练完成，来源/预算核验；EMA原范围选tau0，保留重复保守对照 | 统一frozen评分、active参数和COSTS；预设M7差值与区间 | 未证明额外历史贡献；匹配max/patience规则不等于实际更新数相同；registered参数不等于active参数；共享encoder成本只计一次 |
| 历史是否有内容与时间特异贡献？ | 非重叠双parity历史支持检查通过；reset/content_blind/shuffle/older/两类错配的数学与固定metadata计划完成 | 六干预全部五seed输出；各自eligible keys上true配对的INTERNAL_HISTORY_DIAGNOSTICS | 工程时间支持不证明历史有用；shuffle可含未来观察、older同时改recency/parity，均仅离线分布诊断，不能直接称机制或因果证据 |
| 共享刺激与受试者依赖是否被如实处理？ | 原2/3seen+1/3heldout、跨seed/person共用抽样、共享family Exp敏感性、per-family/LOFO覆盖规则均冻结 | INTERNAL_MAIN、逐人/逐record/story/family CSV、LOFO与敏感性；同key不丢seed核验 | 原Sparr测试均已暴露，仅POST-HOC INTERNAL EXPLORATORY；近似bootstrap非有限样本覆盖保证，Exp区间不是校准CI |
| 外部泛化是否成立？ | DTU18人1080trial、唯一当前3s适配及白名单冻结；TARGET_MASK/READY已在预测前冻结，既定统一推理启动 | 固定全部主模型×五seed×1080trial的EXTERNAL_DTU_MAIN；独立raw/metrics review | 可检验该外部数据的固定适配迁移，不证明刺激/房间独立泛化；不得改适配、排除S11或据外部成绩选择模型 |
| 是否排除了指标/模型角色/批次重放错误？ | v2数学38核对、synthetic caller77482核对、3760合法VAL的R²数值门检查及真实P/M2/FT1024合法VAL重放；18source/8proof冻结，唯一review队列233443 | 全部实际评估结束后的RAW_INPUT_REVIEW与METRICS_REVIEW，以及INDEPENDENT_REVIEW_COMPLETED_V2；协调读取实际轻量结果 | 现有PASS为工程/合法VAL证据；实际review仍待执行。raw仅固定3条Sparr+6条DTU×5seed×全部main；复用冻结NN forward，不冒充全量独立神经数学 |
| 是否有足以支持投稿的方法新意与证据强度？ | SCIENTIFIC_DESIGN_REVIEW及RELATED_WORK_COMPARISON逐项核验9份一手来源与实际代码；明确output_prior非DECAF复现、M3非CPC、M4非物理延迟、M5无Kalman校准等边界 | 上述完整效果/控制/泛化结果、实际失败范围、完整竞品同协议直接比较的缺口、独立科学接受判断及投稿缺口决策 | 新意尚未成立。小幅单数据集r改善不等于ICML/NeurIPS证据，也不因技术PASS、某seed阳性或论文风格包装而完成goal |

指标必须与结论范围一致：3s验证准则用于训练选择；主结论用原60s有效块→人→seen/heldout原权重；外部以trial attended r→人→五seed。验证曲线不替代上述测试终点，辅助2AFC仅描述。原内部主M6/7、五family探索M5、历史诊断和外部比较各自解释，不主张跨所有表的联合FWER。

初始化/失败归因边界见[FAILURE_MECHANISM_CHECK](FAILURE_MECHANISM_CHECK.md)及15行[配对metadata](INITIAL15_PAIRED_VALIDATION_METADATA.csv)：同seed共享本轮encoder，P与五family同随机head起点，候选联合训练head/残差；未检验从训完P head接入历史。14/15的3s VAL也较低，M4/31415却3s较高而60s较低；信息贡献、初始化/优化与目标/连续评分差异尚无法区分。在线train曲线不当同checkpoint泛化误差，阴性不单独证明过拟合或历史信息不存在。

实际数据和计算继续仅在2203，Windows/Git仅保存代码、协议、轻量结构化汇总、静态图和记录；不回传raw、权重、预测数组或交互载荷。SS07和持续goal均ACTIVE，父任务及其他支线不因此启动或完成。
