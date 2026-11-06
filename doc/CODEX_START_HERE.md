# 当前入口：D24 三折训练来源对照已封存

## 当前执行入口

D24 三折训练来源对照已封存。当前节点为 PUB-01-S03_SS07，仍为 ACTIVE。先从 [CURRENT_TASK](CURRENT_TASK.md) 查看 D24 的完整开发结果与验收，再读取协调决定。

2026-11-06。D23 S 的两分支只用当前冻结预测，三 seed 各 20 pass 完成。S 的全部 60 个训练轮次高于锚点，选中均值 0.183899636。D22 B 的各自最佳在三 seed 高于 S，平均 +0.000681269；S 在 52/60 个同轮和三个末轮高于 B。B 的峰值差来自纯活动块，S/B 的输入维数和初始化仍限制历史机制解释。完整轨迹和独立审阅已归档，SS07 总体目标仍 ACTIVE。[D23 报告](../reports/pub_01/s03/incremental_continuity/memoryless_output/run01/REPORT.md) · [归档](../reports/pub_01/s03/incremental_continuity/memoryless_output/run01/ARCHIVE.md)

# 历史入口：D22 输出端非线性对照已完成

2026-11-06。D22 B 只接收当前冻结预测标量及 D20 同一紧邻预测历史。三 seed 各 20 pass，均选 epoch 0，开发 r 均值 0.184580904，较锚点 +0.002240965、较 D20 +0.000490394，三 seed 同向。约 86% 的 B−D20 均值差来自纯活动块。三组末轮均低于锚点，选中活动 MSE 高于锚点和 D20。D18 P 开发均值略低，逐 seed 仅 31415 高于 D18 P。独立复核和完整归档已交付。SS07 科研目标仍 ACTIVE。[D22 报告](../reports/pub_01/s03/incremental_continuity/output_query/run01/REPORT.md) · [归档](../reports/pub_01/s03/incremental_continuity/output_query/run01/ARCHIVE.md)

# 历史入口：D21 输出端仿射与 FIR 对照已完成

2026-11-06。D21 的 18 个精确岭解全部完成，未运行原 EEG encoder 或任何新 NN。仿射 A 与 FIR F 三 seed 均选 λ=0.0001，均值 0.182570113 / 0.182905028，F 三 seed 均低于 D20，均差 −0.001185483。A 的小幅 r 增益位于锚点前缀衔接块。独立终验、两张实际系数与成绩图及完整网格已交付，SS07 总体目标仍 ACTIVE。[D21 报告](../reports/pub_01/s03/incremental_continuity/output_ridge/run01/REPORT.md) · [D21 归档](../reports/pub_01/s03/incremental_continuity/output_ridge/run01/ARCHIVE.md)

# 历史入口：D20 紧邻预测历史已完成

2026-11-06。D20 只将 D19 P 的历史 lag 从 288 点改为 1 点。三组各 20 pass、86,820 更新已完成，选择 epoch 0/1/0。开发 VAL r 均值 0.184090511，高于冻结 FT+Ccap 0.001750571，高于 D19 P 最佳已训练均值 0.002040659，低于 D18 P 的 0.184379162。末轮均低于锚点。固定模型的近远替换均下降，原始支持重叠和分布变化仍限制解释。独立复核通过，报告已归档。SS07 总体目标仍 ACTIVE。[D20 报告](../reports/pub_01/s03/incremental_continuity/recent_prediction_context/run01/REPORT.md) · [D20 归档](../reports/pub_01/s03/incremental_continuity/recent_prediction_context/run01/ARCHIVE.md) · [DECAF 公开实现审计](../reports/pub_01/s03/incremental_continuity/metadata_review/code_review/decaf_protocol_20261106/REPORT.md)

# 历史入口：D19 完整检验与归档已完成

2026-11-06，PUB-01-S03_SS07。D18 的 12 个正式 fit、科学审阅和归档均完成。H 对匹配当前 C 的开发 VAL r 三 seed 平均差为 −0.000004008，H 三个 seed 都低于 P。[D18 报告](../reports/pub_01/s03/incremental_continuity/preprojection_memory/run01/REPORT.md)

D19 每物理点只输出一个可训练残差。H/P 使用完整 6 秒历史与 96 个有序四点箱，C 使用当前点的正 Hann 贡献窗。九组各 20 轮、260,460 更新均完成。180 个已训练轮次均低于冻结 FT+Ccap，九组正式选点全部回退。H 的最佳已训练 r 三 seed 均低于 P 和锚点。独立保存证据与实际末轮有限重放通过，完整四图中文 HTML 已归档。历史收益门未过，SS07 总体目标仍 ACTIVE。[D19 报告](../reports/pub_01/s03/incremental_continuity/physical_stream/run01/REPORT.md) · [归档](../reports/pub_01/s03/incremental_continuity/physical_stream/run01/ARCHIVE.md)

# 历史入口：实现审计已交，projection前memory四臂源/协议准备

D17封存后的有限实现审计已交：reports/pub_01/s03/incremental_continuity/implementation_path_audit/run01/REPORT.md，25函数/类SOURCE_INDEX与AUDIT_SCOPE明确0数组/构模/NN/fit。核实warm18fit已联合训练原encoder，不能重复声称首次end-to-end；新缺口限定64→8前192×64逐时间token/query-dependent memory。D12–D16仅冻结FT缓存，D9近null输出抵消不能靠feature fusion自动消除。DECAF作者README实际指carankt，所读teacher/eval入口与本项目output_prior不同，SHINE尚未代码发布，均未运行作者入口。协调下一已选H/C/T/P四臂×3seed、共同FT+Ccap(.01)/可微固定readout、当前Eθ/head可训/历史E0eval冻结、zero-output memory、near5/3/older9/7/i>=9、首9冻结anchor、D16共同目标shuffle与60s选点；拟20轮/24h/两GPU各一fit。现在只进入实际源/协议/必要工程准备，正式fits须具体冻结/实际工程与源审阅通过；不重开旧外部，D9坐标替代不混入本候选。

# 当前入口：D17局部诊断完整封存，下一实质方法待具体界定

2026-11-02 D17固定TRAIN历史敏感性诊断已收束：四既有H检查点、单sub-001/8story/48有效窗，384残差NN窗/8输入backward/0fit一次完成。selected2026 near/older输入差RMS1.358、LN后.09491、末h6.396e−6、实际gain差0、最终预测差9.392e−9；三个last19最终预测差.033526/.019122/.047253，不能统称所有模型忽略历史。固定输出投影输入梯度均非零，非有用信息或收益证明。协调0NN标量一致性1325检查PASS、完整中文REPORT/REVIEW科学审阅通过；仅源码/metadata/标量算术，不称神经/激活独立重放。完整HTML run_id20261102_pub01_s03_ss07_history_sensitivity含已审全文/3可读表，34manifest产物、56节点/4cp/8梯度CSV、10限制与链接/source_bundle/receipt绑定交付核验PASS。Technical PASS_WITH_LIMITATIONS / 局部诊断Acceptance PASS_WITH_LIMITATION / Parent PASS。D16及更早阴性保持冻结，历史收益/独立泛化与创新目标未建立，SS07/科研goal仍ACTIVE；下一实质方法由协调依据目标与已有证据具体界定，不自动在梯度小修线上循环、新fit/改LN或LR/旧test及外部重开。

# 当前入口：D16完整封存，下一D17仅固定TRAIN诊断准备

2026-11-02 D16九fit/20epoch、710100更新及59473248正式残差NN窗全部完成；816工程与631239独立另列，总60105303NN。独立48624比较、180排列/梯度日志、45离线TRAIN聚合通过，五PNG/完整中文全文与账本经协调实际科学审阅，回执绑定最终REPORT/REVIEW。完整HTML run_id20261102_pub01_s03_ss07_token_normalization交付核验PASS：170清单产物、5图/5说明/10限制可见、全部HTML/全文链接与最新source_bundle通过，无信号/权重/数组。H抽样通路活动改善和TRAIN拟合增强，但仅2026小增且低于P、另两H回锚；last VAL三臂均低于D15/overlap增大，历史收益门FALSE。Technical PASS_WITH_LIMITATIONS / 诊断Acceptance PASS_WITH_LIMITATION / Parent PASS；goal/SS07仍ACTIVE。D16已收束，不重跑。协调下一D17仅原固定TRAIN48有效窗、四既有H检查点的near/older历史敏感性诊断源/身份/预算准备，待其具体源码审阅后执行；0新fit，不改LN位置/LR或重开外部。

# 当前入口：D16全九完成、独立数值通过，图文归档收束中

2026-11-02 D16九fit×20epoch/710100更新全部成功，正式59473248残差NN窗；816工程、631239独立另列，总60105303NN。独立手写normalization/GRU/Hann48624比较、180排列/梯度日志、45离线TRAIN聚合通过，max r误差2.3282e−7/其他4.3531e−6。H首批epoch1…19 token/GRU均非零、post token RMS约1，抽样活动退化缓解；TRAIN r均值H/C/P=.262342/.260453/.257760改善。H仅2026 epoch5小增+.000116401且低于P .000130032，其他H回锚；非零H near−older约−1.43e−10处于数值误差尺度。三臂last VAL均值低于D15、overlap增大，历史收益门FALSE。五实际PNG已目视并写完整中文REPORT/REVIEW，协调全文审阅与HTML交付检查进行中；goal/SS07保持ACTIVE，不自启新fit/改LN位置/LR或重开外部。

# 当前入口：D16单一token归一化九fit正式执行中

2026-11-02 D16已获协调实际六源/221pins/62输入/816NN工程与独立逐点桥审阅批准，协议56f4be82...，正式queue1832753/UID1012/ticks755734179已启动，九worker按5+4全部running且真实epoch4–5/19725–23670更新确认，无失败。首批detached记录post token RMS约1，GRU梯度非零，但部分状态仍近饱和；仅通路观察不作收益判定。独立依赖runner1834387已由协调部署，九组成功后自动一次≤631239NN，PUB不重复启动。正式59473248NN/710100更新不变；继续全九/图文/账本与独立验收，不按中途结果改source/预算或启其他模型/数据。

# 当前入口：D16单一token归一化v2工程通过，待协调执行审阅

2026-11-02 D16单一token末端GELU后/GRU前无仿射LayerNorm32 eps1e-5已具备v2冻结56f4be8214ec7b5402b1c6ea3e4ff2154f082150b95c30538743bd1fa170b3a2。首528NN必要工程PASS：CPU/GPU公式误差4.77e-7/2.38e-7，9初始化state/参数精确一致，轴/RNG/常量/fresh有效梯度/日志开关等价；后到独立手写参考桥GPU144+manual144逐点≤1.066e-6 PASS，累计816/0fit/optimizer。主四源未改，v1所有source/protocol/工程保留，v2仅绑定桥/证据及补充PLAN。正式预算仍59473248NN/710100更新/9fit，两GPU5+4/6h/160MiB含28在途，正式0更新，待协调绑定实际v2源/工程一次执行review后立即启动；D15及所有旧外部保持冻结。

# 当前入口：D15完整封存，D16单一token归一化对照准备中

2026-11-02 D15全部九fit/20epoch完成，正式59473248残差NN窗/710100更新；H/P三seed全选−1，C仅2026 epoch2微增1.23723e-5；H收益门FALSE。训练负cosine改善至约−.247，windowMSE约.1959较D14.1941差，TRAIN物理r约.2564–.2568无H特异稳定优势；late首批token/GRU仍零或极小，不称已解决退化。独立manual GRU/Hann631239窗、48624比较/180排列/180梯度日志/45聚合通过；480工程及0NN存储/guard成本分列，总已知60104967NN。协调实际全文/四PNG/成本科学审阅通过并绑定REPORT/REVIEW SHA；152产物/4图说明/9限制和真实HTML/全文链接、最新source_bundle版本核验通过。完整HTML run_id20261102_pub01_s03_ss07_objective_control；Technical PASS_WITH_LIMITATIONS / 诊断Acceptance PASS_WITH_LIMITATION / Parent PASS，整体goal ACTIVE。D15已冻结不重启。协调具体D16决定已写token_normalization/run01/COORDINATOR_DECISION.md：仅token末端GELU后/GRU前无仿射LayerNorm32/eps1e-5，全三臂、无新参数；下一仅源/工程/实际冻结准备后审阅，不自动扩候选/外部/患者/PR合并。

# 当前入口：D15固定目标对照九fit按5+4正式并发执行中

2026-11-02 D15 v2经协调实际六源/123pins/63input与480NN工程审阅批准（review2e29bcb6...，protocolb80f9aef...），正式queue1769088已启动；九worker按cuda0五/cuda1四全部登记PID/UID/startticks与实际资源，均已完成epoch0/3945更新并进入epoch1，首batch已有梯度日志可读。原fresh零头上游0为预期；epoch1九组GRU日志均存在非零值，仅活动诊断，不作为选点。正式59473248残差NN窗/710100更新/9fit，原矩阵和NN预算不变；独立review由协调结果后运行一次。当前尚无完整结果/新科学结论，无FAILURE，不改冻结源，不重启旧外部或患者，SS07 ACTIVE。

# 当前入口：D15 v2工程通过，固定目标对照待协调执行审阅

2026-11-02 D15唯一float64负cosine目标对照已具备v2冻结协议b80f9aeffada112c012ce23f944279027d4219250fdf2999a28c4878d336506f。原288NN数值工程/target12个TRAIN eps窗、无常量，追加192NN fresh非零输出通路probe通过；旧late H/C该批token/GRU饱和与零/近零输入梯度保留为限制，不调LR/结构。IO竞态修复0NN mock、28MiB在途reserve及epoch首batch既有梯度日志均无额外forward/update，数学函数AST同v1。累计工程480NN/0fit/opt；v1所有源/协议/证据保留。协调仅将新root存储上限160MiB并批准最多9jobs/两GPU5+4，仍6h/8GiB RSS/2GiB allocated/两盘20GiB；正式59473248NN/710100更新不变。正式仍0，待协调绑定实际v2源/工程执行review后启动，旧D14及外部冻结。

# 当前入口：D14顺序对照已封存，D15唯一目标对照准备中

2026-11-02 D14九fit×20epoch全部完成，710100更新/61710828正式残差NN窗；成功工程1230、失败路径362、独立631239另计，总已知62343659，0新encoder/head。TRAIN固定last windowMSE H/C/P=.194114/.194091/.194101，较旧.202329/.202466/.202579及anchor.196161均改善；TRAIN r=.257599/.256791/.256759。VAL selected H/C/P=.182404586/.182339940/.182468347，H仅2026微增、H-P混合，near/older极小差处于实现数值误差尺度，历史收益门FALSE。独立48624比较/180排列/54TRAIN聚合/18旧对照通过，额外CPU72.036s；四图/完整中文全文协调审阅、HTML162产物/7限制/链接版本交付检查通过。Technical PASS_WITH_LIMITATIONS / 诊断Acceptance PASS_WITH_LIMITATION / Parent PASS；run_id20261102_pub01_s03_ss07_order_control。科研goal ACTIVE，D14已封存不重启。协调授权下一D15仅目标对照的PLAN/source/必要工程准备：沿D14三臂/三seed/初始化/排列/20epoch/Adam预算，仅float64 p/y centered negative cosine eps1e-6，正式拟合仍待其实际冻结源审阅。旧test/DTU/ds004408/D12/D13冻结，患者FORBIDDEN，PR6 Draft不合并。

# 当前入口：D14固定顺序对照正式九fit执行中

2026-11-02 D14已获协调实际源/协议/工程独立审阅（COORDINATOR_EXECUTION_REVIEW approved=true），正式queue1692090已启动，首worker1692186(H1024/cuda0)/1692194(C1024/cuda1)均空闲GPU准入，真实epoch3/4及15780/19725更新已确认。协议83b05eaf...与源码f5134d7不变，正式9×20epoch/710100更新/61710828残差NN窗；后续协调独立数值复核最多631239另计，不加新fit/encoder。当前结果未齐，不下最终科学结论；D12/D13和旧外部冻结，不重启旧PID。

# 当前入口：D14工程通过，固定顺序对照待协调执行复核

2026-11-02 D14仅顺序对照已冻结f5134d7，协议83b05eafcb126952d48750930106b5fb04c56d49f82d2b0128ba7d2c813d0172；engineering02 PASS：1230残差NN窗/0opt/fit/encoder，混record重排loss差0、参数grad≤3.49e-10，60排列双射/非空可学习batch及原record-local past通过。attempt01 eval-GRU backward失败source/protocol/log与映射原样保留，另362路径核算NN窗、资源峰未知。正式九fit尚0，待协调实际执行review后直接按原20epoch/710100更新/61710828残差NN窗运行，不再用户重复许可。源PLAN见 reports/pub_01/s03/incremental_continuity/order_control/run01/PLAN.md。旧D12/D13及外部冻结，SS07 ACTIVE。

# 当前入口：D13固定误差诊断已封存，D14最小顺序对照准备中

2026-11-02 D13已完成固定九last19误差分解：2508111残差NN窗、0新encoder/head/fit/update，61704主块投影；21375辅助行中21240投影已定义、135不足点保留。VAL MSE改善来自bias而r下降；TRAIN物理MSE升高、r下降，三对输出梯度在总体/record/person均正。独立真实TRAIN autograd3834record cos差5.55e-16、全块恒等及逐人/phase聚合通过，四图全文审阅与HTML归档交付检查通过；Technical PASS_WITH_LIMITATIONS / 诊断Acceptance PASS_WITH_LIMITATION / Parent PASS。完整报告 reports/pub_01/s03/incremental_continuity/error_decomposition/run01/REPORT.md；run_id20261102_pub01_s03_ss07_error_decomposition。D12/D9和旧test/DTU/ds004408冻结。协调已授权D14唯一逐epoch全窗口顺序对照的最小方案/源准备与审阅，尚未新训练；不换loss/LR/容量，SS07/科研goal保持ACTIVE。

# 当前入口：完整FT历史条件器九fit已封存，收益门FALSE，持续goal未达到

[完整结果](../reports/pub_01/s03/incremental_continuity/full_ft_history/run01/REPORT.md) / [独立验收](../reports/pub_01/s03/incremental_continuity/full_ft_history/run01/REVIEW.md) / [5PNG中文HTML归档](../reports/pub_01/s03/incremental_continuity/full_ft_history/run01/ARCHIVE.md)。9×20epoch/710100更新完成，全部selected−1；最佳已训练H仍低于强锚点.0063–.0079。技术及图文审阅通过不等于方法改进；MSE改善与r下降共存。见CURRENT_TASK顶部，旧PID均结束，不重启旧入口。当前不自动新fit/数据/模型/预算/患者/PR合并，后续问题由协调另行界定；整体SS07 ACTIVE。下方各启动入口仅历史。
# 当前入口：SS07完整FT条件历史九fit正式运行

先读[CURRENT_TASK](CURRENT_TASK.md)和[FULL_FT_HISTORY_STATUS](../reports/pub_01/s03/incremental_continuity/research_gap/run01/FULL_FT_HISTORY_STATUS.md)。协调审定单结构/三seed/九fit各20轮，正式queue1585992已启动；工程/初始Ccap/梯度/严格过去/输入身份检查通过，源commit e62a70b。当前结果尚待，不能用工程PASS完成goal；不重启旧PID、旧test/DTU/ds004408，不追加模型/预算或碰患者。下方“待审定/只读”属于先前快照。
# 当前入口：SS07持续goal，原TRAIN/VAL研究缺口盘点与单候选待协调审定

先读[CURRENT_TASK](CURRENT_TASK.md)、[只读盘点](../reports/pub_01/s03/incremental_continuity/research_gap/run01/REPORT.md)、[具体候选](../reports/pub_01/s03/incremental_continuity/research_gap/run01/CANDIDATE_PLAN.md)。盘点新fit/NN/数组值读取0；完整FT时序条件历史尚未实际测试，候选未冻结未运行。协调可在既有授权内审定原TRAIN/VAL执行，当前不自动启动拟合；旧test/DTU/ds004408已看结果冻结，不再用于选模型/新适配或评分。“无自动新实验”不终止原持续goal。以下外部阶段已完成，科学结论和限制原样保留。
# 当前入口：SS07 ds004408固定读出迁移已完成，独立数值复核通过

Ccap强FT基线r=.164518/.156477，Δ+.008041、Holm2 p5.34e−5，3seed/18人正；F−W区间跨0、F单窗退化/MSE更差保留。完整380条、19人、988块，独立4560记录/11856块/2280FP32复算通过。见[CURRENT_TASK](CURRENT_TASK.md)、[结果报告](../reports/pub_01/s03/incremental_continuity/independent_confirmation/run01/report01/REPORT.md)、[阶段REVIEW](../reports/pub_01/s03/incremental_continuity/independent_confirmation/run01/REVIEW.md)。本阶段有限制通过，不等于连续历史机制/投稿充分性或SS07 goal完成；旧作业已结束，勿重启旧PID。当前无自动下游/新模型/新数据，患者FORBIDDEN，PR6仍Draft。最终全文/图表/表与成本独立审阅已通过；13PNG完整中文HTML归档及链接/限制可见性核验完成（run_id20261102_pub01_s03_ss07_openneuro_gain_transfer）。

## 下方是历史预处理/等待入口，不是当前运行状态
# 当前入口：SS07全380条预处理与最终freeze完成，等待协调准入

实际19人/380条/988块/45429窗，failures=[]；16条复用+364新增、380次无标签gain统计。最终协议SHA `5cb28c2cd219c25fc36ed5ea91d92572a9c6b61b0a3ae864a0f1fa5180cc76c5`；queue_v2 PID1445704等待协调独立绑定protocol及queue SHA，尚未新NN。最终结果呈现用report_v3/PRESENTATION_PROTOCOL_V3（统一示例纵轴、完整成本字段），旧版本保留。以下“预处理进行”及旧PID为历史状态，不重启。完整记录见CURRENT_TASK顶部与PREPROCESSING_REPORT。

唯一后续推理为evaluate_v3/queue_v2 PID1445704；旧waiting队列已核验终止留痕。数据预处理1395718继续；未生成Z/P cache放同2203的/sata5自有目录，两盘各20GiB及两root合计3GiB/全SS07 NPY28GiB不变，不用19GiB备选。最终实际freeze及协调绑定新protocol/queue SHA前，真实新NN仍0。所有当前路径/证据见CURRENT_TASK顶部，不能按下方旧PID重启。

当前传输v2 PID1395718（8网络/最多2路DSP/BLAS1）复用原16条完整输出；原v1已核验终止，source/protocol/log保留。19历史模型资产及54历史关联、读出/统计检查已协调预审。queue PID1426629仅等待全量380→最终实际freeze→协调绑定protocol/queue SHA审阅，之后才运行3seed冻结四角色一次推理；当前真实NN/新学习模型fit0，每record无标签gain统计另列。最新实际入口和证据见CURRENT_TASK顶部。

当前唯一共享适配已按[AMENDMENT](../reports/pub_01/s03/incremental_continuity/independent_confirmation/run01/AMENDMENT_GAIN_CALIBRATION.md)改为预测前无标签全record整体gain，物理单位仍UNVERIFIED；offline/transductive，不称纯零校准确认。实际canary/数学/常量/paired block支持检查通过，380条内存获取/校验/预处理PID1379762进行。20音频处理和[四图中文概览](../reports/pub_01/s03/incremental_continuity/independent_confirmation/run01/PREPROCESSING_REPORT.md)完成；最终实际mask/模型/统计freeze和一次协调审阅前不预测。当前细节以CURRENT_TASK顶部为准，以下早先单位等待为历史。

2026-11-02真人授权指定ds004408 v1.0.8范围，见[授权原文](../reports/pub_01/s03/incremental_continuity/metadata_review/OPENNEURO_SCOPE_AUTHORIZATION_20261102.json)。[共享适配与统计PLAN](../reports/pub_01/s03/incremental_continuity/independent_confirmation/run01/PLAN.md)已协调设计审阅通过；760header/119坏道表、20WAV+首EEG已完整核验，固定音频包络处理进行。首EEG按header与MNE一致但幅度异常，单位provenance未解决前不预测、不按成绩或TRAIN std配缩放。无单位依赖的描述性图、mask和统计实现继续。唯一现行路由为[CURRENT_TASK](CURRENT_TASK.md)顶部；以下“等待数据授权/禁止新载荷”是历史快照，不覆盖此新明确授权。D11及以前保持冻结，SS07目标未达到，不合并PR6/不碰患者。

## 历史：D11完整阴性结果已独立核验；三图归档完成

D11 [REPORT](../reports/pub_01/s03/incremental_continuity/observation_state_probe/run01/REPORT.md) / [REVIEW](../reports/pub_01/s03/incremental_continuity/observation_state_probe/run01/REVIEW.md)：固定15fit/6干预/24评分完成。Past r=.182063<Ccap .182340，−Prefix/Person/Older/Neutral及Add/F各三seed负；Past MSE改善保留。六条件不满足，不支持额外历史机制。独立26/551/475/15/60/48/24核验和全文三PNG实际审阅通过，PCA缩放措辞修正；三图HTML实际归档及说明/副本/链接通过，goal ACTIVE。原TRAIN/VAL以外不新拟合/旧test/DTU/新载荷，具体最新记录见CURRENT_TASK顶部。

以下为历史结果与执行记录。

D10完整[REPORT](../reports/pub_01/s03/incremental_continuity/finetuned_internal_audit/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/finetuned_internal_audit/run01/REVIEW.md)与三图HTML已实际review/归档，36聚合/72缓存通过，旧FT raw桥全0。PFT/Ccap/F加权r=.168945/.173348/.174284，内部点增益和F单窗退化并存；旧test暴露标签不变，无独立确认/历史/创新，D4-D10均冻结退出。D11精简PLAN/源码/工程门已独立通过，source5c50719、PROTOCOL a238ed63…b5f11，PID656537实际运行15固定岭解+6无拟合干预；三个原Ccap完整VAL桥通过。仅原TRAIN/VAL，i>=41/当前与静态控制/全部评分mask冻结，NN新增前向与优化0；旧test/DTU关闭，结果和独立数值复核尚待。OpenNeuro新scope仍待用户明确答复，不取payload；goal ACTIVE，CURRENT_TASK顶部为最新入口。

D6[REPORT](../reports/pub_01/s03/incremental_continuity/readout_attribution/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/readout_attribution/run01/REVIEW.md)实际独立review与中文HTML归档已完成有限制。当前只读盘点/准备D7冻结36角色-split方案，原3seed/已选λ/强current控制、仅既有seen/heldout，全部POST-HOC INTERNAL EXPLORATORY；先交协调审定，尚不测试/新拟合，不碰DTU/下载。旧D4-D6不按新结果重开，最新路由CURRENT_TASK顶部。

D6[REPORT](../reports/pub_01/s03/incremental_continuity/readout_attribution/run01/REPORT.md)已实际交付27新解+18精确复用与三图；P-only已有current再读出增益，完整latent映射相对同local U32稳定正向，低overlap不是准确度保证。结果review/归档收束中，暂不新拟合，不扩旧矩阵/下载/test/DTU。最新路由CURRENT_TASK顶部；以下准备记录为历史。

D5实际36解与独立review已完成有限制并归档，[REPORT](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/REVIEW.md)，未建立历史交互增益。当前仅准备最小shared local32 / unshared同local32 / dense C / Ccap归因方案供协调审定，原P*残差/MSE/精确D4复用/连续性次要指标，尚不拟合。旧D4/D5网格不扩大，无新NN/test/DTU/下载，goal ACTIVE。以下启动快照为历史。

D5已审定并冻结cdb970b，PID525296在2203执行[PLAN](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/PLAN.md)固定36解，锚点D4强current Ccap，全部learned共有加性A，CC/near/older交互与Add对照。仅CPU、原TRAIN/VAL，实际结果仍待，源码静态review并行；无新NN/encoder/head前向或test/DTU。下段D4交付历史保留，当前路由见CURRENT_TASK顶部。

以[CURRENT_TASK](CURRENT_TASK.md)顶部为唯一现行路由。D4固定54解已实际完成、独立审阅有限制通过并归档，[REPORT](../reports/pub_01/s03/incremental_continuity/conditional_history_probe/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/conditional_history_probe/run01/REVIEW.md)。强current Ccap VAL=.176897，完整历史=.174090，未建立额外历史收益。当前只准备另一个最小历史×当前映射交互设计，保留强current锚点，先交协调审定；不扩大已完成D4网格。只原TRAIN/VAL，无新NN架构/encoder forward/数据/test/DTU推理；不合并PR#6，不启动患者或其他支线。goal保持ACTIVE。

报告：[REPORT](../reports/pub_01/s03/incremental_continuity/run01/REPORT.md) / [REVIEW](../reports/pub_01/s03/incremental_continuity/run01/REVIEW.md)。SS07及持续goal保持ACTIVE。

---

## 历史：SS07 round01执行过程

用户最新明确授权尽快全量实验、设goal、使用2203全部可用算力，研究目标超过原baseline并积累ICML/NeurIPS级证据。SS07 ACTIVE；旧STOP/单pass/NO_GO不覆盖本授权，旧结果保持只读。先执行PUB-01-S03_SS07_FULL_SCALE_INCREMENTAL_CONTINUITY.md及本轮INTAKE/BASELINE_PROTOCOL，原完整Transformer/XLSR19两阶段、fold完整train/val、100epoch/pat15预训练与250epoch/pat10head至少3seed；仅在强模型上增加五类有界残差，精确矩阵协调审阅。开发只train/val，旧test均内部暴露，未冻全计划不评估。无阳性/论文接收保证。

唯一计算2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101；本地reports/pub_01/s03/incremental_continuity/run01。新cache≤25GB、余量≥20GB、best/latest有界，不删历史/不动他人进程。沿用SS06 worktree/branch/PR6 Draft，可commit/push、不合并，不动主目录CNN/TRF/stash，无患者/远端Git/擅自新数据。持续goal不以smoke、技术PASS或单seed分数完成。

2026-11-01执行接续：五native完整两阶段及五FT已完成；54/54筛选齐备、五类均H16，初始15full已全部完成，原三seed VAL规则冻结m4_h16（FULL_PRIMARY_SELECTION d43da5b521af0cabfed6a2a6fc4ce968ae39f20824411d07c305fb1fa08b69d2）。五family验证均值及M4三个seed均低于原P；保留阴性证据，不改矩阵。现37/37full全部完成（主比较25+其余family12），协调独立核验全部预算/选点/来源通过，14537正常结束。FINAL_CHECKPOINT_MANIFEST锁定47角色/106成本条目；五seed×2VAL实际最终权重bridge差均0，EVALUATION_READY已在预测前冻结。原source4e33d69、队列88448已自动启动两卡统一推理；primary/M7/五seed保持原冻结。下一步等待完整summary与实际raw v1/metric v2，实际图表使用v2入口，goal仍ACTIVE。

独立复核现用v2：source1153dbf、INDEPENDENT_REVIEW_PROTOCOL_V2（97c7686b94d63836c59d83df561887d926d2d65a3528ea004ddf86cd1d52d28d），18source/8proof已协调独立核验。五native合法VAL3760块复现大R²舍入误拒绝1例；仅R²改用既有MSE的scaled≤1e−10核验，r/差值/CI仍绝对1e−10，raw绝对1e−6、精确常量拒绝均不变。原raw v1几何/源/协议与旧8/3原位保留并备份；旧等待父113154经UID/argv/无children/无actual review核验后仅该父进程退出。新PID233443实际live/WAIT_COMPLETE_FIXED_EVALUATION，上游88448；handoff时仅此一个review queue且14537/88448未动；现14537已正常完成，88448继续实际评估。

复核只在完整final_summary+ready后执行原raw v1→metrics v2；完成标记为INDEPENDENT_REVIEW_COMPLETED_V2.json，实际图表入口为pub_01_s03_ss07_result_figures_v2.py。按既定metadata固定3条Sparr/6条DTU×5seed全main重放raw，全部保存预测/统计另复算；不运行旧queue/旧呈现main。任一失败保留不自动重试，不提前读成绩，不改活跃冻结源。实际计数看服务器状态与进程；DTU仍原两文件白名单。工程修正和排队不代表实际复核或科学目标完成，继续统一评估、实际复核及科学解释。

---
以下是旧交付/STOP历史，非当前路由。
# 当前交付：SS06 run02 COMPLETED / PASS_WITH_LIMITATION（2026-10-26）

用户授权的强基线同规模比较已完成：C/O各3×3262完整60s目标，P完整两阶段3×639pre/2035head，全部训练/验证选点与实质门核验后统一9模型seed assessment。C/O/P r=.123014/.151762/.141466，旧M2=.128164；C对五框架/GRU六项Bonferroni交叉99.167%区间全跨0，不是等效/非劣证明。O/P离线未来信息与P额外预算单列，不能归因为纯future/训练量；全部POST-HOC INTERNAL EXPLORATORY。

入口continuity/eeg_run02_matched_baseline/REPORT.md、REVIEW.md及3图/逐record-person/机器证据。executor聚合误差0、18 raw-derivative CPU r max3.36e-8/MSE2.36e-7；协调独立SUMMARY2.78e-17/六配对均值6.94e-18。P原法修正在P0正式更新前冻结；源5b93ca5/55ddd40/observer82f3001。三seed/5family/单pass、C末端最佳与非纯组件对比等限制明确。**当前STOP，无自动下一READY；#6仍Draft不合并。** run01/旧CNN-TRF/stash/工作树不改，所有计算/张量仅2203，SS04/S04/父S05/GPT/artifact不启动。

---

# 历史交付：SS06 run01 COMPLETED / PASS_WITH_LIMITATION（2026-10-23）

39次完整record监督训练（各3262步）、3次正式SSL（各3511段）及一次前瞻INTERNAL/EXPLORATORY审计完成。M2最高点估计r=.128164，但未稳健胜GRU；八primary只有M1multi−matched long、M3finetune−frozen、M4adaptive−uniformfixed区间为正。预训练相对从头GRU、M5额外收益和内容特异历史收益未建立，不宣称外部确认/时延机制/SOTA。

入口：reports/pub_01/s03/continuity/eeg_run01/REPORT.md、REVIEW.md、FORMAL_PROTOCOL.json、243行RESULTS、四图及逐seed/逐record/逐人证据。后置汇总误差0，222个原型CPU相关max差1.55e-7；协调独立SUMMARY重算max差5.55e-17。完整与精简交付包12项检查均PASS。执行/冻结source commit ea5a1b5，最终报告commit见PR#6。

**当前STOP，无自动下一READY。** SS06完成不等于S03/PUB完成；SS04/S04/父S05/GPT/artifact不启动。#6仍Draft，不自行合并。旧SS02/03/05、主目录CNN/TRF/stash和其他worktree不改；载荷/权重只在2203。下方启动指令是已执行历史，不重新运行。

---

# 历史启动：SS06 真实公开 EEG（2026-10-23）

用户最新明确“启动吧”。**PUB-01-S03_SS06 ACTIVE / REAL_EEG_EXECUTION**。此前仅原型/禁止训练/完成后停止仅描述prototype01历史；当前可以读取既有合法公开EEG，完成真实loader、train/val pilot、协议冻结、训练及合格评估，无需重复询问启动许可。

首先读取 `doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_REAL_EEG_EXECUTION.md` 与 SS06 五框架任务书。执行会话继续使用 `codex/pub-01-s03-ss06-continuity` / PR #6 独立worktree；模型仍按既有指定6.1-sol/high。代码/轻量报告在Windows；数据和训练只在2203新eeg_run01目录。

只使用旧公开资产，禁止SS03/SS05已看test调参；可从原SS05 train按元数据冻结本轮前瞻内部审计划分，诚实注明历史训练暴露，不能当完全独立确认。缺独立test不阻断合法train/val探索。优先启动M1/local真实pilot，然后五框架和既定对照；不要在smoke或准备清单停工。

代码和轻量报告推送PR#6分支，不自行合并；SS02/03/05冻结，不启动SS04/S04/父S05。问题和结果向协调会话汇报。当前真实运行状态见 `reports/pub_01/s03/continuity/eeg_run01/EXECUTION_LOG.md`（由执行者按实际进展创建，未启动的阶段不能写已完成）。

---

## prototype01 历史入口（由上方最新授权取代）
# Codex 启动入口 — SS06 连续信息原型（2026-10-23）

本轮仅 PUB-01-S03_SS06：用户已授权五种框架任务书、基本实现和合成工程测试。**真实数据加载、预训练、pilot、参数优化与正式评估均未启动，也不自动启动。** 工程审阅与交付已完成：SS06 COMPLETED/PASS_WITH_LIMITATION，当前停止。

1. 读取最新用户指令、AGENTS.md 与父任务登记。
2. 读取 doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md（五模型结构/训练目标/对照/验收）。
3. 接口契约为同目录 SS06_IMPLEMENTATION_CONTRACT.md；运行证据/审阅入口为 reports/pub_01/s03/continuity/prototype01/REPORT.md。
4. 代码 src/m6a_public/continuity/，配置 configs/pub_01_s03_ss06_prototype_20261023.json。所有数值测试在2203；Windows仅代码轻量报告。当前无真实loader或训练命令。
5. 独立 worktree 分支 codex/pub-01-s03-ss06-continuity；保护主目录CNN/TRF、stash和旧worktree。main起点60c3404，PR#3/#4已合并；SS02/03/05结论只读，不重跑旧test。

当前 split/声学lag/实际公平预算均未冻结，不能从工程PASS推断EEG性能改善。S03/PUB仍ACTIVE；SS04、S04和父S05不启动。

---

## 历史启动入口（以下均为快照，不构成当前授权）
> **Issue#5已实际完成（2026-10-19）**：SS05 COMPLETED / PASS_WITH_LIMITATION。五组×三seed正式训练及105个历史条件结果已交；新内部holdout21人/47record/7family的r约.091–.092，关联/历史残差额外收益的六预设区间均跨0，为有效阴性。新train-only norm、后置same-agent review和实测图表见reports/pub_01/s03/predictive_memory/codex_run01/REPORT.md。当前停止，PR#4仍Draft，不合并、不启动SS04/S04/父S05。

# Codex 启动入口 — Auditory_Simulation（2026-10-19最新）

**本轮完成任务：`PUB-01-S03_SS05`，有效阴性/完成有限制。** PR#3已按用户新授权由协调者合并main@8d21d0a，SS05在PR#4独立分支交付、base main。当前无自动下一READY节点，不因阴性追加模型；报告和核验不称第三方科学签署。

## 立即读取（按顺序）

1. `doc/TASKS.md` → `doc/tasks/PUB-01/TASK.md` → `doc/tasks/PUB-01/SUBTASKS.md`：父任务与唯一状态源。
2. `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS05_PREDICTIVE_MEMORY.md`：**唯一完整执行任务书及验收规则**。
3. `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md`：公开 EEG 连续历史/split/lag/跨记录 reset；仅作为实验契约，若与 SS05 完整任务书冲突，以后者为准。
4. `reports/pub_01/s03/predictive_memory/REPORT.md`、`SYNTHETIC_RESULTS.json` 以及对应 `scripts/pub_01_s03_ss05_*.py`：**以前由协调助手编写的原型/声称结果，仅作待独立审查资料，不能当作 Codex 已复现的事实**。
5. 冻结历史：SS03 `reports/pub_01/s03/architecture/REPORT.md`，SS02 `reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`；只读，绝不根据已打开的test结果调新模型。

## 直接行动（Codex自己完成）

- 在 Windows 上采用**独立worktree**进入 `work/pub-01-s03-ss05-predictive-memory` / PR #4；先检查工作树，保留本地主目录未提交的 CNN/TRF 和 stash。不要 checkout/reset/force-push/merge `main` 或 PR#3。
- **先独立审计现有源码及合成实验结果**，保留可复现与不可复现项；检查新状态模块与标准GRU、窗口因果性、状态跨记录 reset；不能以已有报告 PASS 代替自己的执行。
- 若 server2203 SSH 可访问，首先只读确认公开数据、subject/story/source_family 和可建立的**新内部分组**。既有 SS03 test_seen/test_heldout 已被打开，**不得再次用于SS05模型选择和确认性研究**。
- 再完成真实 EEG 连续窗口 loader、真实数据train/validation pilot；在触及新的预留 heldout 之前冻结 protocol、3+ seeds、预算与 lag。符合准入才一次评估。若不具备可用新test或服务器不可访问，分别标 `HOLD_NEW_TEST`/`HOLD_RUNTIME`；不要虚构结果。
- **由Codex提交并push** 自有代码、配置及轻量报告到同一SS05分支；新运行结果存server2203公开目录，Git不收原始EEG/音频/权重。每阶段更新 `reports/pub_01/s03/predictive_memory/codex_run01/EXECUTION_LOG.md`，合成与EEG结果严格分列。

## 停止条件

完成SS05限定验收后停止汇报；SS04、S04、父任务S05、GPT重排与患者实验不自动启动。`Auditory_Simulation` 只处理公开数据与通用表征（M6A）；`STN_Decoding_Encoding` 只处理患者实验与STN特异适配（M6B）；仅冻结版本化artifact可单向衔接。

---

## 历史入口（归档，不再作为执行授权）

# Codex 启动入口 — PUB-01 SparrKULee 强基线复现

**研究者明确的第一步是复现，不是设计新模型或追分。**

2026-10-15最终交付：SS02 **COMPLETED / PASS_WITH_LIMITATION**。HDPS76人100Hz/M作者split，Top1/5/10=12.03/30.45/42.10%；Transformer原fold4、70人训练/69seen/13heldout，2AFC=83.53/80.08%（加权82.38%），冻结encoder原Conv32直接包络r=.1688/.1403（加权.1593）。各自一次test及独立review通过；完整论文cohort/隐藏test/ensemble仍未复现。见`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。本轮停止，无自动下一READY节点，不启SS03/S04。

2026-10-13追加执行已完成原raw100Hz、70story Mel及独立论文功能模型batch256工程smoke；602条产物对应作者647对中的600对就绪，必要47条raw仍受限，正式训练/论文测试NOT_RUN。Transformer对应冻结EEGcheckpoint失效、官方测试载荷未齐。SS02为 **WAITING_EXTERNAL / HOLD**，缺口与恢复位置见`reports/pub_01/s03/strong_baselines/RESOURCE_RECOVERY.md`；已启动作业全部结束，无后台替代训练。用户只要同论文对照，不跑缩减cohort/64Hz适配分数交付，不转SS03/S04。

## 权威执行顺序

1. 当前研究者指令；
2. sibling `AuditoryReading/AGENTS.md`（可见时）与本仓库 `AGENTS.md`；
3. `doc/PROJECT_BOUNDARY.md`、`doc/TASK_EXECUTION_STANDARD.md`；
4. `doc/TASKS.md`、`doc/CURRENT_TASK.md`；
5. `doc/tasks/PUB-01/TASK.md`、`doc/tasks/PUB-01/SUBTASKS.md`；
6. `doc/tasks/PUB-01/RESEARCH_BRANCHES.md`；
7. **当前SS03 ACTIVE执行节点** `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`；**SS02历史复现节点** `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`只读；
8. `doc/tasks/PUB-01/BENCHMARK_REGISTRY.md`；
9. 真实现存 S02/S03_SS01 数据/代码/config/reports。

## 具体指令

首先只读盘点当前 Windows Git 工作区（未提交修改、分支、已有资产），并只读检查 `server2203:/home/fanyu/auditory_simulation_m6a` 上现有公开 SparrKULee 数据、依赖与可复用的声音模型；**不覆盖、删除或自行修复任何既有产物**。

然后逐项目核对 HDPS、Transformer-CLIP-EEG、2023/2024 Challenge 的原始代码和协议；固定官方 protocol 与可运行条件，先做轻量 smoke，再执行 `S03_SS02` 允许的实际复现。遇到作者权重、非公开数据、许可、采样率/时间轴、GPU/内存/服务器连接不可用，留下可复现阻塞，不能通过换测试协议自称复现成功。

本次必须交付 `PROTOCOL_MATRIX.md`、`REPRO_REPORT.md`、`RESULTS.csv`、`REVIEW.md`，即使部分失败也要交。逐个结果标明 `OFFICIAL_PROTOCOL` / `INTERNAL_PROTOCOL`、实际 test split、候选数和是否含语言模型。若无法完成两套，给出各自的 `HOLD` 原因与继续所需条件；**不得自行启动新模型结构或 S04**。

Git 仅在获准 Windows 工作区维护代码/配置/轻量报告；服务器只作数据/训练/评估，不初始化另一个 git。完成后停止并报告。不得把执行日志写成已证明 ICML 新颖性或已达到 SOTA。
