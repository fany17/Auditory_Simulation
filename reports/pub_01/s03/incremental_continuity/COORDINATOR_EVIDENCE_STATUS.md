# SS07 持续目标的累计证据与缺口

2026-11-01，协调端基于已读实际产物更新。此表跨轮汇总，不修改旧协议或结果，不新增实验、成功阈值或审批门。**目标未完成：当前有可核验的读出开发增益，但连续历史贡献、独立泛化及方法新意仍未建立。**

## 已经得到的直接证据

| 科学问题 | 实际结果与来源 | 能支持的判断 |
|---|---|---|
| 原强基线是否认真训练过 | [round01 报告](run01/REPORT.md)：五个 native 完整两阶段、五个微调控制、54 个筛选和37个唯一 full 条件；真实 raw/metric 复核 | 可用公开条件下的全预算强基线已完成。250 epoch 上限不证明收敛；不能将失败归咎于只跑 smoke |
| 五种连续框架是否稳健提高 | [round01 证据矩阵](run01/EVIDENCE_MATRIX.md)：冻结主模型 M4−P 内部均差−.000621、DTU+.000149，所报调整区间均跨0，匹配 current 对照也未建立优势 | 没有稳健历史增益，既不证明历史无信息，也不证明等价/非劣。DTU已暴露，不能再次当新确认集 |
| 从已经训练好的原模型接入历史是否有效 | [warm round02](run02/REPORT.md)：18格真实5 epoch；joint past 对 current 三seed均负，frozen past 对 current 两seed负；全部实际VAL重放核验完成 | 随机head起点并非唯一未检验解释；本轮也未建立稳定历史收益。固定短预算不代表最优训练或收敛 |
| 摘要压缩是否可能遗漏信息 | [TRAIN诊断](train_diagnostics/run01/REPORT.md)：六bin保留约13%的窗内中心化latent时间能量；跨半窗无监督仿射校正恶化误差 | 能量比例不是有效预测信息比例；不能据此直接宣称增容会改善。失败的仿射方案保留 |
| 保留更多历史信息是否就能改善 | [D4](conditional_history_probe/run01/REPORT.md)：54闭式解；强current Ccap VAL r=.176897，完整near-history CF=.174090；CF对C/Ccap三seed均负 | 该线性读出族未获额外历史收益。current 改进真实存在，但与原P比较同时改变目标、求解器、容量与正则 |
| 历史是否改变当前映射 | [D5](history_interaction_probe/run01/REPORT.md)：36闭式解；Near对共同加性控制仅1/3 seed正，对同容量CC三个seed全负 | 所测低维双线性交互未获预设支持，不排除其他状态时标/非线性机制。不能用极小的Near−Anchor正值替代控制 |
| current 增益来自哪些读出变化 | [D6](readout_attribution/run01/REPORT.md)：27新解+18原样复用；P/Ponly/S32/U32/C/Ccap VAL r=.162362/.171837/.174284/.173867/.174789/.176897 | 原输出后处理已有重要增益；latent映射另有增量。解除共享不是选定收益来源；完整latent支持与容量共同改变；平方增益依赖正则 |
| 输出一致性能否代替准确度 | [D6图三及review](readout_attribution/run01/REVIEW.md)：P* overlap误差最小而r最低；固定Hann误差恒等已实际验证 | 不能由一致性单独推断准确度，亦不能据此直接训练一致性loss；r、target MSE与融合分散收益需同时解释 |
| 固定读出增益是否在旧内部测试仍存在 | [D7](internal_generalization/run01/REPORT.md)：36/36固定角色–split；P*/P-only/Ccap三seed加权r=.160248/.164642/.168943，Ccap对前两者在seen/held-out每个seed均正；全部36聚合和72缓存重建已独立核验 | 支持相对原冻结P和输出再读出的内部可重复点增益；旧测试已暴露，没有新的独立确认或历史机制证据 |
| 是否已经超过原完整微调强基线 | [同三seed历史参考](internal_generalization/run01/COORDINATOR_HISTORICAL_FT_REFERENCE.json)：原P_FT均r=.1689447264，Ccap=.1689434862，配对均差−.0000012402且逐seed符号混合；同2044块/原2:1口径 | 点均值几乎一样，不能宣称优势、等效或非劣。P_FT额外5epoch联合微调与冻结encoder的TRAIN岭拟合/λ搜索成本不同；此项是事后历史参考，未改变D7预设矩阵 |
| 读出改进在完整微调后的强encoder上是否仍存在 | [D8](finetuned_readout_transfer/run01/REPORT.md)：27/27固定岭解；P_FT/P-only/C/Ccap三seed VAL r=.174141/.177842/.180541/.182340，Ccap−P_FT均差+.008199、对P-only+.004498，各seed均正；3个原FT桥差0、全部30评分及18独立缓存重建通过 | 已得到定义明确的强FT之上的重复VAL开发增益，不能全归因原encoder较弱。尚未测该FT+Ccap的新测试表现；没有历史输入、新意或独立确认，不能把.182340直接与D7测试值相减 |
| 直接监督连续融合输出是否有额外收益 | [D9](physical_objective/run01/REPORT.md)：W/F×3seed六解全部收敛；Ccap/W/F VAL r=.182340/.182474/.183874，primary F−W均差+.001400且三seed均正，融合MSE下降.0001266；独立6个全TRAIN物理梯度/残差、全部VAL聚合和12缓存重建通过 | 支持相同强读出/物理目标/参数权限下的融合监督VAL小增益。F单窗bias约247–253、均匀窗MSE352、3秒r约.0034为严重单窗表示退化；固定Hann内部bias却≤.00587，不能将虚拟窗口近抵消直接称融合输出崩坏。仅固定离线融合适用，没有额外历史输入或独立确认 |

D4–D6使用原3个seed、重复source VAL（49人/5个clip标识，部分与TRAIN同书家族）。它们不是独立复现实验，也不把多组λ或重复窗当独立样本。D6当前最强读出相对原P的开发差值不等同连续性收益或同容量、同优化成本优势。后续必须保留P-only及C/Ccap等强控制，不能退回只与原257参数head相比。

## 持续目标尚缺的证据

| 目标要求 | 当前状态 | 可证明它的证据，而非替代品 |
|---|---|---|
| 可重复超过原强baseline | D8已在重复VAL上超过定义明确的完整P_FT，三seed方向一致；该新方法的测试泛化尚未验证 | 固定强baseline下的逐seed/逐人泛化，以及进一步可信确认。不能混用D8的VAL与D7的测试分数，或把开发增益当独立确认 |
| 连续性独立贡献 | 历史输入增量仍未建立；D9已有匹配融合监督的VAL小增益，两者须分开 | 先核固定融合读出的泛化和适用边界；若进一步主张历史记忆，仍需强current/输出prior/容量/预算控制与保持当前EEG不变的历史内容/时间干预 |
| 方法新意 | 未建立；常见模块、岭读出及平方扩展本身不构成充分贡献 | 对照一手相关工作明确新的可检验方法贡献，辅以相应消融/效率/适用边界。换名称不解决缺口 |
| 泛化与统计 | 原Sparr测试及DTU已有暴露；同人/同刺激依赖存在 | 新的可信确认范围、预先冻结适配/评分，明确被试与刺激层级；不能将旧test改名新holdout |
| 公平性与成本 | 各轮真实训练/搜索/额外求解及共享成本已分列 | 新方法继续公开样本呈现、更新、参数、额外预训练、时间和继承成本。只同epoch或只报最后一次fit不够 |
| ICML/NeurIPS投稿评估 | 当前不具备充分证据 | 上述性能、机制、新意、泛化、统计与可复现性共同支持；不作论文接收保证 |

## 当前下一步的精确状态

[D7 PLAN](internal_generalization/run01/PLAN.md)的P*/Ponly/S32/U32/C/Ccap × 3 seed × seen/heldout共36角色-split已实际完成。只用已选λ及既有参数，不新拟合；69名seen、13名heldout被试，共144记录。原P未缩放波形的432record桥差均0，独立72缓存重建最大block r差2.78e−16，全36聚合差0；协调已实际读完整中文报告及三图。主run wall75.177秒，没有encoder/raw EEG前向或新训练。

D7是**POST-HOC INTERNAL EXPLORATORY**。预设主要比较Ccap−P*、Ccap−Ponly的均差分别+.008695、+.004301，沿用原2/3 seen+1/3 heldout计权；不根据结果调整本轮模型或扩大网格。原完整P_FT在相同三seed/块/计权下与Ccap点均值几乎一样。该正向审计未补足超强baseline、独立确认、历史机制或创新缺口；完整review及归档由D7产物记录实际完成状态。

D8已完成对应P_FT的TRAIN/VAL读出转移诊断：原426 TRAIN/49 VAL、三seed、三个角色×三个原定λ共27解，重新拟合FT TRAIN统计与a,b。原global64前向与历史FT验证loss差均0，主run wall386.589秒、848862主前向窗+300工程窗、NN更新0；全部27正规方程、30评分聚合及18独立缓存重建已核验，block r最大差3.33e−16。全文/三实际PNG已审阅，最终root review为PASS_WITH_LIMITATIONS，goal仍未完成。

以已建立的强FT+Ccap为共同起点，[D9具体方案](physical_objective/run01/PLAN.md)已在51abaea版本获协调审阅通过，科学代码/协议在a27c620冻结。固定W/F两目标×三seed六个FP64解已全部完成：共同24112044个TRAIN物理目标、相同参数起点、首4窗固定上下文、Hann端点约束与λ=.01/192，只比较逐窗加权MSE和融合MSE。355CG更新/386Hessian动作、主run185.396秒，6个真残差全低于原1e−8，无失败重跑、新encoder/head前向或NPY。原D8完整窗支持不同，不能把其差值解释为纯loss缩放；D9全程只TRAIN/VAL，旧test/DTU未开。

协调在2203独立重建了六解全TRAIN物理轴目标/梯度，未调用本轮core adjoint：目标差≤2.69e−14、真残差差≤1.31e−16且6个仍全过原门；275 source/535 input pins无漂移，全部VAL聚合差0，12条固定缓存重建block r差≤2.22e−16。额外审核69.714秒，无新fit或NPY。实际四图和完整中文报告已阅读，[最终review](physical_objective/run01/COORDINATOR_RESULT_REVIEW.json)为PASS_WITH_LIMITATIONS，parent goal未完成。

F的单窗异常主要由病态但有效正定的未惩罚bias坐标解释；保存系数的物理内部bias≤.00587，首可学习边界≤.12165、尾部≤.30603。此代数证据不等于新拟合消融或完整鲁棒性实验，不能否定已核验的固定融合增益，也不支持独立窗口可用、长历史机制或充分创新。

下一步仅已批准准备固定P_FT/P-only/C/Ccap/W/F×三seed的原seen/heldout事后内部泛化审计PLAN和只读资产/空间盘点。须先明确原FT测试实际前向签名/波形桥、必需缓存增量和原评分；不得改模型、λ或预算追测试结果。它不能充当新确认，DTU仍不再前向。具体新编码/评分待该矩阵审定，当前无新增测试结果。

所有信号、权重、系数和数值计算仍在2203；本地仅代码与轻量报告。旧各轮结果冻结，PR#6不自动合并，不启动患者/STN或无关任务。
