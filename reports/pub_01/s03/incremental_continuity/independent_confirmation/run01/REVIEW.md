# ds004408 固定读出外部迁移阶段验收

本阶段 **PASS_WITH_LIMITATION**；SS07仍ACTIVE，科学goal未达到。有效结果允许当前窗口读出的迁移增益结论，不允许连续历史机制、方法新颖性或投稿充分性结论。未新增任务节点或自动下游。

## Technical validation — PASS_WITH_LIMITATION

- 全380公开EEG载荷及20音频按固定版本核验；19人、380条、988完整块、45429窗。16个既有processed复用、364新增；failures=[]。物理增益UNVERIFIED，以固定无标签全record单一gain完成适配。
- 最终预测前协议 `5cb28c2cd219c25fc36ed5ea91d92572a9c6b61b0a3ae864a0f1fa5180cc76c5` 经协调独立准入，绑定唯一queue `c67f1808…392a4b`；3seed四角色一次执行，无自动重试。每seed主45429窗+181工程原Decoder桥窗；九个桥batch最大波形差0。
- 协调独立输入复核1017 pin、380完整EEG/掩码、119 TSV及20 target；完整独立数值复算4560 role-seed-record、11856块、2280 FP32诊断及两主推断通过。不是只读汇总表；复核源不import执行者evaluate/scoring。
- 原TRAIN mean/std、冻结模型/读出、统计家族与固定支持不变；380次gain是新增无标签输入统计，与新学习模型fit=0分开。
- 正式五图已由执行者逐图查看，协调实际通读Results与查看六个PNG（含历史tick版），仅将“显著低于”的描述改为“数值上低于”，避免未检验的统计措辞。补正当前主比较图刻度密度，原图保留，数值不改。

完整机器证据：[最终输入核验](COORDINATOR_FINAL_INPUT_AUDIT.json)、[独立数值核验](COORDINATOR_INDEPENDENT_NUMERICAL_REVIEW.json)、[资源快照](FINAL_RESOURCE_SNAPSHOT.json)。协调最终全文/表/成本审阅文件随归档保留。

[最终全文/视觉/轻量表与成本审阅](COORDINATOR_FINAL_REPORT_REVIEW.json)已完成，实际SHA `53894fcfe85c6cd1b73d16b73298b9c500f783e53d40fe7eab897a360c0a1d64`，PASS_WITH_LIMITATIONS、stage_accepted=true、scientific_goal_complete=false。228/12/2280/38个轻量表行、唯一性与每人52块通过；全部成本manifest引用及90个直接字段对照通过。只补审阅状态与证据链接，实际结果数字不变。

## Acceptance validation — PASS_WITH_LIMITATION

Ccap−P_FT* Δr=+0.008040823，family95近似CI[0.00511387,0.01097443]，Holm2 p=0.0000534058，三seed全正、18/19人改善。支持在该固定输入适配和共享叙述条件下，当前静态读出增益能迁移。

F−W Δr=+0.000151871，CI跨0、p=0.34562683，seed2正1负；没有达到预定统计阈值，不证明等效或不劣。F窗口MSE343.2005、r3s0.01053、overlap大，融合MSE较W/Ccap更差，边界与tail方向不一致，全部保留。完整FP32结果对本轮主结论影响极小，不消除窗口退化或泛化风险。

主要限制：整条未来EEG用于标量gain；物理单位未确认；标准模板而非实测电极；零偏移依README而无硬件验证；19个人共享单一书籍叙述；原音频跨库同一性及XLSR预训练接触未穷尽排查；旧包络revision未知；推断仍有符号对称/独立人假设及近似区间。所有限制在报告及HTML中明示。

## Parent consistency — PASS

PUB-01公开数据/公共方法范围不变；患者/STN/私有资料继续FORBIDDEN。只执行真人授权ds004408 v1.0.8固定范围，无额外数据/权重、成绩驱动参数修改、新学习模型拟合、旧test/DTU新推理或PR合并。D11及更早冻结结论不改，S04/SS04/父S05未启动。

所有数据/权重/信号/缓存/计算均在2203；本地仅代码、清单、结构化表、静态图和报告。两个服务器root合计2797859025B<3GiB，全SS07 NPY28343750260B<28GiB，两盘各保留20GiB且home另留128MiB未来报告预算。没有删除/搬迁历史资产。

原queue8小时上限只约束轮询，未独立timeout同步freeze/statistics子进程；这是已记录的工程限制，本轮已正常结束且队列/预处理/实际worker退出。未为此重跑或改已冻结源。

独立最终输入wall2.833秒；全量数值审阅wall61.426秒/CPU221.339秒/RSS395880KiB，另列于[含审核成本账](report01/COST_LEDGER_WITH_REVIEW.json)，不隐去继承训练成本或把缺测字段记零。
