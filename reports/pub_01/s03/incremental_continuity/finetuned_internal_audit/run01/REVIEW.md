# D10实际独立审阅

固定P_FT*/P-only/C/Ccap/W/F×三seed×原seen/heldout的36项评分全部完成，PID599747正常退出；无新神经网络/岭拟合、TRAIN统计重拟合、λ或模型选择、下载或DTU前向。

- **Technical：PASS_WITH_LIMITATIONS。** 原global64跨record/双stack/layout/flags路径及同次P/Z得到实际核验。36cached VAL映射通过，300VAL gold Z/P、636test原Decoder及432条旧P_FT未缩放fused波形桥差均0，原FT全block r差≤4.44e−16、同三seed历史加权r=.168944726368672恢复。协调独立306来源/600输入、全部36原mask/逐人/2:1聚合差0，72项预定缓存重建block r差≤2.78e−16、MSE/Hann≤2.70e−14、首4精确。完整REPORT及三张实际PNG已读看；两处角色/拟合文案澄清已实际核验，不改变数值。
- **Acceptance：PASS_WITH_LIMITATION（冻结方法的旧内部审计）。** Ccap−P_FT*/P-only原加权均+.004403/+.002973，F−W+.000892，三项在两组各seed均正。F固定Hann点增益与巨大单窗MSE/低3秒r/近相消bias仍并存；C−P-only在heldout三seed全负、W−Ccap也不稳定。全部POST-HOC INTERNAL EXPLORATORY，不能称独立新确认、额外历史收益、显著性、等效、SOTA或通用稳健可用。goal_complete=false。
- **Parent consistency：PASS。** 仍为PUB-01-S03_SS07公开强current读出/连续输出研究；D4-D9冻结结论和所有旧参数保留。原patient/STN禁令、2203计算/信号权重缓存边界、无远端Git/其他支线及PR6 Draft不合并不变。新OpenNeuro范围仍待用户明确授权，未取payload。

原样[最终review](COORDINATOR_RESULT_REVIEW.json)、[实际数值audit](COORDINATOR_NUMERICAL_AUDIT.json)、[代码review](COORDINATOR_CODE_REVIEW.json)和[独立审计源码](independent_review/COORDINATOR_NUMERICAL_AUDIT.py)保留。audit SHA475caa270497f9ee5f9f0e61dea9b595c33d82d92367bdafd08af2445b584f88；protocol14b2cedd0df026637624afbc63004c3803644d420e7966bad234cab33babb78e/source2ce7df72c7082b7ff482121a3c076227eeff65c9d9d5c0915cef77f80a23f675为正式冻结版本。原[各seed实际桥证据](BRIDGE_EVIDENCE.json)仅将已有JSON原样汇集，没有重算或新增前向。

原FT只有fused历史输出，本轮不声称已有旧单窗raw gold。独立audit仍复用冻结神经来源/缓存，不是外部实验室或独立神经数学复现；新cache生成的实际255012 encoder/head窗已计次。主wall147.210秒/CPU539.147秒，额外audit3.087秒/CPU17.644秒分列；全NPY24.4245GiB<25GiB，2GiB新root/20GiB余量守界，未删旧cache。

图表首版字段适配KeyError与partial CSV/旧source/log原样保留在服务器presentation_attempt01，[记录](PRESENTATION_ATTEMPT01.json)回传；修正仅作用于renderer，正式36评分/系数/source/protocol没有重跑或改动。所有阴性和F窗口表示异常在三图/完整表文中保留。

完整[REPORT](REPORT.md)、三图中文解释、全表/窗口细节、REVIEW及官方轻量HTML归档交付。SS07/S03/PUB与持续goal保持ACTIVE，归档不等于论文或科学目标完成。后续仅按已授权原TRAIN/VAL范围准备最小“长过去EEG观测/噪声状态”假说，先核现有M2/M5/D4/D5及EA/AEA先例、强current/静态状态控制和固定当前输入的历史干预；具体实验审定前不新拟合、不读旧test/DTU或下载新数据。
