# D7实际结果独立审阅

固定6角色×3seed×seen/held-out的36项评分已全部正常完成，PID535287退出；不重启或按结果追加测试。全程无新拟合、λ选择、encoder/raw EEG前向、DTU或下载。

- **Technical：PASS_WITH_LIMITATIONS。** 独立协调实际核对286来源、893输入header、全部36项共同mask与聚合，保存结果重聚合差0；按预测前scope重建72项缓存读出，block r最大差2.78e−16、MSE及3秒r最大差5.55e−17，首4窗回退差0。全部432个原P未缩放波形桥差0；P* block r最大差3.33e−16。完整中文REPORT、三张实际PNG及新增同三seed历史FT段落均实际审阅，无须额外重跑。
- **Acceptance：PASS_WITH_LIMITATION（本轮固定内部审计交付）。** Ccap相对P*/P-only在两组每seed均正，原2:1加权均差+.008695/+.004301；支持当前读出的内部重复点增益。所有旧测试已暴露，仍为POST-HOC INTERNAL EXPLORATORY。相同三seed历史P_FT r=.168944726，Ccap=.168943486，配对均差−.000001240且seed方向混合，未证明超过强微调baseline、等效或非劣；不同预算/边界策略不构成单因素消融。没有建立额外历史贡献、独立确认或方法新意，goal_complete=false。
- **Parent consistency：PASS。** 仍为PUB-01-S03_SS07公开EEG强baseline与连续输出研究；D4-D6冻结结论保留。权重、latent、系数及预测仅server2203，本地只有轻量汇总/代码/静态图。无患者/STN、其他任务、新数据、远端Git或PR合并。

原样[最终review](COORDINATOR_RESULT_REVIEW.json) SHA57072ffe1fdfa56bee94072809833c9bec3ea1b0373d3639a38fc0e8cc09b45f；[数值audit](COORDINATOR_NUMERICAL_AUDIT.json) SHA1128a53b4609f0b33e1f8ef4c8587c03e9121f5250978856289721267bbb8da4；[同三seed历史FT参考](COORDINATOR_HISTORICAL_FT_REFERENCE.json) SHAd178483c28be5847bad1c74e4c5c8b7dea32cd451e2ad18638ec23d667dd8a2f。protocol eed8be9fbd1d5c2a182f90f10bc3fcb4128468da15f0eff84554e55e3d149886；全部原文件保留，不改冻结评分或把FT追加为D7角色。

重建使用冻结缓存输入及系数，未重放raw EEG/encoder，亦非外部实验室复现。3秒undefined规则保持null+覆盖，本轮undefined全0。三seed重复相同数据；C−U32的held-out均值小负，低overlap不代表高准确度，Ccap比C的fused MSE更差等限制均披露。

[完整报告](REPORT.md)、全表、三图中文解读与轻量HTML归档一并交付。D7收束后仅准备对应P_FT的TRAIN/VAL最小读出诊断：实际盘点资产/空间/一次前向预算，重拟合FT TRAIN统计和a,b，不套native系数、不访问D7旧测试/DTU。SS07/S03/PUB及持续科学goal仍ACTIVE。
