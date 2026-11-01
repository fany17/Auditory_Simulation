# D8实际结果独立审阅

固定3角色×3seed×3λ的27套岭解和3套P_FT*参考正常完成；唯一PID541227已退出，无新NN更新、旧test/DTU或下载。三seed原路径完整VAL scalar与首尾raw波形桥均差0，实际848862主encoder/head窗+300工程窗，未重启正式run。

- **Technical：PASS_WITH_LIMITATIONS。** 独立协调实际核对247来源/523输入、全部27系数有限/维度/正规方程与截距、全部30项原mask/共同计权及共享λ，聚合差0，正规方程最大残差2.24e−17、截距差0。按预先scope独立重建所选3角色×3seed×原VAL首/尾记录共18项，block r最大差3.33e−16、MSE/Hann差5.10e−18、3秒r及首4窗差0；undefined全0。完整中文REPORT及三张实际PNG均查看，无需追加测试/改图。
- **Acceptance：PASS_WITH_LIMITATION（本轮强FT迁移诊断）。** P_FT*/P-only/C/Ccap三seed VAL r=.174141/.177842/.180541/.182340；Ccap−P_FT*均差+.008199、Ccap−P-only+.004498，每seed均正，支持强FT encoder下的当前读出开发增益。反复使用known-person/source VAL、额外监督岭/标定/搜索成本、容量/λ差异明确；未建立独立泛化、额外历史贡献、纯loss因果或ICML/NeurIPS方法新意。goal_complete=false。
- **Parent consistency：PASS。** 同一PUB-01-S03_SS07公开强基线与连续输出研究，D4-D7历史结论冻结；信号、权重、Z/P、Gram、统计与系数仅server2203，本地只代码/轻量汇总/静态图文。无患者/STN、新数据范围、远端Git、其他节点或PR合并。

原样[最终review](COORDINATOR_RESULT_REVIEW.json) SHAa059476ad1ef1f73f16e0c8e8b995d674442e31c2399e97701766dc251cb26a9；[数值audit](COORDINATOR_NUMERICAL_AUDIT.json) SHAf800ba4a3a860254ee9396bdb3c56e7ca056d43adb58cf48d8c2986e839009bd；[audit实施说明](COORDINATOR_AUDIT_IMPLEMENTATION_NOTE.json)及其[实际源码](independent_review/COORDINATOR_NUMERICAL_AUDIT.py)/[首版源码](independent_review/COORDINATOR_NUMERICAL_AUDIT_ATTEMPT01.py)一并保留。protocol54d424d5c26f1ab4aee4712c2efb149739a9e70ec876d2698391e4e5b6d0c980与source3f4d4786daa1533b504d48300dd287ef36044d3e498425a9ce4b7456955365a0保持原冻结。

原P_FT“完整预算”指原冻结5个额外epoch，不证明已收敛或所有可能baseline的最优；不能称SOTA。新读出共同当前输入，最弱正则下Ccap−C三seed负，selected Ccap/C还改变λ；Ccap r较高但相对C融合MSE更差。一致性不能代替准确度。source_family缺失时使用story/clip fallback，不是核验过的独立书籍/说话人家族；audit首版metadata KeyError在重建/评分前修复，正式模型/分组/门限不变。prepare批准schema修复亦在任何数值前保留原失败，不冒充事前CODE gate。

重建复用冻结FT缓存而未另做神经前向，不是独立raw预处理或外部复现。原FT没有全VAL历史波形，当前桥只证明所声明完整scalar与首尾实现；D7旧测试分数不可与D8 VAL跨比分差。完整[REPORT](REPORT.md)、全表与三图中文解释及轻量HTML归档交付，SS07/S03/PUB与持续goal保持ACTIVE。

下一轮仅准备强FT+Ccap共同起点、相同Hann物理点/target/当前输入的weighted-window MSE与fused-physical MSE最小比较，写清正则尺度、初始化、求解/目标计次和资源，具体PLAN冻结后才执行。无历史输入的融合监督不称历史机制或方法创新；不扩D8旧λ/seed，不重新打开旧test/DTU。
