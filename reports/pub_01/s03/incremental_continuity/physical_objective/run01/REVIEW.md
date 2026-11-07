# D9实际独立审阅与限定验收

固定W/F×3seed六解全部完成，唯一queue563655及三个worker均正常退出，正式source/protocol保持；共355 CG更新/386 Hessian动作，无失败重跑或按成绩扩预算。无encoder/head前向、新NPY、旧test/DTU或下载。

- **Technical：PASS_WITH_LIMITATIONS。** 协调独立实际核对275来源/535输入，重新从缓存和保存参数构造六套完整TRAIN物理轴Hann目标/梯度（不调用本core adjoint）。目标差≤2.69e−14，真实预条件残差差≤1.31e−16且六个仍通过原1e−8门；全部六套VAL原mask/49人752块聚合差0，12个预定首尾记录缓存重建block r差≤2.22e−16、MSE/Hann≤2.70e−14。首4回退差0、共同固定端点精确；完整REPORT和四张实际PNG均实际审阅。
- **Acceptance：PASS_WITH_LIMITATION（匹配固定离线目标诊断）。** F−W的VAL宏r均+.001400且三seed全正，fused MSE下降.000126614，事实获独立重建支持。同时F窗口MSE352.068、3秒r .003435、overlap706.180为严重单窗表示退化。保存bias和边界代数支持近null坐标相消，未证明固定Hann连续输出无效/数值崩坏，也未证明通用稳健可用；单窗独立使用、其他融合/边界和新数据均未验证。科学goal未完成。
- **Parent consistency：PASS。** 仍是PUB-01-S03_SS07公开强current框架上的连续输出监督诊断，无新增历史推理输入，不改D4-D8冻结结论。所有信号、特征、参数、工作张量和预测只在2203；本地只有轻量表图/记录/代码。无患者/STN、远端Git、其他支线或PR合并。

原样[最终review](COORDINATOR_RESULT_REVIEW.json) SHA4e3a81632d852583617d00fe04d819c00f2f92636d62f33408c1887e005d2e16；[实际独立audit](COORDINATOR_NUMERICAL_AUDIT.json) SHA339a8be287b59cf951724c18e0ad01b317bcff10e5aed634ddd31953fb434211；[bias核验](COORDINATOR_BIAS_DIAGNOSTIC.json) SHA8c36e80e96be7bac07bba497a8610e61f1a77f1c096b74c4fc221962e9a1e4ff；对应[审计源码](independent_review/COORDINATOR_NUMERICAL_AUDIT.py)、[bias源码](independent_review/COORDINATOR_BIAS_DIAGNOSTIC.py)保留。source61f80f5c…413f38f/coredaa4d5c7…f058628/protocol86704273…e4f1499为正式冻结版本。

严重窗口异常不是未收敛的代名词。F不惩罚W目标内含的disagreement；F单窗bias峰247–253，内部物理bias≤.00587、首可学习边界≤.12165、尾部≤.30603。bias-only解释大部分disagreement仍非新拟合消融，coef-only float32差≤1.45e−8不是全模型低精度/鲁棒性试验。只读代数核验和第四展示图均为新发现风险的必要解释，不改变六解或新增选择。不同TRAIN/VAL计权、D8与D9不同支持、重复known-person/partial-book VAL及额外计算成本均披露，不称显著性、等效、SOTA、历史机制或方法创新。

完整[REPORT](REPORT.md)、三主图和第四退化补图、全表、真残差历史及轻量HTML归档交付。主run185.396秒与额外独立audit69.714秒分别计，不能以PASS或微正r关闭SS07/S03/PUB与持续goal。

后续仅准备固定已选P_FT/P-only/C/Ccap/W/F×3seed的旧seen/heldout事后内部审计PLAN及资产/准确空间盘点；必须明确原FT测试输入路径/签名/未缩放波形桥和25GiB总NPY/20GiB余量，具体矩阵审定前不进行真实测试编码或评分。该旧测试继续POST-HOC INTERNAL EXPLORATORY，不称新确认，不按结果再训练/换模型；DTU关闭，原D9不追加bias约束/λ或重跑。
