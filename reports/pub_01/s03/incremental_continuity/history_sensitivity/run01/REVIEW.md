# D17三级验收

2026-11-02；PUB-01-S03_SS07，持续goal保持ACTIVE。

## Technical：PASS_WITH_LIMITATIONS

v2 protocol1791ab56…22d883e，source commit788029c；协调执行review实际核229源pins、15输入身份、4 checkpoint SHA、48有效位置/单受试者metadata及统计轴。源直接使用D16 forward，权重全冻结、model.train()/history输入梯度、无optimizer。

实际一次384窗/8forward/8输入backward，0fit/encoder/head，无失败重跑。四检查点current/anchor相同、权重不变断言通过；pure axis0/零分母/已知差值合成测试0NN。56节点比较、4检查点汇总、8梯度行保留，无新数组/激活/权重文件。

协调独立review source13d34e2b…e1c8e5实际1325检查PASS、0NN/0backward，CPU.167573s另列。仅源码/metadata/已存标量算术一致性，不是独立神经/激活回放，也未独立重做运行中的权重/current断言。原v1准备与v2修订证据保全，两轮prepare均0NN。

## Acceptance：PASS_WITH_LIMITATION（局部诊断）

H2026 selected5在此批中输入/LN后token配对变化非零，但GRU末h变化约6.40e−6、FiLM gain完全相等、最终预测差9.39e−9；局部输入梯度仍非零，不称严格独立历史。三个last19的预测差.03353/.01912/.04725，有明显有限配对响应，不能笼统归为全部塌缩。此诊断解释所选与末轮的不同数值行为，未建立历史任务收益、独立泛化或创新。只有sub-001/8story/48窗、固定输出投影与near/older联改内容/recency，不能外推人群或识别全部性能原因。

## Parent consistency：PASS

无新训练、目标读取/VAL/test、完整TRAIN遍历、数据/权重下载或模型位置/LR网格。全部信号/模型计算仅2203，本地轻量统计；D16与更早阴性结果和外部保持冻结。总新增384NN/8输入backward/0fit，成本及未测管理范围明确，PR6 Draft不合并。完成诊断后只由协调界定下一具体方法，不自启新fit。
