# 预测前修订：唯一无标签整体增益校准

2026-11-02；归属PUB-01-S03_SS07同一run。修改时新数据NN前向、模型分数和训练均为0。真人的数据范围授权不变，四角色、三seed、两个primary、统计和split不变。协调决定见COORDINATOR_GAIN_CALIBRATION_DECISION_20261102.json；这是共享输入适配的科学修订，不是新增模型或单位修复。

原header增益方案及其实际失败线索保留在PLAN_HEADER_GAIN_PRE_AMENDMENT.md、HEADER_GAIN_PLAN_SNAPSHOT.json、EEG_UNIT_CANARY.json、EEG_UNIT_DIAGNOSTIC.json和原adapter源码。快照包含审阅要求的精确统计澄清，不冒充与最初被审PLAN哈希相同。MNE与手工header解析逐值差0，只能证明两个reader一致遵循声明；处理后仍达百万数量级的声明µV。发布端512Hz转换链未找到，真实物理增益保持UNVERIFIED。

新契约只允许一个正标量：固定坏道/模板映射、0.5Hz高通、CAR、重采样64Hz全程FP64；按EEG/audio实际共同长度裁为64×L。用该record全部[0,L) EEG样本（包括Hann未覆盖边界）计算

`A_record = sqrt(mean_c(mean_t((x[c,t] - mean_t(x[c,t]))**2)))`，ddof=0；

`A_source = sqrt(mean(original_TRAIN_std**2))`；

`gain = A_source / A_record`，输出为`gain * x`，之后才cast C-contiguous float32，并沿用原TRAIN mean/std。

去时间均值仅用于估计方差，不从最终输出另行减去各道时间均值。不进行逐通道缩放、新norm、目标/预测驱动拟合、可调clamp或倍率网格。固定模板/curator bad和每条记录的相对空间幅度保持；artifact与预处理差异仍会影响这个整体幅度估计。所有good道在filter前各自时间常量，或A_record非finite/<=0时是预测前机械失败，保留记录与原因，不按模型成绩排除。该标量适配不解决个体电极位置、参考、artifact、语言或硬件同步差异。

数学上，当一条记录整体乘任意正单位倍率k，A_record也乘k，最终校准信号保持不变（浮点实现容差单独核验）。这提供对未知整体正增益的不变性，不识别真实物理单位；校准输出标为“source-amplitude calibrated numeric units”。旧TRAIN单位provenance只描述原数据合同。

全record EEG用于估计gain，包含相对于某个预测窗的未来EEG。因此本轮必须称“预冻结、每record无标签整体增益校准后的外部迁移评估”，属于offline/transductive输入校准；不能称原TRAIN固定增益下的纯零校准独立确认、在线因果或已修复SI单位。该变化发生在任何新模型成绩出现之前，不能据最终结果改回原方案或增加另一个适配候选。

工程验证只用已有首canary及合成输入：正倍率不变性、不同record独立性、常量机械拒绝、FP64计算→最终float32顺序；不运行新模型。全部380条获取/处理可在此明确合同下继续，最终模型/来源/实际mask/统计freeze与一次正式预测审阅仍需完成。
