# DTU 无目标域模型拟合的外部适配方案（待协调冻结）

2026-11-01。依据已下载数据、输入量纲审计和官方处理代码提出，仅单一预定adapter；没有计算模型分数，也不依据外部表现选择适配、lag或排除trial。Sparr主模型/controls/五seed/EMA锁定后才做一次外部评估。本文件未授权执行该评估。

## 实际差异与量纲审计

原train426固定normalization的channel mean范围−0.01271至0.003177，std为3.8876至12.1795；源数据按公开provenance为MNE V×1e6的µV契约，历史实际generator revision仍UNKNOWN。官方Sparr源码可核实LoadEEGNumpy(unit_multiplier=1e6)、0.5Hz高通、MWF、64头皮CAR、64Hz；这是公开管线说明，不伪称独立复现了历史每一步。[官方源码](https://github.com/exporl/auditory-eeg-dataset/blob/master/preprocessing_code/sparrKULee.py)

DTU官方cfg/脚本为0.1Hz单向高通、EOG denoise及含mastoid的66列average。按名称选64并scalp CAR是固定坐标适配。18人全部cfg/dim递归搜索没有发现unit/volt/scale/calibration/gain字段或文本，物理单位仍未证实。全18人×60trial×64channel的50s精确常量通道数为0。scalp-CAR后各人的channel×trial std中位数约6.30–43.95，S11含显著幅度离群（最高std约17050.38，source-channel std比值最高约2889.66）。这些仅是数值尺度事实；不能由生理范围/相近幅值推断单位，也不自动称S11损坏或据此删人/删trial。

INPUT_DIMENSIONAL_AUDIT仅计算描述性整trial均值/std，没有把整50s统计应用于任何模型输入，没有计算包络相关或拟合归一化参数。整trial标准化若用于较早3s窗，会引入较晚EEG信息，不应静默称原3s权限。

## 建议唯一primary adapter

为了不依赖未证实的µV换算，建议每个native **当前3s窗**独立、无标签标准化；保留所有1080trial，不追加幅度排除或剪裁。顺序为：按64物理名称重排→逐sample scalp CAR→原192点窗/96点hop→各channel用该192点自身mean和population std标准化。std精确为0时该channel置0，其他不设按单位变化的绝对epsilon，不插值坏道、不增高通/低通、不重做MWF。输出float32，所有baseline、primary及controls用同一个适配和相同窗。

该变换不使用整50s统计、未来窗、其他trial、包络、模型成绩或目标域训练参数；全局正比例单位缩放会相消。它是预声明的当前窗输入变换，**改变了原baseline固定train-only标准化的输入定义**，必须称“外部固定适配条件下的无目标域模型拟合”，不称逐项原生预处理复现。不另外叠加源µV mean/std到已无量纲结果，不擅自运行第二种adapter并择优。若协调更偏向物理单位已确认后的固定source-norm，需要在任何外部成绩前另定，本轮不以成绩决定。

工程已用真实S1 EEG检查：正比例×1e−6/×1e6后float32输出最大差0；改变首窗之后的EEG不改首窗；任意重排66列后按名称选通道仍逐点相同；常量输入全0有限。没有调用任何神经模型或使用MAT中加载的包络数值。

## 精确时间、覆盖与状态

每trial3200点；只用原regular starts=0,96,…,2976，共32窗，最后支持[2976,3168)。不追加start3008的补尾窗，避免i−2历史原始支持变为重叠。1.5s步距、两parity流各3s更新，i−2窗结束恰为当前窗开始；trial间reset，头两个无历史窗照常评分。所有输入变换也只依赖每窗原始支持。

Hann192、covered>1e−8，固定可评分样本为0-based 1…3166（3166点）；未覆盖[0,1)及[3167,3200)，共34点。此为50s trial的完整native有效覆盖，不能声称评分了全部3200点或只挑好片段；trial→person指标显式标注3166/3200覆盖。所有模型同一mask，不填充未覆盖点。ADAPTER_GEOMETRY_ENGINEERING_CHECK已经逐点验证该边界、parity和冷启动。

## 外部endpoint与推断建议

不做目标域模型拟合时不强行新建DTU train/dev；全部18人60trial作为锁定模型的外部评估，权重、宽度、EMA tau不重选。主endpoint：每trial在同一3166点上Pearson(pred,wavA attended)，60trial等权到person，18person等权，再五训练seed等权。次指标仅预设attended-vs-unattended辅助正确率：同trial比较rA和rB，rA>rB计1、相等计.5，不增lag搜索；与主相关指标分开。

主比较和固定controls可沿用预先确定的6/7 contrast族，但使用18person外部平均，不能套Sparr seen/heldout的2/3+1/3权重。建议另冻交叉训练seed/person20k重采样、主要r的固定M近似Bonferroni区间；辅助正确率仅支持性描述。此数据集的trial共享刺激依赖仍在，完整story/room信息未恢复前只称条件于这些外部样本的泛化，不能用1080trial当独立n或宣称独立刺激/房间泛化。

任何恒定/非有限模型预测导致相应condition失败，不能选择性丢trial。输入幅度离群保留并描述，不为提高结果临时删S11。可报告预声明逐人、leave-one-person-out敏感性及实际覆盖；不据此重新选模型。任何DTU训练/校准需要另冻同预算baseline+方法方案，与此无目标域模型拟合评估分开。
