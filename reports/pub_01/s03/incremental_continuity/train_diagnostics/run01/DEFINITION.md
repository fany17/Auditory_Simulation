# SS07 SOURCE-TRAIN 表示压缩与重叠窗诊断

2026-11-01协调端明确追加本有界诊断。定义先于数据分析冻结；独立目录，不改warm round02的18fit、源码、预算、选择、报告或复核。只用原426条TRAIN、seed1024/2026/31415的既有train latent与训练完成P head，以及既有TRAIN目标。无VAL/test/DTU读取、标签拟合、新NN训练/encoder前向、模型选择或新数据。头部CPU推理是原P功能实现，不保证与GPU逐bit一致。

唯一计算位置2203：`/home/fanyu/auditory_simulation_m6a/pub_01/s03_train_diagnostics_20261101`。CPU计算线程固定1（不超过2），CUDA_VISIBLE_DEVICES为空，nice15/idle I/O；缓存流式读取，batch256，只保留上一窗预测与标量统计，不写预测/latent数组或整段cache。若影响warm吞吐则仅顺延本诊断，warm继续。原源文件/权重只读，失败记录保留，坏记录注明并继续其他记录，不补下载/替换。

## D1：原P预测的窗内与重叠一致性

原P为Conv1d(8,1,kernel32,padding=same)+LeakyReLU(.01)，加载同seed训练完成head。每窗192点、hop96。记录每窗预测均值、population std的逐record分布（n、均值/标准差、5/50/95分位），并记录窗均值变化相对median窗std、窗std变异系数。P输出尺度任意，非物理单位，不跨seed直接比较原始MSE大小。

相邻窗对应同一物理时间的96点：A=前窗[96:192]，B=后窗[0:96]。所有合法相邻对保留raw MSE、均值差、Pearson及归一化RMSE；后者分母为两预测各自centered population variance的均值，分母0时undefined，不设0。每条record独立，绝不跨边界配对。

固定两个方向，不按结果选择：以overlap前48点拟合B=aA+b、后48点评估；再以后48点拟合、前48点评估。不用真实包络标签。要求拟合段A精确非恒定、方差>0、a>0且a/b有限；否则该方向对无有效正尺度拟合，保留失败数量，raw对不丢弃。不裁负斜率为0、不加人工epsilon。只在同一批有效正尺度对上比较held-half aligned MSE与raw MSE，聚合ratio=sum(aligned MSE)/sum(raw MSE)，分母0为undefined；ratio可>1，不截断。完整覆盖率/正尺度比例、scale/offset分布一并报告。

这是同一模型对同一时间两次预测之间的描述性拟合，即使分半检查也不是跨record/被试/测试泛化，不证明差异只由均值尺度造成；两窗encoder上下文也不同。

## D2：六bin均值重建保留的centered latent能量

每窗每channel独立去192点均值，然后按6个连续32点bin求均值并重复32次重建。E0=sum(centered latent²)，E6=sum(reconstruction²)，Er=sum(residual²)。检查E0≈E6+Er（float64相对1e−10）。报告每record及channel的energy-weighted E6/E0、每窗channel比例分布与零能量数量；E0=0为undefined，不写0。图显示record比例的全范围ECDF，保留数值容差范围内的原始比例。

这是48维summary压缩的时间变化能量证据，不等于有用预测信息丢失，更不证明增加表示容量会改善解码。各record/seed分别报告，记录数不是独立受试者数，不做显著性或因果外推。

## D3：TRAIN目标的自相关（共享目标仅算一次）

同一426条record内，仅lag192/384/768 samples（64Hz下3/6/12s），截断到既有common_samples，各lag的两段各自center后计算Pearson。不跨record，短于lag或精确常量标undefined。目标与seed无关，计算一次，不将三个seed当独立目标复制。低自相关不证明条件历史无用。

## 产物与验收

冻结DIAGNOSTIC_PROTOCOL/仅TRAIN清单与来源；D1/D2逐record、D2逐channel、D3逐record CSV；逐seed与coverage轻量JSON；D1/D2/D3三张静态图。均为描述性统计，不新增模型、阈值搜索、p值或新方法声明。流式/公式/正尺度与无标签检查使用合成例子，实际报告完整覆盖与失败。实际图需查看、中文解释并协调独立审阅；本诊断不代表科学goal完成。

18fit的完整真实结果仍优先交付，本诊断服务后续科学决策，不能据其途中结果改变warm矩阵。
