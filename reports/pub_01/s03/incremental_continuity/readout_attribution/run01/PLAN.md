# D6：强current读出增益的共享约束与支持范围归因

状态：PROPOSED_NOT_RUN，PUB-01-S03_SS07内独立冻结run。D4/D5已交付且不扩大网格。2026-11-01。

## 问题与共同起点

D4 C/Ccap开发VAL高于原P，但同时改变原Conv32共享约束、latent支持、平方特征、目标/solver/正则。本轮只检验：在完全相同P*残差、MSE、标准化和正则泛函下，解除latent读出权重共享、扩大latent支持、增加平方特征各对应什么表现？不把闭式MSE相对旧Pearson head的差写成纯loss因果效应；严格同架构/初始化/优化预算的loss消融仍未做。

继续原TRAIN/VAL及seed1024/2026/31415。P*=aP+b、target Y、D4 current latent/P特征的TRAIN逐坐标标准化与块缩放、共同i≥4拟合、首4窗P*回退均原样固定。P*是原P正仿射，保留原3秒离线权限。所有条件不读历史输入；连续性只在评分时描述。无NN优化/encoder或head前向、新数据/权重、test/DTU。

记D4 x^Z_i∈R1536、x^P_i∈R192，均已按D4 TRAIN统计及sqrt(1536/192)缩放。所有线性臂输出 P*_i + x^Z_i B_Z + x^P_i B_P + b。**B_P始终为共同、位置不共享的192×192分支，b始终192维自由截距**。因此S32/U32的local只指Z残差分支，不是整个模型只看局部原始EEG；纯P*另作零残差锚点。保留该共同分支是精确复用D4 C的必要条件，不隐藏它。

## 最小矩阵与精确复用

| 条件 | Z残差约束 | 共同P分支/截距 | 参数总数 | 本轮求解 |
|---|---|---|---:|---|
| P* | 无新残差 | 无 | 0 | 不拟合 |
| S32 | 相对offset−15…+16的8×32系数跨192输出位置共享 | dense P192×192 + b192 | 37312 | 新拟合 |
| U32 | 与S32逐输出相同local32支持，各位置独立系数 | 同上 | 84160 | 新拟合 |
| C | 完整current Z1536→192，无共享约束 | 同上 | 331968 | **精确复用D4** |
| Ccap | C再加完整标准化Z平方1536→192 | 同上 | 626880 | **精确复用D4** |

输出位置t的两种local分支只使用latent索引t−15…t+16，超0…191以0填充，不循环、不移动坐标。S32/U32同支持，没有同时解除共享和扩RF。U32仅保存真实有效局部系数：8×[192×32−256]=47104，加共同36864+192得84160；无效pad位置不是可训练参数。

“共享”在固定的D4逐坐标标准化后的Z坐标中定义。其均值/std可随latent位置不同，因此不是未经标准化raw latent上的原生Conv32等价实现，更不是原native head复现。当前encoder本身可在3秒内全局交互，扩大latent读出RF不等于扩大原始EEG时间权限。

固定λ={.0001,.01,1}，same (1/N)||Y−P*−XB−b||²_F+λ(||B_Z||²_F+||B_P||²_F)，b不惩罚。Ccap另含对应平方系数范数，完全沿用D4。S32约束下的正则必须是**展开到完整B_Z后的Frobenius范数**，不能给共享系数另用未乘重复数的λ而混入不同收缩。

若offset k在192输出中出现m_k=192−|k|次，以θ_(channel,k)=sqrt(m_k) w_(channel,k)参数化，共享feature为local Z/sqrt(m_k)，于是||B_Z||²_F=||θ||²。每个位置的P分支可用同一(G_PP+λI)消元，256维共享θ通过Schur complement求解；再回代各位置P系数。U32每个位置仅解有效Z local列与共同P列对应的中心化ridge，统一一个λ，不对192个输出单独选择λ。

直接复用D4 TRAIN_MOMENTS的centered G/H、x_mean/residual_mean，无新TRAIN latent/target统计遍历；只取current Z/P相关块。C/Ccap的18套原D4系数、λ和评分来源原样复用，不为D6重新拟合。新S32/U32×3seed×3λ=**18套model-λ解**，另18套D4精确复用，共36套评分表；U32内部1728个按输出的小线性系统、S32九个共享Schur系统及P回代的实际开销单列，不把18套模型虚称18次标量求解。PCA/新normalization拟合0。

有效df统一报告整个192输出系统（不含192截距）：U32按各输出之和，C/Ccap以D4每输出df×192，S32按其耦合正则化系统hat trace。名义参数匹配不保证有效容量匹配；S32与U32参数数有意不同，差值是解除共享约束的开发证据而非等参数性能优越。

## 原选择规则、次要连续性指标与科学解释

每臂一个跨3seed共享λ，原VAL60秒/person宏均值选择，零残差P*亦为候选，严格高于P*才选拟合；同分优先P*，其次更大λ。C/Ccap必须重现原D4共享选择，否则先定位实现差异，不能改λ找一致。保留全部三个λ/逐seed/逐人/逐块/3秒与60秒，不据次要指标挑选。

主要预设描述比较：U32−S32（同支持、解除共享），C−U32（扩大Z latent读出支持），Ccap−C（平方特征扩展，容量亦增）；分别报告相同三个λ及各臂共享选择后的配对结果。S32−P*只说明整个新MSE残差实现的变化，**不单独归因loss、原生卷积能力或共同P分支**。若共同P分支可能解释收益，当前矩阵不进一步分离，明确为缺口，不悄悄加P-only模型。

预设次要描述：所有原VAL相邻窗同物理96点overlap在Hann融合前的预测差异，每record reset，共30500−49=30451对/seed。报告同单位pooled overlap MSE、record分布/有效非恒定覆盖，必要NRMSE按双方中心化方差均值计算并明示分母无效覆盖，不按预测删掉准确度评分样本。比较所有模型与P*（不是任意scale原P）；不做逐窗/逐record仿射拟合/校正，不用未来窗帮助预测当前窗。

**连续性不能单独当收益**：同时报告原3秒/60秒Pearson及同一输出单位的target MSE（3秒窗MSE与60秒→person宏MSE，说明overlap重复计数和块计权）。常数/过平滑可有低overlap误差，却可能低r/高target MSE。连续性仅描述，不参与λ/候选选择，也没有新增一致性训练loss。

同一原相邻窗/固定Hann已有预测可直接补充精确分解：归一化权重α、1−α和真值y下，α(p1−y)^2+(1−α)(p2−y)^2 = [αp1+(1−α)p2−y]^2 + α(1−α)(p1−p2)^2。报告同物理覆盖点的Hann加权单窗MSE、fused MSE及disagreement项（单窗覆盖时该项0），按同60秒块/person聚合并核对恒等式。后项也表示融合相对加权单窗误差的分散收益，不能将更一致自动等同更准。仅复用现有读取/固定权重，无新拟合/学习融合权重/逐窗校准/一致性loss或选参。

原VAL仍为已知被试/不同clip/部分同书来源的反复开发集。参数数、归一化坐标、正则/solver与原P不同；本轮不证明独立泛化、loss纯因果、历史机制、实时因果、显著性、等价或ICML/NeurIPS新方法。Ccap−C也同时增加函数类与容量，不独称非线性因果效应。

## 冻结、必要验证、预算与交付

新2203 ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_readout_attribution_20261101。源码/PLAN、D4原协议/实际review/moments/transforms/全部被复用coef/P_CHECK、公共mask及VAL latent/P/targets shape/dtype/size/mtime、record长度在正式拟合前冻结。无覆盖D4/D5、服务器Git、患者或其他节点。

必要验证仅：小合成多输出系统共享Schur解对直接增广受约束最小二乘（含m_k重复权重）；S/U相同padding/offset和非局部Z扰动不影响该Z输出的canary（共同P输入保持固定）；U32 band约束、共同P/截距与首4回退；复用C/Ccap实际原评分/共享选择一致；独立review源码/实际18新解+18复用/共同mask与关键聚合。不得另建NN raw重放。

CPU/BLAS/OMP≤8、GPU0、RAM≤8GiB、wall≤30分钟、新派生磁盘≤512MiB，余量≥20GiB、cache总上限不变；顺序seed，流式VAL预测/overlap统计，不写N×d新特征cache。仅复用TRAIN moments求解，VAL新计算36组模型和3个P*的描述评分，精确计次数；source/hash、求解/预测/score/secondary指标与共享上游成本分列。达到上限或实际错误保留失败，不自动扩预算/重试。

本地仅轻量代码/协议/结构化表/静态图/完整中文REPORT/REVIEW；moments、coef和预测仍2203。方案先交协调审定，当前NOT_RUN；无论正负，D4/D5结论不改，持续goal不因工程PASS或反复VAL阳性完成。
