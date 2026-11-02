# D17：固定当前输入的历史敏感性诊断（不训练）

2026-11-02；PUB-01-S03_SS07 持续目标下的协调具体决定。用户已授权公开 EEG 实验和2203可用计算，本项不扩数据/模型范围，不需要再次请求用户许可。D16先完成科学报告与归档；本诊断只补足下一步方法选择所需的机制证据，不能替代超越强基线或独立泛化目标。

## 已核事实与待检假说

D16九fit全部20轮，独立复算REVIEW SHA c6696649ecec5d0090b042b1ce51423a6f3bdc0783367e257fef73a4a8dcc77d，48624数值对比通过。H/C/P所选VAL平均r分别0.1823787401/0.1823399396/0.1824220842；H仅seed2026选epoch5，另两seed回零输出锚点。该非零H的near-minus-older为-1.4275e-10。H各seed epoch1–19抽样首批token/GRU模块梯度均非零，而TRAIN改善未带来稳定VAL历史收益。

这些事实不能区分：历史输入经编码后变化很小、GRU/FiLM压缩了变化、输出投影抑制了变化，或所检两段历史恰巧产生相近作用。参数梯度非零也不等于输出对历史内容敏感。不得直接宣称表示塌缩、历史无信息或生理机制。

## 唯一诊断与检查点

只读D16既有H_full的四个训练检查点：
- seed2026 selected(epoch5)；
- seed1024/2026/31415各last(epoch19)。

两个其它selected为已知零输出锚点，无需重做其神经推理。不是挑选新候选或选点；这四个检查点分别回答唯一非零所选模型及三seed训练终点。所有权重只读，不改原文件、不重训、不改LN或loss/LR。

只用原TRAIN元数据预先固定的同一小批：前三轮工程已使用的前8条manifest记录，每记录局部索引(0,8,9,10,11,12,13,14)，64个metadata位置中48个合法历史位置。按可用性固定，不读取分数选择记录、不增加VAL/test/DTU/ds004408访问。每seed沿用其既有TRAIN变换与Ccap。

对每检查点仅两次forward：
1. 原near历史(i-5,i-3)；
2. 同record older历史(i-9,i-7)。

两次的当前z、Ccap、样本顺序逐tensor相等；记录源record/local/global索引并核历史严格过去边界。不能随机拼患者、跨record donor或引入真实过去包络。保留既有CUDA/float32算子与最终float64锚点合成；选择一张合法空闲GPU顺序执行。无dropout/持久跨窗state。输入梯度沿D16实际验证的model.train()反传路径；此网络没有dropout或running统计，权重全部requires_grad=False，唯独历史输入requires_grad=True，无optimizer。避免将cuDNN GRU切到eval后才尝试反传而造成不必要失败和重复forward。

## 观察量（仅标量/元数据输出）

在这两次原forward通过hooks或显式读取既有中间节点，不增加网络遍历。记录：
- 输入历史、末GELU前置token（LN前）、LN后token、GRU两步状态及末h、FiLM原gamma/beta和实际tanh调制量、输出残差、最终预测。
- 各节点全局RMS/max_abs，固定token位置跨48窗中心化RMS（明确其平均轴，不混同跨token时间方差），near与older逐元素差的RMS/max_abs及相对量；分母为0时明确null，禁止隐含epsilon制造结论。
- 逐窗差值的min/median/max或计数，及near/older完全相等元素比例；float32数值下等于0只作局部计算观察。
- 固定当前时的历史输入导数：每次forward只做一次autograd.grad，标量探针为residual与固定sin(arange(192)*0.017)向量的内积除以有效窗数。不用目标loss挑探针或计算新指标。仅对历史输入求导，权重不更新；记录nonzero_scalars/L2/max_abs/finite。这是一个局部方向导数探针，非所有方向、全数据或信息论证明。

不保存任何新预测/特征/激活数组；中间张量只驻本轮进程内存。报告必须逐检查点展示结果，不能只选有利seed。near/older同时改变内容与recency，older也可能保留相关信息；有限样本、所选checkpoint和浮点精度限制必须保留。

## 精确上限与执行

4检查点×2条件×48有效窗=384残差NN forward，最多8次输入梯度backward，0学习fit/optimizer更新/encoder/head forward，0新数据/权重下载。不重做D16全VAL复算或完整TRAIN遍历。必要的纯标量统计轴/恒等性小测试另计CPU，不能藏额外网络forward；若实际需求超出此界，先交具体原因给协调，不自行增候选。

新ROOT：/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_sensitivity_20261102
本地：reports/pub_01/s03/incremental_continuity/history_sensitivity/run01
现有D16源、协议及4checkpoint SHA/输入统计身份全部绑定。prepare/实现后由协调核源码及统计轴，实际通过即执行，无需用户再确认。CPU/壁钟/RSS/GPUallocated/真实NN/backward分别计账；运行<=15分钟，RSS<=8GiB、GPUallocated<=2GiB、两盘各20GiB余量、新ROOT<=8MiB、无新数组、无服务器Git。没有空闲合法GPU时等待，不杀他人任务。

交付一份可复算的小表与中文解释，区分观察、假说和未识别机制；不能从该诊断直接宣布新方法有效。不启动新训练/正则或归一化位置网格，不重开已冻结外部数据。下一项方法决定由协调基于实际证据另行具体界定，持续总目标仍为可重复超越强baseline且具备投稿所需证据。
