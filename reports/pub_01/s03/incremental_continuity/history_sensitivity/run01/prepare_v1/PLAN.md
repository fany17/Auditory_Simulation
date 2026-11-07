# D17：固定当前输入的四检查点历史敏感性诊断

2026-11-02；PUB-01-S03_SS07同一执行节点内的有界诊断。D16已封存commit7a17fc2，收益门FALSE；范围与授权见[协调决定](COORDINATOR_DECISION.md)。本阶段不是训练或新方法性能比较。

只读H2026 selected5及三个seed last19，原前8条TRAIN manifest每record local0/8/9/10/11/12/13/14的64metadata位置，48有效窗。当前z/Ccap恒定，near i−5/i−3和older i−9/i−7各一次，4×2×48=384残差窗、8次forward/至多8输入梯度backward，0fit/optimizer/encoder/head。无新VAL/test、完整TRAIN遍历、目标读取或新数据/权重；两个所选零头不重推理。

实现只从原memmap索引48 current及指定两段history，按原TRAIN transforms/Ccap系数计算这些current，避免调用构建完整TRAIN的Data。两条件同一current/anchor tensor并逐tensor核相等。直接调用冻结D16 Conditional，hooks读取既有forward节点，不改网络数学；model.train()保持cuDNN反传路径，全部参数requires_grad=False，仅history可导。固定探针为sum(residual*sin(arange192*.017))/48，一次autograd.grad，不用目标loss。所有权重state_dict前后精确核验，0optimizer。

每节点报告global RMS/max、axis0跨48窗逐坐标中心化后RMS；对两token节点另列两个固定token位置的跨窗中心化RMS，不能混作跨时间方差。near/older差RMS/max、逐窗差RMS min/median/max、完全相等元素比例与窗口数。相对差分母为sqrt((mean(near²)+mean(older²))/2)，另列axis0中心化后的pool分母；精确0返回null，无epsilon。输入、LN前/后token、GRU两步/末h、FiLM gamma/beta/tanh与实际.1调制、current q、调制后current、residual及float64最终预测全部只保存统计。额外tanh/差值/归约属诊断算术，非额外网络遍历，实际CPU/wall计入。

先0NN prepare冻结实际source、D16协议/独立/科学审阅、四checkpoint SHA和TRAIN caches stat身份/继承SHA；小文件TRAIN变换/系数/calibration核已有SHA。大cache不为本轮再次全文件hash读取；不读数组值，纯合成标量轴/零分母测试在2203完成且另计CPU。实际冻结源/协议交协调核轴与预算，批准后直接执行384NN，不重复向用户要许可。

新ROOT `/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_sensitivity_20261102`，上限900s/RSS8GiB/GPUallocated2GiB/ROOT8MiB/两盘各20GiB；选一张合法空闲GPU顺序运行，不杀他人。无空闲卡在同900s内等待，0NN；超界保留FAILURE与实际计数，不自动神经重跑。无新数组/预测/权重文件，内存激活随进程退出释放；只回轻量JSON/表/必要静态图及中文解释。

Technical核源/输入/检查点、48位置边界、同current/anchor/权重、8/8/384精确计数、finite、统计轴与相等/零分母恒等；协调独立review pass核源/标量算术及实际完整四检查点，不新增NN。Acceptance为诊断PASS_WITH_LIMITATION，不直接宣称表示塌缩、历史无信息或方法有效；内容与recency共同改变、older可能保留信息、单方向局部导数/有限样本/float32限制明确。Parent保持公共数据和原split/claim边界。完成后记录当前SS07/owner，后续方法由协调具体决定，不自动训练或改LN/LR。
