# D18：逐token跨窗适配的协调决策

2026-11-03。状态：**实现和有界工程预检已授权；正式训练待具体源码/输入/账本冻结和必要检查通过后直接执行，不再等待用户许可。** 本文件不是已完成实验或冻结的机器协议。执行者需将实际张量、资源和数值桥接补成可核验实现；不得按VAL成绩改变下列单候选矩阵。

## 证据与问题

D16/D17已封存。历史敏感性本身不等于泛化收益。warm round02曾联合更新encoder，但历史经48维摘要/H16；D12–D16只训练冻结8维latent上的压缩条件器。尚未覆盖的具体问题是：保留64→8投影前的192×64逐时刻表征，以当前query读取过去token，并让当前encoder受相应任务梯度，能否超过同容量当前输入和强模型继续训练？

这是segment-memory adaptation，不预称新颖。既有Transformer-XL等已经建立跨段记忆先例；正向开发结果也不等于ICML/NeurIPS证据。来源审计见[连续性代码核对](../../metadata_review/CONTINUITY_CODE_AUDIT_20261103.md)。

## 唯一模型方向与四臂

三seed1024/2026/31415分别复用同seed已选FT encoder/head和D8 Ccap(.01)，全部共享最强锚点定义。禁止将H接Ccap、而继续训练控制退成弱257参数head。Ccap系数、TRAIN统计、标定固定，读出实现须可微地将任务梯度传给当前encoder及原head；不重新拟合统计/系数。

1. **H_history**：当前Eθ由已选FT初始化并训练；历史E0为同源冻结eval copy。在原64→8 projection之前，以当前192×64 query读取两段384×64历史K/V，attention输出投影零初始化。不得先把整段历史压成单个32维token。
2. **C_current**：与H相同新增结构、shape、初始化和训练规则；K/V内容由冻结E0的当前窗token构造，两个副本，不增加历史EEG。它是容量/输入内容控制；相对时间位置标签等必须与H固定一致。
3. **T_continue**：同FT+Ccap可微读出继续训练当前encoder/head，无新增memory attention。相同监督目标、样本、更新、选点权限；额外历史前向成本另列。
4. **P_prediction_history**：同类attention读取冻结Ccap预测包络历史，当前Eθ/原head同样训练。以一个预声明的时序投影映射到64维，避免scalar复制后channel LN将输入消去。它是语义控制，参数/有效容量差异明示；不是完整DECAF，不输入真实过去目标。

历史E0不接收任务梯度，不称历史encoder也端到端更新。H/C必须隔离额外初始化和history eval操作对current dropout RNG的影响。显式保留原frozen-reference/epoch−1 incumbent；encoder更新后仅关attention不能恢复原θ0，禁作这种退化声明。

## 时间权限与评分

原每窗192点、hop96、64Hz。主历史使用i−5/i−3，原始支持并集为[s_i−480,s_i−96)，严格位于融合current输入并集之前；不用i−4/i−2最后1.5秒的重叠输入冒充新增pure past。两段仍由原3秒模型独立编码，不延长带LayerNorm((64,192))的原模型窗口。

统一仅i>=9应用学习适配；首9窗返回原Ccap，完整原记录/原49人752个VAL块保留。固定同一H检查点的older干预用i−9/i−7，其当前EEG、current encoder结果和全部其他处理不变；这同时改变内容和新近程度，不是纯内容因果干预。

继承原Hann192/hop96连续融合→完整60秒块→person宏r。预先报告所有臂逐seed/逐人/来源、MSE/3秒r、overlap/disagreement与边界情况。固定新当前输入未来权限，不把native3秒离线表示称逐sample因果。不同时修改D9的物理输出坐标或目标。

## 完整数据与固定上限

只原TRAIN426/252454窗、VAL49/30500窗及原train-only归一化；无新数据/权重/患者。12fit各计划20epoch，每轮原252454个目标窗呈现（含无新梯度的fallback窗），global64、3945个更新位置，共946800个更新位置、60588960目标窗重复呈现。执行时须分别列真正有效适配窗和实际optimizer更新，不能把fallback或空梯度伪报有效学习。

当前encoder LR=1e−5、原head/新增模块LR=1e−3；Adam、无新增wd/clip，统一D16负centered-cosine定义和确定性全局shuffle语义。原VAL60秒/person严格最早最大值选点，候选含旧Ccap incumbent−1及20轮，不在3秒/60秒准则间择优。所有新臂使用同规则；不因初期阴性换宽度、长度、LN、LR、损失或增预算。

H/C/T/P的实际encoder前向、历史访问、反传、cache复用、唯一数据、sample重复呈现、optimizer步、GPU/CPU/墙钟和共享上游成本分别计账。同监督目标/更新不等于总FLOPs相同。

只server2203计算。当前两L40空闲，可两fit并行；建议每worker GPU≤32GiB、RSS≤48GiB，总wall24h、新run≤768MiB、home与SATA各保留20GiB。正式协议以预检实际资源/吞吐校核后冻结，不用猜测耗时停工。仅存可变参数best/last和轻量表；frozen E0/Ccap以来源pin复用，不复制到每个checkpoint；不物化大历史cache或保存全VAL预测数组。可在服务器RAM缓冲只读现有数组支持shuffle。禁止删除旧资料、抢占无关进程或服务器Git。

## 必要工程门与科学判读

代码和已知输入身份独立核查；真实原TRAIN小批验证初始锚点、Ccap Torch/NumPy同latent读出、raw重编码数值桥、finite梯度和attention内部学习、frozen E0/统计不变、record/phase/未来支持、P历史信号可达、参数和预算。区分数学同latent等价与不同GPU批次路径误差，门限在新VAL结果前确定；失败留下证据修复，不反复神经重放以追逐无意义逐bit相同。工程更新不继承到正式fit。

具体源码/协议审定后执行完整矩阵，不停在smoke。H若不能在三个配对seed均超过C、T、P及冻结Ccap，并在同检查点真实/older比较中一致获益，则不能认定该候选建立额外历史贡献。这个标准是反复开发VAL上的有限推进门，不是显著性、全局否证或独立确认；全部失败和回退incumbent保留。

旧内部test、DTU及ds004408均冻结。本轮不以其分数调参、复测或宣称新的外部泛化。持续研究goal保持ACTIVE；D18即使阳性仍须进一步审查竞争方法、机制、可重复性及独立证据。
