# 单候选：完整FT时序条件残差及两个匹配训练控制

状态 **COORDINATOR_APPROVED_IMPLEMENTATION_PENDING_FREEZE**。协调已批准该唯一结构原TRAIN/VAL执行；盘点阶段无fit/NN，正式训练前必须冻结源码/参数/支持与必要工程证据。不是新节点、模型网格或外部结果后的适配重选。只用原TRAIN426/VAL49；旧test/DTU/ds004408冻结，不进入源码loader、选择、诊断或新评分。

## 可证伪假说和冻结锚点

H：给定当前完整FT latent与已冻结强FT+Ccap，严格过去的完整时序能经非线性条件映射降低当前包络预测误差，且其贡献超过同容量current-only与预测包络历史prior。这里不把历史预先命名为神经状态、噪声或刺激机制。

固定既有三seed1024/2026/31415的FT encoder/head（均原VAL选epoch4）和D8 Ccap λ=.01不更新。Ccap完整TRAIN变换、系数及P_FT*标定原样复用；不换小backbone、不重新编码。新增模块输出为`Ccap_i + 0.1 * residual_i`，末输出层权重/偏置全0初始化，初始函数精确回锚点。0.1沿用既有残差约定，不从外部效果选取；仍须技术核初始锚点逐块一致。

只选两个完整3秒token构成6秒有序历史，这是能保留“跨两个过去片段顺序”的最短序列；不试3/6/30秒网格。其选择源于最小可证伪问题，不来自ds004408或旧test成绩。

## 同一函数、同一活跃容量的三个训练臂

当前`z_i`为8×192，每个history token也保留8×192；无预先六bin均值、PCA降维或目标历史。下列具体结构只有一个宽度/深度，不搜索：

1. H/C共享结构的token编码器E：Conv1d(8,16,k9,pad4)→GELU→Conv1d(16,16,k9,pad4)→GELU→flatten3072→Linear(3072,32)→GELU。flatten到可学习投影前保留192个位置，不以均值替代；E共101824参数。**输入完整不等于无损表征**：3072→32是监督学习的瓶颈，GRU/FiLM也仅32维，阴性只约束这个条件器。P_prior直接用1通道包络，首层Conv1d(1,16,k9)只160参数，E共100816，不复制/零填充伪造容量相等。
2. 过去两token按时间顺序经同一GRU(input32,hidden32)，每个预测窗从零状态开始，只跑两步，取末h32；GRU共6336。没有跨batch/record隐藏状态或截断BPTT歧义。
3. 当前完整z展平1536→Linear(1536,32)→GELU，得q32，共49184参数。
4. Linear(h32,64)拆为γ32/β32，共2112；`q' = q*(1+0.1*tanh(γ)) + 0.1*β`。Linear(q',192)产生残差，共6336，末层零初始化。显式乘法使当前映射依赖历史；β也允许输出prior，不先宣称增益来自乘法本身。

H/C总新增可训练量 **165792/臂/seed**；P_prior为**164784/seed**，是输出历史语义控制，不是严格参数匹配臂。encoder/head/Ccap新增可训练量0。这里是按层公式的预期registered数，未实例化网络；实际requires_grad、第二次更新后的梯度可达数和当前控制GRU recurrent weights须在原TRAIN工程核验，不把预测值写成实测。

|训练臂|E/GRU两个输入token|当前q/FiLM/输出|作用|
|---|---|---|---|
|H_full|同record z_(i−5), z_(i−3)|三臂相同完整current和参数数|完整严格过去时序条件器|
|C_match|z_i,z_i两次；两GRU步均执行|同上|只有当前信息；匹配结构、更新、token次数和名义参数，避免额外容量冒充历史|
|P_prior|同record两段冻结Ccap预测包络，各为1×192点，不输入真实过去Y|同一current/FiLM/输出；E首层只读1道|有强当前及预测包络历史的语义对照；不是官方DECAF复现|

H/C token用既有D8 TRAIN latent mean/std逐坐标变换；当前q同一变换，原constant坐标仍置0。P_prior预测先用既有D8 TRAIN P_FT*输出mean/std变换，不重估新VAL统计。H/C虽同registered参数和两个GRU步，但C两次输入相同，信息维数、状态分布和有效函数容量不等价；不能只数参数就声称完全控制容量。P_prior输入信息秩和首层参数更少，关键容量归因以H对C_match为主，P_prior回答输出历史能否解释。若工程发现预期活跃参数不一致，应先rework实现，不能按表现掩盖。

## 过去支持、错配干预及完整评分

当前原始支持s_i=96i，当前窗[s_i,s_i+192)。参与该预测窗Hann坐标的current输入保守最早起点为s_i−96。H_full两个token按时间为i−5/i−3，合计[s_i−480,s_i−96)，6秒，完全位于该边界之前；内部两个token本身不重叠。历史结束在当前窗起点之前1.5秒，并非紧贴当前单窗的前6秒。

唯一无拟合错配：对**同一H_full权重**保持current/Ccap/所有处理不变，用i−9/i−7的更早同record连续6秒替换历史，即[s_i−864,s_i−480)。这同时改变recency和内容，不能当纯内容随机干预；older不是零信息。不给短donor重复补齐、不引入未来或其他record。

为全部条件和干预共同可用，统一新残差只在i>=9拟合/应用；前9窗完整返回Ccap，所有记录/原752块保留。TRAIN248620个有效更新窗、3834个fallback窗；VAL30059个有效残差窗、441个fallback窗。每batch仍按原global64顺序，record-local lookup，不能全局减offset跨记录。输出prior同样只看已存在且位于过去的冻结预测，不递归使用真实Y或当前/未来目标。

主评估沿用原FP64 Hann192/hop96→完整60秒块→person宏r、49人/752块。共同cold/transition/warm切片按实际非零Hann支持在预测前生成，不按分数筛；MSE、3秒r、单窗极值/overlap/disagreement及边界/tail同时报告。无逐窗min/max重标、无lag或test-time校准。

## 一个预算与一个选择规则

- 三seed×三个训练臂=**9个fit**，每fit固定20epoch，原global64、252454窗/epoch（含fallback位置）、78900次optimizer step；总180epoch/710100步，45441720个目标窗口呈现。有效残差目标窗/epoch248620；计账区分fallback与有效窗，不能只报全部窗口为有效历史学习。
- Adam lr=.001、weight_decay=0、无grad clip，沿用已有frozen-head/warm优化约定；H/C只训练新增165792参数，P_prior164784。三臂同seed相同shape模块拷贝同一初始化，按组件隔离RNG，使prior首层维数差不改变其他模块初始权重；H/C全部权重逐tensor相同，batch顺序相同，无dropout。所有fit跑满20轮，不按中途分数追加epoch或删seed。
- 训练统一**未加Hann权重的逐点窗口MSE**：每窗192点都对其固定目标负责，在原global64平均（fallback贡献常数、无新残差梯度）。不只优化融合loss，也不同时试Pearson/MSE网格。选择依据是此前原TRAIN/VAL [D9窗口退化证据](../../physical_objective/run01/REPORT.md)：纯融合目标允许相消大窗口值；本轮目标逐点约束输出可辨识性。不是根据已见ds004408分数选择loss。它与原FT负Pearson目标不同，三新增臂同用，既有强Ccap原本也是窗口MSE残差拟合；差异如实披露，不归因为单一历史因素。
- **建议**所有新臂只按原VAL60秒/person r选checkpoint，候选为incumbent−1及0…19，最早最大值；完全相同优先incumbent。这一选点相对旧FT的3秒loss规则有所改变，须由协调在训练前明确冻结，不能事后在两准则间取优。锚点权重仍原样，不重新选FT。
- 不扫宽度/层数/历史长度/正则/LR/残差scale；不先用screen挑格再只full赢家，不以外部成绩选参数。缓存原位memmap分批读取，新增history cache=0；仅保存每fit已选模型、last模型和日志，新增run总量建议<=128MiB。最多两张空闲GPU并行不同fit、每worker GPU<=2GiB/RSS<=8GiB、总wall<=6h，BLAS<=8；这些是协调批准的待冻结上限、非实测耗时。超界记录失败、无自动扩大或重试；每次准入复核双盘20GiB及当前全SS07数组边界。

NN encoder/head新前向0，但新增残差NN训练和VAL前向不是0，须完整计账；原FT预训练/head/FT/Ccap成本继承并另列。输出prior生成既有Ccap预测只是固定读出计算，单列CPU，不称全流程零前向。

## 何种结果能否证

预定开发判读至少同时满足：H_full相对冻结Ccap、C_match及P_prior的60秒/person差值三seed均正；且同一H_full的true历史相对older干预在共同支持上有一致正向差。全部逐seed/逐人/逐块、full与固定warm切片完整报告。任一条件失败就记“本有界候选未建立额外历史贡献”，不将H−旧弱P的阳性替代主比较；不能把未满足当历史普遍不存在。

人/来源依赖与反复VAL使用使这仍是开发门，不是显著性或独立确认。建议同时报告按被试配对差与按五个VAL family的方向/覆盖描述，两个共享TRAIN family与三个未共享family分开描述，不能把block或seed当独立人数，也不临时对新family重加权后称确认。

本候选有意只回答这一有限条件贡献，不证明FiLM乘法、神经状态或连续性原理独特；β/prior、刺激连续性和被试状态仍有替代解释。若全部门满足，再由协调决定具体机制问题及独立证据，不自动重开已看外部集、追加竞争方法或扩模型网格；若失败，保留完整九fit，不再换名追分。

## 执行前尚待协调冻结

确认这一单结构/九fit及60秒选点、统一i>=9与严格支持、梯度可达/初始锚点桥、参数/预算实际可行性、固定mask和全部输出/成本验收。审定后在原SS07授权范围实现，不重复向用户索取已有启动许可；**当前进入实现/冻结/工程核验；必要检查通过后直接启动正式九fit，不停在smoke或重复等用户许可**。


2026-11-02协调明确修订：预算由5轮改为20轮，理由是既往warm多数末轮仍最优，避免极短优化混淆阴性；不是追分或收敛承诺。总710100更新，两卡可并行、墙钟6h，其余网络/支持/目标/三seed不变。工程临时前向和两步检查更新全部另计，正式fit从干净共同初始化重建；吞吐预计超界时报告实际吞吐后再处理资源，不按分数延长。
