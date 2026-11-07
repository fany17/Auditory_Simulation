# SS07只读盘点：历史阴性结论的覆盖范围与下一可证伪问题

2026-11-02。本轮仅核已有源码、TRAIN/VAL JSON/stat/NPY header和依赖元数据；**新fit=0，NN前向=0，数组数值读取=0**。不读旧test、DTU或ds004408载荷/预测，不作数值筛分；这些评估已冻结。本目录是SS07内部research_gap/run01，不创建新节点。

**结论：完整FT的8×192时序latent、非线性条件历史、冻结强FT+Ccap锚点与同容量current控制的组合尚未实际测试。** 既往阴性覆盖各自受限表征/函数/目标，不是这一组合的阴性，更不是全部连续历史无用。可复用三seed既有FT缓存直接实现一个有界条件器，无须缩小backbone或重编码。具体单候选/九fit/选择规则见[CANDIDATE_PLAN](CANDIDATE_PLAN.md)，供协调审定，本轮不启动训练。

## 1. 已实际测过什么

下表以已完成报告、实际训练completed/gradient元数据和源码为准；历史PLAN中的PROPOSED_NOT_RUN是其早期状态，不覆盖完成证据。参数是每seed，闭式系数数含192输出截距，不与可训练NN参数混称。

|阶段|历史实际入口与分辨率|backbone/锚点、目标和可训练量|强current控制与阴性边界|
|---|---|---|---|
|round01|当前仍完整8×192；过去先adaptive mean到8×6=48，约0.5秒/bin，再进入H16状态；按i−2/i−4奇偶流，记录/phase/epoch reset、更新后detach|native冻结encoder；P和残差从同随机head起步，不是已训练P上加历史。原3秒负Pearson；M3另加.01的48维未来latent摘要预测MSE及当前重建teacher控制。M1/2/3/4/5实际registered且gradient-reachable为3169/5841/6657/5161/6657（含257 head）|M4同core current=5161，静态容量=5161；output-prior=2167、静态alignment=281。当前控制只用于对应阶段，不是FT+Ccap。M4未稳健超过P/current，不外推未保留的完整过去时序。M3/M5也不是完整CPC/任意未来建模或Kalman。|
|warm round02|仍8×6过去摘要和M4 H16；current臂把当前六bin作为六步序列，recurrent weights实际参与；非完整1536历史|从训练完成P起步，冻结encoder或joint更新；所有18fit固定5epoch/19725更新，原3秒负Pearson。frozen P=257、current/past=5161；joint P=3372617、current/past=3377521，三seed元数据一致|past对同容量current不稳定，joint三seed均负。joint训练中encoder会改变，不等于在已选强FT缓存+Ccap锚点上测试完整历史。冻结/联合行还改变随机行为，不能当纯梯度因果效应。|
|D4条件历史|near完整8×192=1536维，i−2；older i−4；summary48与TRAIN PCA48。完整时序经固定线性展平进入加性岭|**仅native P encoder/cache与其已训练head**；P*正仿射、current dense C/Ccap。目标Y−P*的192维均方残差+岭正则，54解。C/CS/CF/CO/CF48/Ccap系数分别331968/341184/626880/626880/341184/626880|CF/Ccap同名义3264维但函数类/有效df不同。CF未超过Ccap；这是真正full1536历史已测试，但仅native加性线性残差。i−2不重叠当前单窗，不完全在融合current union之前。不能记成“FT full-history已阴性”。|
|D5交互|native current TRAIN PCA32，current前16×near后16的256个双线性乘积；并非完整1536任意条件映射|冻结**D4 native Ccap**；共同A=current32+near16+older16。Add12480、CC/Near/Older61632闭式系数，36解；窗口MSE+岭，3秒/60秒均报告|CC实际含共同历史加性A，不是纯current。Near未超过Add/CC的预定三seed一致条件。只约束该PCA32/16×16二阶函数，不代表全FT表征或一般非线性条件器。|
|D11观测状态|原64道EEG局部协方差→2080个logvar/correlation descriptor→TRAIN pooled PCA8；30秒过去在融合current union之前；current latent另PCA32|冻结**FT+Ccap**，current q32+state h8+q⊗h256。Add6336、四state各57024闭式系数；15解+6无拟合干预。窗口残差MSE+(.01/192)系数惩罚|CurrentCov/RecordPrefix/PersonTRAIN及Older/Neutral控制已做，Past未满足六条件。FT强锚点已测，但历史是8维二阶统计、低阶交互，未保留8×192完整时序。不能外推所有观察状态/非线性条件历史。|

证据：[round01](../../run01/REPORT.md)、[warm](../../run02/REPORT.md)、[D4实际完成](../../conditional_history_probe/run01/REPORT.md)、[D5实际完成](../../history_interaction_probe/run01/REPORT.md)、[D11方案与实现边界](../../observation_state_probe/run01/PLAN.md)、[实际参数逐seed核验](PARAMETER_AUDIT.json)。round01当前窗完整与过去摘要应明确区分；D4“完整历史已测”和“完整FT历史未测”可以同时成立。

未测组合包括：冻结已选FT encoder/head+Ccap上、完整时序过去的加性dense探针；完整时序过去的非线性条件残差与同架构current；在这个强锚点上、同一训练/选点权限下的预测输出历史prior及内容错配。这里列的是证据缺口，不是必须逐一补跑的实验清单；下一步只提一个候选，不复制D4/5整个网格。

## 2. 既有FT缓存确实可用，但不能把header检查称为数值复核

2203只读核验三seed1024/2026/31415的TRAIN/VAL Z/P共12个NPY header，及2个共同窗口target header、46个独特全record envelope header；shape/dtype/C-order/长度、475记录offset与同源manifest一致。Z分别为(252454,8,192)/(30500,8,192)，P为(N,192)，均float32；三seed现有Z/P共5867335680B。未在本盘点读取latent、目标或统计数组值，更没有新桥接/评分。

原D8 `TRAIN_TRANSFORMS.npz` / `TRAIN_MOMENTS.npz` / Ccap(.01)系数和FT checkpoint身份已在既有冻结协议中记录；本轮只查元数据/路径，不重新拟合、加载checkpoint或复验数值。正式候选工程若获审定，应按旧pin验证并重建合法TRAIN/VAL锚点，不重编码、不读取任何外部集。完整清单见[FT_CACHE_METADATA_AUDIT](FT_CACHE_METADATA_AUDIT.json)，服务器路径为`/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_readout_transfer_20261101/seed{seed}/`。

每窗原始支持为[96i,96i+192)，64Hz。i−2结束在96i，不重叠当前单窗，却可能与参与当前融合的i−1输入重叠。对内部窗保守采用融合current输入并集的最早起点96i−96：K个互不重叠历史token按时间顺序为i−(2K+1),…,i−5,i−3，原始并集[96i−96−192K,96i−96)。不要在已拼接大cache上直接减全局index；先检查record-local i和offset，绝不跨record或phase。仍是以原3秒双向encoder为当前输入的离线方法，不是逐sample实时因果声明。

盘点中的K=1/2/10仅是3/6/30秒几何/资源例子，不是拟合网格。保留原Ccap首4窗规则后，最低i=max(4,2K+1)，对应TRAIN可加残差250750/250324/243508窗，VAL30304/30255/29471窗；完整30500窗、49人/752块主评分不删除，cold-start返回锚点。候选另因older干预统一从i>=9开始，见具体方案。

**不应物化N×K历史副本。** 例如K=2仅TRAIN单seed副本就需3102154752B；当前home高于20GiB保留量约1690824704B，单这份复制已超可用余量。按memmap分批取索引，global64/K2历史输入只需786432B float32（仅输入，不是总RAM实测），新增历史cache=0。保持所有既有数据与外部缓存原位；新训练预算必须另在协调冻结时核全NPY/双盘限制，不能借此盘点自动放宽资源。

## 3. 实际依赖与内容错配的可行性

|划分|记录/窗口|人数|story标签|source family|
|---|---:|---:|---:|---:|
|TRAIN|426 / 252454|70|41|37|
|VAL|49 / 30500|49|5|5|

49个VAL人全部见于TRAIN，VAL每人仅一个record；story交集0，但source family共享`audiobook_5`和`audiobook_13`（不同parts），另外三个VAL family是podcast_15/21/30。family依据发布命名归并，不是本轮音频指纹或真实独立性审计。共享被试/来源、相邻窗和重复开发VAL共同限制推断；49人、752块、30500窗不能互换为独立样本数。

仅元数据计数：VAL49/49都有同story异subject且足长的donor；TRAIN425/426有，1条没有。VAL在本phase没有同人异family donor（每人一条）；若改用该人的TRAIN record，49人虽都有donor，但仅42/49有足长记录，且改变了phase/刺激来源权限。不能默默补齐、重复/拉伸短donor，或看分数选错配。最小候选优先固定同record更早完整block干预，承认其recency与内容一起变；跨人/跨family donor不先塞入额外矩阵。[DONOR_METADATA_FEASIBILITY](DONOR_METADATA_FEASIBILITY.json)只是存在/长度盘点，未选具体donor。

详见[DEPENDENCY_COUNTS](DEPENDENCY_COUNTS.json)、[475行依赖表](DEPENDENCY_RECORDS.csv)。本轮未读取这些划分之外的信号、目标或分数。

## 4. 科学缺口不等于创新

最小问题是：**在强FT+Ccap及同容量当前映射已控制后，完整严格过去FT时序是否仍提供可重复的、超出输出历史prior的条件预测贡献？** 可以被匹配current/输出prior/错配历史三类结果直接否证。正结果也只到重复使用TRAIN/VAL的开发证据，不自动提供独立泛化、因果或生物机制。

协调一手文献/固定源码审阅见[LITERATURE_CHECK](LITERATURE_CHECK.md)及其七个纯文本DECAF源/许可快照。既有output_prior不是官方完整DECAF；该公开trainer存在真实context分支，普通推理另从过去EEG生成context，完整配置尚未全部追踪，不能据局部差异认定论文错误。GRU/gate/条件化或未测组合本身不新颖。后续候选的作用是识别有限科学问题，而非以组件名字宣称投稿贡献。

## 5. 本轮状态

技术：只读清单及边界/维度算式通过，不是任何新模型的工程或科学验收。Acceptance：盘点和具体候选已交协调，候选**PROPOSED_NOT_FROZEN_NOT_RUN**。Parent：SS07内部追加记录，范围/患者FORBIDDEN/server2203边界不变；持续goal继续，旧“无自动新实验”只约束未获具体授权的外部扩展，不代表终止原TRAIN/VAL研究。
