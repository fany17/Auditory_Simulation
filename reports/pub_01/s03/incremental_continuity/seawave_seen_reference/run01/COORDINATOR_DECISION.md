# D27 Sea-Wave-Small 已见被试强对照

2026-11-06。用户已授权全量公开EEG实验，并明确「有性能提升就行，核心目标是SOTA」。本项落实已有SOTA对照范围，不是继续添加历史模块追阳性。D26正式队列保持原冻结规则；D27可并行准备代码和有界CPU工程，正式训练须实际源/工程审核通过，且不得占用D26预留的两卡。

## 科学问题与实际任务条件

同一合法426 TRAIN / 49 VAL上，已完成原强Transformer系列仍缺少实际运行的公开强方法。Sea-Wave正式论文在已见被试新故事条件S1采用个体微调。协调从冻结清单核得：TRAIN70人，VAL49人**全部在TRAIN出现**，因此不能把本VAL称跨人泛化。来源家族29已见/20未见是故事来源维度，不是被试是否已见。

据此，D27覆盖论文S1所需适配阶段的最小方法为 **3个共享initial模型＋每seed49个个体微调，共147个personal FT**；每个个体从本seed的同一个initial-best分叉。当前VAL含29条已见来源家族记录，并不等于论文S1测试集。不能只跑群体模型便冒称论文S1强条件，也不需要先做主要服务S2的全TRAIN群体FT。顺带报告initial群体结果，明确它**未做群体FT，不是完整S2/强化群体Sea-Wave**。未来完整群体/未见人比较仍是剩余工作，不因本项省略而声称已完成广义SOTA比较。

本项所有结果仍来自反复开发VAL，是同数据/指标的探索性已见人比较。personalized相对原共享参数Transformer具有个体适配和更多逐人选点自由度，必须分栏；不得称同适配条件的纯架构收益。方法为已发表Sea-Wave，不把复现它的增益当本项目原创贡献。

## 固定模型、输入与实现来源

唯一结构为正式论文Small：64输入通道、32 residual/skip、1输出、20层、dilation cycle5、kernel3、非因果对称padding，实际感受野249（左右各124点）。复用已逐字节核对的作者模型数学形式与weight_norm/初始化次序，保留GELU、gate、dropout、残差/skip及零输出初始化。实际参数量在2203核验，不用论文四舍五入的150k替代程序计数。不增加D4、大模型或结构网格。

来源和代码/论文差异见[一手审计](../../metadata_review/code_review/seawave_protocol_20261106/REPORT.md)，作者commit `910138cdac01c58c007459711008ae4991757ff2`。FT按论文.8/.2与.1秒，而非旧公开入口的.6/.4与.117秒；必须逐项标注适配。禁止照搬硬编码checkpoint62、加载失败继续和每两轮才保存的入口行为。

仅用D26同一冻结426/49清单、同一原raw EEG/包络共同长度及现有合法原TRAIN-only通道统计；同一TRAIN未变，无需为形式重拟合相同统计。不能在VAL或个人完整记录重做归一化、改目标尺度、调lag；不加载旧EEG checkpoint，不下载新权重/数据。Sea-Wave仅EEG和包络监督；Transformer还有既有XLSR教师特征，其外部预训练信息与成本须披露。

每条输入按`common_samples`截断。原包络float32不另标准化。训练记录按原作者单GPU路线使用record名称排序、记录内时间递增，固定顺序、不shuffle；不要按文件glob重新定义TRAIN/VAL。原作者多GPU排序条件不同，作为实现差异记录，不混称完全相同运行。

## 训练、选择与预算

三seed固定1024/2026/31415，每个正式fit均从规定起点开始；不用seed赢家拼接。

**initial：** 5秒=320点、hop32、global batch32、保留尾批19，755187窗/23600更新每完整epoch。float32 MSE；AdamW lr1e-4、原始wd1e-5、无clip。作者自定义CyclicLRWithRestarts，restart10/t_mult1.2/min_lr1e-7/cosine，保留它对实际weight_decay的缩放和epoch首step/每batch_step顺序。固定epoch0..200共201轮，不把论文40–50轮达到某阈值当收敛依据。每轮按本项目统一完整VAL指标选最优initial，平分取更早，保存真实权重和全部选择轨迹。

**personal FT：** 各seed从该seed选中initial独立建立49个人模型，每人只读该人的原TRAIN记录。重新设同一run seed、清空Adam/scheduler状态，严格载入共同initial权重，dropout改.65。8秒=512点、hop6（.1秒在64Hz下向下取整）、B8且保留尾批。loss=.8 MSE+.2作者negative Pearson；Pearson沿时间中心化，使用作者eps1e-5定义。AdamW lr1e-5、原始wd.01，无clip；同一作者scheduler但epoch_size必须是该人的实际窗数。最多epoch0..20共21轮，预定patience5，epoch−1 initial是合法候选；按该人VAL所有原eligible60秒块的平均r选点，平分取更早。这里patience5为明确执行适配，不能写成原公开patience1。

每个人完全独立重置模型、优化器、调度器及RNG，加载失败停止；不沿用上一个人的参数，不从全人global-FT起点开始。即使奇数epoch最佳也要保存。初始共享模型仅训练3次，不对每人重复训练全TRAIN。

协调实际元数据计数（清单SHA `6385bde08161beb1dd507568aeb3c6d32d2f2be9087f423aec04107c1e943697`）：

|阶段|单个全体pass|三seed正式上限|
|---|---|---|
|initial|426记录、755187窗、23600更新|201轮/seed，共14230800更新|
|49人personal合计|305个TRAIN记录、2846186窗、各人独立尾批后合计355800更新|各最多21轮，共22415400更新|

个人实际早停各自记账，不能将147个个人fit当147次全TRAIN训练。共享initial成本只记3次，逐人摊销另列；所选checkpoint前缀、实际全部执行及未提交失败成本分开。完整披露唯一点、重复样本、更新量、模型/多个人参数存储、每阶段时间、内存、验证与准备成本。不要把“同epoch”说成与Transformer同训练规模。

## 推理与评价

保留原生整记录推理。长记录可用固定core4096与左右halo124计算，先在训练fixture验证与整段eval等价，不在块边界重置有效上下文。Transformer旧Hann贡献窗的支持随时间相位变化；两方法不声称同时间权限或严格同算力。不能将Sea-Wave截成3秒来削弱它。

主评价仍49人/752个原eligible60秒块，逐人等权r及MSE，采用旧Hann的**预测无关正支持交集**。Sea-Wave本身无需Hann拼窗；直接从连续输出在共同支持计分，并与旧评分函数做数值桥。不能因某人弱、某源负而排除记录。personal结果使用每人对应模型；initial群体行使用单一共享模型。保留所有seed、每人、每块、29/20来源家族以及阶段轨迹。

主结果并列原强FT+Ccap锚点、D26各完整臂、Sea-Wave initial（未globalFT）、Sea-Wave personal。只比较实际共享开发协议，不把论文.2258直接放入实测排名。personal不同适配自由度、XLSR信息、原生上下文及预算必须显示。旧test/DTU/ds004408不重开；不能凭本VAL宣称独立泛化或广义SOTA。

## 最小工程与执行衔接

执行者先完成PLAN、具体loader/模型/完整预算及冻结源，协调复核实际实现。已有用户授权覆盖本项，不再等待新的用户确认；技术审核是双方完成质量核验。

D26双卡运行期间，D27仅使用2203 CPU最多2线程/RSS4GiB做有界工程：seed1024、固定前两条合法TRAIN记录，最多2次B2 initial-MSE更新＋1次B2 FT目标更新，共3更新。权重丢弃，成本独立列出，不看VAL成绩选结构。其余纯前向预算先由执行者在PLAN列明，控制在16个模型batch调用以内；需要扩大只针对实际新失败，不重复完整试验。

必要验证：作者原模型到运行封装的同状态数值等价、实际参数/感受野；同共同支持的EEG/包络与train norm坐标；第二MSE步后的实际早层/末层有限梯度与FT目标有限性；训练fixture整段/halo块与共同支持评分桥；每个人权重/Adam/scheduler/RNG独立复原；作者自定义lr/wd跨restart与mid-epoch恢复一致。scheduler可做0NN的微型epoch轨迹验证，不假造训练更新数。不要把泛化结果、论文分数或一个前向PASS代替正式实验。

仅server2203数值、模型及绘图。新root为 `/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_seawave_seen_reference_20261106`；home原资产只读。本项writer≤16GiB、SATA保留≥64GiB，正式每worker CPU≤4线程/RSS≤32GiB/CUDA reserved≤40GiB，每GPU最多1worker。原始连续数组按需缓存，不物化全部重叠训练窗。Windows仅代码/轻量报告，目录为本run01，脚本前缀`pub_01_s03_ss07_seawave_seen_`。

正式可启动的条件：实际冻结源与工程经协调审阅；D26队列和子worker按PID/UID/startticks证实终结，GPU调度不冲突。不因观察超时重启任何任务；不为D27终止D26。D27可将不同seed/person fit独立调度到两卡，但严格保存共享checkpoint与每人独立恢复状态。完成本矩阵后交完整实测、失败、选点、成本和科学限制，由协调独立核验；不因负结果自行加模型、重开旧test、合并PR或启动其他父阶段。

