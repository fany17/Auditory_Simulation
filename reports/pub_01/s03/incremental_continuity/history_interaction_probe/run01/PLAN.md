# D5：历史是否改变当前EEG到包络的映射？

状态：PROPOSED_NOT_RUN。归属PUB-01-S03_SS07内的新假说/独立冻结run；不扩大已完成D4，不改变父问题、数据或科学验收。2026-11-01。

## 假说、锚点与最小性

D4加性历史未超过强current读出，但没有检验历史×当前交互。D5检验有限的二阶形式：给定强current Ccap和所有臂相同的加性主效应，past因子是否改变current低维因子到包络的线性映射？这是原连续增量/条件映射问题，不是新NN架构或对D4的网格追分。

**冻结D4 Ccap λ=.01为强current锚点**，三个seed1024/2026/31415，全部D4 TRAIN-only统计、P正仿射及系数原样复用。锚点三seed VAL60秒=.1748812524/.1769144594/.1788963654，均值.1768973591。其λ在已暴露VAL上选择过，明确是开发基线，不当独立确认；不用257参数原P冒充充分current控制。

锚点保留不更新；在TRAIN拟合Y−Ccap的残差。当前Ccap特征/系数由原D4变换和保存权重重建，不做head/encoder前向，不训练NN。正式任何新拟合前用原49人752块评分核对锚点与D4逐块/总体r差≤1e−10，失败保留并停止受影响seed，不按分数换锚点/λ。

## 固定时间与共同mask

原TRAIN426record/252454窗、VAL49record/30500窗；共同i≥4，拟合250750窗/seed、VAL30304窗可加残差。首4窗所有臂新残差0，返回原D4锚点（其首4窗为P*），保留原Hann重建/原60秒target-only eligibility与30500窗/752块/49person，评分坐标不移动。

z_i原始支持[96i,96i+192)，64Hz。near=z_(i−2)为当前前3–0秒；older=z_(i−4)为前6–3秒，同record/phase，无递归、无跨record，当前和历史EEG始终固定、不含真实历史声音或包络。原P当前仍离线3秒权限，不称在线因果或长程记忆。

## TRAIN-only公共坐标和交互公式

仅在共同TRAIN i≥4的current latent上逐坐标中心化/标准化，拟合无监督PCA前32分量。按特征值降序，符号固定最大绝对载荷为正，正分量按TRAIN标准差白化；非正分量置0并报告，不挑维数。该同一个current坐标变换用于current/near/older，不分别为历史挑更有利的PCA。无VAL拟合或有监督维数选择。

记变换q(z)∈R32。u_i为current的前16分量，v_i^C为current后16分量，v_i^N为near后16分量，v_i^O为older后16分量。u、v^C/v^N/v^O各自使用共同TRAIN样本的均值/总体std标准化，常量置0。历史变换仍只应用当前TRAIN拟合的PCA，后续历史因子均值/std也只来自TRAIN。

所有拟合臂**相同**的加性主效应A=[q(current)32, v^N16, v^O16]，共64维；三个块分别TRAIN标准化并除sqrt(32/16/16)。这样near/older可通过加性读出进入所有臂，比较交互时不把单纯历史主效应误当历史改变映射。

固定三种256维交互：J^C=vec(u_i⊗v_i^C)，J^N=vec(u_i⊗v_i^N)，J^O=vec(u_i⊗v_i^O)。每种product逐坐标以TRAIN中心化/标准化再除sqrt(256)。current的两组16因子不重叠，避免u⊗u的对称重复坐标造成显然不同的名义rank。仍报告各臂常量数/有效df，不能保证三类product实际rank相同。

J^N的线性读出Σ_jk u_j(current) v_k(near) B_jk等于让current因子的系数依赖near，正是本轮有限二阶映射假说；它不是任意历史条件网络。通过加性A和同维J^C控制，降低把额外自由度或current跨坐标二阶项误认作历史机制的风险。

## 固定矩阵与上限

| 臂 | 共同A之外的输入 | d | 含192截距的系数数 |
|---|---|---:|---:|
| Anchor | 冻结D4 Ccap，无新拟合 | — | 0（既有626880另列） |
| Add | 无，只有A | 64 | 12480 |
| CC | J^C：current×current | 320 | 61632 |
| Near | J^N：current×near | 320 | 61632 |
| Older | J^O：current×older | 320 | 61632 |

每臂固定λ={.0001,.01,1}，float64中心化多输出岭解，目标 (1/N)||Y−Anchor−XB−b||²_F+λ||B||²_F，b不惩罚。4臂×3seed×3λ=**36个闭式解**，另3个无监督PCA32；没有SGD/NN、维数/距离/λ网格扩展、额外seed或失败自动重试。Anchor重建/校验和共享PCA/统计成本全部计账。

每臂一个跨三seed共享λ，按原VAL60秒/person均值选择；冻结Anchor零新残差亦为候选，必须严格超过才选拟合，否则回退Anchor，完全相同优先Anchor，其次更大λ。不是按seed选λ。保存全部36格/anchor/逐seed/逐人/逐块/3秒与60秒，以及有效df和常量覆盖，不只报胜者。

## 预设比较与判读边界

主要Near−Add和Near−CC；辅助Older−Add、Older−CC和Near−Older，全部同时报告各臂共享选择及相同三个λ的配对差。不因某λ好看改primary。只有Near相对Add及CC在三个seed一致正向时，才讨论本固定二阶表征/预算下的开发性历史映射线索，仍不当显著性/独立因果证明；否则不把Near−Anchor单独当历史证据。

Older不是零信息负对照。near/older都超过Add/CC可能涉及持久状态；near超过older另涉及时间局部性，不要求near必胜older才能讨论条件历史交互。当前函数对照只覆盖这组PCA二阶项，不等于全部current非线性已控制。负向仅约束本PCA32/16×16双线性形式、原encoder/距离/目标，不能排除其他非线性交互。正向也可能受估计方差、PCA方向、共享刺激和反复使用VAL影响；没有硬件同步、生物机制、信息论或泛化结论。

## 冻结、资源、必要验证与交付

新ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_interaction_probe_20261101；代码/协议/原D4 transforms/coeff/P_CHECK、latent/P缓存指纹、TRAIN/VAL targets shape/dtype/size/mtime与全部record长度、共同mask在新拟合前固定，明确不补称D4已预冻结。原D4/旧结果只读、不覆盖、不重跑。

只CPU，BLAS/OMP/MKL≤8线程、GPU0（复用2203上D4 P预测）；RAM≤8GiB、wall≤15分钟、新派生磁盘≤512MiB，余量≥20GiB且既有+新cache≤25GiB。逐seed流式处理，不写N×d feature或新signal cache。公共A64+三种product共832维Gram约5.54MB，36套系数含bias总14211072B；锚点预测可在单seed内RAM暂存，不持久化成新大cache。实际head/encoder forward计数0；校验/统计/PCA/Gram/solve/score CPU与wall尽可能分项记录，缺失不编造。

必要验证：小矩阵增广岭公式、乘积张量与手算双线性等价；current/near/older同record支持与边界；所有臂共同加性A、首4窗/评分mask；TRAIN-only变换；真实锚点原D4评分一致。独立review检查实际代码/来源/36解/维数/共同mask与关键聚合，既定通过后不另建整轮NN重放。

本地仅代码/协议/来源/结构化轻表/静态图/完整中文REPORT与REVIEW；载荷、latent、预测/系数/PCA/Gram只2203。方案供协调审定，当前NOT_RUN；审定后在既有授权内实现冻结与执行，不重复询问用户。D5无论正负，D4保持原结论，SS07与goal不因工程PASS或VAL单轮阳性完成。
