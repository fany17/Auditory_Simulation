# D4：控制当前表征后的历史条件预测探针

状态：PROPOSED_NOT_RUN。归属 PUB-01-S03_SS07；不是新节点或新方法训练。2026-11-01。

## 问题与证据来源

给定原完整 P 的当前 EEG latent 和原 P 输出，完整非重叠历史 latent 是否提供六段均值摘要之外的可预测包络信息？D2 的六段重建只保留约13%中心化时间能量，不能直接证明丢失有用信息；D1跨半段仿射校正未获支持，本轮不推进该方法。warm18fit已结束，其结果、权重、失败和复核不改。

只使用2203上原fold TRAIN426record/252454窗和VAL49record/30500窗，三个已训练原P seed=1024/2026/31415。复用round01原encoder的既有train/val latent缓存及各自训练完成的P head，不能换为warm联合微调encoder。无encoder forward、新NN架构、反向传播、下载、test或DTU推理。

## 时间、公共mask与评分

每record局部窗口i对应原始64Hz支持[96i,96i+192)，当前允许原P完整3秒离线权限。近历史取i−2，支持[96i−192,96i)，恰好结束于当前起点；远历史取i−4，支持[96i−384,96i−192)。不跨record、phase，不使用未来EEG，也不使用真实历史包络。历史lookup无递归状态。

所有拟合条件严格共同使用i≥4；已核对最短TRAIN409窗、VAL566窗，所以TRAIN=250750个拟合窗，VAL=30304个可加残差窗。首4窗所有条件都返回共同P、残差0，不移动原评分起点或删60秒块。拟合前冻结record顺序、offset、窗口支持及旧VAL target-only eligibility mask；非有限输入在首次全覆盖检查失败即保留失败，不按模型预测筛样本。非法预测/常量指标使该条件失败，不删seed/块或自动重试。

沿用原192点Hann/96步长的整record重建、原60秒固定块→person宏平均，预计49人/752有效块；实际先按原mask核对。同时报告完整3秒窗Pearson宏均值。所有条件和λ使用相同mask和首4窗回退。TRAIN相邻窗/共享刺激有依赖，拟合目标按窗计权，不把样本数当独立观测数。

## 共同P与残差单位

原head在原缓存上eval/no_grad、global64顺序前向，所有条件共享同一P预测。使用原精度/backend设置；不重新编码、不改变P。每seed只拟合一个全局正仿射a,b：在TRAIN各record Hann融合且weight>1e−8的物理时间点上，每点计一次，a=cov(P,Y)/var(P)，b=mean(Y)−a mean(P)。必须a>0且有限，否则停止受影响seed并报告，不裁剪或用VAL替代。

所有条件用同一个P*=aP+b，残差目标R=Y−P*。a,b只对齐岭回归的MSE物理单位，不提供Pearson增益：正全局仿射与归一化Hann融合可交换，须核对原P/P*的3秒及60秒Pearson差≤1e−10，并报告实际a,b。没有逐窗/逐record/VAL校准。

## 固定矩阵

每个块先按TRAIN共同拟合mask逐坐标中心化和总体标准差归一化，常量坐标记0、除数记1，再除以sqrt(块维度)。不能用VAL均值、方差、PCA或目标选择表征。当前基础C包含current latent展平1536维及current P*输出192维，两个分别归一化的块。

| 条件 | C之外的输入 | 总维度d | 含192维不惩罚截距的系数数 |
|---|---|---:|---:|
| P | 无拟合，残差0 | — | 0 |
| C | 无 | 1728 | 331968 |
| Ccap | 标准化current latent逐坐标平方，再以TRAIN单独中心化/标准化的1536维块 | 3264 | 626880 |
| CS | 近历史8通道×6段（每段32点）均值48维 | 1776 | 341184 |
| CF | 近历史8×192完整latent 1536维 | 3264 | 626880 |
| CO | 远历史8×192完整latent 1536维 | 3264 | 626880 |
| CF48 | 近历史标准化full latent的TRAIN-only无监督前48 PCA分量，各以TRAIN分量标准差白化 | 1776 | 341184 |

PCA按TRAIN covariance特征值降序；非正分量置0并报告，不挑维数/seed。符号固定最大绝对载荷为正。CF48与CS同维、同系数数；Ccap与CF/CO同维、同系数数。名义维度不保证相同有效容量，报告每λ有效自由度tr[G(G+λI)^−1]、常量坐标数、PCA谱摘要。full/summary也改变表征，不把CF>CS直接归因于有用信息损失。

每臂固定λ∈{0.0001,0.01,1}，求最小 (1/N)||R−XB−b||²_F + λ||B||²_F；b不惩罚。使用float64 TRAIN中心化Gram和cross-moment，B=(G+λI)^−1 H，b=mean(R)−mean(X)B。每臂一个192输出联合岭解，非192次选参。共6臂×3seed×3λ=54解；无SGD/早停/额外搜索。

## VAL选择与判读

每臂三seed共享一个λ，按三seed VAL60秒/person宏均值选择；原P零残差也是候选。严格高于P才选择拟合，否则保留P；完全相同优先P，λ之间完全相同优先更大λ。全54格及P、逐seed/逐person/逐块3秒和60秒指标全部保留，不能只报胜者或按seed挑λ。VAL既选参又报告，是开发证据，不是独立确认或显著性检验。

预设主要描述比较：CF−CS、CF−Ccap、CF−CO、CF48−CS；辅助CF−C及各臂−P。近/远历史均胜C可能符合持久被试/record状态；近>远另支持时间局部性。CO是合法时间距离对照，不是预定零信息负对照，也不要求CF必须胜CO才承认条件历史信息。CF−CS混合容量与表征差异；CF48提供同维补充但不能覆盖full rank。Ccap为固定当前非线性容量对照，不证明所有当前非线性已被控制。

阴性只约束本固定原encoder、历史距离、线性残差和预算，不排除历史×当前的非线性状态交互。阳性也不证明物理因果/同步、信息论独立性、跨数据泛化或新方法成立。当前与历史均来自离线3秒encoder，不能称实时因果解码。旧测试和DTU不再推理；不因阴性扩矩阵、λ、NN预算或历史距离。

## 实现、资源与可审阅交付

server ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_conditional_history_probe_20261101；不初始化远端Git。按seed顺序运行、最多8 CPU/BLAS线程、最多一张空闲L40仅原head前向、GPU≤1GiB、进程RAM≤8GiB、wall上限120分钟。不得终止其他作业；超界保留失败、无自动延长/重跑。运行前保证磁盘余量≥20GiB、既有+新cache≤25GiB。

流式TRAIN遍历：先P全局校准及基础块统计；再current平方统计/PCA48；最后缩放特征的共同Gram/cross-moment。联合基础块[current1536,P192,summary48,full1536,older1536,quadratic1536]共6384维，float64 Gram326043648bytes；PCA48子矩阵由full矩变换导出，不写N×d特征cache。54组完整系数含bias共208438272bytes；新派生moments/系数/临时P预测总磁盘≤2GiB。实际遍历数、CPU/GPU/wall、峰RAM、磁盘、求解数和全部失败入账；共享GPU计时不声称隔离延迟/FLOPs优越。

必要验证仅覆盖岭解公式（小矩阵对照直接增广最小二乘）、train-only统计、固定近/远支持、record reset、共同mask、P正仿射Pearson不变、实际原P评分一致；不另建整轮NN重放。先冻结本方案和源码/来源/eligibility，再正式求解。独立协调review检查实际代码、解数/维度/来源、逐seed和共同mask，不由“跑通”宣称科学完成。

本地仅代码、协议/成本JSON、结构化CSV/汇总、静态图与中文REPORT/REVIEW；latent、P预测数组、Gram、PCA载荷、ridge系数均仅2203。交付完整固定矩阵和限制，更新SS07完成记录/当前入口；goal继续ACTIVE，独立确认、新机制/方法创新和论文证据仍未满足。
