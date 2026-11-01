# D11：慢观测状态的最小条件探针（待独立方案审阅）

状态：PREPARED_NOT_APPROVED_NOT_EXECUTED。所属 PUB-01-S03_SS07；不是新节点。
目的：仅在原 TRAIN/VAL 检验，encoder 前的长过去观测二阶统计，是否在冻结强 FT+Ccap 之后仍提供任务相关增量。它不是已测噪声、因果状态或新方法。

## 1. 已有证据与区别

- 实读 incremental.py/history.py：M2/M5 使用 8×6 latent 均值摘要，M5 的先验和 8×8 修正不显式观察 64 通道二阶统计；过去与当前 Hann 输入并集需另行区分。D4 全历史和 D5 PCA 双线性交互均来自 native latent 的近邻窗，不是本轮 30 秒 encoder 前协方差。但本轮仍是通用条件双线性岭读出，输入/时标改变不构成新颖性。
- [EA](https://arxiv.org/html/1808.05464v2) 用被试平均协方差的逆平方根变换输入（已读 III.A 公式10–12）；[AEA 作者版本](https://repository.essex.ac.uk/39235/1/Lightweight_Source-Free_Domain_Adaptation_based_on_Adaptive_Euclidean_Alignment_for_Brain-Computer_Interfaces.pdf) 的输入投影/冻结预训练模型已有先例（当前独立检索确认摘要，完整方法尚未复核）。本探针保持当前 EEG/latent 不变，仅改变读出条件；没有复现 EA/AEA，不作优于它们或算法新颖的结论。
- [自适应 auditory reconstruction](https://simongeirnaert.com/publication/timeadaptive2022geirnaert/timeAdaptive2022geirnaert.pdf) 已使用 EEG 加竞争语音包络做时变估计；本轮推理条件只允许 EEG，无目标/音频历史，不将无 attention 标签误称无音频。
- 官方 e617dd2 [预处理源码](https://github.com/exporl/auditory-eeg-dataset/blob/e617dd211de60e976835a15b3ec025dad5f371ab/preprocessing_code/sparrKULee.py) 含 SosFiltFilt、MWF、CAR、64Hz resample；实际 derivative 历史 revision UNKNOWN。本输入称“64通道发布预处理 EEG”，不称原始 raw；数组严格过去不等于端到端实时因果。
- D10 不参与本轮结构、时长或超参数选择；旧 test/DTU 全部关闭。D8 Ccap 为主 anchor，D8 P_FT/Ponly/C 与 D9 W/F 的已有 VAL 只作冻结参考。

## 2. 固定时间访问及静态控制

采样率64Hz，窗口 i 从0开始，当前输入 [96i,96i+192)。保守完整 Hann baseline 输入并集对内部窗最早为 s_i=96(i−1)，而非96i。各区间均左闭右开，不跨 record：

| 状态 | 协方差输入 | 部署语义 |
|---|---|---|
| CurrentCov | [96i,96i+192)，3秒 | 已在当前输入中的同容量二阶统计控制 |
| RecordPrefix | [0,1920)，30秒 | 当前record开头的静态观测，不能称动态连续性 |
| PersonTRAIN | 同一人的其他 TRAIN record 前30秒协方差均值 | 已知被试的静态参考；TRAIN 对当前record leave-one-record-out，VAL只用该人全部TRAIN记录；不读VAL未来/标签 |
| PastCov | [s_i−1920,s_i)，30秒 | 真正位于当前融合输入并集之前的滑动状态 |
| OlderCov（干预） | [s_i−3840,s_i−1920)，30秒 | 更早同record、同长度、同身份的时间错配；改变时效性，不能分离时效与内容 |
| Neutral（干预） | TRAIN pooled 状态标准化后的零向量 | 不含当前record历史；不是新拟合 |

所有新残差训练/应用共同 i>=41，保证 OlderCov 完整。TRAIN 234988 / VAL 28491 有效窗；前41窗分别17466/2009全部原 Ccap fallback，不丢record或评分块。原 D8 Ccap 本身 i<4 为 P_FT* fallback。
475条记录最少409/566窗，均满足；原VAL完整30500窗、49人、752块保持不变。
TRAIN有70人426record、每人3–8record；VAL49人全部见于TRAIN，因此 PersonTRAIN 合法但仅代表known-person设置。

## 3. 状态与读出（不改 encoder）

1. 继承原 EEG TRAIN-only 通道归一化，逐record读取既有64×T float64 derivative，转T×64；严格原raw_windows数值路径：原始值先float32、使用保存mean/std归一化、最终float32，再转FP64累计；不得重新预处理或重新估计原归一化。对每个区间局部去均值，K=mean(xxᵀ)−mean(x)mean(x)ᵀ。
2. d(K) 为64个 log(max(Kcc,1e−8)) 加2016个上三角相关系数 Kab/sqrt(max(Kaa,1e−8)max(Kbb,1e−8))，共2080维。每坐标负variance舍入界为1e−10×max(1,未中心化二阶矩绝对值,均值平方)；超界技术失败，界内修正至0后再应用floor，不静默裁掉record。相关数超出[-1,1]的舍入量同容差，报告floor/clip计数；明显异常停止受影响流程。
   只对对角取log，不做协方差逆、矩阵log或SPD假设，允许CAR秩亏。协方差包含神经信号和其他成分，不直接命名噪声。
3. 仅用 TRAIN i>=41 上 CurrentCov/RecordPrefix/PersonTRAIN/PastCov 四类、等臂等窗权重，学习共同 descriptor 均值/方差及 PCA8；静态状态按该record有效窗数加权，不重复写大矩阵。OlderCov/VAL不进入统计拟合。PersonTRAIN先对其他record的前30秒协方差等记录平均，再取descriptor，不能平均descriptor。PCA8先统一缓存float32；后续h/product TRAIN统计、拟合和评分均基于同一缓存值转FP64，不能混用量化前后数值。标准化方差0坐标置0；特征值<=0的PCA列置0且披露有效秩，不换维数。
4. 每seed从该 D8 TRAIN_MOMENTS 当前 latent 子块恢复 TRAIN PCA32（G[:1536,:1536]×1536），继承 D8 TRAIN_TRANSFORMS 的latent归一化；实际D8源码以标准化current latent/sqrt(1536)计算centered G，因此乘1536恢复标准化latent协方差；其支持是旧TRAIN i>=4，后续factor/product/残差支持才是i>=41。不用VAL重估。特征向量按最大绝对元素正号固定，非正特征值列置0，不另寻方向。
5. q_i 为32维current latent PCA，h_i 为8维state PCA。q 与 h、q⊗h 的归一化统计仅来自共同有效 TRAIN；h及乘积使用四状态等权 pooled 统计，所有状态臂共享同一变换。原始标准化量为0表示TRAIN pooled均值，Neutral在h归一化后置0，乘积仍按既定TRAIN product统计转换，不把常数项误删。
6. Add 特征 q/sqrt(32)；四状态特征 [q/sqrt(32),h/sqrt(8),standardize(q⊗h)/16]，296维，均加未惩罚192维截距。统一窗口 residual 目标 y−Ccap，loss=mean_windows(mean192(error²))+(.01/192)||B||F²。故求解 (G+.01I)B=H；不是物理fused目标，不重新引入D9近零偏置解。
7. 每seed×每臂只求解一次 λ=.01；不做新网格、不选择窗口/时长/PCA维度。零残差 Ccap 始终并列显示，不据VAL删负分seed或隐藏原解。

## 4. 最小完整矩阵与判读

seed固定1024/2026/31415：
- Add、CurrentCov、RecordPrefix、PersonTRAIN、PastCov：5×3=15个新岭解。
- 已拟合 PastCov 分别以 OlderCov、Neutral 重新预测：2×3=6个无拟合干预。保持当前EEG、全部current latent、P_FT、Ccap、q、模型系数、输出mask完全一致。
- 无额外NN前向/优化、无新权重、无目标历史、无test-time拟合；每seed5拟合+2干预+Ccap=8个实际VAL评分角色，合计24个新评分集合；既有F等VAL参考只读单列。TRAIN可做协方差状态稳定性/任务残差描述，不能用稳定性替代任务增益。

primary描述性对比为 PastCov−Ccap、PastCov−CurrentCov、PastCov−RecordPrefix、PastCov−PersonTRAIN，以及 true−Older/Neutral。
只有上述六个差值在三个seed均正，才记“本有界探针支持额外观测状态的任务相关条件信号”。任一不满足，记未满足；完整逐seed/逐人差值照报，不新增CI或显著性检验，不改矩阵追阳性。
这不是显著性/独立泛化/慢噪声机制或创新的充分证据。PastCov−Add只作secondary；PastCov与既有F并列，即使超过Ccap却未超过F，也不能称最强当前方法已超越。
判读仅限所选8维二阶状态的任务条件信息；未拆分h加性与q⊗h，不能证明交互本身或排除全部协方差信息。按人配对差值、每seed、family描述性、cold-start和warm-only敏感性共同报告；warm-only固定使用原60秒网格中全部Hann贡献窗i>=41的完整块，保守start>=4032，原网格首个为7680。首个start0块作cold-start精确一致性；start3840过渡块只计full主评分。先按实际非零Hann支持核对，不据结果改mask；既有F仅full参考。主评分仍原完整60秒/person宏平均。报告fused/window MSE、3s r、Hann weightedSingle/disagreement/identity，避免仅报r。
Current3秒与Past30秒存在估计样本数差异，RecordPrefix/PersonTRAIN控制静态与身份但不完美匹配动态分布；Older/Neutral干预也不能单独排除分布偏移。协方差的刺激成分仍是替代解释。

## 5. 既有资产、精确新增与限额

server BASE=/home/fanyu/auditory_simulation_m6a/pub_01。
只读 D8 s03_finetuned_readout_transfer_20261101：PROTOCOL中原TRAIN/VAL records/mask、三seed latent/P缓存、TRAIN_TRANSFORMS/MOMENTS、Ccap(.01)系数与已有指标；原TRAIN envelope cache，及其指向的475 EEG derivatives。
D9只读既有VAL JSON参考，不读其test。新独占ROOT=s03_observation_state_probe_20261101（代码冻结前不创建科学产物）。
代码拟新增 observation_state_core / observation_state_run / observation_state_report 三入口及一个inventory，重用已有指标函数；不修改历史冻结源码/系数。

资源盘点只读元数据见 OBSERVATION_STATE_INPUT_FEASIBILITY.json：
- 原EEG TRAIN/VAL文件合计13979146624B，仅2203逐record读，不回传。
- 5类state(Current/Prefix/Person/Past/Older)×TRAIN+VAL全长×8float32缓存，payload45272640B；10个128B NPY header后45273920B。cold-start行固定0并显式mask。Neutral不缓存。所有seed共享这些state，q从已有latent按需生成。
- 2080²FP64 pooled moment34611200B；PCA载荷/统计<1MiB，三seedcurrent PCA约1.2MB。
- 15份系数float64共5626368B（未计小型NPZ封装），含Add6336、state57024参数/解；不得本地回传。
- 不持久化2080维逐窗descriptor、current q、预测全数组或额外EEG副本；分块2048，逐record累积96样本块的一阶/二阶矩，验证rolling与直接区间算式。
- 新ROOT硬上限256MiB；其中所有新增数值数组（不限扩展名）<=128MiB。累计NPY由26225606304B增至最多26270880224B，仍<25GiB，剩572665376B；free盘当前25757130752B，预留整个256MiB后25488695296B>20GiB。
- CPU BLAS最多16线程，GPU0，峰RSS硬上限8GiB，wall硬上限60min；实际wall未测，不把预算说成实测耗时。新求解15，NN forward/update0，新增下载0。
- 每stage检查资源；技术失败保留原记录，不自动重复或换条件。无永久删除/覆盖任何历史资产。

## 6. 执行与验收门

先独立审阅本最小PLAN；获明确通过后实现并冻结完整15fit+6干预矩阵，不在smoke处停下。
执行前核对ID/parent/FORBIDDEN患者/原数据许可/上游D8冻结/允许修改范围；全部满足才生成源pin和PROTOCOL。
允许文件为新增四脚本、本run01轻量摘要/报告、三份当前入口/SS07 Completion Record/唯一SUBTASKS登记的阶段描述；不改parent科学问题、分支/原split或旧结论。

工程校验只用隔离synthetic及原TRAIN/VAL：区间边界与无current支持相交、record reset、同人leave-record-out、维度/常数/floor、rolling/direct covariance、PCA矩与ridge augmented-lstsq、相同current干预、原Ccap完整VAL复现<=1e−10及cold-start精确相等。不为工程重复NN前向。
正式完整运行后独立重算15解normal equation和状态支持/来源、原Ccap桥、全部聚合、代表性缓存预测；三静态图与中文逐图说明/失败与成本齐备，随后独立acceptance及parent consistency review/归档。
无论阴阳均不自动扩时长、PCA维数、正则、预处理、baseline或旧test。本PLAN的技术通过不等于SS07科学goal完成。
