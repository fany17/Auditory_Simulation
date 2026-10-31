# SS07 scientific design review（2026-11-01）

协调独立审阅输入与执行者核验记录。此文不是性能交付；完整原强基线已正式运行，研究goal ACTIVE，未宣称新方法创新或阳性。

## 原模型与公平比较

保留SS02 native EEG Transformer/XLSR19对比预训练/SpeechLSTM及Conv32+LeakyReLU head；原EEG/包络坐标都为s:s+192，lag0，无SS06 lag8/60s warmup。426/49/69/75已实际核实。100epoch是SS02可用公开数据资源上限，非官方500epoch/180训练人/ensemble完整论文条件。head250/pat10。native训练入口只train/val；旧main的自动2AFC和读出全phase编码/自动test均不调用，XLSR载荷亦按phase过滤。

基线全量P是主锚点。后续必须有原生O直接监督或原encoder微调强控制，预算在train/val有界冻结并单列，不能仅战胜较弱冻结head而宣称超过最强同协议基线。增量保留原head、匹配初始化与同seed完整encoder；关闭模块在eval数值恒等。只把最终残差投影置零，不同时将门/残差全置零；还须验证首步末投影和后续内部参数可学习。

## Pearson可辨识性

3s negative Pearson对正缩放和常量偏置基本不敏感。慢状态仅输出每窗标量gain/bias，可能没有有效梯度，不能作为有效历史模型。主增量改变8通道组合、时间形状或滤波内容，例如历史条件8×8低秩混合；真实梯度/active容量另验。若改MSE/60s目标，则baseline同目标/同预算，不仅改候选。

## 时间支持与状态契约

native每个3s latent受整个EEG窗影响。当前窗起点s，主历史仅允许原始支持end≤s的已完成窗。50% overlap上一窗[s−96,s+96)与当前重叠，不是纯过去；主历史应至少隔两个window索引，即[s−192,s)。overlap-only可单列控制，不能把重复观察当长历史。

计划按hop96的奇/偶两条非重叠3s流维护状态：i的历史由i−2/i−4…更新，预测i后仅更新该parity，给i+2使用；记录/phase/epoch重置。batch64仍是64个重叠窗，非一条不重叠时间轴；TBPTT边界/状态保留必须明示。Hann后每个评分sample的原始输入支持为所有覆盖窗支持的并集，增量不能扩展其未来界限。M3 future teacher原始支持必须与其student当前窗不重叠，teacher仅train-only且stopgrad，额外学习成本须匹配控制。

历史干预每评分窗重新构造指定真实/错配历史，固定当前EEG/encoder latent；不能只改60s起始状态后被真实历史洗掉。区分同人错故事、同故事错人、顺序打乱；后者跨人仅离线诊断，非部署输入。应另有static active-capacity、current-window-conditioned同容量、output-history prior/平滑控制，不将冗余未用参数算匹配容量。

## 有界选择与seed

每类≤4配置，筛选必须按1024/2026/31415三配对seed的train/val，完整试验账本保留失败。最终主候选及baseline预先固定五seed为1024/2026/31415/42/12345，新增两seed已在看任何SS07 test前冻结，可提前训练baseline。不能按显著性增seed、只挑赢家或无限试val。候选精确矩阵另交协调审阅后执行，native full baseline不等待此门。

## 一手相关工作与新意缺口

- [SHINE v2](https://arxiv.org/html/2602.23960v2)已采用多尺度层级及局部/上下文动态门控；其30s窗、record标准化、pooled r、SparrKULee263/207/192与本轮不同，不能直接比r或把组件改名当创新。
- [DECAF](https://arxiv.org/html/2602.19395v1)已有强EEG分支、历史预测包络GRU与门控融合；其500ms偏移、80%训练重叠、3s测试和目标不同。M5须区分latent EEG-history与output-prior，并保留同条件adaptation控制，不能照搬其分数或fully-causal表述。
- [Transformer-XL](https://aclanthology.org/P19-1285/)的segment recurrence是长上下文背景，并非本任务新增理论。
- [CPC](https://arxiv.org/abs/1807.03748)以latent未来预测组织表征学习；M3不是因名称相同就复现原对比目标。
- [KalmanNet](https://arxiv.org/abs/2107.10043)为部分已知动态下学习滤波增益提供背景；M5并非已校准Kalman filter。
- [Cawley & Talbot](https://www.jmlr.org/beta/papers/v11/cawley10a.html)说明有限验证准则亦会过拟合；有界配置/预定seed/全失败记录不能省略。

上述入口已由执行者2026-11-01实际浏览一手页面，协调亦独立审阅。需要回答的是：原强模型已充分使用当前窗时，**真实且不重复的历史是否提供超出容量、输出先验、平滑与额外学习预算的可重复增益**。仅包络单数据集的小幅增益不足以宣称ICML/NeurIPS程度。所有现有测试已暴露，外部确认/其他任务/新竞品公平比较仍是明确证据缺口，不因暂无新holdout阻断train/val研究。
