# SS03 单独post-execution review

2026-10-19。**Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS。** 这是同一agent在执行后另行对照任务书、冻结协议、源码、训练记录、selected checkpoints、逐段与逐人统计所作的separate review pass；**不是第三方独立审查、外部签署或生理机制验证**。

## 实际复核与证据

1. 输入：原619记录身份/phase/common support一致，train-only归一化原样复用；每个readout window的原始包络切片与冻结target float32逐元素一致。train/val/seen精确story隔离、heldout人不进train；数据/权重只在2203。
2. 结构：5项unit tests PASS；四组真实CPU/GPU forward/backward finite，参数均6001，名义conv MACC均1124352/3s窗，long和multiscale最大计算感受野相同。
3. 冻结与训练：100步pilot只用train/val且丢弃其权重。formal_protocol在test前保存；12run均25000步/10次完整val，best step严格等于最大val60s subject macro r的最早位置。源码/config SHA256与冻结表一致；三seed、批序/初始draw规则一致，无test驱动修改。
4. Test：`all_training_review.json`在打开test前核对全部12run；每selected checkpoint只在test01中一次评分，12个identity canary均通过，无候选声音/预训练encoder输入。每次seen69/heldout13人完整，未静默删人；完整60s段无无效support/常数排除（不满60s尾段按冻结规则不评价）。
5. 数值：从保存的逐segment记录重新计算所有逐人/组/weighted，max误差0；另独立从原始EEG读取、套用冻结normalization并CPU预测，再以np.add.at作Hann融合及Pearson，与每run两组首个60s段对照，共24段，最大r误差7.4126e-6，低于预设1e-4技术容差。该检查不是原GPU评分函数自验；未声称全部test段逐个重跑CPU。
6. 对照：wrong EEG同录音循环错配，平均r接近0；train-only常数mean在test前记录，Pearson零方差未定义，未伪填0。没有把它们泛称所有audio/prior-only风险均消除。
7. 配对推断：每person先平均3seed，按person bootstrap10000，四个预定contrast给Bonferroni family95%CI；全部下界>0。本轮H1只在四种原型、可用fold4与25000步固定预算内支持；没有泛化为生理多尺度机制或超过SS02。

机器记录：[review.json](review.json)、`evidence/bootstrap01/`、`evidence/formal_runs/`，逐人[per_subject_all_seeds.csv](per_subject_all_seeds.csv)，对照[paired_comparisons.csv](paired_comparisons.csv)。最终轻量包schema/路径/数值核对见`package_review.json`。

## 验收限制及停止

一个公开可用fold、13名heldout人、共享故事/三seed/固定训练预算的局限保留；physical_sync UNKNOWN。long较差还可能反映稀疏采样/频率响应及优化偏置，不等于更长时间信息本身无价值；相对最佳单尺度middle约+.01r才是更克制的增量。原型加权r=.1066低于SS02预训练参考.1593的数值，训练目标/规模/预算不匹配，既不作公平排名，也不称新SOTA。

SS03 COMPLETED / PASS_WITH_LIMITATION。S03/PUB-01仍ACTIVE；SS02、SS01/S02 frozen结论不改。主目录未提交CNN/TRF和stash未动，工作仅在PR#3独立worktree/分支；不合并main，PR保持Draft。数据、cache、权重和逐窗数组不回传，不产出ART-AUDREP-v1，不启动SS04/S04。可选线性参考、时序诊断未运行并明确标记，不以额外追分分支代替当前交付。
