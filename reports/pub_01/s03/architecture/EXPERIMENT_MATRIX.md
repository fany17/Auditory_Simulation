# SS03 多时间尺度单假设：正式冻结设计

在任何SS03测试分数出现前，先完成2203真实数据对齐核验、5项结构单元测试、四组CPU/GPU forward/backward、seed1024 multiscale **100优化步** train/val工程预实验。pilot未读test成绩、未保留权重供正式运行。冻结证据：`evidence/bootstrap01/{formal_protocol.json,freeze_review.json,pilot.json,smoke.json}`。

| 组别 | 三branch dilation | Branch感受野（sample count） | 参数 | 每3s窗口nominal conv MACC |
|---|---|---|---:|---:|
| short | 1/1/1 | 9/9/9 | 6001 | 1124352 |
| middle | 4/4/4 | 33/33/33 | 6001 | 1124352 |
| long | 16/16/16 | 129/129/129 | 6001 | 1124352 |
| multiscale | 1/4/16 | 9/33/129 | 6001 | 1124352 |

共同结构：64→48 pointwise空间投影，分成三组16channel，各9tap depthwise temporal卷积+GELU，concat→48channel pointwise融合/残差+GELU→1channel输出。只改dilation，点数/参数/名义MACC匹配；不宣称相同实测GPU latency。上述sample count约0.14/.52/2.02s，首末sample时间差为.125/.5/2s；这是计算支持范围，不是神经生理时标。

正式4×3：seed **1024/2026/31415**，全部从头训练；同seed相同初始权重抽样顺序与相同window index批序。每步用numpy.default_rng(seed)均匀无放回抽64个窗口，每步之间允许重抽。此训练按window数权重，不声称被试平衡。

- 共同预算：**25000 optimizer steps**，batch64，AdamW lr=.001/wd=.01；无scheduler、无早停，每次均跑满相同预算。
- 损失：3s时间维负Pearson，batch平均；同SS02读出目标形式。
- 每2500步完整validation，按原始60s Hann重建→49人subject macro r；选择最大r的checkpoint，exact tie取最早。共10个候选检查点。
- 所有12次训练结束且独立核对预算/检查点后才打开test；每个selected checkpoint一次完整seen/heldout测试。不能根据test延长预算、换dilation、窗口或目标。
- Primary：seen/heldout分别报告subject-macro60s r；Secondary：2/3seen+1/3heldout。不能用加权值遮盖未见人组失败。
- Wrong EEG：同record内将预测窗循环移动半段，等价于错配输入窗后再同样融合；保持录音/被试粗信息，破坏局部对齐。
- 常数/均值control：训练target全局mean作为恒定输出，其Pearson数学上未定义（零方差），报告null/原因，不编成0。
- 身份canary：模型forward只接受EEG；相同EEG重复预测相同，EEG时间扰动改变预测；无pretrained encoder或候选输入。

H1要求multiscale超过**同最大感受野long**，并超过middle。对每人先平均3seed r，再算multiscale−long和multiscale−middle；每组分别按person bootstrap10000次（bootstrap seed20261019），报告描述性95%CI及四个预定contrast的Bonferroni family95%区间（每个98.75%）。组内两contrast下界均>0才记该组支持；两组均支持才总体支持，否则明确group-specific/mixed或H1未支持。不把仅胜short/middle解释为多尺度机制。

同时报告每seed及seed间SD，不将三个seed当作独立被试扩样本。subject bootstrap不覆盖全部共享story、训练集或软件不确定性。未自动加线性TRF参考、调制/jitter诊断、retrieval或其他架构。SS02和旧10人TRF仅上下文参考，训练预算与split不同，不是本次参数/算力匹配对照。
