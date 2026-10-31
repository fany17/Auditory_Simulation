# SS06 run02：同规模强基线的实际结果

2026-10-26。**COMPLETED / PASS_WITH_LIMITATION；POST-HOC INTERNAL EXPLORATORY**。

本轮已将既有强Transformer放到我们的监督规模，完成三个seed，而不是只讨论训练量差异。结果是：**因果适配强Transformer C的r=.1230，原生离线直接监督O为.1518，完整两阶段低预算参考P为.1415；原M2为.1282、GRU为.1252。** C与五框架/GRU的六项校正区间全部跨零。当前证据不支持“连续框架普遍明显不如强因果模型”，也没有证明这些模型等效。

同样监督目标次数下，原生离线方法仍获得更高点估计，因此“只是训练次数少”不足以解释全部差距。但C/O同时改变时间归一化、卷积、attention和head的信息权限/实现，**不能把.1518与.1230的差值归因于单一未来EEG因素**。O/P是描述性参考，不是同未来权限、同总训练量排名。P单轮也不代表旧SS02充分训练或原论文性能上限。

## 1. 共同数据与实际比较对象

复用run01冻结249train/63val/41audit记录、新train-only EEG/目标统计与lag8（125ms）。audit为36人、5个source family、538个完整60s块；每块先计算Pearson/MSE，再等权record→person，最后三seed均值。人可跨phase，此处不评估未见被试泛化。所有记录来自既有公开合法资产，无新下载、患者资料或新模型名单。

三个正式seed固定1024/2026/31415。旧SS02 EEG encoder、对比模型和head见过当前audit，故**完全不用于本轮初始化**。C/O每seed独立从头；P每seed从头预训练，head只使用该seed本轮新train的选点encoder。可复用的旧资产仅固定XLSR19声学特征及公开来源/许可。run01全部39个模型及其结果原样只读复用，没有重跑或改写。

| 条件 | 实际网络与权限 | 正式监督 | 额外学习 |
|---|---|---|---|
| C | 强Transformer causal adaptation，含head3,128,137参数；逐sample channel LN、left conv、含当前192sample的因果attention；EEG样本坐标≤目标+8 | 3262次完整60s loss/step，每seed12,526,080目标sample呈现 | 无SSL |
| O | 原生强EEG Transformer+Conv32，共3,372,617参数；整3s time-LN/sameConv/unmasked attention，离线未来信息 | 同C的目标、顺序、loss、optimizer、验证步点；39个重叠3s窗先Hann重建完整60s再一次loss | 无SSL；额外重叠EEG计算另计 |
| P | 原生EEG↔SpeechLSTM/XLSR19对比预训练，再冻结EEG、训练257参数Conv32；推理整体仍3,372,617参数 | 一轮130,231个重叠3s读出窗，25,004,352目标sample呈现 | 一轮639步对比预训练，65,240个唯一record-time 3s窗 |

P不是“只有257参数的基线”：257只是读出head，EEG encoder有3,372,360参数；预训练还包括428,352参数audio branch及logit参数，总3,800,713。P为EEG—声学对齐预训练，run01 M3为未来EEG自监督，学习目标不同。

## 2. 三seed实际结果

| 条件 | seed1024 r | seed2026 r | seed31415 r | 均值r | seed SD | normalized MSE均值 |
|---|---:|---:|---:|---:|---:|---:|
| C 因果强适配 | .122625 | .124924 | .121494 | **.123014** | .001748 | .842176 |
| O 原生离线直接监督 | .151887 | .150615 | .152785 | **.151762** | .001091 | .822210 |
| P 两阶段参考 | .143173 | .141933 | .139291 | **.141466** | .001982 | 3.431352† |

全部9个condition/seed均有538评分块、41记录、36人，无undefined-r块。†P只优化3s Pearson，输出mean/scale未校准，MSE仅如实描述，不能据此当作公平的MSE优化比较。

图1：[共同记录上的三seed点与均值](figures/performance.png)。灰色为run01冻结比较对象，C为本轮同监督/因果样本坐标控制，O/P另列离线参考。点为三个实际seed，不是置信区间。M3和P有额外不同预训练，不能因同图而视为总预算相同。

图2：[C与六个比较对象的探索性配对区间](figures/paired_contrasts.png)。先在每记录上平均三个seed，按subject/source family交叉bootstrap 10,000次，六项Bonferroni，各区间99.167%。仅5个family。

| C减比较对象 | Δr | 校正交叉区间 |
|---|---:|---|
| M1 multi | +.000102 | [−.014713, +.005439] |
| M2 | −.005150 | [−.011411, +.000306] |
| M3 finetune | −.000732 | [−.012559, +.003668] |
| M4 adaptive | +.009614 | [−.010117, +.016558] |
| M5 full | +.008836 | [−.007055, +.014020] |
| GRU | −.002207 | [−.014354, +.002319] |

六项均跨0：本轮没有建立C稳健优于或差于任一对象。区间仍允许有意义差异，不能读成等价证明；seed SD小也不能消除跨family不确定性。O/P未追加事后显著性比较，不从图上升级因果结论。

## 3. 训练量究竟是否一致

C/O与run01严格同每seed3262×3840目标、同numpy RNG seed record permutation和记录内chronological block顺序、同AdamW(.001,wd.01)/clip1、同60s `(1-r)+.1MSE`。每记录先3840sample无梯度warmup，再连续更新；C逐层conv63/KV191/head31缓存跨块保留，跨step detach，record reset。内部192sample微块不detach、不独立算loss。

| 每seed实际数量 | C | O | P |
|---|---:|---:|---:|
| 监督目标sample呈现 | 12,526,080 | 12,526,080 | 25,004,352 |
| 对比预训练EEG/声学sample呈现 | 0 | 0 | 各12,526,080 |
| 训练EEG呈现，含warmup或frozen encoder cache | 13,482,240 | 26,290,368 | 37,530,432 |
| optimizer steps | 3262 | 3262 | 639 pre + 2035 head |

O的目标只在聚合后算一次loss，但每60s由39×192=7488个EEG窗sample参与计算（1.95倍），没有藏作同算力。P的读出重叠重复、对比阶段和encoder cache分别记账，不以“一epoch”伪称总预算匹配。run01 M3另有每seed13,482,240 EEG-SSL sample，也不能称与C总学习量相同。

12,526,080指唯一EEG **record-time目标位置**或其实际呈现，不是独立声学信息：同story-time的包络跨被试重复。训练15story/13family在评分区间的story-time union只有771,840sample/4,020个3s窗（ACOUSTIC_SUPPORT.json）。P实际65,240个record-time窗全部覆盖、重复anchor=0，batch4–128，同story不同timestamp；不能把同一音频timestamp的跨人重复当作新声音证据。

P保持原方法预训练AdamW无clip，读出Adam(.001)、无weight decay/clip、manifest记录/时间顺序、batch64/shuffle=false，最后batch55。首版把C/O optimizer扩到P的偏离在P任何正式更新前被原协议审查纠正，冻结两个amendment；旧pilot与源码保留，正式不复用。详见PROTOCOL_NOTES.md与P_NATIVE_AMENDMENT_02.json。

## 4. 验证曲线与预算限制

图3：[四个阶段实际固定验证点与选点](figures/validation_curves.png)。C/O完整验证980块，step1000/2000/3000/3262，最高macro r、earliest tie。C三seed全部选3262；O选3262/3262/3000。Ppre各639步全跑完，三seed都选479；head各2035步全跑完，选1017/2035/2035，依据原3s Pearson而非C/O的60s macro r。

C三seed的最好点都在预算末端，**本轮不能宣称已充分收敛或达到模型上限**。O/P的曲线也只说明这一轮固定验证点的结果。我们按用户所选“强框架在当前规模下完成”收束，未因低分、末端改善或高分而追加epoch/seed/模型。P低于O的点估计不证明对比预训练无用：两阶段/冻结、loss、重复目标和优化器均不同，且单轮不是充分训练。

## 5. 已做的实质核验

训练前真实train/val pilot及检查通过：C未来扰动max1.79e−7、full/chunk max8.35e−7、跨块carry max8.94e−7；跨microchunk梯度非零，真实60s C/O全部应训参数gradient finite；Hann identity max2.38e−7、全部3840目标支持。C的逐sample LN比原生time-LN减少244480个affine参数，故标adaptation。

全部12正式阶段完成后，复核精确record/block顺序、P实际访问覆盖/读出连续索引、三seed新pretrain-checkpoint身份、统计来源、冻结source及P修订，才记录ASSESSMENT_OPENED并统一评估。随后单独observer逐block→record→person重算，最大汇总误差0。另从发布方原预处理文件重做normalization，在CPU复算首末两记录×三条件×三seed=18个同条件段；C改127sample微块验证，最大r差3.3624e−8、MSE差2.3583e−7。

这是executor separate observer pass，加协调独立协议/工程/原方法审阅；不伪称第三方完整科学签署。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS_SS06_ONLY。原生same-padding偶数kernel的PyTorch提示保留，不是数值失败。

协调最终还在2203以标准库独立从369行per_record重做record→person→seed聚合，SUMMARY最大误差2.7756e−17；与run01逐record重算六项配对均值，最大差6.9389e−18。协调实际核对预算、amendment与assessment时序；没有重跑全模型或原始CPU。以上范围与机器/执行者复算分开记录。

## 6. 结论边界与保存位置

本轮填补了缺少同数据/同监督规模强控制的缺口，保留了阴性证据。当前最可支持的描述是：**在既定单pass、提供的预处理EEG样本因果权限下，较大强Transformer没有建立相对现有连续框架/GRU的稳健优势；原生离线流程仍有更高点估计。** 单纯归因于训练量、预训练、网络参数数目或未来EEG都超出本对照。以后若研究纯lookahead或充分收敛，需另行授权协议，本轮不启动。

audit早已查看，历史来源也曾参与旧方法开发；全部结果POST-HOC INTERNAL EXPLORATORY，非新独立外部确认。仅5family、人员跨phase、固定一个split，bootstrap不能创造新证据。C因果只对既有64Hz derivative样本坐标成立，上游预处理/硬件物理同步仍UNKNOWN；不宣称端到端在线因果系统、普遍优越性或患者效果。

代码/轻量报告在同SS06 worktree与PR#6 Draft。共同训练源5b93ca5、P原法修正55ddd40、observer82f3001；最终报告提交见Git。2203 canonical根：`/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/eeg_run02_matched_baseline`。完整访问序列/缓存/latents/权重全部留服务器；本地约1.21MB允许清单仅CSV/JSON/compact元数据及3张静态汇总图，另有C盘轻量备份。未回传信号/权重/数组。

结果入口：audit01/RESULTS.csv、per_record.csv、per_subject.csv与各538行block表；review01/SUMMARY.csv、primary_contrasts.csv、training_ledger.csv、REVIEW.json；FORMAL_PROTOCOL.json以P_NATIVE_AMENDMENT_02覆盖P特定字段，全部source身份/选点/限制可追踪。run01原文/结果不改。**SS06本轮完成后STOP；不合并#6，不启动SS04/S04/父S05/GPT或artifact。**

交付包25项核验均PASS（DELIVERY_REVIEW.json），compact保留字段与canonical完全一致、9个block表各538行。协调独立审阅范围另记COORDINATOR_REVIEW.json，未混称执行者CPU复算。原main c7b2364、CNN/TRF/stash保留；本worktree之外未修改科研原目录。
