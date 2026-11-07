# SS06 run02 — 强基线与训练规模匹配

日期2026-10-26。**当前COMPLETED / PASS_WITH_LIMITATION / STOP**，实际交付见run02 REPORT/REVIEW。用户明确：“按照你说的吧，起码我们的训练规模要和之前一致，或者让之前的框架也在我们的训练规模下完成吧”。执行时此授权覆盖run01 STOP、恢复SS06 ACTIVE，唯一run为`eeg_run02_matched_baseline`，现已完整交付。旧run01/prototype01及SS02/03/05证据只读，不新建研究节点、不自动启动下游、不合并PR#6。

## 目标与边界

优先将既有SS02强Transformer放到SS06规模，补足真实同协议强控制，不同时扩五框架预算。只用run01固定249train/63val/41audit、新train-only EEG/目标统计、相同538评分块与record→person聚合。audit早已查看，本轮全部为 **POST-HOC INTERNAL EXPLORATORY**，不是新独立测试。严禁旧SS02 EEG encoder/对比模型/head checkpoint；所有新权重seed1024/2026/31415从头，P的head只可使用本轮同seed正式新train预训练。

仅2203原公开根，新增`pub_01/s03_continuity_20261023/eeg_run02_matched_baseline`。已有合法XLSR19特征可只读复用，拟合只用新train，validation只选点；无下载/患者/远端Git。本地同SS06分支/worktree，重型证据留2203，轻量完整副本可放既有C备份，Git仅必要摘要。

## 最小候选矩阵（协调已核对，正式协议已冻结）

| 条件 | 实际模型与学习 | 监督预算/时间信息 | 用途 |
|---|---|---|---|
| C | SS02深10/width64强Transformer的明确causal adaptation，全部从头联合监督；Conv32 head | 同run01 3262×3840目标呈现/seed、同record/block顺序、同60s Pearson+.1MSE/AdamW/clip；EEG≤target+8samples | 主匹配控制，比较run01五主框架及GRU |
| O | 原生SS02 EEG Transformer+Conv32，全部从头联合监督，无对比预训练 | 每60s用39个192sample/hop96重叠窗，Hann聚合后一次3840目标loss；与C同目标/step/顺序，额外重叠计算单列；离线未来EEG | 综合方法差异，不称纯lookahead因果效应 |
| P | SS02完整两阶段方法的低预算参考：新train EEG–XLSR19对比预训练、冻结encoder后Conv32读出 | 两阶段单独账本；对比覆盖新train合格3s窗，读出按3s Pearson及50% overlap一轮。额外预训练/重复目标**不称总budget匹配** | 原方法参考，不与C混称同信息/同总预算 |

所有旧run01模型/逐record结果只在输入、target、评分/监督预算一致时直接复用，不重跑39模型。C主配对六项（五框架+GRU），subject/family交叉bootstrap10000draws及六项Bonferroni；O/P单独描述。M3额外EEG SSL仍披露，不能称全部方法总训练预算相同。

## 因果性与精确账本

C必须处理所有未来泄露：原跨time LayerNorm改逐sample channel LayerNorm；所有Conv/readout改left causal；attention只允许过去/当前，并保留有限past KV与卷积缓存。原生剩余通道LN本已逐sample。跨record reset、60s块间detach保留状态，60s内拼接一次loss/step，不在3s内部detach。LN affine参数减少须报实际数，明确为adaptation而非原文模型。O/P的整3s attention/whole-time LN/sameConv仍有未来信息，不因lag相同而宣称权限相同。

每阶段分别记录唯一record/时间支持、有效EEG与目标sample呈现、重复/overlap、optimizer steps、梯度参数、时间/内存。C/O正式目标均为12,526,080sample/seed；P的原生重叠读出呈现另计，不能用“一epoch”掩盖额外量。对比batch保持同story不同time；尾batch处理/anchor重复如有必须显式记账。输入坐标沿用cache x[s]=rawEEG[s+8]，声学XLSR目标使用同story坐标s。

原生P overlap-add须保持与run01相同全部评分点，Hann边界零权重用唯一原生窗预测补全并明确为工程适配，不能静默删评分点。P的Pearson-only读出不校准mean/scale，MSE限描述，不伪称优化过该项。

已冻结具体实现：C含head3,128,137参数，O3,372,617；差244480个whole-time LN affine参数。C每层attention窗口192（含当前）、pastKV191、conv输入缓存63，head缓存31。原生无位置编码，滑动mask用相对sample差；缓存跨60s保留、record重置。内部192sample微块保留全部图，整3840一次loss。O/P共同评估窗仅在每原60s块内起点0:96:3648，39窗，不跨评分块；保留所有目标、两端零Hann权重按唯一窗补全。这是共同评分工程适配，不能称逐项复刻原SS02全record边界。

C/O每seed目标12,526,080；O额外重叠EEG窗呈现24,425,856，warmup另计。P预训练639step、65,240个唯一3s窗、12,526,080 EEG及声学sample，batch4–128、同story不同time、实际重复0；head2035step/130,231重叠窗/25,004,352目标呈现（最后batch55）。P原生3s loss/对比目标，单轮不是充分训练上限。

协调2026-10-26消息明确通过矩阵并授权通过必要测试后冻结/正式，无需再次等待回复。正式FORMAL_PROTOCOL.json、INTAKE.json及ENGINEERING_CHECK.json在run02轻量报告与2203同名根。真实pilot：C/O各20step、Ppre20step/head3step；未来扰动1.79e-7、full/chunk8.35e-7、跨块8.94e-7、Hann identity2.38e-7、跨microchunk梯度非零、实际60s全部应训梯度finite。pilot权重不复用。正式三seed已启动，所有训练/val完成且实质检查通过后才统一audit。

P生效补充：协调原协议审阅后，P尚0 formal更新时以P_NATIVE_AMENDMENT→_02冻结独立native入口；Ppre恢复原AdamW无clip，head恢复原Adam无wd/clip、manifest顺序shuffle=false，centered-cosine eps1e-6及本seed head初始化。C/O源与预算不改，旧pilot/源保留，新native03 pilot成功；不存在受影响P正式权重复用或按成绩选择。详见run02 PROTOCOL_NOTES.md；首版Protocol对应P字段以amendment为准。

## 执行与验收

先只读核验固定metadata/normalizer及特征覆盖，完成新loader/三条件真实train-val工程pilot。每阶段初始最多100更新步，pilot权重不进入正式结果。依据真实吞吐/内存/曲线完善矩阵和有限正式预算，**协调完成共同协议/预算核对后才正式训练**，不再索用户启动许可；此前可独立推进工程。

正式前冻结所有候选、seed、样本顺序、val选择规则、预算与统计；所有训练/选择完成后统一评估已看audit，不依audit延长/换模型。技术检查包含因果后缀扰动、分块一致性、同目标样本账本、初始化/预训练来源、无旧权重、metric独立重算和parent一致性。既有强架构/方法确实执行、三seed/真实比较和限制齐备后可PASS_WITH_LIMITATION；阳性非验收前提。

允许改本补充/必要任务登记、新SS06强基线适配/loader/trainer/tests/config、run02轻量报告。完整run02交EXECUTION_LOG、矩阵/ledger、FORMAL_PROTOCOL、RESULTS、REPORT、REVIEW和逐seed/record/person证据，commit/push同PR#6，完成后STOP。旧run01文件不改。

## Completion Record

三seed C/O各3262监督，P各639pre/2035head，全部12阶段完成，统一9模型seed评估与独立后置复核通过。C/O/P均值r=.123014/.151762/.141466；C六项交叉99.167%校正区间均跨0，不声称等效/非劣或未来信息单一因果。C三seed最佳末端，预算照冻结停止，P单轮/额外预算不称原文上限。技术PASS/验收PASS_WITH_LIMITATION/parent一致PASS_SS06_ONLY。

executor聚合误差0、18原derivative CPU r max3.36e-8/MSE2.36e-7；协调独立SUMMARY2.78e-17/六配对均值6.94e-18，范围非协调重跑全模型/CPU。三图视觉核对、1.21MB允许轻量包、中文REPORT/REVIEW/完整参数与访问账本齐备。source5b93ca5/P原法amendment55ddd40/observer82f3001，最终报告commit见Git/PR#6。run01冻结不改，全部数组/cache/权重留2203，只有元数据/汇总/静态图回本地及C备份。SS06完成，#6 Draft不合并；PUB/S03仍ACTIVE，无自动下一READY或下游artifact。
