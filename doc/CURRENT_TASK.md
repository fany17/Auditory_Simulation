# 当前执行：SS07 D9已冻结；D10旧内部FT读出36项正式执行（2026-11-01）

D7固定36/36角色-split已完成，PID535287正常退出。Ccap对P*/P-only原2:1加权r均差+.008695/+.004301，seen/held-out每seed均正；但同三seed历史P_FT=.168944726、Ccap=.168943486，差−.000001240且seed方向混合，未证明超过P_FT、等效或非劣。全部POST-HOC INTERNAL EXPLORATORY，不据旧测试重选。实际独立286来源/893输入/36聚合及72缓存重建通过（块r差≤2.78e−16，432原P未缩放桥差0），全文/三图已审阅；原样review57072ffe…b45f，Technical PASS_WITH_LIMITATIONS / Acceptance PASS_WITH_LIMITATION / Parent PASS。完整[REPORT](../reports/pub_01/s03/incremental_continuity/internal_generalization/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/internal_generalization/run01/REVIEW.md)及HTML run_id20261101_pub01_s03_ss07_internal_generalization已归档，三图说明/副本/链接通过。无新fit/encoder/rawEEG/DTU/下载，wall75.177s、CPU504.670s；科学goal ACTIVE。

D8固定27/27岭解已完成，PID541227正常退出不重启。P_FT*/P-only/C/Ccap三seed VAL60s/person r=.174141/.177842/.180541/.182340，Ccap对P_FT*/P-only均差+.008199/+.004498且各seed全正；支持强FT encoder也有当前读出开发增益，不是连续历史/独立泛化或新方法证据。三个完整历史VAL loss及首尾raw桥差0，无NN更新/旧test/DTU/下载。独立247来源/523输入、全部27系数/30聚合及18缓存重建通过（聚合0、block r≤3.33e−16），全文/三实际PNG已审阅，review a059476a…26a9原样回传。Technical PASS_WITH_LIMITATIONS / Acceptance PASS_WITH_LIMITATION / Parent PASS；完整[REPORT](../reports/pub_01/s03/incremental_continuity/finetuned_readout_transfer/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/finetuned_readout_transfer/run01/REVIEW.md)与HTML run_id20261101_pub01_s03_ss07_finetuned_readout_transfer归档完成，三图说明/副本/链接检查通过。wall386.589/CPU1183.994秒、RSS5.69GiB、全NPY24.469GB/free27.709GB守预算。prepare01接口与audit首版metadata失败保留，正式run不重跑/门不放宽，旧D4-D7冻结。

科学goal保持ACTIVE。D9固定W/F×3seed六解均实际收敛，queue563655和workers均已退出，不重启。Ccap/W/F VALr=.182340/.182474/.183874，F−W+.001400且每seed正、fused MSE−.000126614；但F窗口MSE352.068/r3s.003435/overlap706.180，巨大bias在固定Hann下近相消。内部/首学习边界/tail物理bias≤.00587/.12165/.30603，未证明融合输出无效或通用稳健可用；无历史输入/新确认/创新claim。独立275来源/535输入、六套完整TRAIN物理轴梯度/目标/真残差、全VAL聚合和12缓存重建通过（目标≤2.69e−14、残差差≤1.31e−16、block r≤2.22e−16），全文四实际PNG已审阅；review4e3a8163…d2e16原样保存，Technical PASS_WITH_LIMITATIONS / 诊断Acceptance PASS_WITH_LIMITATION / Parent PASS。完整[REPORT](../reports/pub_01/s03/incremental_continuity/physical_objective/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/physical_objective/run01/REVIEW.md)和HTML run_id20261101_pub01_s03_ss07_physical_objective归档完成，四图/说明/副本/链接通过。wall185.396s/CPUworker和347.091s、355CG/386Hessian；独立audit另69.714s。encoder/head/新NPY/旧test/DTU0，六解/λ/bias不再改，D4-D8冻结。

D10[精确PLAN](../reports/pub_01/s03/incremental_continuity/finetuned_internal_audit/run01/PLAN.md)b5d5e37已明确审定，57a09ff source2ce7df72…/protocol14b2cedd…在真实test前冻结；36 cached VAL读出桥全过。唯一PID599747（UID1012/start_ticks745101269）正式执行PFT*/P-only/C/Ccap/W/F×3seed×seen/heldout=36角色-split；seed1024新capture旧VAL Z/P gold均差0，原global64 actual layout/grad/eval签名匹配，进入原两phase测试编码。主254076窗/3972batch+300VAL/636test工程窗；每seed先全144原FT未缩放fused桥及旧blockr门，再5固定readout，不重启/换λ/新fit。新12NPY1.756GB、全NPY预计26.226GB<25GiB（26.844GB），root≤2GiB/free≥20GiB；完整结果/实际review/图文待。所有旧test仍POST-HOC INTERNAL EXPLORATORY，不称新确认，不据排名再训练；DTU关闭，科学goal ACTIVE。

以下准备记录为历史，原D4-D6不改。

D6最终实际独立review有限制通过（52来源/12输入/27coef约束/45共mask；9个选定新模型独立缓存输入重建，块r差≤4.44e−16），原样audit/review齐备。完整[REPORT](../reports/pub_01/s03/incremental_continuity/readout_attribution/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/readout_attribution/run01/REVIEW.md)、三图中文解释与全部表、轻量HTML归档run_id=20261101_pub01_s03_ss07_readout_attribution已完成（E:/实验室-科研工作-文件/00_ChatGPT/CodexResult/Result/runs/20261101_pub01_s03_ss07_readout_attribution/report.html，三图/副本/链接检查通过）。current开发增益有限归因成立，科学goal仍ACTIVE。

协调同意下一D7一次有界内部泛化审计的只读盘点/精确PLAN准备：P*/Ponly/S32/U32/C/Ccap×3已定seed×seen/heldout=36角色-split，只用D6已选λ/冻结权重与TRAIN transforms，无新拟合/选点/下载/DTU。先查既有cache/原P输出/bridge/mask来源，冻结具体矩阵与预算交协调审定后才能实际测试。所有结果必须POST-HOC INTERNAL EXPLORATORY，不称独立确认，不按D7重新选择/训练。当前尚无D7测试推理。

以下D6完成/启动描述均为当时进度，不重启其矩阵。

D6最终PLAN已协调审定含P-only；源码/协议2534b28在任何D6拟合前冻结，实际PID529076已正常结束。27新model-λ解+18原D4精确复用/45候选+3P*评分，完整REPORT/三图/逐seed逐人逐块及Hann分解已bbf7f08交付。P*/P-only/S32/U32/C/Ccap选定VAL60s均r=.162362/.171837/.174284/.173867/.174789/.176897。C−U32全部3λ×3seed正，解除共享U−S选定三seed负，平方收益正则依赖；P*overlap最一致却r最低，不能把低overlap当目标。source b0e4336…/core5b9cb9de…；wall62.767s/CPU348.898s，GPU/NN/新TRAIN载荷遍历0。独立实际review和轻量归档收束中，旧D4/D5不扩，goal ACTIVE。下一阶段仅给有界研究建议供协调评估，不自动新拟合。

以下D5交付与D6准备是历史进度，当前以本段为准。

D5实际独立review PASS_WITH_LIMITATIONS，21来源/60输入/36系数/共同mask/共享λ核对，关键重聚合差≤1.11e−16，报告全文及三图实际查看；原样review SHA97b6d4a3545eafad9e4b4ea742de6f0b93787cf36093543d0849f6a32079f833。完整[REPORT](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/REVIEW.md)及全36格/成本/三图中文解释已收束。轻量归档run_id=20261101_pub01_s03_ss07_interaction_probe，E:/实验室-科研工作-文件/00_ChatGPT/CodexResult/Result/runs/20261101_pub01_s03_ss07_interaction_probe/report.html，三图说明/副本/HTML链接检查通过。未建立额外双线性历史映射收益，goal仍ACTIVE。

下一步只准备最小current读出归因方案：相同MSE/P*残差起点下，shared local32与unshared local32必须同支持，再与完整dense C/Ccap比较；可复用D4一致解/moments，不重跑旧矩阵。Hann前同物理点overlap一致性只作预设次要描述，必须同时报告r/MSE、不据此选模型。尚未新增拟合，先交协调审定。

以下D5完成/启动快照保留，PID均已退出。

D5实际36解已正常完成，PID525296已退出，不重启；Anchor/Add/CC/Near/Older共享选择VAL=.176897359/.176902435/.176905911/.176902047/.176903531，全部选固定网格最强λ=1。Near−CC三seed全负、Near−Add仅1/3正，未达到预设交互线索门；不能用Near−Anchor微正证明机制。完整三图中文REPORT/全λ/逐seed逐人逐块及成本已685850e交付；实际结果review和轻量归档收束中。wall69.045s、CPU493.421s、GPU/encoder/head forward/NN更新0。PCA先于锚点检查的顺序差异已披露，所有岭解前锚点评分差0。goal保持ACTIVE，下一轮先协调有界假说，不扩大D4/D5网格。

以下D5启动描述为历史，原冻结source/协议不变。

D5具体[PLAN](../reports/pub_01/s03/incremental_continuity/history_interaction_probe/run01/PLAN.md)已协调明确审定，cdb970b源码/协议/实际目标及D4锚点输入指纹冻结；必要双线性等价/增广岭/record边界/共同主效应检查通过，sourceSHA fb40aec382b560df1e31ec7bea36733372de5e335ee4f5961b964f63c4912df2，PID525296在2203 ROOT pub_01/s03_history_interaction_probe_20261101正式执行固定4臂×3seed×3λ=36解。冻结D4 Ccap(.01)作纯current anchor，所有learned共享current/near/older加性A；Add无交互、CC当前交互、Near/Older历史交互。CC仍含共同历史加性，不称纯current。只CPU8线程、GPU0、无新NN/test/DTU/下载，完整结果待；goal不以工程PASS完成。

以下D4交付保持冻结，旧“准备D5”描述为当时进度，不重启D4。

D4固定54解已全部正常完成，实际独立结果review PASS_WITH_LIMITATIONS，重聚合最大差1.11e−16。P/C/Ccap/CS/CF/CO/CF48选定VAL60s均r=.162362/.174789/.176897/.174538/.174090/.173939/.173970；current读出开发增益三seed为正，CF对C/Ccap三seed均负，未建立额外历史收益。完整[REPORT](../reports/pub_01/s03/incremental_continuity/conditional_history_probe/run01/REPORT.md)/[REVIEW](../reports/pub_01/s03/incremental_continuity/conditional_history_probe/run01/REVIEW.md)、54全λ/逐人/逐块与三图齐备。轻量归档run_id=20261101_pub01_s03_ss07_conditional_probe，E:/实验室-科研工作-文件/00_ChatGPT/CodexResult/Result/runs/20261101_pub01_s03_ss07_conditional_probe/report.html，三图/说明/副本/HTML链接检查通过；科学goal仍ACTIVE，不称独立确认或新方法。

本D4矩阵冻结，不新增解/距离/NN。下一步只提出“历史是否改变当前EEG→包络映射”的最小二阶交互probe，保留D4强current Ccap作锚点和同容量current控制，另冻结run供协调审定，不直接追阴性扩容。无新test/DTU/下载/encoder forward/患者或其他节点。

以下为D4实际启动历史，PID已正常结束，不重启。

D4方案4f59651已协调明确审定；源码a8e4c12、PROTOCOL及必要公式/无未来/PCA矩变换检查已冻结，实际PID521141/source0182fb055b7d684026ad3a21d13d715e70e7cd952831b42c436bcd6a4f54a410在2203正式执行。ROOT=pub_01/s03_conditional_history_probe_20261101；本地[PLAN](../reports/pub_01/s03/incremental_continuity/conditional_history_probe/run01/PLAN.md)。原TRAIN/VAL、三seed、C/Ccap/CS/CF/CO/CF48六臂×三λ共54闭式解；先原P评分核对，TRAIN-only统计/正仿射/PCA，共同i≥4拟合、首4窗共同P回退，原49人752块评分。无新NN/encoder forward/test/DTU/下载。源码静态审阅并行，不增新用户等待门。全部结果/成本/失败与完整报告仍待，不把工程PASS/正式启动当goal完成。

以下warm轮次已交付，旧“仅准备D4”是当时进度，以上段最新执行为准。

最新实际状态：18/18fit全部5epoch完成，90epoch/355050更新/22720860监督窗，全格CSV/三图齐备；训练不重启。wholeVAL v1首格失败原样保留，恢复真实cache/RawWindows执行路径的独立v2（source2df9f4f、协议6fc0f00a…4d2c9）已实际完成18格/549000窗/13536块，wave绝对差全0，原1e−6主门全部通过，未改训练/权重/分数/选点。协调原样写入COORDINATOR_RESULT_REVIEW，technical PASS_WITH_LIMITATIONS / scientific NO_STABLE_INCREMENTAL_HISTORY_BENEFIT_ESTABLISHED，goal仍ACTIVE。run02/REPORT.md逐图解释及完整限制已写；以下启动/失败进度均为历史。

当前仅准备最小D4可审方案，检验控制当前EEG/原P后的dense历史相对summary条件预测贡献；尚未执行新拟合。依据是已完成SOURCE-TRAIN D1/D2/D3，不凭13%能量直接扩容，也不推进负向的仿射校正。只既有TRAIN/VAL、固定3seed、有限线性/岭probe、同容量当前/更早past控制；方案交协调审定后再执行，无新NN架构/下载/test/DTU推理。warm中文报告/轻量归档/提交仍优先。

唯一当前节点仍PUB-01-S03_SS07 / ACTIVE，持续goal尚未达到。round01全部54screen、37唯一full、五native/FT、五seed统一内部/DTU评分及实际raw v1/metric v2复核已完成；该轮controller/eval/review正常结束。实际M4−P内部−.000621、DTU+.000149，M7近似区间均跨0；未建立稳健超越、历史机制或投稿充分性。当前后续warm round02正式训练见下段。

本轮Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS；报告入口[REPORT](../reports/pub_01/s03/incremental_continuity/run01/REPORT.md)、[REVIEW](../reports/pub_01/s03/incremental_continuity/run01/REVIEW.md)。九张实际图已逐图解释并协调独立审阅；轻量HTML归档run_id=20261101_pub01_s03_ss07_round01，路径E:/实验室-科研工作-文件/00_ChatGPT/CodexResult/Result/runs/20261101_pub01_s03_ss07_round01/report.html，九图/说明/副本/链接检查通过。

当前见[NEXT_TRAIN_VAL_PLAN](../reports/pub_01/s03/incremental_continuity/run01/NEXT_TRAIN_VAL_PLAN.md)及run02/WARM_PROTOCOL.json。8701673方案及7fb74d6最终源码、a04实际18格VAL/合成证据已协调独立审阅通过；review SHA0d208252…c87587。正式协议已冻结a9e1d220170a9c6a166c71c8113ed6ede0b6570a5aab3152e5a3ee5a685481a8，六格三seed、固定5epoch/epoch−1候选/原3s选点/所选VAL60s，预先固定18新fits共同实现；旧FT仅历史参考，不按新分数选择是否重跑。队列416910及首批frozen_p worker416911/416912的UID/实际argv/start_ticks已核实，START_VERIFIED在run02；当前两GPU正式train/val，未完成全部矩阵，不在工程PASS处收工。新ROOT=2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_warm_20261101，本地run02；round01产物冻结。旧内部测试及已暴露DTU无新推理/调参授权。沿用SS06 worktree/branch/PR#6 Draft，不合并；患者/STN FORBIDDEN，信号/权重/预测只在2203。无新下载、患者或其他支线启动，goal ACTIVE。

---

## 历史：SS07 round01执行过程

用户最新明确授权尽快全量实验、设goal、使用2203全部可用算力，研究目标超过原baseline并积累ICML/NeurIPS级证据。SS07 ACTIVE；旧STOP/单pass/NO_GO不覆盖本授权，旧结果保持只读。先执行PUB-01-S03_SS07_FULL_SCALE_INCREMENTAL_CONTINUITY.md及本轮INTAKE/BASELINE_PROTOCOL，原完整Transformer/XLSR19两阶段、fold完整train/val、100epoch/pat15预训练与250epoch/pat10head至少3seed；仅在强模型上增加五类有界残差，精确矩阵协调审阅。开发只train/val，旧test均内部暴露，未冻全计划不评估。无阳性/论文接收保证。

唯一计算2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101；本地reports/pub_01/s03/incremental_continuity/run01。新cache≤25GB、余量≥20GB、best/latest有界，不删历史/不动他人进程。沿用SS06 worktree/branch/PR6 Draft，可commit/push、不合并，不动主目录CNN/TRF/stash，无患者/远端Git/擅自新数据。持续goal不以smoke、技术PASS或单seed分数完成。

2026-11-01执行接续：五native完整两阶段及五FT已完成；54/54筛选齐备、五类均H16，初始15full已全部完成，原三seed VAL规则冻结m4_h16（FULL_PRIMARY_SELECTION d43da5b521af0cabfed6a2a6fc4ce968ae39f20824411d07c305fb1fa08b69d2）。五family验证均值及M4三个seed均低于原P；保留阴性证据，不改矩阵。现37/37full全部完成（主比较25+其余family12），协调独立核验全部预算/选点/来源通过，14537正常结束。FINAL_CHECKPOINT_MANIFEST锁定47角色/106成本条目；五seed×2VAL实际最终权重bridge差均0，EVALUATION_READY已在预测前冻结。原source4e33d69、队列88448已自动启动两卡统一推理；primary/M7/五seed保持原冻结。下一步等待完整summary与实际raw v1/metric v2，实际图表使用v2入口，goal仍ACTIVE。

独立复核现用v2：source1153dbf、INDEPENDENT_REVIEW_PROTOCOL_V2（97c7686b94d63836c59d83df561887d926d2d65a3528ea004ddf86cd1d52d28d），18source/8proof已协调独立核验。五native合法VAL3760块复现大R²舍入误拒绝1例；仅R²改用既有MSE的scaled≤1e−10核验，r/差值/CI仍绝对1e−10，raw绝对1e−6、精确常量拒绝均不变。原raw v1几何/源/协议与旧8/3原位保留并备份；旧等待父113154经UID/argv/无children/无actual review核验后仅该父进程退出。新PID233443实际live/WAIT_COMPLETE_FIXED_EVALUATION，上游88448；handoff时仅此一个review queue且14537/88448未动；现14537已正常完成，88448继续实际评估。

复核只在完整final_summary+ready后执行原raw v1→metrics v2；完成标记为INDEPENDENT_REVIEW_COMPLETED_V2.json，实际图表入口为pub_01_s03_ss07_result_figures_v2.py。按既定metadata固定3条Sparr/6条DTU×5seed全main重放raw，全部保存预测/统计另复算；不运行旧queue/旧呈现main。任一失败保留不自动重试，不提前读成绩，不改活跃冻结源。实际计数看服务器状态与进程；DTU仍原两文件白名单。工程修正和排队不代表实际复核或科学目标完成，继续统一评估、实际复核及科学解释。

---
以下是旧交付/STOP历史，非当前路由。
# 当前交付 — SS06 run02 COMPLETED / PASS_WITH_LIMITATION（2026-10-26）

用户授权的匹配比较已完成：C/O各3×3262、P各639pre/2035head三seed、统一9模型seed/538块、原始derivative CPU/独立汇总复算与3图齐备。C/O/P r=.123014/.151762/.141466；C相对五框架/GRU六校正区间全跨0，不能认定普遍不如强因果控制或等价。O/P未来信息、P额外学习/不同loss与单轮限制明确，POST-HOC INTERNAL EXPLORATORY/5family。报告continuity/eeg_run02_matched_baseline/REPORT.md、REVIEW.md。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS；协调独立SUMMARY误差2.78e-17。当前STOP，无自动下一READY，同#6 Draft不合并。run01冻结与原资料/CNN-TRF/stash不动；无下载/患者/远端Git，下游不启动。

## run01历史交付（已被新启动授权覆盖STOP）

# 历史交付 — SS06 run01 已完成有限制（2026-10-23）

用户“启动吧”授权的SS06真实实验已完成：39×3262监督、3×3511 SSL，新train统计/完整连续record/一次内部审计/后置复核齐备。M1multi .122913、M2 .128164、M3finetune .123747、M4adaptive .113400、M5full .114178；GRU .125221。八primary三项为正，其余跨0；不把M2点估计当稳健胜出，未建立SSL相对GRU或M5额外收益、内容特异历史收益。只5family且旧训练暴露，INTERNAL/EXPLORATORY，非外部确认。

报告reports/pub_01/s03/continuity/eeg_run01/REPORT.md及REVIEW.md；243真实结果/四图/逐seed人表及机器review可读。原型CPU222r max差1.55e-7、协调SUMMARY独立重算max差5.55e-17，完整与精简包检查PASS。SS06 COMPLETED/PASS_WITH_LIMITATION；当前STOP，无自动下一READY。分支codex/pub-01-s03-ss06-continuity / PR#6 Draft、不自行合并；S03/PUB ACTIVE，SS04/S04/父S05 BACKLOG。E盘回传空间不足已用本轮C盘备份及精简JSON收束，原资料/旧worktree不删；数据/权重继续只在2203。

## prototype01 历史状态（以下非当前训练禁令）
# 当前任务 — PUB-01-S03_SS06（2026-10-23）

用户批准五种连续信息框架的任务书和基本代码，仅做合成工程检查。五模型已在2203完成前向/梯度/状态测试，独立代码/运行证据审阅已完成，原型包COMPLETED/PASS_WITH_LIMITATION；真实EEG、预训练、pilot、正式训练/测试全部NOT_RUN。本原型包已完成，当前STOP，不自动训练。

- 任务书：doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md。
- 证据：reports/pub_01/s03/continuity/prototype01/REPORT.md；状态以PUB-01/SUBTASKS.md为准。
- 独立worktree分支codex/pub-01-s03-ss06-continuity；main起点60c3404。PR#3/#4均已按既有授权合并，旧模型/结果冻结；主目录CNN/TRF与stash不动。
- S03/PUB保持ACTIVE，SS04/S04/父S05 BACKLOG；未生成ART-AUDREP-v1。

## 历史工作板快照（以下非当前执行指令）
> **Issue#5 / SS05已完成有限制并停止**：15run真实训练/一次新holdout测试、7种同当前EEG历史条件及后置review已完成。五组r约.091–.092，六预设比较区间均跨0，未检出复杂记忆额外收益。报告predictive_memory/codex_run01/REPORT.md。PR#3已按新授权合并8d21d0a，PR#4保持Draft，不合并/不扩实验。

# Auditory_Simulation Current Workboard

更新时间：2026-10-19（Codex 分工纠偏）。父任务状态由 `doc/TASKS.md` 定义，子任务状态由 `doc/tasks/PUB-01/SUBTASKS.md` 定义。**以下当前区是唯一执行路由，其下为历史记录。**

## 当前交付与停止

- **任务：`PUB-01-S03_SS05`，COMPLETED / PASS_WITH_LIMITATION。** 已交真实内部holdout阴性证据及同agent separate pass；协调只核对轻量文件，不称第三方科学签署。当前STOP，无自动下一READY节点，不启动SS04/S04/父S05。
- **执行归属**：Codex 写代码、运行 tests/合成验证/2203公开EEG、复核与提交；协调助手维护总任务并核对指定轻量文件。旧协调助手合成初稿已另由Codex实际复核，accuracy复现而A-assoc MSE有差异，详见CODE_AUDIT；不称第三方科学签署。
- **工作位置**：独立managed worktree pub-01-s03-ss05，分支work/pub-01-s03-ss05-predictive-memory/PR#4 Draft、base main。PR#3已由协调者按新授权合并8d21d0a；当前Codex不更动主目录CNN/TRF/stash或合并#4。
- **证据入口**：codex_run01/REPORT.md、REVIEW.md、FORMAL_PROTOCOL.json、105行RESULTS及15run轻量证据；新holdout来源限旧train，重拟合统计防泄漏，没有用旧SS03 test调参。原始EEG独立CPU90相关复核max差8.13e-6，全部汇总误差0。
- **不可外推**：本轮没有支持关联/历史残差的额外收益，不等于证明所有历史记忆无用；不能声称超过GRU/旧SS02/SS03、预测误差机制成立或第三方科学签署。PR#4继续Draft，后续由研究者另行决定。
- **其他状态**：SS01/SS02/SS03已完成有限制并冻结；SS04 BACKLOG；S04公开颅内支线BACKLOG；S05版本化artifact整合BACKLOG；公共项目与患者/STN项目单向版本化衔接，禁止患者数据进入本项目。

## 历史执行/状态快照（仅供追溯）

### 历史：SS05启动前入口（已被上方交付状态取代）

**2026-10-19新授权 SS05：ACTIVE / SYNTHETIC_PROBE_COMPLETED；正式公开EEG TRAIN/TEST NOT_RUN。** 已写PI-MRAM（预测误差＋多速率状态＋关联delta记忆）及同窗口GRU控制、10项本地CPU单元测试与两套合成实验。延迟两键干扰160步：full/assoc_no_pred 100%，GRU62.5%；GRU600步也100%，不能声称不可替代的算法能力；预测误差独立贡献尚不成立。代码和结果仅在PR#4独立工作分支。下一步按 `PUB-01-S03_SS05_PREDICTIVE_MEMORY.md` 与 `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md`，先做公开source-family和新内部测试划分可行性审计，SS03旧测试集不可再用于SS05选模型。**不自动启动SS04/S04/S05，不产生artifact。**


2026-10-19交付：`PUB-01-S03_SS03` **COMPLETED / PASS_WITH_LIMITATION**。真实输入对齐/CPU-GPU smoke/100步pilot→测试前预算冻结→4×3正式训练及每模型一次test→同agent separate review已完成。multiscale seen/heldout r=.10305/.11380；long .06411/.06979；middle .09297/.10413。当前四原型/固定数据预算下的H1获支持，不是生理机制或超过SS02。报告`reports/pub_01/s03/architecture/REPORT.md`。

当前STOP：PR#3保留Draft，工作分支work/pub-01-s03-ss03-multiscale/独立managed worktree；主目录未提交CNN/TRF和stash未动。S03/PUB仍ACTIVE，SS04/S04/S05保持BACKLOG，未产生ART-AUDREP-v1。后续须研究者指定，无自动下一READY节点。下方10-15为SS02历史交付。



2026-10-15交付：用户授权“缺几个就缺几个”后，两套指定强基线及Transformer直接包络readout均完成真实训练/验证/一次测试与独立review。SS02 **COMPLETED / PASS_WITH_LIMITATION**；当前执行停止，没有自动下一READY节点。报告`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。

- Parent：`PUB-01 — Public Auditory–Neural Representation`，`ACTIVE`。
- 当前优先支线：**A / SparrKULee / EEG 听觉直接解码**，归属 `PUB-01-S03`，`ACTIVE`。
- **本轮完成节点**：`PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction`，**COMPLETED / PASS_WITH_LIMITATION**。
- 执行入口：`doc/CODEX_START_HERE.md` → `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`。
- 任务：按冻结的可执行公开条件复现HDPS与Transformer-CLIP-EEG，明确原论文条件差异；当前HDPS仅M，Transformer仅单模型及其原方法包络readout，不启动新架构/GPT重排序。
- 实际结果：HDPS 76人、100Hz/M、作者split及valid索引，Top1/5/10=12.03/30.45/42.10%，shuffle Top1=0.53%。Transformer作者fold4、70人训练、69seen/13heldout，64Hz/XLSR19；2AFC=83.53/80.08%，加权82.38%；冻结encoder原Conv32直接包络r=.1688/.1403，加权.1593，错配近0。三运行均review PASS_WITH_LIMITATION；官方完整cohort/hidden test/ensemble未复现。
- `S03_SS03` 架构与 `S03_SS04` 统一验证均 `BACKLOG`，需要前一环节审阅后再决定。

## 两条数据支线

| 数据与职责 | 当前状态 | 阻塞关系 |
|---|---|---|
| A / SparrKULee EEG decoding / `S03` | ACTIVE；SS02 COMPLETED / PASS_WITH_LIMITATION | 两套公开可用条件基线已交；SS03已授权原型启动，实验待运行 |
| B / ds004703 public intracranial / `S04` | BACKLOG；SS01/SS02 尚未执行 | 自身 sEEG baseline 与 A 线可独立；**仅 EEG→sEEG 迁移需 A 线 frozen 模型** |
| `S05` artifact integration | BACKLOG | 两条支线的结果与失败/许可验收后评估 |

参见 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md` 与 `BENCHMARK_REGISTRY.md`。

## 已经完成且不重跑的证据

- `S01`：公开 SparrKULee / ds004703 数据可用范围下载、审计、初步虚拟清洗、图形化探索已阶段验收；历史 ds004703 319 候选片段**不等于正式模型输入合法性**。
- `S02`：83 名参与者、659 条记录 acoustic/log-mel/frozen wav2vec2 → EEG encoding，完成有限制；HuBERT HOLD。
- `S03_SS01`：10 人、93 条记录 64 Hz EEG→声音包络 backward TRF，test r≈0.1819、R²≈0.0178、错配 r≈−0.0033；2026-10-11 PASS_WITH_LIMITATION，证据 `reports/pub_01/s03/envelope_decode_20261011/REPORT.md`。

## 硬边界 / Stop

- 所有数据、权重和实际计算 **server2203**；本地 Windows Git 只存轻量代码、配置和汇总；先只读盘点已有工作，不覆盖用户未提交内容。
- 不读取任何患者/STN 数据与 embedding；公共模型只通过冻结、版本化 `ART-AUDREP-v1` artifact 向患者侧 `INT-01` 单向交付。当前尚未冻结。
- 当前两套方法的可用条件复现不等于完整论文数值一致性；完成SS02三级审阅后停止，不自启SS03/S04或GPT新模型。
