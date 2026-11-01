# D7：冻结强current读出的内部泛化审计

状态PROPOSED_NOT_RUN。PUB-01-S03_SS07内独立冻结评估run，无新模型拟合；需协调审定具体协议/代码后才测试。D4/D5/D6交付保持不变。2026-11-01。

## 问题与暴露边界

已在反复使用的source VAL找到P-only/current latent读出增益，短历史加性/双线性未获额外收益。D7只问：已冻结的current增益是否也出现在原seen/heldout被试/其他clip集合？这是**POST-HOC INTERNAL EXPLORATORY**，原测试已在round01等阶段暴露，不能重命名独立确认、前瞻test、无偏泛化或ICML/NeurIPS充分证据。

不新增拟合/调λ/改窗口/增seed/删人/按成绩选胜者；不读取DTU、患者、新载荷或未来metadata候选的数据，不根据D7排名重开D4-D6。真正独立确认仍是未解决缺口。

## 只读资产盘点：可复用，无需encoder前向

2203只读JSON及NPY header盘点已完成，未读预测数值、运行模型或新评分；详见INPUT_INVENTORY.json。

| split | record | person | source_family | windows | 原有效60秒块 |
|---|---:|---:|---:|---:|---:|
| test_seen | 69 | 69 | 9 | 36573 | 872 |
| test_heldout | 75 | 13 | 8 | 48119 | 1172 |

原ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101。三个seed各144个已冻结test latent、原native_P fused预测全部存在，header/shape正确、错误0，cache_manifest encoder SHA与原最终checkpoint一致。latent在final_latent/seed{seed}/，P fused在final_results/seed{seed}/predictions/internal/native_P/。它们已经过round01实际独立review；不重新提取encoder特征、不重做raw EEG/预处理。

原P只保存完整fused预测，未在该final_results中保存全量window P。为冻结读出的P192输入及首4回退，需从这些既有latent**仅重放原冻结P head**，保持原eval_common.predict_windows/aligned_chunks(global phase_offset)代码/精度/stride/grad标志上下文，并以全部144record原P fused作桥；不是新训练，也不是完整NN/encoder重放。原D4 head训练/VAL使用global64而旧test reader按record边界分割的执行上下文差异明确保留，D7忠实沿用原test P路径，不能悄悄替换为另一批大小/CPU近似。

只读盘点现有SS07各root NPY合计18406879264B；新float32 window P预计195130368B payload（另432个轻header），合计约18602009632B，低于25GiB；盘点磁盘余量约34.19GB。只新增必要P window cache，latent/fused旧P不复制；首实际运行仍复核余量/计数。

上述原inventory实际只列四个具体root，不能称完整allSS07。协调全七个s03_*_20261101根独立补数18406910112B，额外warm NPY30848B；正式PROTOCOL前的INPUT_INVENTORY_SUPPLEMENT逐root列清全部7根（含warm/diagnostics/本D7），预算与runtime guard以该完整范围为准，原盘点文件不覆盖。

## 冻结36角色-split矩阵

seed固定1024/2026/31415，原P encoder/head、D4 TRAIN transforms/正仿射a,b、D4/D6已选系数只读。

| 角色 | 来源 | 固定λ | 新拟合 |
|---|---|---:|---|
| P* | 各自原P window输出的既定TRAIN正仿射 | — | 0 |
| Ponly | D6 P-only原系数 | .0001 | 0 |
| S32 | D6共享local32原系数 | .0001 | 0 |
| U32 | D6同local32 unshared原系数 | .0001 | 0 |
| C | D4全current dense原系数 | .0001 | 0 |
| Ccap | D4全current+平方原系数 | .01 | 0 |

6角色×3seed×2split=36角色-split评分，18个角色-seed流式覆盖两个split；不是36组训练。没有新λ比较，P-only不是无监督校准，S/U共有dense P路径、current encoder离线3秒权限都不变。C/Ccap保持D4逐坐标TRAIN标准化、P*残差、自由位置截距，不以test重拟合均值/std/尺度/阈值。

所有角色保留原phase顺序/offset、64Hz/192窗/96步长；每record局部i<4一律P*，其余按固定系数加残差。原SPARR_TEST_MASK的target-only有效块/覆盖/record集合原样复用，Hann与60秒坐标不移动。seen首4回退276窗、heldout300窗/seed；不删除边界块或预测失败样本。MSE用共同P*的已有target单位，无逐窗/record/目标域校准。

## 先桥接，再候选推理，失败保留

先冻结新源码/PLAN/全部角色与λ、旧FINAL_CHECKPOINT_MANIFEST/EVALUATION_READY/SPARR_TEST_MASK、old raw/metric review证据、原head/native预测helper及依赖、D4/D6协议/实际review/TRAIN transforms/P_CHECK/3套原head及15套选定ridge系数的来源。缓存/旧P/target的shape/dtype/size/mtime与record长度在新预测前固定；沿用旧已核验provenance，不新增全库hash流程。

正式测试前协调审定source/协议与必要合成/VAL桥。实际模型评分门为：各seed重放原head全部144record并存只在2203的window P→原helper Hann融合与旧原P fused比较最大绝对差≤1e−6（**原P输出单位**，含全部既定坐标）→同一原mask的P*逐块r与旧P差≤1e−10及正仿射不变性→整个seed门通过才读出5候选，不以任何分数好坏开关门。bridge不是允许放宽阈值或重选路径；失败原位记录，不自动重跑/换实现。

old P raw bridge沿用冻结head代码的torch.no_grad、原精度/backend flags、requires_grad标志、record tensor布局和global phase-aligned64切分。头推理仅既有8×192 cache；encoder/raw EEG访问计数0。若实际发现缓存/桥错误，停止受影响seed与完整审计科学汇总，保留其余可核工程证据、交协调具体判断，不改mask凑齐。

必要实现检查仅fixed系数重建、S/U约束、首4回退、相同record/phase key、原Hann/严格常量拒绝、冻结C/Ccap/Ponly等在VAL的原分数一致；不做新参数搜索或完整raw NN replay。实际独立review检查全部36评分共mask/来源/固定角色、关键聚合，使用新window P/cache重建代表性readout即可，不再重放encoder。

## endpoint与预设比较

主要比较预先固定**Ccap−P***与**Ccap−Ponly**；Ccap−C、S32−Ponly、U32−S32、C−U32为解释性次要。逐seen/heldout/person宏r及原2/3 seen+1/3 heldout加权，逐seed/人/record/60秒块完整报告，不按最高分选模型。所有角色-seed用完全相同872/1172块、69/13person；无预测依赖排除。

同时报告3秒r、同单位target MSE（块→person）与必要source-family描述，始终区分角色/seed/person/family；不将17来源或82人冒充独立试验。仅描述配对点差/seed方向，不新增确认性p值/CI或根据test分层调整模型。连续性/固定Hann分解可按D6同公式作预设次要描述（不参与选择），必须并报r/MSE，不能把更一致当胜出。

3秒r仅辅助：若任何窗口target或预测恰为常量，其record/split全量3秒r记null并给defined/undefined覆盖数，不挑有效子集均值，也不因此删除/停掉仍合法的60秒主endpoint；MSE仍全量计算。60秒主块常量预测/真值仍严格失败，含防止恒定输入经Hann舍入伪造微小方差的检查，不改变原mask。3秒window target按原监督缓存float32表达，连续60秒truth沿用原文件数值。VAL桥同时核对3秒/60秒和target MSE/固定Hann分解，合成覆盖完整60秒块、person宏计权/2:1及辅助undefined口径。

## 预算、产物与停止边界

新ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_current_readout_internal_audit_20261101。最多1张空闲L40只冻结P head、GPU≤1GiB；CPU/BLAS/OMP≤8、RAM≤8GiB、wall≤30分钟、新派生≤512MiB、combined NPY≤25GiB、余量≥20GiB。约254076个P head窗口、432record bridges；其他均代数读出和评分，encoder/训练0。每阶段CPU/GPU/wall/缓存/窗口数与共享旧训练成本分列，不以低额外成本冒充零成本模型。

完整原预测/新P window cache及固定系数留2203，本地只代码、协议/来源/结构化全结果、静态图、中文REPORT/REVIEW。保留失败及旧证据，所有删除进回收站/备份，不覆盖旧ROOT。完成本次审计后冻结结果；无论正负均不据此改D4-D6、不自动启动DTU/新数据或再训练，科学goal仍不自动完成。
