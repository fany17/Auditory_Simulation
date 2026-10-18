# PUB-01-S03_SS02 单独审阅记录

## 最终结论（2026-10-15）

Technical **PASS**，Acceptance **PASS_WITH_LIMITATION**，Parent consistency **PASS**；SS02 **COMPLETED**。按用户事先授权的可用公开条件，两套指定方法均已完成实际训练/验证/一次测试，且原Transformer直接包络读出也完成。HDPS Top1/5/10=12.03/30.45/42.10%；Transformer seen/heldout2AFC=83.53/80.08%，加权82.38%；直接包络r=.1688/.1403，加权.1593。所有模型选择仅使用validation。

包络独立review从全部test latent在CPU重新预测并重建每个60s段，max相关误差8.18e-6、汇总误差0，0无效段。250轮达到原定上限时val仍改善，报告此预算限制，不补跑/追test成绩。2AFC/包络均固定70 train、69 seen/13 heldout、原fold4和train-only scaling；未把不同任务/不同cohort百分比排名。

限制不撤销：完整论文数据/hidden packet/ensemble未复现、HDPS原M数值仍未核实、公开候选revision与论文描述有差异、单seed/共享刺激不确定性、物理同步UNKNOWN。原10行official/internal未跑结果保持空指标。实际3个运行各自review均PASS_WITH_LIMITATION；轻量交付包schema/证据对照见`available_package_review.json`。本审阅是同一agent的单独post-execution pass，不是外部科研签署。

父S03和PUB-01保持ACTIVE，SS01/S02结论不改、未产生ART-AUDREP-v1。停止当前SS02，后续架构须研究者另行指定；不自动启动SS03/SS04/S04。

## 2026-10-15：HDPS正式结果审阅

Technical **PASS**；该HDPS闭环 Acceptance **PASS_WITH_LIMITATION**；Parent consistency **PASS（同SS02的用户子集amendment）**。76人/599对拟合前按输入完整度冻结；seed1024/validation Top10 checkpoint/early stopping后一次test：Top1 12.0327%、Top5 30.4454%、Top10 42.1004%。独立rank→逐被试→macro重算max误差4.77e-8；phase feature IDs无交集，候选身份无重复且数量与冻结组一致；不完整源对象已排除，5项strict/available模式tests通过。详见`evidence/available_20261015/hdps_m/review.json`。

限制：76 versus原release81/paper描述85；M-only且原表格数字未核实；候选100–235原本存在于固定公开index，不因测试分数改它；单seed的subject bootstrap不覆盖seed/共享刺激变化。不能据此断言与原paper一致或用MWG20%做差。第二基线已完成，整体结论见顶部；旧10-13 HOLD是历史记录。下方工程/首次审阅原文保留。

## 2026-10-15：Transformer二选一正式结果审阅

Technical **PASS**；该二选一闭环 Acceptance **PASS_WITH_LIMITATION**；Parent consistency **PASS**。70训练被试，seen69/heldout13；subject-macro2AFC分别83.5295/80.0786%，加权82.3792%；同record EEG shuffle49.9607/49.9722%。训练24轮，val CE选第9轮，连续15轮无改善后一次test。14项独立检查全部通过：固定query/candidate不变、候选同record±4s/合法support、唯一query、argmax/标签重算、逐被试与macro/weighted、heldout被试隔离、seen train/val/test精确story ID隔离、train-only scaling、固定停止规则、未声明official等价。指标重算误差0，见`evidence/available_20261015/transformer/contrastive/review.json`。

原hidden packet、完整180人训练及15模型ensemble未完成，不比较论文87.01%的数值差；available public derived protocol明确单列。包络readout亦已完成并通过CPU独立复核，整体结论见顶部；未启动SS03/S04。

## 追加执行的独立review pass（10-13历史）

Engineering **PASS限定**，Acceptance **HOLD**，Parent consistency **PASS**。独立NumPy Pearson验证、实际batch256 backward与原raw/M时间支持见`evidence/continuation01/engineering_review.json`。first bit-exact检查未通过；核验差异为BLAS线程下1.69e-10μV/relativeL2 3.97e-12，不能宣称逐位一致。

81个raw作业均结束，602个100Hz产物；按原feature长度尾截断规则，作者647对有600对工程就绪，47必要raw仍受限。sub022的public events容器修复显式记录且未请求APR，恢复其公开train记录；47=37train/5val/5test，49受限BDF中的2个no-match不计必需。数据与索引已用实际输出核对，不依据文件存在或任意fixture认定性能。

独立复查训练入口含inputfailures硬拒绝，实际正式训练/测试NOT_RUN，RESULTS所有性能留空，评价人数0。未把工程loss、M变体或76人结果与MWG20.0%作对照；原论文M数字仍未核验。原Transformercheckpoint失效和官方测试定义不足是局部事实，未重复查询或用新模型替代。原方法GPT2可以作为冻结特征，但使用条款不从关联code自动推定。

实际2203 guard fixture验证：不完整作者输入会在读candidate/array与创建train目录之前拒绝，1项测试通过，fixture仅JSON并保留，不作科研数据。新增代码AST/compileall和Git diff空白检查通过。最终9项轻量metadata检查通过，见`evidence/continuation01/final_metadata_review.json`；不等于性能验证。

以下首次review原文为历史快照，其无100Hz/no依赖/no索引等资产描述不再代表当前状态。未改父验收、S01/S02/SS01结论、Git历史或原frozen artifact。无后台HDPS作业，无SS03/S04启动。恢复同条件资源后仍只继续SS02。

审阅日期：2026-10-13。执行者完成报告后，另做spec→固定源码/配置→实际日志/metadata→RESULTS的review pass。此为**同一agent的独立步骤审阅**，不是另一个agent或外部研究者的科学签署。审阅工具为`scripts/pub_01_s03_review_strong.py`；机器检查结果见`review_checks.json`。

## Technical validation

**PASS（限定：协议审查/preflight/诚实记录），不包含模型复现通过。**

- 37个pinned source/text全部实际读取，Python源码AST通过；作者源码未import/执行，故不能声称模型API、loss/backprop或权重加载smoke通过。
- 2203实际sub-001十条记录的finite、64通道shape、共同区间与64Hzcontract检查通过；物理同步仍UNKNOWN。GPU tensor runtime通过；4项路径/执行边界/部分split覆盖技术tests通过。
- run01网络失败证据完整，run02没有覆盖它；官方协议与内部未启动项分列。RESULTS所有未运行指标/实际候选数/实际seed为空，评价subject数为0；预期2/5候选与真正生成过的candidate_n明确区分。
- 尚无checkpoint、prediction、最终候选manifest、training日志或per-subject性能；server manifest只是文件路径/字节数清单，不做checksum/hash审计。无科研数组回传。

## Acceptance validation

**HOLD。A+B“两套独立强基线实际结果”的目标未达到，不能标PASS或COMPLETED。**

已交付4个mandatory文件、真实代码/config/tests、2203产物清单、协议差异与精确阻塞；这一文档/可读性工作包可审阅，但不改变原实验验收。

主阻塞依任务pre-execution gate：A/B独立代码/权重使用许可尚无可核验授权；HDPS对应100Hz原流程/目标/有效片段未就绪；Transformer官方评估包不在现有合法资产中，完整paper训练数据非全部公开。Challenge2023标签在repo公开，未写成“隐藏标签”；Challenge2024的新增数据/目标schema/对齐revision未被假定可用。缺包可安装，但不能绕过这些主要条件。

同样不能用旧64Hz CNN/TRF、自选候选池、缩小公开cohort或随机初始化分数替代原协议。本轮无性能结果，无法完成误差/对照/seed不确定性分析；均如实NOT_ESTIMABLE，不作为阴性科学证据。

## Parent consistency validation

**PASS（范围一致；父任务仍ACTIVE）。**

- 保持PUB-01 public-only，patient/STN FORBIDDEN；计算和数据/权重留在2203，Windows仅源码文本、配置和轻量metadata报告。
- 未改S01/S02/SS01历史结论、frozen产物、旧CNN资料或Git历史；无数据集扩展、GPT计算、架构研发、SS03/S04或患者侧工作。
- SS02登记为WAITING_EXTERNAL / HOLD，Completion Record记录本轮与继续条件；不把该节点完成误当S03完成。没有模型冻结/ART-AUDREP-v1，SS03依赖门未打开。
- 遵循停止规则，本次审查包交付后停止。下一步仍归属SS02，需许可/资源条件明确后重新做模型smoke与实际复现，不能自动改起SS03。

## 审阅发现与处理

1. 初版盘点包含85人666条index记录；READY cohort只有83人659条。报告已分开，避免将全index称为有效训练集。
2. HDPS作者split是81人647对；全部有64Hz路径不等于100Hz就绪。候选数是subject-phase有效片段数，不是固定100。Top5原test入口未输出，需独立显式评价。
3. Transformer代码91 train subjects与论文180训练人数不同；现有仅71/91，14 held-out普通记录存在也不等于官方test样本可重建。
4. HDPS的val/test跨part不跨source-family；不改官方split，但撤销“严格source-family泛化”解释。2024的30s Mel提交轴方向在两页不同，记录待loader核验。
5. 当前主阻塞使模型smoke未运行。审计工具不是完整训练adapter，交付中明确说明，防止“源码可读”被当成“已复现”。
