# PUB-01-S03_SS07 — 全量强基线上的连续性增量研究

| Field | Value |
|---|---|
| Subtask ID | PUB-01-S03_SS07 |
| Parent Node / Task | PUB-01-S03 / PUB-01 |
| Status | ACTIVE — 执行者接管；精确baseline协议核验后直接全量，增量矩阵另审阅 |
| Patient Data | FORBIDDEN |
| Upstream | SS02 与 SS06 已冻结；不改写其结果 |
| Inputs | 2203 上既有合法 SparrKULee 资产、原 Transformer/XLSR19 功能实现、原 fold4 清单及完整训练配置 |
| Deliverables | 原强基线全预算复跑、五类增量框架筛选、配对多 seed 验证、历史贡献与容量控制、可追溯结果及投稿证据缺口 |
| Acceptance | 有效的同协议性能增益与机制证据；阴性实验照实交付，不以跑通或最高单 seed 作为研究目标完成 |
| Validation | Technical / Acceptance / Parent consistency + 协调端独立复核 |
| Files Allowed to Change | 本任务书、SS07 新代码/config/tests/reports、父任务 amendment/登记/入口；旧结果只读 |
| Last Updated | 2026-11-01 |

## 1. 研究者决定与持续目标

用户明确：『我希望尽快推进全量实验。我的目标是找到超过原有baseline的方法框架。设定goal，允许调用2203的全部权限和算力。目标就是，超过基线，能发表ICML/NeuralPS的程度。』此处按 ICML/NeurIPS 研究目标执行。

这是 SS06 结束后的新研究授权，允许协调端在本公开 EEG 方法范围内安排实现、train/val 方法探索、完整训练、多 seed、冻结评估及纠偏，不再被此前 STOP、单 pass、小规模或只准原型的历史条目阻断。授权不等于承诺阳性或论文接收。最终 scientific acceptance 由研究者决定。

持续目标：在严格对照中找到可重复超过原强基线的 EEG-only 方法，解释连续信息的增益来源，并积累足以评估 ICML/NeurIPS 投稿的创新、泛化、消融、效率与可复现证据。不得把技术 PASS、数值偶然胜出或有偏测试分数当作目标完成。

## 2. 首要实验：保留强基线，再增加模块

主锚点是 SS02 已实际复现的 native Transformer-CLIP-EEG 两阶段方法，而非 SS06 的小 encoder 或因果适配 C。保留其原 EEG backbone、XLSR19 对比学习、native 3 s 时间权限、原读出与目标。所有改动逐项标记。

首先核对 SS02 原始 config/source/run，恢复完整训练预算及可用训练范围：历史清单为 train426 / val49 / test_seen69 / test_heldout75 records；这些数字只在逐项实际核验后采用。全量指既有合法可用训练集合，不将 val/test 加入训练，不凑原论文不可得 cohort。

原预训练上限100 epoch、validation patience15、AdamW 1e-3 / wd .01、StepLR10/gamma .1、batch128；历史24轮结束是早停结果，不应硬编码成所有新 seed 的固定24轮。原 head Adam1e-3 / wd0 / no clip、negative Pearson eps1e-6、batch64、manifest/time顺序、50% overlap、max250 epoch / patience10，验证选点。以冻结原配置/实际代码核验为准，差异先记录。

至少3个完整 seed（1024 / 2026 / 31415）重训原强基线；优先两GPU并行运行不同 seed。每个 seed 的完整预训练 encoder 可在同一固定 split 内共享给所有匹配增量条件，冻结来源；不能将见过某一留出组的旧 encoder 用作该组的独立泛化证据。

第一阶段以冻结完整预训练 encoder 的增量读出比较为主，所有条件保留同一个原 head、相同初始化和目标，增量支路可零初始化，使关闭模块时数值回到原模型。若增量需要额外训练、encoder微调或目标，必须有对应等预算基线/容量控制并单列。不能只训练附加模型更久再归因于连续性。

五类待实现/复核的增量条件（在原强模型上添加，不另换弱 backbone）：

1. 多时间尺度历史特征残差：短/中/长历史聚合，保留当前 native window 表征。
2. 持久快慢状态残差：跨窗口保留状态，记录边界 reset，明确状态衰减与实际有效历史长度。
3. 预测连续表征残差：合法 train-only teacher/预测目标；额外自监督成本与同成本非预测控制单列。
4. 历史条件的自适应对齐残差：只使用原 baseline 已有的当前时间权限及更早 EEG；增加最优静态对齐对照。
5. 潜状态预测与 EEG 创新校正残差：预测仅来自历史 EEG/模型输出，不输入真实历史包络；保留 prior-only 与当前 EEG-only 对照。

确切宽度、历史窗口、训练顺序和最大试验数由工程 pilot/文献审查后在 train/val 上冻结；不得以 audit/test 排名决定这些选择。第一轮筛选预先有界，最多五框架各4个配置，使用共同验证规则、样本量和预算，保留全部尝试及失败。不能无限向同一验证集试到阳性；新增研究轮次须协调端记录假说和选择历史。

## 3. 数据、时间权限和评估

原 fold4 为完整规模的首要可追溯对照，SS06 249条训练不再作为唯一训练范围。既有 test 和 SS06 audit 已有开发暴露，全部标记 POST-HOC / INTERNAL EXPLORATORY，不能重命名为独立确认。训练与调参代码不得读取这些测试分数；验证完成、模型/统计计划冻结后才进行既定一次评估。

并行只读核查所有现有公开资产的实验暴露记录，寻找真正未进入训练/模型选择的有效确认数据。仅靠重新划分旧数据不能消除历史暴露。若没有可信新确认集，继续全量 train/val 研究，不把全项目 HOLD；明确论文级外部证据缺口，需要扩大数据范围时向协调端提出具体最小方案，不自行下载新数据或请求受限资料。

归一化、PCA/teacher/预训练均遵守相应 train-only 范围。完整数据与旧数据的统计不得混用。source-family、story、record、subject 分组和重复刺激暴露单独报告，跨人不冒充同时跨故事。

保留原生3 s offline baseline 的全部当前窗口权限。历史模块只读取当时已经可用的 EEG 表征；按每个 latent 的实际原始输入支持检查，重叠窗口/双向 encoder 不能悄悄增加未来权限。真实历史/错配历史干预保持当前 EEG、当前 encoder 输出和评分坐标不变。不能只改60 s开始状态、再让模型吸收真实历史整段后，就声称充分检验历史无用。

主要 endpoint 沿用可比的包络重建：完整60 s段→被试→seen/heldout→原加权汇总；Pearson为主，校准/MSE为描述或独立目标。冻结逐seed、逐人、逐story/来源统计，matched baseline配对差值；显式处理多个候选选择和共享刺激依赖。检索/语义结论需对应实际训练目标与固定候选评估，不由包络r推导。

## 4. 全量推进步骤与资源

1. 实际盘点代码/数据/暴露、可用GPU与磁盘，形成 SS07 INTAKE 和精确 baseline matrix；先完成可执行基线入口/最小数值一致性检查。
2. 向协调端提供精确 full-budget baseline protocol。已核验一致的 baseline 可立即启动，不需要再次等待用户许可；遇科学协议差异先协调，其他独立工作继续。
3. 基线训练同时实现五类增量条件，完成关闭模块恒等、梯度、状态边界、历史可用时间与当前EEG固定干预检查；提交精确 pilot matrix。
4. 有界 train/val pilot 后冻结候选、完整预算与多seed比较。至少3seed；验证选定主候选后可扩到5seed用于稳定性，并给基线相同配对seed。所有新seed在看本轮test前确定。
5. 全训练/选择完成后一次规定评估；补容量匹配、同成本、平滑/输出历史prior、历史错配/置零/重置/距离与内容控制。收益不成立即如实报告，对下一轮假说只使用train/val证据。
6. 独立读取实际代码/配置/日志/逐人结果，复算关键指标和代表性原始输入预测，提交技术/科学限制与下一轮决策。

server2203:/home/fanyu/auditory_simulation_m6a 是唯一计算/数据执行根，新的运行使用 pub_01/s03_incremental_continuity_20261101/，旧目录不覆盖。用户授权使用该服务器全部可用GPU/CPU/RAM与所需项目运行权限；可并行填满空闲卡，不终止他人/其他项目作业，不更改全机安全或服务配置。当前盘点2×L40均空闲，文件系统仅约55 GB余量，必须实行新缓存/检查点空间预算：优先已有共享特征、必要best/latest保存、留运行余量；删除依用户规则进回收站或备份，不能永久清除历史资料。

Windows只保留轻量代码、JSON/CSV汇总与图表。复用既有SS06独立worktree/分支进行新SS07代码，使用独立文件命名与报告目录；保护主仓库CNN/TRF/stash。提交推送获授权，不自动合并PR#6。没有服务器Git、患者/STN、GPT重排或无关下游启动。

## 5. 交付与接受

交付目录：reports/pub_01/s03/incremental_continuity/run01/；至少含 INTAKE、冻结协议、全部试验账本、曲线、逐seed/逐人/逐来源指标、matched差值、历史与容量控制、实际命令/source、失败与限制、REVIEW 和投稿证据矩阵。

研究成功必须分别回答：是否稳健超过完整强基线；是否有超越容量/平滑/额外算力的历史贡献；是否有足够新方法与相关工作差异；是否有真正可接受的泛化确认。小幅单数据集包络提升本身不等于 ICML/NeurIPS 水平。此goal不因某轮负结果或工程PASS而标complete。

三级审阅：技术可复现；科学结论与证据相称；父任务公开/患者边界和原有冻结结果不变。仅就已授权公开EEG方法目标持续推进，不把本任务完成等同PUB-01全部完成或论文接受。

## 6. 执行记录

2026-11-01：协调端已创建持续goal；核对SS06结束、worktree干净、2×L40空闲。当前为授权后准备/启动阶段，尚未宣称任何SS07训练完成或性能提升。
