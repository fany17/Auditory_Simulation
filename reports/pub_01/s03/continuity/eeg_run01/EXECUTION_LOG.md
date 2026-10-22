# SS06 eeg_run01 实际执行记录

2026-10-23，用户“启动吧”的授权，执行补充SS06_REAL_EEG_EXECUTION.md。接管worktree后status为空；本地和远程codex/pub-01-s03-ss06-continuity均55c87a65a3b90e7448ece2709cf1e038463b244e。旧结果、主目录CNN/TRF、stash、其他worktree不改。PR#6仅交付，不自行合并。

2203可访问（nercn），两L40初检空闲。新根/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/eeg_run01；仅复制代码/配置，数据/归一化/cache/权重均在2203，无下载或远端Git。

执行前明确：只从SS05原train23family按元数据与固定seed20261023划分5audit/5validation/13train。所有来源曾用于既往模型训练，结果最多称前瞻INTERNAL/EXPLORATORY audit，不是独立外部确认。共同lag固定8sample（既往train/validation依据，不读旧test选lag）；目标和EEG均新train重拟合。每条record首60s热身，之后完整60s一次loss/step，batch1，TBPTT detach保留数值；训练每record前两个连续60s评分块，评估全部完整块。所有边界与尾段规则在拟合前写入SPLIT_PROVENANCE.json。

## 已实际启动的阶段

- 新split先冻结；249train/63validation/41audit records，13/5/5family、70/45/36人。train3262、validation980、audit538个完整60s评分块。准备进程3526003完成312条train/val缓存，audit载荷未读。
- 真实pilot进程3528497已完成13候选各100更新步、seed1024，以及100步M3预训练。pilot每record只用首两个评分块；原代码存2203的pilot_source，保留为有限工程/选预算证据。M3旧pilot记录fit_records为249条eligible，实际仅访问100条；应以pilot_scope_correction区分，不能称预训练全覆盖。pilot没有复用到正式训练。
- 同模型初始化下原型38项/71子测试及最初3项适配测试通过；正式前另跑最终7项适配测试，包括CUDA64ch×3840输出/参数梯度/分块、M5观测辅助loss和初始状态一阶BPTT梯度、prior纯度/模式隔离、M4zero-lag缓存。全部PASS。TorchScript弃用与GRU权重非连续警告保留；不影响已测数值等价性，不称无警告。
- 实际train/val坐标和新统计核对通过；CPU/GPU/60s分块及梯度检查PASS。pilot采用默认TF32，正式在拟合前固定关闭TF32，精度差异已单列，不将pilot当正式成绩。

## 独立源码审阅关闭与正式冻结

协调会话只读核查了划分/缓存及源码，指出正式训练必须覆盖完整记录、状态control隔离、预训练实际访问记录、formal来源守卫、全训练完成后才能开audit及CUDA适配核验。均在formal freeze前修正。原始split成员和身份不改；其training_stream首两个块是pilot初案，FORMAL_PROTOCOL.stream_amendment明确取代正式训练部分。

正式完整遍历每条train记录全部评分块，每60s一次loss/step、块间detach保留状态、record间reset；一轮3262步，39个模型×seed同记录/block顺序。M3每seed正式预训练3511步，覆盖新train全部完整60s时间段（含起始段），每段fresh-state以满足teacher辅助接口；不是跨段流式自监督。frozen/finetune共享同seed正式checkpoint，额外样本/时间分开报告。

M1三组各四个独立活跃分支，short全dilation1、long全256、multi1/8/64/256，消除以失活参数伪装容量匹配的问题。M5实现同方程显式一阶BPTT避免大量细粒度autograd节点，已与原型参数/输入/初始状态及观测loss梯度核验；不支持高阶导，不改变训练方程。

FORMAL_PROTOCOL write-once冻结后，调度PID3535778通过input/pub_01_s03_ss06_run.py实际启动。三seed1024/2026/31415；每1000步及最终3262步完整val，仅val选checkpoint；CPU六个M5任务并行、其余两GPU任务队列。全部训练/预训练的预算、source/split/norm、实际样本、预训练checkpoint和最优val选择均须核验，之后输出先验系数仅val选择，最后才允许原子开audit。当前未宣称audit性能或验收完成。

## 最终实际结果与审阅

39监督和3SSL全部跑满，all_training_complete及独立post-training gate核验PASS后才冻结EMA/AR选择、写ready；waiter3545290随后原子写audit_opened并读取41条新audit。每selected模型一次assessment含全部预定条件，39assessment完成后observer重算243行结果。五主框架r：M1multi.122913、M2.128164、M3finetune.123747、M4adaptive.113400、M5full.114178；GRU.125221、local.107432。八primary仅M1multi−long、M3ft−frozen、M4adaptive−uniformfixed的交叉分组/八项校正区间为正；其余跨0，不把M3ft胜frozen当预训练胜GRU，也不称M2稳健最高或M5必要。五主框架真实−错配state的secondary crossed95%均跨0。

后置逐段→record→person重算误差0；原始64Hz输入资产重新标准化+原型CPU222个有效r max差1.55e-7、234个MSE max差6.52e-8、12未定义相关一致。协调会话另外在2203从per_record_all_seeds独立按seed/person重算全部SUMMARY，max差5.55e-17，核对全部primary/history/开封门与统计代码。这不是外部完整科学签署，原始输入CPU复算由执行者另作。

四张静态图已在2203由真实review CSV生成并逐张可视检查；只修正了prior未定义r注释位置，不改科学数值。REPORT/REVIEW、全部逐seed/逐record/逐人表及冻结/运行/验证证据完成。执行/冻结source提交ea5a1b5b50293f36a4d55ab602f39ba4f096140e，最终observer/报告/状态提交见PR#6。

元数据打包首次误用系统python3缺NumPy，切回已指定env01后成功，无科学管线/结果改变。回传完整轻量包时E盘0空闲导致SCP部分写入失败；仅将本轮未提交partial evidence移到C盘任务备份，未永久删除。完整delivery01另保留2203与C:/Users/fanyu/.codex/backups/Auditory_Simulation_ss06_20261023/delivery01；Git精简逐run JSON只去重复访问序列（保留路径/身份/真实覆盖核验），所有指标/四图与source/config不变。完整包与精简包分别技术校验，信号/权重/normalizer/cache未回传。E盘接近满的收尾不涉及原始资料、旧worktree或主目录CNN/TRF/stash清理。

Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS。SS06完成并STOP，父S03/PUB仍ACTIVE；#6保持Draft不自行合并，不启动SS04/S04/父S05/GPT/artifact。
