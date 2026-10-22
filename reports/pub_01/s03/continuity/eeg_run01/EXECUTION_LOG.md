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
