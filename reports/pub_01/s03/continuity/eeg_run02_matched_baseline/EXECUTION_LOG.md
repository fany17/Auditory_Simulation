# SS06 run02 实际执行记录

2026-10-26用户新授权，执行补充SS06_MATCHED_BASELINE_EXECUTION.md；同SS06/PR#6，起点de15d3757a5bbb4b010fd5ebee3d99f7c2de7c9c，接入status为空。E约7.7MB/C约21GB，重型证据仅2203、必要轻量副本优先C，不清理原资料。

已向协调提出C（strong causal adaptation同监督预算）、O（native offline同监督预算）、P（完整两阶段额外参考）的最小矩阵；正式协议经协调核对后开训。run01固定split/statistics/score与已完成模型只读复用。全部新增结果POST-HOC INTERNAL EXPLORATORY，旧encoder见过全部audit，禁止加载。

2203新增run02，INTAKE与ENGINEERING_CHECK实际通过。249train/3262块、63val/980块及旧newtrain-only统计、XLSR19身份/覆盖核验；未读取audit信号。C/O20step及Ppre20step/head3step真实pilot完成，初始每阶段≤100step。因果/分块/梯度/Hann检查与精确账本见同名JSON。

协调已核对C/O/P矩阵并明确授权通过必要测试后冻结启动，无须等待额外回复。正式FORMAL_PROTOCOL.json已冻结：C/O3262step、1000/2000/3000/3262验证；Ppre639/head2035step各季度验证；三seed从头。C/O单次60s目标完全一致，O重叠计算另计；P额外声学预训练/重复监督单列。参数C3,128,137/O3,372,617/head257，C逐层pastKV191/conv63/head31，60s内无detach。

正式已启动：C1024 driverPID2170298，C2026 PID2170299，C31415 PID2170300，offline_reference PID2170301串行O三seed再P每seed两阶段。两L40共享：C1024/31415在cuda0、C2026与offline lane在cuda1。墙时仅实际资源记录，不作独占效率比较。formal_started.json和日志留2203。保持run01冻结，统一audit等待全部训练/选择/来源与覆盖检查完成。

协调原方法审查指出P不应沿用C/O优化器：已核实SS02原pretrain无clip，head=Adam/无wd无clip/无shuffle。P尚0 formal更新时冻结P_NATIVE_AMENDMENT→_02、新独立native入口及实际native03 pilot；原pilot与所有源/日志保留（旧native01字节hash匹配恢复）。不改C/O、不改预算、不依据成绩。仅终止旧offline driver父、保留当时O子进程；新driver2179322完成/跳过O后用nativeP入口三seed。具体PROTOCOL_NOTES.md。source5b93ca5已push，P新source另提交。C/O和本轮统一audit仍按原协议；新增observer只做复算，不改模型/选择。

正式全部12阶段完成：C/O各3×3262，Ppre3×639/head3×2035；正确native P修正source提交55ddd40，统一门/observer/图source82f3001。全部训练、val选点、精确顺序/覆盖与checkpoint身份检查后才统一assessment。9个模型seed均538块、41记录/36人，无undefined-r；C/O/P均值.123014/.151762/.141466，C六项99.167%校正交叉区间全跨0。没有新条件、延长或挑seed。

executor独立后置observer聚合差0、18个发布方原预处理EEG CPU同条件复算r max3.3624e-8/MSE2.3583e-7（C改127microchunk）；协调另在2203从369行record标准库复算SUMMARY max2.7756e-17及六配对均值max6.9389e-18，核实预算/源/时序，不称协调重跑CPU。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS_SS06_ONLY。

三图视觉检查通过。约1.21MB delivery01仅允许CSV/JSON/compact访问元数据/静态汇总图，回传C轻量备份及Git报告目录；无数组/信号/权重/normalizer回传。完整访问序列、latents/cache/权重留2203。中文REPORT/REVIEW与当前登记更新，最终commit/push同#6 Draft后STOP；不启动下游。
