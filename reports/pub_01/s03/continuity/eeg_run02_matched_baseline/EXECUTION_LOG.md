# SS06 run02 实际执行记录

2026-10-26用户新授权，执行补充SS06_MATCHED_BASELINE_EXECUTION.md；同SS06/PR#6，起点de15d3757a5bbb4b010fd5ebee3d99f7c2de7c9c，接入status为空。E约7.7MB/C约21GB，重型证据仅2203、必要轻量副本优先C，不清理原资料。

已向协调提出C（strong causal adaptation同监督预算）、O（native offline同监督预算）、P（完整两阶段额外参考）的最小矩阵；正式协议经协调核对后开训。run01固定split/statistics/score与已完成模型只读复用。全部新增结果POST-HOC INTERNAL EXPLORATORY，旧encoder见过全部audit，禁止加载。

2203新增run02，INTAKE与ENGINEERING_CHECK实际通过。249train/3262块、63val/980块及旧newtrain-only统计、XLSR19身份/覆盖核验；未读取audit信号。C/O20step及Ppre20step/head3step真实pilot完成，初始每阶段≤100step。因果/分块/梯度/Hann检查与精确账本见同名JSON。

协调已核对C/O/P矩阵并明确授权通过必要测试后冻结启动，无须等待额外回复。正式FORMAL_PROTOCOL.json已冻结：C/O3262step、1000/2000/3000/3262验证；Ppre639/head2035step各季度验证；三seed从头。C/O单次60s目标完全一致，O重叠计算另计；P额外声学预训练/重复监督单列。参数C3,128,137/O3,372,617/head257，C逐层pastKV191/conv63/head31，60s内无detach。

正式已启动：C1024 driverPID2170298，C2026 PID2170299，C31415 PID2170300，offline_reference PID2170301串行O三seed再P每seed两阶段。两L40共享：C1024/31415在cuda0、C2026与offline lane在cuda1。墙时仅实际资源记录，不作独占效率比较。formal_started.json和日志留2203。保持run01冻结，统一audit等待全部训练/选择/来源与覆盖检查完成。
