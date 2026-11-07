# run02协议执行注记

## P原方法一致性纠偏（正式P开始前）

协调复核发现首版Protocol将C/O共同AdamW/clip1及随机顺序不必要地扩展到P。已亲自核验本仓SS02`pub_01_s03_transformer_envelope.py:123`与连续offset训练loop：原读出Adam(lr.001)、无weight decay/clip、shuffle=false；`pub_01_s03_transformer_train.py:131`原预训练AdamW(.001,wd.01)、无clip。

C/O冻结源和在跑任务完全不改。P尚0 formal更新时，新增独立`pub_01_s03_ss06_matched_native_p.py`，以`P_NATIVE_AMENDMENT.json`→`P_NATIVE_AMENDMENT_02.json`绑定source SHA并覆盖原Protocol中的P optimizer/order。有效规则：Ppre AdamW原参数无clip；Phead Adam无wd/clip、按新train manifest记录/时间窗顺序batch64；centered cosine eps1e-6对齐原pcc，在构建encoder后重新按本seed初始化head。预算/窗/固定验证点/最高val与earliest tie不变。

旧pilot01/native02不复用并原样保留，最新native03真实pre20/head3步已通过；初始P工程累计pre60/head9≤100。旧native01源在2203 source_history/native_p_amendment01.py恢复并与原amendment hash一致。不存在“受影响P正式结果中择优”；新Pformal均从头，head仅用同seed本轮正确预训练选点。此次修正来自原协议审阅，未看本轮audit或P正式成绩，不延预算。

旧offline driver父2170301单独终止，正在跑的O子进程保留；新driver2179322等待/跳过已完成O，使用正确native入口续P三seed。全部旧日志、启动清单、修订原因留2203；旧formal_started列出的未来P命令已被native_lane_started取代，不代表实际执行。

## 坐标与因果性适用范围

`rawEEG[s+8]`中的raw只指“相对本轮cache之前的EEG”，实际源为发布方64Hz预处理derivative，并非未经处理的硬件原始信号。C因果性只保证该既有derivative的sample坐标中不使用target+8之后的EEG。上游发布方预处理是否具有未来支持、物理同步均未在本轮证明；不能据此宣称端到端在线或硬件因果系统。O/P保留整3s未来信息，C/O综合差异不能解释为纯lookahead效应。

O/P共同评分每原60s块独立39个3s重叠窗；Hann两端零权重使用唯一窗预测补全，538块/全部目标保留。这是明确的共同评分边界适配。P仍只有单轮预训练+读出，不能据阴性否定对比预训练，也不是SS02充分训练或原文上限。
