# 完整FT历史条件候选：工程完成、正式矩阵接续

2026-11-02。协调已批准唯一结构，原TRAIN/VAL九fit×20epoch、710100更新，最多两张空闲GPU并行，整体6h。详细规则见[CANDIDATE_PLAN](CANDIDATE_PLAN.md)、[协调决定](COORDINATOR_DECISION.md)、[实际冻结协议](PROTOCOL.json)。本轮不是已完成科学结果；SS07/持续goal ACTIVE。

## 正式前必要核验

[工程检查](ENGINEERING_CHECK.json)实际通过：H/current各165792、prior164784参数，三seed全部gradient reachable且第二步每个参数梯度均非零；GRU_hh3072坐标均非零。H/C全state逐tensor相同，prior首层之外共同模块逐tensor相同。三seed实际初始完整VAL逐值等于FP64 Ccap，全部原块r和被试宏r桥差0；当前不变/历史不同/h和输出改变在非零临时权重上核验。严格同record融合输入并集边界、完整752块和选点逻辑通过。

成功工程另计18次临时optimizer更新、92046个新残差NN窗口前向，encoder/head前向0；wall47.613秒、CPU124.276秒、RSS5892972KiB、GPU allocated44781568B。临时权重全部仅RAM且弃用，正式重新构建干净共同初始模型。

首次工程因初始predict把H留在eval而触发cuDNN GRU backward拒绝，正式fit0/临时optimizer更新0。已保留[attempt01失败记录](engineering_attempt01/FAILURE_RECORD.json)、原source/protocol/log；执行路径可追溯的临时残差前向30169窗，wall/CPU未测不编造。修正只恢复临时train状态，并补输入流式SHA/GPU空闲与自有worker超时清理，不改科学矩阵，不把它记作性能失败。

## 执行契约

原三seed全部TRAIN/VAL Z/P、共同窗口target及46个record target在准备阶段核流式SHA并检查stat稳定；worker在启动时核全部实际输入size/mtime及小型冻结资产/source SHA。只使用已有合法TRAIN/VAL cache，无raw重编码、旧test/DTU/ds004408入口、新外部权重或患者资料。

训练网络/参数FP32；冻结anchor、原float32目标的FP64表示、残差scale/add/MSE和最终Hann均FP64，不按成绩选precision。前9窗Ccap fallback；完整原VAL752块不删除，固定warm辅助、逐人逐块、窗口/overlap/极值和边界tail随既有selected预测数组记录。最终selected重放及H的older干预前向事先纳入成本，不为诊断另重编码。

queue启动前检查GPU compute PID为空且free>=4GiB，至多每卡一worker，不停止他人进程。异常/总6h超时按自有Popen PID/UID/start_ticks/argv核身份后有界终止自有worker并保留失败，无自动重试或追加预算。正式模型按20轮最早VAL60s/person最佳（含incumbent−1）选择，全部跑满；任何终点/支持或源发生变化需留痕，不能按中途成绩修改。


## 正式启动快照

code commit e62a70b90f726ed029f86378f494a8caf116a5b6；PROTOCOL SHA b8e3552f33557be86609aa244c6d837ef63d9d765f518b5e3df58b46d89289f3。queue PID1585992/UID1012/start754213642，首H1024 PID1586000/cuda0/start754213754，C1024 PID1586008/cuda1/start754213761；prelaunch compute PID均空。正式20轮全部执行和结果/独立review仍待，不在启动处完成goal。源/科学协议不再修改。
