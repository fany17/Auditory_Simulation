# D15：唯一训练目标对照，原公式float64适配

2026-11-02；PUB-01-S03_SS07同一节点。D14已封存648ee1c，固定顺序对照修正共同TRAIN退化，但未建立完整历史收益；协调依据完整九fit与实际独立review批准本具体目标对照的方案/源/工程准备。**正式训练仍须协调绑定实际冻结协议/源的独立执行审阅，当前不是新训练已批准启动的声明。**不向用户重复索取已有目标授权。

## 唯一科学改变

沿用D14三个seed1024/2026/31415、H_full/C_match/P_prior、同冻结完整FT+Ccap、同165792/165792/164784参数条件器、同component RNG初始化、同60个逐epoch排列digest、同Adam.001/weight_decay0、20epoch、global64/尾38、全部252454窗含3834 fallback与原TRAIN/VAL支持；只将训练windowMSE换为：

```python
pc = pred - pred.mean(-1, keepdim=True)
yc = y - y.mean(-1, keepdim=True)
loss = -F.cosine_similarity(pc, yc, dim=-1, eps=1e-6).mean()
```

直接使用现有float64 p/y，**不cast float32**；是原native公式的float64适配，不称原float32路径位级相同。模型/特征仍float32，p=a_float64+0.1*res.double，y为原float32目标cache转float64。不是新loss、混合loss或方法创新；不搜索LR/eps/幅度惩罚。D14原MSE/物理/VAL统计精确复用作为参照，暂不新增旧D14 TRAIN NN回放补3s loss。

history仍按原record-local i−5/i−3，older i−9/i−7，不读真实过去Y；shuffle batch邻接不成为历史。i<9返回原Ccap，loss均值分母保留全部fallback。常量目标不删除、不重加权；在当前dtype中心化后精确0时loss/预测导数0，近零范数由两个范数各自max(norm,1e−6)处理。这是有限的数值目标，不等于严格Pearson已定义。准备阶段只描述全原TRAIN/VAL窗口的raw-constant/centered-zero/norm≤eps计数，不生成新norm或学习统计。

换目标会改变偏置/尺度约束、逐窗统计权重、输出梯度方向和量级；Adam同LR不代表相同参数步长。非eps区输出梯度范数为sqrt(1−r²)/(B||p−mean(p)||)，与MSE不同。沿用既有优化器是受控比较，不借工程小批梯度倍率选择新LR。

## 固定评价与输出幅度诊断

VAL0…19与incumbent−1仍按原完整60s/person r，strict >和最早tie；每fit selected完整VAL一次，H selected older一次。H对anchor/C/P的三seed方向及近/远共同支持如实报告，零残差选点的相等不作历史阳性。保持原752块/49人及逐人记录，主指标不改为3s training loss。

TRAIN在−1/0/4/9/19以同一次预测记录：新的未加权window negative cosine、原windowMSE、6104块/70人physical60s r/MSE、常量/eps计数；−1直接锚点0NN，四个非零检查点各一次残差预测。另记录全TRAIN窗口prediction mean/RMS/maxabs、各record/fused maxabs及unweighted overlapMSE/预测小范数计数。VAL保留原score的窗口r/MSE、overlap/disagreement、边界/tail、window/fused extrema。所有诊断不参与选点、校准或新增loss；输出无幅度约束的变化不冒充收益。每epoch在线negative cosine和windowMSE轨迹与固定参数离线值分开。

## 精确正式成本及工程

| 用途 | 残差NN窗口 |
|---|---:|
| 九fit×20epoch TRAIN更新 | 44,751,600 |
| 每epoch VAL | 5,410,620 |
| 四个非零离线TRAIN点 | 8,950,320 |
| selected VAL | 270,531 |
| H selected older VAL | 90,177 |
| 正式总计 | **59,473,248** |

每C/P为6,598,119，H6,628,178；每fit78900更新，总710100；9新fit、新encoder/head前向0。没有旧last参考额外NN。工程另计上限4096残差窗、0optimizer/fit，预期288窗（固定metadata指定64混record窗中48有效×fresh/nonzero×三臂），实际计账。target-window描述统计另列282954窗，不声称零统计计算。

冻结前核实际D14 protocol/source/缓存身份，并绑定D14已验收指标与新source/PLAN。必要工程：独立NumPy centered-dot/分别norm floor与实际float64 PyTorch值一致；正尺度/平移/符号、精确零中心化及非二进制常量舍入/eps有限性；真实TRAIN固定小批的fresh锚点精确等价、非零D14权重下新目标loss/输出导数有限差分/全参数梯度finite，报告同可学习mask的PCC/MSE梯度范数比与cos但不据此调参。借用D14非零权重仅工程，正式全部干净初始化，不保存工程模型。不重复已冻结且未改变的全量history/gather验证。

## 并发执行资源（独立于科学矩阵）

server新ROOT `/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_objective_control_20261102`；本地轻量输出 `reports/pub_01/s03/incremental_continuity/objective_control/run01`。源码 `pub_01_s03_ss07_objective_control_{core,prepare,train,queue}.py`。所有数据/权重/计算仅2203；无信号/张量/权重回传，不建远端Git。

按协调对实际CPU/RAM/两张L40空闲盘点的执行授权，最多九独立进程并发，cuda0≤5/cuda1≤4，每worker torch2/BLAS4。每次launch要求GPU free≥4096MiB、MemAvailable≥16GiB；该卡所有compute PID必须属于本轮已登记且UID/start_ticks/完整脚本与seed/arm/device argv实际一致的worker。有陌生PID则不在该卡追加任务，等待可用slot，不终止他人；不改MPS/全局GPU设置或batch大小。

queue按PID/job管理，保留总6h deadline；每worker原RSS8GiB/GPU allocated2GiB上限，新root160MiB/两盘各20GiB（实际保守估计134376176B略超旧128MiB，协调仅调存储上限，原估计保留并绑定STORAGE_DECISION），无新预测/历史/梯度磁盘数组。保存best/last checkpoint仅2203和结构化统计。失败即停止本轮已核身份自有workers，保留错误/source，不自动扩预算/retry。资源忙只等待空slot，不以成绩调整算力。并发更快的queuewall不称方法算力减少，总workerCPU/前向/更新单独报告。

并发改动以无训练mock校验5+4九slot、满slot/陌生PID/身份复用/错误device/低RAM或VRAM的拒绝；实测launch资源和UID/start_ticks逐条记录。scheduler检查0NN/fit/optimizer。实际工程与源冻结后交协调一次独立审阅，再正式九fit。

## 三级验收与停止边界

Technical核完整计数、loss dtype/常量语义、全排列/选点、离线聚合、幅度/overlap与独立数值；Acceptance回答这一个目标敏感性问题，不把已知PCC称创新，阴性合法；Parent consistency保持公开数据边界/原split与历史冻结。旧test/DTU/ds004408/D12/D13/D14不重开、不用其分数调候选；无患者，无额外架构/混合loss/LR搜索或PR6合并。后续科学决定仍由协调根据完整结果界定，不自动扩展。
