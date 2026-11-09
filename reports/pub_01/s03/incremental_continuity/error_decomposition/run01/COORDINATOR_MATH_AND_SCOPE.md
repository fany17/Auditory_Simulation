# 固定模型误差分解：协调定义与数学边界

2026-11-02。上一目标轮完成完整 FT 条件器的九 fit、实际独立回放、报告和远程提交37271a0，是实质进展。本轮保持整个科研目标；新增的共同退化证据需要解释，不能当作“历史普遍无用”，也不凭此直接改 loss 训练。

## 已授权的具体新工作

仅原 TRAIN426/VAL49、seed1024/2026/31415，冻结 FT+Ccap 和 D12 全部九个 last epoch19。统一最后轮与分数无关，不另选较好检查点。不用 selected 零头代替训练后模型。每个模型/phase 只一次有效残差网络回放，计划9×(248620+30059)=2508111窗；无新 encoder/head 前向、optimizer 更新或学习模型 fit。解析诊断投影系数确实要计算，不能以“无学习 fit”冒充零统计处理。

所有数值/信号/模型计算只在 server2203 的新目录 `/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_error_decomposition_20261102`；本地仅代码、统计与报告。旧D12/D9、旧test/DTU/ds004408不重开、不改动。最多两张确实空闲GPU、各RSS8GiB/GPU2GiB，总墙钟30分钟，新目录128MiB、无新历史/梯度磁盘数组、两盘20GiB余量。具体输入、支持和源先冻结，修改所需数学与梯度检查通过后可直接执行，不重复等待已有授权。

## 1. 逐块分解，不把均值代回非线性公式

在同一有效60秒物理块内，冻结锚点为a，新预测为p=a+d，目标为y；所有均值、方差、协方差统一使用分母n。令

```
u = a - mean(a)
v = d - mean(d)
alpha = dot(u,v) / dot(u,u)
e = v - alpha*u
```

于是mean(e)=0且dot(u,e)=0。记Va=var(a)、Ve=var(e)、Vy=var(y)，Cay=cov(a,y)、Cey=cov(e,y)。预测新方差为`(1+alpha)^2*Va + Ve`。当相关所需方差非零时，精确得到

```
r_new = ((1+alpha)*r_anchor + Cey/sqrt(Va*Vy))
        / sqrt((1+alpha)^2 + Ve/Va)
```

alpha可正可负；不能默认1+alpha正而隐去符号反转。Ve=0时仍可使用该式，不需要强行给e的独立相关赋值。

MSE差精确分为

```
offset = 2*mean(d)*(mean(a)-mean(y)) + mean(d)^2
along  = (alpha^2+2*alpha)*Va - 2*alpha*Cay
orth   = Ve - 2*Cey
MSE_new - MSE_anchor = offset + along + orth
```

这三项只是描述性代数分解。alpha由预测自身计算，不是训练得到的部署适配参数；协方差项使用真实目标，仅可作事后解释。不能把按块依赖未来/目标的分析操作包装成新推理模型。只在每块完成重构检查后按record/person汇总；先平均alpha、Ve等再代回r公式通常不等于平均r。

精确零方差、非有限或低方差需要完整标记；不按成绩删除记录、静默加epsilon、裁alpha或重定义主支持。TRAIN支持须在新预测前按覆盖和目标冻结；VAL保持旧752块/49人。TRAIN目标继承已有float32再转float64，VAL保持原连续float64；区别明列。

## 2. 三种输出空间梯度

gM：在固定锚点处，全TRAIN按窗口/点等权的MSE数据项，`2*(a_window-y_window)/(N*192)`，i<9置0。其点态primitive与D12每batch一致；它不等于20轮动态参数、Adam状态及最后短batch组成的实际优化轨迹，也不冒称参数空间梯度。

gF：同一固定TRAIN60秒有效块/person宏权重下的融合MSE。单块物理梯度为`2*(a-y)/n`，乘共同person/block权重，经完整Hann归一化算子H的转置映射回window输出，最后遮i<9。

gR：在与gF完全相同块、目标、person权重下的负Pearson。对单块中心化u、yc，

```
gR_physical = -yc/(norm(u)*norm(yc)) + r_anchor*u/dot(u,u)
```

同样乘权重、映射H转置、遮i<9。完整记录中跨60秒边界的窗口必须累积来自相邻块的梯度，不能在块边界重建独立Hann。未评分tail对gF/gR为0，但gM仍覆盖合法窗口点；这个差异必须保留。

流式计算三对norm/dot/cos及record分布，不物化全量梯度缓存。gF与gR控制了融合、支持与person计权；仅比较gM/gR会同时改变这些条件，不能归为纯loss效应。若另列固定参数下按实际64/尾38 batch拼接的梯度，也只能作为计权描述，不称重现历史SGD轨迹。

## 3. 单块角向分量的恒等式

令P_perp投影掉常数向量和当前中心化预测u方向。直接代入上述公式，有

```
P_perp gF_physical = (2*norm(u)*norm(yc)/n) * gR_physical
```

非零方差下比例为正。故相同物理块上的MSE角向分量与负Pearson方向一致，不能预设存在固有反向冲突。MSE还有均值/尺度分量；不同块的比例、Hann转置、固定mask、样本权重与模型Jacobian会改变总体输出/参数空间关系。单块恒等式和全局实测cos回答不同问题，均不证明换成Pearson必然改善泛化。

这部分是协调的直接代数推导，执行端应以小合成恒等式与Hann-adjoint有限差分检验实现。原D12的MSE改善/r下降只提供诊断动机；在新的实际分解和梯度证据前，不先写具体失败原因。

## 4. 一手文献与新颖性边界

- [ICASSP 2023 Task 2官方说明](https://exporl.github.io/auditory-eeg-challenge-2023/task2/description/)明确用固定包络、60秒相关及被试聚合。本项目已沿用其内部开发口径；这一指标定义不允许事后以较好MSE替换r。此处未重新打开测试数据或提交榜单。
- [Ciccarelli等，2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687829/)已有EEG语音重建线性/神经网络比较，并采用Pearson损失对齐训练；直接采用相关损失不是新原理。本次检索获得其正文及相关方法摘录，不将两说话人AAD性能搬来与当前单目标开发r比较。
- [Liang等，2026，§2.3–3.3](https://arxiv.org/html/2601.20542v1)的contrastive PCC同时使用attended/unattended目标，回答两说话人区分问题。当前单目标包络没有相同的竞争输入条件；不能把随机错配包络默认为对应真实unattended，也不能把其多数据集结果当本项目复现。
- [Kolbæk等，作者预印本摘要](https://arxiv.org/abs/1806.08404)讨论特定条件下语音增强MSE与envelope correlation的关系。仅核摘要，不把该领域/条件下的结论直接外推到本EEG、有限网络和person宏指标；上面的逐块恒等式来自这里明确给出的推导。

内置web工具本轮连接失败，已改用可用的搜索连接器获取以上一手来源。未下载新EEG/音频/模型或绕过访问限制；IEEE全文未通过访问验证，未作为已读全文证据。

## 本轮交付和决定边界

交付完整预定TRAIN/VAL分解、三个梯度比较、所有未定义标记、成本和实际独立审阅。原强锚点、同容量current及输出prior都保留。诊断本身不是新增性能结果、连续性机制证明或ICML/NeurIPS贡献。

待实测之后再判断是需要处理目标/计权、非线性泛化、表征或其他已识别缺口。不得依据这份尚无实测结果的说明自动增加模型、超参数或重开外部数据；科学goal保持ACTIVE。
