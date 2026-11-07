# 固定last19误差分解与三输出梯度诊断

归属PUB-01-S03_SS07，同一工作树；2026-11-02。此轮只执行协调已授权的固定诊断，不是新研究节点、训练、模型选择或可部署后处理。D12/D9旧结果和全部旧test/DTU/ds004408冻结。

## 固定输入与计数

原三seed、冻结FT+Ccap及全部九个last epoch19，不按分数选checkpoint。复用D12已核验的Data/Conditional及既有TRAIN426/VAL49缓存，不重编码。每arm/phase恰一次残差前向：9×(248620+30059)=2508111个有效窗；旧encoder/head0、optimizer0、学习模型fit0。逐块alpha与正交分量明确是新计算的描述性投影系数，计数另列，不能称零统计计算。

先对同覆盖/target定义冻结TRAIN/VAL60s mask，VAL与原752块/49人精确一致；TRAIN与既有mask对照。所有record、block、person保留，指标每块计算后块→人→phase宏平均，三seed另平均，不把平均alpha代入r式。原TRN/VAL元数据与源实际SHA/stat冻结，九last身份固定。

## 分解与非有限规则

以同一物理块covered点的anchor a、increment d、target y，u=a−mean(a)、v=d−mean(d)，alpha=dot(u,v)/dot(u,u)，e=v−alpha*u。全部方差/协方差population ddof0，记录mean(e)与orthogonality误差。

逐块验证
`r_new=((1+alpha)*r_anchor+cov(e,y)/(sd(a)*sd(y)))/sqrt((1+alpha)^2+var(e)/var(a))`。

MSE差拆bias=`2*mean(d)*(mean(a)−mean(y))+mean(d)^2`，parallel=`(alpha^2+2alpha)*var(a)−2alpha*cov(a,y)`，orthogonal=`var(e)−2cov(e,y)`；与实测差闭合。另记录r的anchor-normalization项与orthogonal-covariance项，二者逐块重构差后再平均。偏置不出现在Pearson中心化恒等式中；不据解析投影重构新的部署预测。

精确零方差/非有限显式undefined，不平滑、ridge或按结果删人；任一block的某指标undefined则其person及完整phase该指标保留undefined，其他独立指标继续。低方差阈值固定为`var<=64*float64_eps*mean_square`，仅标记，不排除正且有限的分母。TRAIN目标继承原float32窗口值并转float64（物理目标由原record cast f32再f64），VAL保持原连续float64目标，差异不混称同一标度。whole-covered/prefix/首重叠/tail/末single区域为辅助，不能取代完整60s支持。

## 三输出空间梯度

仅在TRAIN固定锚点、每seed一次。gM=`2*(a_window−y_f32)/(N_train_windows*192)`，为全TRAIN按window/sample等权的固定数据项；原fallback仍在分母但其可学习导数为0。原D12每global64/末38 batch使用相同primitive，但本量不是实际20轮SGD/Adam梯度或其平均。

gF、gR在完全同一块/covered target/person权重计算：每人权重1/P，每块该人权重1/B_s。gF=`2*(a_fused−y_f32)/(n*P*B_s)`；gR=`[r*u/dot(u,u)−yc/sqrt(dot(u,u)*dot(yc,yc))]/(P*B_s)`。

先在每原record的完整物理坐标累积所有块梯度（包含跨60s块的窗口贡献），再以`h[t]/weight[96i+t]`作Hann转置，最后同遮i>=9。gM保留完整192点监督，gF/gR对非评分tail/未覆盖点为0；零Hann端点和fallback差异单列。流式累norm²/dot再算三对cos，不能平均record cos。保留record/person分布及变换前物理gF/R参照；不物化全量梯度cache。

单块物理空间的`P_perp gF=(2||u||||yc||/n)gR`为正比例，将独立核验；总体角度差可来自校准方向、块尺度/权重、Hann转置/跨块与mask，不先认定MSE与Pearson反向，也不冒称参数梯度或换loss有效的因果证据。

## 冻结、检查、资源及交付

入口`pub_01_s03_ss07_error_decomposition_{core,prepare,run,queue,report}.py`；server ROOT=`/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_error_decomposition_20261102`。实际源码/输入身份/支持/公式/计数在首个真实残差前向前写PROTOCOL，合成投影恒等式、1+alpha负、constant/nonfinite标记、三梯度的随机/跨block/零Hann端点/tail/fallback有限差分通过后直接执行。协调独立autograd证据原位保留不覆盖。

按seed worker，至多两张compute PID确实为空且free>=4GiB的GPU，每worker RSS<8GiB/GPU allocated<2GiB；ROOT<128MiB、无历史/梯度磁盘数组，两盘保留20GiB。共享从协议冻结起30min deadline；队列异常/超界核PID/UID/start_ticks后有界停止自有worker，不影响他人，不自动重试/扩范围。预计RAM约5–6GiB、主CSV/汇总/图文<80MiB，属于预算估计，不是实测。

输出固定全量block/record/person/phase统计、三个seed梯度norm/dot/cos与record/person分布、恒等式闭合/标记计数、成本/必要静态图和完整中文报告；所有科学计算仅2203。完成后送协调独立审阅并报告实际诊断，不追加训练/超参、不重放older、不新建其他节点。
