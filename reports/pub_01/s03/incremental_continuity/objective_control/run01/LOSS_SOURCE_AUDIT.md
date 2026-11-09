# 只读来源核对：完整FT+Ccap条件器与原3s相关性目标

2026-11-02；PUB-01-S03_SS07。协调在D14固定九fit运行期间要求的并行只读方法核对。**在当前已登记SS07源码、协议和完整运行证据中，没有找到“同一冻结完整FT+Ccap、D12完整历史条件器/同容量current/P_prior三臂、原3s centered negative cosine eps1e−6”这一完整组合的既有试验。**原native head/FT和旧round01/warm使用过该相关性目标，但锚点、历史表示或结构不同，不能冒充该组合。此为登记证据范围内的缺口，不是全服务器绝对不存在的证明；也不构成换loss实验的必要性或执行批准。没有运行新NN/fit/optimizer。

## 精确来源与差异

| 路径/函数 | 实际目标与数值路径 | 与当前组合的差别 |
|---|---|---|
| `pub_01_s03_ss07_baseline.py:208` head.loss_fn | 每个192点窗先沿最后维中心化，`-F.cosine_similarity(pc,tc,dim=-1,eps=1e-6).mean()`；head/latent/target为float32，形状B×1×192 | 原预训练encoder的native卷积head，非完整FT+Ccap条件器；无固定Ccap锚点/前九fallback |
| `pub_01_s03_ss07_train.py:85` objective；`pub_01_s03_ss07_finetune.py:99` | FT直接import同objective，原raw输入float32、缓存目标float32、输出float32；EEG/head分别1e−5/1e−3，最多5轮，原global64顺序 | 更新完整encoder+native head；它产生强FT来源，但不训练冻结FT+Ccap之上的D12三臂 |
| `pub_01_s03_ss07_train.py:185`、`train_full_v2.py` | round01 negative cosine，某些family另有既定teacher辅助项；full v2只改finite检查实现 | 历史上下文先由`summary(z)=adaptive_avg_pool1d(z,6).flatten(1)`压为48维（output_prior为6维），width16的GRU/EMA；并非D12的完整8×192时序Conv→GRU32两token条件器 |
| `pub_01_s03_ss07_warm.py:213` | import原objective；frozen/joint × p/current/past，共18fit、各5轮；原global64，PCC选点 | 初始是原pretrained encoder+已训练native head（original_p构造），不是已完成强FT+Ccap；past仍summary48维M4位移系数，joint可更新encoder，无Ccap固定锚点 |
| `pub_01_s03_ss07_full_ft_history_core.py` Data/Conditional、`..._train.py:29` | normalized current/history float32；冻结anchor float64，p=a+0.1*res.double；目标cache float32→float64；完整B×192点`(pred-y).square().mean()` | D12正是当前锚点/完整时序H/C/P组合，但训练目标MSE，统一i≥9可学习，前九Ccap fallback仍在loss分母 |
| `pub_01_s03_ss07_order_control_train.py:39` | D14保持D12全部上述数值/目标与模型，只按预冻排列改变TRAIN顺序 | 仍然MSE，不是相关性目标对照；D14不得在运行中改loss |

D12实际PROTOCOL的loss字段为未加Hann权重的逐点192窗MSE。当前已有Ccap/条件岭读出与D9物理目标的解也不能充当该非线性三臂PCC试验。旧warm的“完整native latent作为当前输入”不等于“完整时间形状作为历史上下文”，两者在源码中应严格分开。

2026-11-02在2203仅读取NPY文件头确认：原`train_envelope.npy`为(252454,1,192)/float32；旧native latent与新完整FT latent均为(252454,8,192)/float32，新FT P为(252454,192)/float32。没有读取新信号载荷或做新前向。实际runtime为torch2.11.0+cu128；安装版本docstring与[PyTorch2.11官方cosine_similarity定义](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)一致：两个L2范数**分别**取max(norm,eps)，不是把范数乘积统一加eps；supports type promotion。源码显式eps1e−6不同于API默认1e−8。

## 若以后仅换loss，哪些量仍会改变

这部分为数学/方法界限，不是执行方案。保持D14模型、初始化、排列、Adam与预算，只把MSE改成原函数，可隔离一个明确的训练目标变化，但不能称“其他优化数学量都一样”。

1. **精度必须明说。** 将原objective直接用于D14的float64 p/y会按float64中心化与计算cosine；代数公式相同，但不是旧native的float32数值路径。若另将p/y cast float32，则又引入loss内量化/中心化舍入变化；不可静默宣称位级复用旧目标。预测/Hann/主评分精度可保持D14原样，不必改为float32。
2. **fallback贡献改变。** 同3834个i<9窗依然在每batch均值分母、对新参数导数为0，但其常数MSE会换成常数负cosine。不能借换loss排除fallback或重定义可学习支持。原head/FT没有这一固定fallback机制。
3. **常量目标定义不同。** MSE对常量目标仍给出有限幅值误差及梯度；负cosine通过eps给出数值定义，不等于严格Pearson已定义。若中心化后yc精确0，则该项loss/预测梯度0；float32均值舍入可能让原始常值窗得到微小非零中心化值，不能保证旧实现每个常量窗恰为0。不能事后删这些窗或把eps行为称为真实相关。主60s评分已有独立的非零方差规则，不受训练eps替代。
4. **均值/幅度约束改变。** 对非epsilon区域，centered cosine对加常数和正尺度不变，MSE则同时约束偏置、幅度和形状；去掉MSE会失去原逐点幅度锚定。eps区域的尺度性质又不同。已有固定Hann下相消/单窗退化风险并不由换PCC自动消除。
5. **方向、权重和梯度量级改变。** 设单窗u=p−mean(p)、v=y−mean(y)，B窗、T=192，两个范数都高于eps时：

```
g_MSE,p = 2*(p-y)/(B*T)
g_PCC,p = (r*u/(u·u) - v/(||u||*||v||))/B
||g_PCC,p|| = sqrt(1-r^2)/(B*||u||)
||g_MSE,p|| = 2*||p-y||/(B*T)
norm_ratio = sqrt(1-r^2)/(2*sd_population(p)*RMS(p-y))
g_residual = 0.1*g_p（两者共同链式因子）
```

该式为非eps区域解析关系，不将D13的60s块方差代替3s窗口量，也不给未经计算的实际倍率。低预测方差/低目标范数会触发不同eps分支，上式不能直接延伸。梯度经过相同非线性Jacobian和Adam的一二阶矩，保持lr=.001仍不等于相同参数步长；Adam默认eps及逐窗口方向/尺度变化使其不能视作纯整体loss重标。D13的物理负Pearson梯度是完整60s/person权重，与这里的3s等窗负cosine也不是同一目标。

## 能回答与不能回答

若以后经协调判断确有必要，三臂同目标可检查特定训练目标下H相对C的差异，C_match控制当前输入的匹配容量；P_prior是预测包络历史语义对照，少1008参数而非严格容量匹配。共同锚点/初始化/排列/支持保留时，比较才有可解释性。H优于其自身MSE旧版本只表明目标敏感性，不能分辨历史信息与额外容量；须看H对C/P及相同H权重near/older共同支持。

older同时改变recency与内容，不是纯内容随机干预；selected−1零残差时near=older属于结构相等，不能作历史依赖的阳性证据。当前VAL反复用于开发，哪怕以后出现增益也不是独立泛化确认；PCC是已知目标，更换它不是方法创新。本核对不根据D14部分VAL决定后续试验，不批准训练、不更改D14，等待完整矩阵及其独立验收后由协调判断科学必要性。
