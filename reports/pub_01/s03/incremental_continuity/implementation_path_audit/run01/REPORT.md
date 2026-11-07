# 连续适配的实现路径审计：已覆盖什么、尚缺什么、最小直接比较

**只读结论：早期warm已做端到端历史训练，不能再把它列为未测；D12–D16则确实只在冻结FT缓存上学习。当前可核实的结构缺口，是原encoder最后64→8投影之前的192个64维时间token、query-dependent跨窗memory与当前encoder联合适配的组合。**本文件只是源码可行性反馈，未实现/冻结新方法，无新数组读取、模型构造、神经前向/反传、拟合或权重下载，不分配D18编号，不修改既有结果。

## 1. 实际输入与学习路径

源码位置与函数行号集中在[SOURCE_INDEX](SOURCE_INDEX.csv)。本地审计起点为`fe70a051548933e8824926f17daa51d76baaf552`；唯一额外2203读取为既有Python源及冻结config JSON文本，没有导入执行这些模块。

| 已执行路径 | 真实输入/学习对象 | 历史任务能否更新原encoder/attention | 不能扩大解释的边界 |
|---|---|---|---|
| native预训练/原head、P_FT | RawWindows为B×192×64 EEG，64Hz、hop96；原EEG backbone输出B×192×8，head为Conv1d(8→1,k32)+LeakyReLU | `finetune.Decoder.forward/main`把encoder与head都放进Adam；原FT阶段确实更新 | 当前窗口仍整3秒双向offline支持，没有跨窗K/V memory |
| round01五框架 | 冻结预训练encoder的8×192缓存；过去adaptive pooling为8×6=48，H16/32状态；head与模块共同从随机head起点学 | 原encoder不在该优化路径 | M4已有8通道特征变换，不能说此前全是“输出后处理”；但未保留preprojection64 token |
| warm round02：frozen/joint×P/current/past，18fit全部实际完成 | `WarmDecoder`加载训练完成P；joint从raw重算，M4仍在8通道latent后用H16历史调4/8/16位移 | **joint有**：`Data.batch`调用`model.encoder(x)`，`optimizer`含encoder，历史在同global64内可回传；更新后`state=detach(state)`，跨更新历史图被截断 | 过去仍48维摘要；跨更新状态来自旧参数时刻，非重新编码的可训历史bank；joint还改变dropout/train模式，不能当纯梯度因素 |
| D8强FT+Ccap | `encode_phase @no_grad/eval`将已选FT的Z=(N,8,192)、P=(N,192)写缓存；Ccap用当前Z、P和Z²的冻结TRAIN变换/ridge | 缓存生成后没有新的encoder梯度 | “完整FT基线”指上游曾完整监督FT，不等于此后每个历史候选都联合训练原模型 |
| D12、D14、D15、D16 | `Data`读D8 Z/P；H的每个8×192过去窗经Conv/Flatten/3072→32，全窗变32 token，再两步GRU；当前1536→32，经FiLM/32→192 residual加到冻结Ccap | **没有**：训练只构造`Conditional`，Adam只含其参数，原encoder/head未加载到图中；D16只加LN | 原始历史完整1536虽进入支路，最终仍过32维全窗瓶颈；D13/D17是诊断，无新训练 |

### 原encoder的具体切入点

`pub_01_s03_method_models.py::AuditoryTransformer.forward`：spatial64→width64，10层`TimeNormConv`与`AttentionBlock`后，`state.transpose(1,2)`是B×192×64；最后`self.output=Linear(64,8)`才得到B×192×8。已读冻结config为depth10/width64/heads8/latent8。原attention在每个192点窗口内双向计算，并无跨record/窗口memory。

`TimeNormConv.norm=LayerNorm((64,192))`含固定时间维及对应参数。**不能直接把3s raw片段拼成9s再送原forward，并称结构/初始化不变**；那会遇到归一化shape/参数及原窗口支持改变。最小改动应继续逐3s编码，在最后projection前接memory，不改变原10层和head的shape。

### warm结果必须保留

[实际run02报告](../../run02/REPORT.md)不是仅有草案：18fit×5epoch/355050更新已完成并验收。joint P/current/past的VAL60s均值为.174139935/.174100032/.173740247；past−current三个seed全负，均值−.000360。冻结past对current只有1/3正。因而本候选不能再包装成“第一次让encoder受历史监督”，也不能用“加联合FT”重复warm M4。可询问的新组合仅是**从已选强FT起点、projection前64维逐时间token的query-dependent内容检索**；它仍是待检适配，不是创新结论。

## 2. 一个最小可实现的segment-memory候选（未冻结/未运行）

优先评估一个版本：当前encoder为可训Eθ，历史为初始化于同seed已选强FT的冻结E0 copy，eval/no_grad；两者都取projection前B×192×64，不先压成32或48维全窗摘要。history copy的选择是单独设计决定，不能与共享可训历史encoder混称同方案。

仅在`AuditoryTransformer.forward`返回最终`self.output(...)`之前拆出原循环为`preprojection(x)`，保留同一运算、参数、输入layout和batch。概念接口如下，**非已实现或已通过数值桥的代码**：

```python
Hq = E_theta.preprojection(current_raw)          # B,192,64
with no_grad():
    Hm = E0.preprojection(past_raw_flat)         # (B*2),192,64
KV = Hm.reshape(B, 384, 64)                     # chronological two segments
delta = memory_attention(query=Hq, key=KV, value=KV)
Z = E_theta.output(Hq + delta)                  # original Linear64->8
P = head_theta(Z.transpose(1, 2))               # original pretrained head
pred = fixed_Ccap_map(Z, P, local_index)         # differentiable, no refit
```

memory attention可用64维/8head单层Q/K/V；输出投影weight/bias零初始化，其余Q/K/V正常同seed初始化。不要再同时把外层gate也置0造成双零支路。初始Q/K/V与history路径梯度可为0，输出投影先学；这不是永久断路。当前encoder通过原baseline/readout已有任务梯度；输出投影离开0后，历史条件损失还能沿query回传当前64维token与原10层attention。新增attention直接跨窗口；原10层attention本身仍各在192点内处理，只是参数受这个联合目标监督。

E0 copy无历史encoder梯度，memory attention参数仍可从固定K/V学习。若改为共享Eθ并对两段过去也反传，就增加历史encoder学习路径、额外反传图/成本，且不再是同一个最小版本；若缓存其token又不失效，则会错误地把旧θ历史冒充当前θ。当前不建设frozen/shared两版网格。

### 必须保留最强base及三种“恒等”语义

当前最强锚点为**FT+Ccap**，不是仅native P_FT head。若H经Ccap而继续FT对照只用native head，就混入读出差异。各臂应共用同一A0定义：正仿射FT head加`[normalized Z /√1536, normalized P /√192, normalized Z² /√1536] @ coef + bias`，首4窗保持原fallback。旧TRAIN统计/系数作为固定buffer，不重拟合；Torch float64算子应允许输入梯度，不能用`no_grad`/NumPy/detach包住整个Ccap map。当前encoder/head可训，Ccap系数固定，是**固定读出坐标下的继续适配**。

1. θ=θ0、head=head0、memory输出0时，模型在数学上退化到原base。保持原head参数，不能重新随机初始化。
2. 数值层面仍须以后在2203核同global64、原stack/astype/transpose布局、mode/RNG及原FP64 Ccap路径。warm曾出现相同数值但不同batch/layout导致GPU路径不一致；本次源码审计没有做新bridge，不声称位级恒等已通过。
3. **训练后只关闭memory，并不能恢复已改变的θ。** 若要求随时严格返回冻结base，`enabled=False`必须直接走E0/head0/Ccap0 reference（TRAIN/VAL可复用原身份cache/anchor），且保留−1 incumbent。不能把“关闭附加attention”偷换为“恢复原强基线”。

### 最小可信直接比较：基本三臂加prior语义控制

| 臂 | 当前E/head | memory内容 | 可回答的问题 |
|---|---|---|---|
| P_continue | 从同已选FT继续更新，共同固定Ccap map | 无新增memory | 继续FT本身能否解释改善；冻结A0另保留incumbent |
| C_current | 同初始化/同可训练路径 | 同结构/参数/384长度memory槽，填两份冻结E0的当前192×64 token | 新增容量、query-dependent当前映射及其计算能否解释H收益 |
| H_past | 同初始化/同可训练路径 | 两段真实过去192×64 token，经冻结E0 | 在上述控制下，过去内容是否带来额外开发收益 |
| P_prior（协调已选第四臂） | 同初始化/同可训练路径 | 冻结Ccap过去预测包络，经单独1→64时序投影再同类attention | 区分高维EEG memory与预测输出先验；不称参数/有效capacity严格同H |

四臂共同使用原全部252454 TRAIN/30500 VAL窗、global64/tail、record归属与全部cold-start目标，不按可取历史删窗；同seed、更新/epoch上限、目标dtype/eps、optimizer组与学习率、dropout/mode、验证选择及最早tie/−1规则都应在具体决定中一次冻结。本次不指定新epoch/LR或启动fits。前4窗可以一致走各臂当前路径，无memory；不能一臂丢弃、一臂参与分母。共同Ccap/目标/物理评估定义不变，不能同时把3s目标换D9物理目标。

H/C需显式对齐新增attention形状、初始化、监督/更新及RNG，并分别记真实encoder调用成本。C可只做一次冻结E0当前前向、把token复用到两槽；H需两段过去的E0前向，成本不能假称相同。**不为虚构FLOP公平而额外计算真实过去再丢弃**；若实际选择两次当前E0填槽，也只按真实调用记账。T=P_continue没有adapter计算，作为继续训练控制不冒称逐FLOP相同；H−C控制可训容量/结构和输入内容，仍需披露历史访问成本。基本三臂只回答直接memory比较；协调为区分输出先验已明确加入第四臂P_prior。它可复用旧Ccap预测缓存，必须披露额外投影与encoder成本差，不能把单scalar复制64份后channel LayerNorm（会抹掉该scalar）。固定1→64的k9 Conv是待写入具体协议的单一投影候选，需实际验证P内容扰动进入K/V，而非在零输出attention上要求最终输出立即变化。四臂仍不是完整论文消融集，也不完成独立泛化或新颖性证明。

### 时序支持：合法过去不等于全是fusion baseline之外的新信息

当前窗半开支持W_i=[s_i,s_i+192)，s_i=96i。候选i−4/i−2为[s_i−384,s_i−192)、[s_i−192,s_i)，连续6s、互不重叠，end≤当前start；须逐record索引、i≥4，不跨记录/phase。所有位置仍经过各自原3s双向encoder，不能称逐sample实时因果。

但原Hann融合的当前窗输出还会与邻窗i−1/i+1共同进入同一物理区间，其保守raw支持并集从s_i−96开始。i−2末1.5s的[s_i−96,s_i)与这个baseline已有输入并集重叠。**三臂可检验该明确定义的window-level memory适配，不能把全部6s叫成融合baseline之外的纯新增过去。**若科学问题要求纯before-union历史，需要协调另选例如i−5/i−3（并集[s_i−480,s_i−96)）或增加支持匹配核查；这里不擅改候选lags、增加格或将两种权限混为一谈。

新history没有超过当前窗末端的raw支持；原3s内部attention/归一化仍使用整窗，未来权限不能降格表述为因果。不可把当前token/未来window塞进history bank，不能把global batch相邻当同record过去。frozen E0在线取两窗，无需持久recurrent state；若采用缓存，64通道preprojection约为原8通道latent的8倍，三seed全量新cache显著超出现有磁盘余量，不能默认物化。小批在线/受限内存缓存是可行实现方向，实际峰RAM/显存和吞吐尚未测量。

## 3. D9重叠抵消：feature fusion本身不能根除

`PhysicalQuadratic.A`仍输出每窗192坐标中的190个自由坐标，`residual_adjoint(role='F')`只看归一化Hann融合；W另含不同视图disagreement。对内部一对视图，F见到`a*p_left+b*p_right`；任意成对扰动`(b*d,-a*d)`可在该物理点抵消。实际D9有跨样本共享参数、边界约束和ridge，不能因此把其完整Hessian说成严格奇异；既有报告支持的是**近null坐标/病态bias识别**，不是已证明输出数值崩坏。

[D9实际结果](../../physical_objective/run01/REPORT.md)中F的VAL融合r小幅升高与window MSE352.068、overlap MSE706.180同时保留；本次不重算/更改它。把历史放进表示层、共享卷积head或继续FT，都不保证同一物理点从相邻3s窗得到相同值，因为其上下文、padding与双向attention不同。只要仍自由产生每窗192输出后再融合，原冗余输出坐标风险就还在。

### 能从坐标上消除新增抵消自由度的方案

令r_t是每个物理点仅一个残差值；输出`p_i(u)=a0_i(u)+r_(s_i+u)`，其中a0逐窗baseline冻结，所有重叠视图引用同一r_t。则在原有效覆盖上`F(p)_t=F(a0)_t+r_t`；相邻新输出之差始终等于冻结baseline之差，新增支路无法另造相反大误差来互相抵消。r=0可逐窗严格回到a0。**它消除的是新增支路的跨视图抵消自由度，baseline原有disagreement仍在**；若只输出唯一连续`F(a0)+r`，物理base恒等成立，但不再保留native逐窗输出身份，必须分开窗口和物理验收。

r_t必须先由唯一owner/token上下文定义，不能先学任意两份window residual再Hann平均并改名“一个r”，否则前融合的null自由度仍未消失。

支持安全的一个具体owner候选是：窗i只发出其后半96点，赋给[s_i+96,s_i+192)唯一物理坐标；r由该窗preprojection后半token和严格过去生成，prefix/尾部/尚无合法历史处r=0并沿用原mask/原Hann baseline。latest raw end≤s_i+192，不超原baseline在对应点的支持上界。反之，直接用最新起始窗的前半token发物理预测，在该窗Hann起点权重为0时，可能引入原baseline该点没有的更晚raw支持；必须逐点核正贡献集合，不能只看平均延迟或把Hann新拼块当纯过去。

这个坐标约束同时改变目标点权重、窗口输出表示、warmup与可用上下文；需要自己的current-only/输出坐标对照与共同比较集合，不能与projection前memory三臂同时混改再归因历史。它是可实现的备选结构约束，不是已验方法；不在本审计构造数据/模型或执行D9重跑。

## 4. 我们的output_prior不是作者DECAF复现

原round01 `NativeResidual(family='output_prior')`用**本模型head(z)预测**的6-bin摘要进既有状态，输出6个残差bin线性插值到192，再加到当前head；head仍联合学习。它不是独立预训练envelope forecaster，也没有把真实过去包络喂入训练。D12–D16的`P_prior`又不同：输入过去冻结Ccap的完整192点预测，经同类token/GRU条件器，而非原6-bin prior。两者不能混称一个已实现DECAF。

[JHU-LCAP作者入口](https://github.com/JHU-LCAP/DECAF/blob/main/README.md)明确指向carankt/DECAF。本次已读代码的main提交为`f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332`。[递归模型](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/src/models/two_branch_model_recursive.py#L351)的eval forward先由EEG分支重建过去包络，再用独立forecaster，与当前EEG分支通过可选融合结合；这个函数不是持久反馈上一最终fused输出。其动态融合网络可依赖两分支预测，不等于本项目加性6-bin残差。

[trainer](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/src/train/trainer.py#L211)所读递归训练分支的teacher forcing比率始终1，VAL调用GT teacher函数，默认test使用预测历史；另一测试函数调用的forward_recursive并未在所读recursive类定义。模型eval当前EEG输出窗内minmax，teacher路径则sigmoid。这说明已读入口/论文协议仍待对齐，**不能推断整仓库无正确paper脚本、作者结果无效或存在测试标签泄漏**；真实过去包络训练不自动等于测试泄漏。本项目没有运行其自动test入口或下载Hugging Face载荷。具体已读边界与相关源码blob见[协调一手审计](../../metadata_review/CONTINUITY_CODE_AUDIT_20261103.md)。

[SHINE当前README](https://github.com/xuxiran/SHINE/blob/main/README.md)仍说代码待发表后公开；本次看到的是README入口，不能记为作者代码复现完成。有限代码检查不承担方法新颖性判定；已知segment-memory/attention组合不因“此前本项目没测”自动构成创新。

## 5. 交付与下一决定的具体边界

实现可行性：projection前插入点及梯度路径明确，现有3s网络可原样复用；数值恒等、梯度实际可达、资源峰值和有效训练尚未实测。三臂及frozen-memory版本只是最小候选，不是已冻结实验。

审计交付时协调已具体选定下一方向：四臂H/C/T/P、三seed，共同最强FT+Ccap(.01)、冻结history E0/可训当前Eθ与head、projection前zero-output memory；主near改为i−5/i−3，older为i−9/i−7，共同i>=9，首9窗冻结旧anchor且全部目标/VAL保留。这样避开前述i−4/i−2与fusion当前支持交叠的解释问题。统一D16负centered cosine、globalshuffle及60s最早选点含−1，拟20轮；E/head/adapter学习率1e−5/1e−3/1e−3。正式运行仍需具体可执行协议/预算与必要工程、实际source审阅，不能把本文当已通过工程或已启动fit。

后续工程合约：先用固定TRAIN窗区分same-latent Torch/NumPy Ccap读出桥与raw重编码差，旧Ccap仍是incumbent，不按新VAL好坏定容差或无休止追逐逐bit一致；Ccap按`u=norm0(vec Z), v=norm1(cal P), w=norm2(u²)`及1536/192/1536尺度拼3264→192，float64读出/目标、float32 encoder/head且autograd贯通；history eval不消耗current dropout RNG，额外构造隔离初始化。prior必须保留可识别内容的时序投影。raw shuffle可只读mmap/RAM缓冲，不新增完整64通道磁盘cache。位置/segment标识及mask应在实际实现中明确，不能把无时间标识的attention置换不变性包装成已学会recency。

协调资源初界为两卡逐fit并行、每worker GPU32GiB/RSS48GiB、24h、ROOT768MiB/两盘20GiB，只有Eθ/head/adapter best/last，不重复保存frozen E0/Ccap大buffer或全VAL预测。以上是已给具体方向及待核实上界，不是实际吞吐/峰值结果；本只读审计没有构造科学数组或新fit。D9唯一物理坐标方案与当前memory实现分开，不同时混改。D17/旧test/DTU/ds004408保持冻结，父任务和患者FORBIDDEN边界不变。
