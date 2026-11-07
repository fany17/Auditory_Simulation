# D17：所选模型历史响应很小，末轮模型仍有明显内容敏感性

2026-11-02；PUB-01-S03_SS07。**固定当前输入的384窗诊断完成，0新增训练。H2026 selected5中，near/older历史差经过token、GRU及FiLM后传至最终预测的RMS差仅9.392×10⁻⁹；三个last19的预测RMS差则为.033526/.019122/.047253。所选模型与末轮模型行为不同，不能把D16阴性归为所有检查点都不响应历史，更不能据此宣布历史没有信息。**本轮只定位固定小批中的数值行为，不证明生理机制、任务收益或方法创新。

## 冻结范围与实际执行

沿[协调决定](COORDINATOR_DECISION.md)，仅四个既有H_full检查点：seed2026 selected(epoch5)及三个seed last(epoch19)。另两个selected是零增量锚点，不重复推理。D16已封存`7a17fc2`，不改模型、LN位置、loss/LR、权重或原VAL选点。

原前8条TRAIN manifest每record local0/8/9/10/11/12/13/14，共64metadata位置、48合法历史位置；**全部来自sub-001的8条记录/8个story**，不是8名受试者。保存原subject/story/record/local/global索引，无source_family原字段就不臆造。沿各seed原TRAIN变换及Ccap，只索引48 current和指定history，不构建完整TRAIN Data、不读目标或新VAL/test。

每检查点当前z、Ccap、顺序完全相同，仅替换同record的near i−5/i−3与older i−9/i−7，两段均严格过去且不与当前窗口重叠。直接调用冻结D16 Conditional.forward，hooks取既有节点；沿已验证model.train()路径，此网络无dropout/running统计。全部权重requires_grad=False，只有history.requires_grad=True；每次已有forward后只对固定标量`sum(residual*sin(arange(192)*.017))/48`求一次输入梯度，无optimizer和目标loss。它是固定输出投影对历史输入的局部梯度，非全Jacobian或所有输入方向。

生产源冻结commit `788029c82e7170a8847bb73a51a7087fc23cec0c`，v2 [PROTOCOL](PROTOCOL.json) SHA `1791ab561d6faf1505855df1d691be83b7e2bd3eb26bc091276ab110422d883e`。协调实际核229 source pins、15输入身份、4 checkpoint SHA及统计轴后[批准执行](COORDINATOR_EXECUTION_REVIEW.json)。首0NN prepare/source/protocol完整保留于[prepare_v1](prepare_v1/PROTOCOL.json)；v2只补subject/story及Ccap原加法分组，未增加神经工程。原/新prepare分别0NN/0backward。

## 四个检查点的逐层差异

下表均为near−older逐元素差的RMS，不是分数或置信区间。不同节点尺度/维度不同，绝对量不直接作可比较的“信息损失率”。完整[56行节点表](tables/NODE_COMPARISONS.csv)另列节点自身global/跨窗中心化RMS、差max、相等比例、逐窗差min/median/max和相对量。[四检查点汇总](tables/CHECKPOINT_SUMMARY.csv)不挑有利seed。

| 节点差RMS | 2026 selected5 | 1024 last19 | 2026 last19 | 31415 last19 |
|---|---:|---:|---:|---:|
| 历史输入 | 1.358000 | 1.295589 | 1.358000 | 1.292056 |
| LN前token | .812690 | 113.804601 | 395.657848 | 158.129694 |
| LN后token | .094910 | 1.310017 | 1.311967 | 1.325895 |
| GRU末h | 6.396e−6 | .739365 | .781799 | .737513 |
| FiLM实际gain增量 .1tanh(gamma) | 0 | .024648 | .028707 | .013419 |
| FiLM实际bias增量 .1beta | 1.239e−6 | .481248 | .334581 | .633301 |
| 调制后current | 1.242e−6 | .483871 | .365064 | .633860 |
| 网络residual | 9.392e−8 | .335261 | .191216 | .472527 |
| 最终Ccap+.1residual预测 | 9.392e−9 | .033526 | .019122 | .047253 |

输入差确实存在：selected5的历史输入相对差（除以两条件pooled global RMS）为1.398896，LN后token仍有.094910相对差；因此不能把所选模型输出差很小解释成这两段输入原本相同。该检查点GRU末h相对差只有6.396×10⁻⁶，residual相对差3.961×10⁻⁷，最终预测相对差1.708×10⁻⁸。它们描述本批配对响应相对于各节点自身尺度的大小，不是跨层因果传递系数或信息论上界。

在selected5，near/older的GRU末h global RMS约.999930/.999931，跨48窗中心化RMS仅4.457×10⁻⁶/4.100×10⁻⁶。两步state按固定token位置分别计算的跨窗中心化RMS，near为.0027065/4.457×10⁻⁶，older为.0026611/4.100×10⁻⁶。**这里沿axis0跨48窗中心化，每个token位置/feature坐标固定，未把两时刻混在一起估计方差。**这些数值表明末h在本批上高度近似常量；不能据此宣称所有输入的表示塌缩。

selected5的FiLM原gamma RMS约7.54968，仍有约10⁻⁵级跨窗变化；tanh(gamma) RMS约.99999944、max约.99999964，跨窗中心化RMS为0。near/older的实际gain增量逐元素100%相等，RMS约.09999995。与tanh平坦区域及float32分辨率下变化被压小相容；但bias仍有约1.239×10⁻⁶配对差，residual也不是全相等。其residual相等元素比例约10.77%，48窗均有非零差，逐窗差RMS min/median/max约3.053×10⁻⁸/7.570×10⁻⁸/2.001×10⁻⁷。**“输出差很小”不等于严格不依赖历史。**未隔离各门控/投影的因果贡献，也未比较其他位置、eps或正则方案。

三个last19则不同：LN后token相对差约1.310/1.312/1.326，GRU末h相对差.890/.906/.840，residual相对差.804/.411/1.012；三个residual的near/older完全相等元素比例均0。固定当前输入下，这些末轮网络在所检小批上确有明显历史内容响应。它既不证明响应有助于任务，也不推翻D16的VAL阴性；训练终点响应更大与last VAL更差可同时存在。D16所选2026只有微小开发增益、低于P_prior，其他H选回锚点的结论保持。

## 局部历史输入梯度

| 检查点 | near梯度L2 | older梯度L2 | near/older非零历史输入元素 |
|---|---:|---:|---:|
| 2026 selected5 | 3.445e−7 | 3.453e−7 | 各147456/147456 |
| 1024 last19 | 1.056518 | .988828 | 各147456/147456 |
| 2026 last19 | .501071 | .621454 | 各147456/147456 |
| 31415 last19 | 1.280832 | 1.206443 | 各147456/147456 |

八次梯度均finite。selected5的全部历史输入元素在该固定输出投影下仍有非零局部导数，但范数较小，与有限near/older配对的微小输出差共同说明：参数/输入梯度非零并不自动意味着有足够大的内容敏感性或有用预测。三个last的局部梯度也较大；梯度大小受权重、尺度、当前点及固定投影影响，不能直接当性能、有效信息量或通用梯度可训练性指标。详见[8行梯度表](tables/INPUT_GRADIENTS.csv)。

## 验证、成本与边界

[实际完成回执](COMPLETED.json)：一次384残差NN窗、8forward、8输入backward，0fit/optimizer/encoder/head；四检查点及所有节点均保留。两条件current/anchor逐tensor相等，所有权重state_dict前后精确不变；源/输入/4cp身份、metadata严格过去边界、finite与计数通过。合成0NN轴测试验证固定token位置跨窗中心化、已知差值及零分母null。相对差分母明确为sqrt((mean(near²)+mean(older²))/2)，另一列先沿axis0中心化再pool；精确0返回null，没有隐含epsilon。额外hooks/归约有实际算术开销，未把0额外网络遍历说成零计算。

协调预声明独立[标量一致性review](COORDINATOR_SCALAR_REVIEW.json)，source SHA `13d34e2b3c416c2ad8111dcd1ca3aff1c4286c786aad4e32448c42eadae1c8e5`，一次return0，1325检查通过，0NN/0backward。它核源码、metadata与已存标量算术的一致性；运行时current/anchor/权重不变断言经源码审阅，未独立重放。未保存激活，故没有独立重算全部实际激活RMS/梯度，也没有独立384窗神经回放，不将此有限审阅扩大命名为完整数值复现。

| 实测阶段 | NN窗 / 输入反传 | CPU秒 | wall秒 |
|---|---:|---:|---:|
| v1源/stat/纯标量检查 | 0 / 0 | .227605 | .234885 |
| v2元数据/加法分组重绑 | 0 / 0 | .227900 | .230280 |
| 四检查点固定小批诊断 | 384 / 8 | 2.186963 | 1.237380 |
| 只读JSON→三CSV/汇总 | 0 / 0 | .002907 | .002921 |
| 只读成本账本 | 0 / 0 | .000840 | .000847 |
| 协调独立标量一致性review | 0 / 0 | .167573 | .167967 |

成本账本所列前五阶段CPU和2.646215秒，后到协调标量review另测.167573秒，上表分列，未伪装成已含在原账本列表。计时均从各入口标准库/主依赖导入后开始，诊断计时包含核验、空卡准入/等待、子集加载、前向/反传和标量归约，最后receipt序列化在计时范围外。SSH/复制、审批及人工式源码审阅、本地归档和全部管理时间未完整测量，不将这些忽略项称为0CPU。详见[COST_LEDGER](COST_LEDGER.json)与独立review实际receipt；不重复运行任一阶段补计时。

实际单GPU空闲准入，未终止他人；诊断RSS峰1106752KiB、GPUallocated36940288B。账本时新ROOT466974B<8MiB，两盘余量22660694016/178649985024B均大于20GiB，无新数组/权重/激活文件。仅结构化统计与代码回本地，数据/权重和全部科学计算仍在2203。所有统计只驻进程内存后释放，没有为补指标重跑网络。

本诊断仅单受试者48个固定有效窗；near/older同时改变内容和recency，older也可能保留相关信息。精确相等及接近常量均是特定检查点、小批与float32下的计算观察，不是全人群、全方向、全数据的信息论结论；不据此认定tanh/GRU是全部性能差异唯一原因。没有新VAL/test或性能评估，不能从本轮直接证明新方法有效。下一方法决定由协调依据完整证据另行界定，SS07/科研goal仍ACTIVE，历史收益及独立泛化/创新目标尚未建立。
