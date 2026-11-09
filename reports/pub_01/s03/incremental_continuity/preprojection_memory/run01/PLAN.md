# D18可执行方案：单一projection前memory，四臂三seed

本文件落实[协调方向](COORDINATOR_DIRECTION.md)。状态为SOURCE_IMPLEMENTATION/NOT_ENGINEERED/FORMAL_NOT_RUN；正式fits在机器协议、必要工程与实际源/输入/账本独立审阅通过后直接执行，不再向用户要已有授权。SS07同一节点内执行，不扩数据/其他主节点，不修改D16/D17或旧测试结果。

## 唯一模型和输入

新ROOT拟`/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103`，本地本run01。沿三seed已选FT encoder/head、固定D8 Ccap(.01)/标定/三个TRAIN变换。当前raw192×64经原10层到192×64；最后64→8前加单层64dim/8head memory attention，query/KV各自LayerNorm64，dropout0、输出投影0。原native64→8与Conv8→1 head原样保留并可训；Ccap f64图保留，统计/系数buffer固定。E0为同源独立冻结eval copy。H/C相同参数、初始化/shape，额外构造CPU RNG隔离；history eval不耗current dropout RNG。

H_history取E0两段真实过去；C_current只做E0当前prefix一次，复用到两槽，不额外空算past；T_continue无attention；P_prediction_history取冻结Ccap过去预测值，以原stats1标准化后Conv1d(1,64,k9,pad4)再进同类attention。P多640个投影参数及成本实记，不称同有效capacity或DECAF。原head不随机重置。

固定正弦/余弦64维sample位置码，Q位置0…191，memory两槽为−480…−289、−288…−97，加入规范化后的Q/K、不加入V。C/P槽标签与H相同；older只替换历史数据，仍用原near槽码。因而older联改内容/新近程度且保留固定slot标记，不是纯时间延迟/内容因果效应。不扫位置/长度/层/宽/损失。

全部原TRAIN426/252454、VAL49/30500，原TRAIN raw归一化和目标dtype；current完整global64 raw前向后仅i>=9的输出参加适配，首9预测强制旧anchor且留loss分母。近i−5/i−3，older i−9/i−7，逐record检查raw半开支持end≤current start−96，不跨phase/record；不是逐sample实时因果。未来权限仍原整3s；不混D9输出坐标修改。

## 数据接口与baseline身份

RawPhase保留原np.asarray(window,f32)→减mean/除std→np.stack(...).astype(f32)布局及统计原dtype；随机索引由D16同公式permutation决定。正式可把原raw数组按原K-order只读复制到服务器RAM；不写新raw/latent/history磁盘cache。原anchor仅以旧Z/P缓存和冻结Ccap按原1024块/record、首4fallback恢复；不重拟合。工程用lazy subset anchor和mmap，仅碰预设TRAIN索引，不遍历所有数组值。

`enabled=False`直接返回旧anchor；训后只关attention不恢复θ0。epoch−1候选绑定旧Ccap分数/身份；zero-output θ0主干的数学恒等、same-latent数值桥和raw重编码差分开验收，未经实测不称位级恒等。所有可变checkpoint仅named trainable parameters，排除E0/固定Ccap/位置buffer/optimizer，不复制12份只读资产。

## 训练与预设结果

12fit各20epoch，global64/tail38，D16确定性globalshuffle（seed、20261102、epoch），每epoch3945更新位置/252454完整目标窗；实际空梯度位置若有须单列，不伪报optimizer更新。Adam encoder1e−5、原head/memory/P投影1e−3，wd0/no clip。float64 centered negative cosine eps1e−6保留全部fallback/12低范数TRAIN目标语义。源/预算冻结后不根据中期VAL改动。

每epoch一次原VAL60s/person，strict >最早最大值含旧Ccap−1，0…19全跑。固定离线TRAIN检查点−1/0/4/9/19：−1旧anchor0NN，其余各一次完整TRAIN预测，记录目标、window MSE/r3s、物理r/MSE、幅度、overlap/disagreement/边界/逐人/来源；在线轨迹单列。VAL逐record RAM流式评分，不落预测数组；最后selected每臂一次，H在同一次current query上再做older，当前张量不重编码。若selected−1直接复用旧anchor/结构相等，不假装有效历史依赖。

每epoch第一既有TRAIN batch只加detached激活/梯度标量，不增加forward或step；记录模型/原attention/新attention/原head/P投影的finite、nonzero/L2。fresh零输出时QKV/P投影梯度可为0，期望与后续活动分开。所有12fit及阴性、回退与失败保留。

## 精确组件上限（不混称相同FLOPs）

正式监督窗呈现60588960、更新位置946800。按全部selected均非−1的最大路径：

| 组件 | 窗前向上限 | 定义 |
|---|---:|---|
| 当前encoder prefix | 80392752 | TRAIN60588960+逐轮VAL7320000+四非零TRAIN12117792+selected VAL366000 |
| 当前64→8 projection/head，各自 | 80484252 | 上项+H paired older91500，older复用current prefix |
| 冻结E0 prefix | 59563425 | H两过去39769068+C一次当前19794357，无原projection/head |
| memory query窗 | 59473248 | H/C/P有效窗，含H older；T无attention |
| P k9投影过去窗 | 39588714 | P每有效query两窗，P没有older |

这些组件次数不是独立样本数，不能相加冒充唯一窗口或完整模型FLOPs。fallback也在current global64 prefix/head前向中，但不给新参数目标梯度；有效TRAIN适配窗每epoch248620。selected−1会减少实际次数，最终账本按调用计数而非上限伪报。历史共享原训练、cache生成和参考合成自检另作provenance，不称零历史成本。

## 必要工程与桥接合同（新VAL前冻结）

拟一次预声明TRAIN工程，三seed×四臂各两次临时optimizer更新、最多24次且全丢弃；正式均从原FT干净重建。覆盖真实shuffle batch与cold-start/record边界，另用同批旧Z/P/统计作纯readout代数/VJP桥。上限先按current prefix/head各8192、冻结prefix16384、所有其他读出/attention/P算子逐项记录，输入backward次数/24临时更新独立计；不把工程fit和正式fit混为一项。若首次失败，保留实际路径/源/日志/计数，仅在该总界内有依据修复，不为逐bit追逐反复神经重放。

- same-latent Torch/独立NumPy Ccap值：abs1e−10+rel1e−9；独立解析VJP对Z/P：abs1e−9+rel1e−8，固定sin输出cotangent。原reference自身合成检查已由协调一次PASS，CPU.057992472s/wall.014535500s/RSS47404KiB另计，不重复。当前生产梯度桥尚未执行。
- 同一raw/model/mode/RNG下原forward与zero-memory路径：沿旧raw绝对1e−6门；与旧缓存不同batch位置/raw重编码单列，预设maxabs≤1e−4+1e−6×参考maxabs，不使用VAL成绩定门限或替换旧incumbent。失败先定位layout/gradflags/mode/分组，不把合约任意放宽。
- actual Ccap输入/coef/bias/constant mask及head直接标定梯度、Z²链路；前9fallback逐tensor旧anchor、enabled=False恢复旧anchor。
- H/C同初始化/参数/KV shape、current dropout RNG隔离；E0/统计不变、所有梯度finite、第二步内部QKV/query可达；P内容扰动在其投影/规范化/K/V有非零差，零输出主干不要求最终预测立即变。
- position/near/older/record支持、12fit与全部计数、参数-only checkpoint与存储；只测实际TRAIN吞吐和峰值，预测完成时间仅是估计，不按猜测阻断授权。

以上工程cap/阈值在实际ENGINEERING_PROTOCOL中绑定源和TRAIN索引，任何变动留版本，不是已测PASS。

## 资源、审阅与停止边界

两GPU各一fit并行、总24h；每worker GPUallocated32GiB/RSS48GiB，BLAS4、torch2，新ROOT768MiB并保留64MiB在途，home/SATA各20GiB。仅核本队列PID/UID/start_ticks/argv，不杀他人；陌生占卡等空卡，不改MPS/全机设置。实际preflight核当前资源、参数字节/输出大小和吞吐，source、input identities、工程/预算经协调一次实际审阅再开完整矩阵。正式失败保留，停止本轮已核身份worker，不自动改上限或重跑。

大指标采用无损gzip JSON，完整保留顶层及records内逐块信息、边界、逐人/来源和全部小指标；规范路径为`INCUMBENT.json.gz`、`OFFLINE_TRAIN_epoch{epoch}.json.gz`、`SELECTED_METRICS.json.gz`及H的`OLDER_METRICS.json.gz`，读取统一用runtime的`read_result`。最多87份，每份强制≤2MiB（174MiB总上限），其他源/小记录/日志32MiB，24个best/last参数checkpoint按工程实际完整metadata序列化最大值，另加64MiB在途。元数据阶段用现有逐块结果模板及真实426/49记录、6104/752块计数做新schema JSON尺寸压力核验；填入的独立随机full-precision浮点仅用于序列化，绝不作为科学结果落盘或报告。gzip往返验证全部字段相等；不删字段/降精度、不以旧D16紧凑格式外推新文件体积。

Technical含工程及最终独立review pass；科学门为H三seed都超过C/T/P/旧Ccap且同checkpoint near/older一致获益，否则不认定额外历史贡献。反复开发VAL只是有限推进门，无独立确认或创新保证。Parent/public/server2203边界不变；旧内部test、DTU、ds004408不读取评分，不下载患者或其他载荷。任务不在smoke/工程PASS处停止，也不按阴性自动开新候选。
