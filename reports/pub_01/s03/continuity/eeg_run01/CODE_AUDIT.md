# SS06 真实执行：模型、连续性与对照审计

执行源码/冻结和首阶段证据已提交`ea5a1b5b50293f36a4d55ab602f39ba4f096140e`，分支`codex/pub-01-s03-ss06-continuity`。FORMAL_PROTOCOL保存2203实际部署源码字节身份，提交前与当前工作树逐项核对一致。prototype01源码/报告不覆盖；真实执行新增execution适配器及独立loader/train/evaluate/contract。后置observer脚本不选择或修改训练方案。

## 实际比较的13候选

| 候选 | 实际结构/对照 | 注册参数 | 下游主路径参数（非逐标量活动保证） |
|---|---|---:|---:|
| local | 原causal LocalEncoder + linear head | 18817 | 18817 |
| M1 short/long/multi | 各4个独立活跃branch；dilation分别全1、全256、1/8/64/256 | 每组40481 | 每组40481 |
| M2 | GRU fast + learned-rate slow state + local fusion | 29345 | 29345 |
| GRU / M3 no_pretrain | 共同local encoder + causal GRU context + head | 47105 | 推理25153，teacher/辅助头不进入下游推理 |
| M3 frozen | 同seed正式SSL checkpoint，冻结encoder/context，只训head | 47105 | 推理25153，梯度主路径仅head33 |
| M3 finetune | 同一SSL初始化，联合微调下游路径 | 47105 | 25153 |
| M4 fixed/adaptive | 相同0/4/8/16 sample lag bank；uniform固定权重 vs EEG慢门控 | 每组18950 | fixed18817，adaptive18950 |
| M5 EEG-only/prior-only/full | local+tanh、无EEG自主latent、prior→observation→innovation→gain校正 | 每组23009 | 分别18817/1089/23009 |

注册数、图参与参数、首次真实batch的实际非零标量梯度分别记录在training_summary；不能用注册数相等隐藏失活路径。M1专门补齐4活跃branch匹配，long与multi最大span相同约16.27s，short约0.33s。不同框架本身容量/优化难度仍不同，不能称全局等容量。GRU同时是M3 no_pretrain，不重复计数成两个独立基线。

## 连续性和目标

真实loader保留time×64密集轨迹，不提取3s摘要token。batch1，每record首60s热身，正式随后所有完整60s块逐一推进。一次60s完整输出计算一次`1-r+.1MSE`及optimizer.step，绝不按3s各算loss平均。MSE目标只用新train的mean/std。块后detach仅切计算图、保留状态值；跨record/subject/session/split/gap重置。反传跨度60s，不是无限梯度；训练时state来源于上次更新前参数，属于在线TBPTT。评估权重完全冻结且同record持续传state。

M3正式预训练覆盖新train所有3511个完整60s时间段，段内teacher与online同一输入轨迹，future latent只作loss目标、stopgrad，EMA仅optimizer之后更新。当前辅助目标不支持带入历史teacher cache，所以各SSL段fresh-state，不能称跨段流式预测表征学习；下游仍完整record连续。三seed正式checkpoint各自共享给frozen/finetune，reject pilot/异seed/错误source或预算。额外SSL样本/时间披露，不把M3改善自动归因更高算力效率或听觉语义。

M4固定与自适应最大可见EEG相同，门控不读取目标；加入.001权重相邻变化罚。M5full加入.01观测预测MSE，目标EEG特征stopgrad。它有计算意义的innovation，不是脑预测误差/校准Kalman状态；自主prior完全不使用前缀或当前EEG，仅按记录起点时间演化，不能单靠此对照排除输出自相关。

M4本轮fixed为均匀混合，**没有另行优化最优静态非均匀lag权重/逐lag模型**。即使adaptive优于fixed，也只能支持相对这个均匀控制的收益，不能证明真实时延随时间变化或动态门控不可替代。M2未执行fast-only/slow-only，M5latent未作可识别性或校准不确定度验证；这些缺口不靠追加test后分支补齐。

## 对照必须按端点解读

- 全记录主端点：538完整60s→record等权→person等权；M2同权重每3s清**后encoder状态**控制也在全记录评分，局部encoder连续。
- 历史干预端点：每record只用第一个评分60s，共41段；真实/异record-person-family donor前缀均60s，当前EEG和目标完全相同。state-reset/state-wrong保留真实encoder缓存，当前局部特征逐点相同；prefix-reset/prefix-wrong连encoder一起改，最初16sample局部边界效应单列。不能把有限等长前缀干预当整条记录持久记忆的严格等价检验。
- 输出自相关：local+causal EMA仅用validation选alpha；AR先用新train真过去包络拟合lag1/2/4/8/16/32的ridge关系，再仅用过去**预测输出**递推，以当前local预测作校正，validation选gain。测试没有真音频teacher forcing。此为一个明确的output prior控制，不声称排除了所有可能声音先验。

AR使用完整新train包络（包括监督热身/尾部之外的合格过去配对），拟合样本数在output_controls_frozen中另列；不能把这部分标签拟合算成零成本或与相同监督optimizer预算混淆。alpha0和gain1允许退化回local，避免强迫不合适的平滑。
- 自主prior按MSE优化/验证选点；std≤1e-6归一化单位时Pearson为null，绝不填0或把浮点抖动相关写成语音信息。初版pilot的prior r保留为历史，formal数值规则和限制已在开audit前冻结。

## 工程适配与验证证据

M2/M3将原GRUCell相同四个参数张量交给安装版PyTorch序列GRU核；slow state与M4gate用因果仿射parallel scan。没有改变门、状态、输出或增加未来EEG。M5full保留同一递推方程，用显式一阶反向递推减少细粒度autograd图；包含gain、observation辅助loss、EEG输入及跨chunk初态梯度，不支持高阶导。M5模式隔离和M4zero-lag空cache与原型一致。

实际核验：原型38测试/71子测试与首版3适配测试通过；最终7适配测试另行通过，包括64ch×3840 CUDA输出/参数梯度/分块等价、M5观测loss/初态梯度、prior纯度和模式隔离。不是声称一次跑了48个不同方法。TorchScript弃用和非连续GRU权重warning保留；后者可能含参数packing成本，实际耗时已计入。

真实train/val原始索引与新统计核验PASS；CPU/GPU实际60s分块及梯度检查PASS。正式TF32关闭；初版pilot默认TF32作为有限pilot历史，不与正式结果混表。observer复算方法在开audit前用验证集36个原型CPU vs加速CUDA相关做实测，最大r差9.10e-8；错误seed/预算/pilot权重被拒，输出AR递推与因果性检查通过。后置真实audit复核另见REVIEW与review.json，不能把此方法检查当后置结果已完成。

## 成本、范围与审阅身份

所有监督候选相同3262×3840目标sample/seed、record/block顺序和val选择；M3额外3511×3840 EEG sample/seed。M5prior的MSE-only是自主对照例外。CPU/GPU与活动参数不等，训练wall time、GPU峰值、逐seed实际样本及同设备推理计时各自列出；CPU的GPU bytes=0表示不使用GPU，不是CPU内存为零。MACC不计bias、非线性、逐点scan、数据搬运，不能当完整效率结论。M5训练用CPU，其他主模型用GPU；另有同设备实测供核对，不能混设备计时排名。

协调会话实际只读审阅source/split/cache/runtime日志，执行者另做后置逐级汇总重算及原始EEG+原型CPU复算；这两种证据分开，不称外部第三方科学签署。仅已有公开研究授权资产，无患者/STN、新下载、远端Git或旧模型checkpoint输入；SS02/SS03/SS05 frozen结果不改。旧来源历史暴露、5 audit family、人与训练重叠、一个监督pass、固定lag/目标及physical_sync UNKNOWN始终限制结论。

协调会话另在2203从逐record/逐seed证据独立汇总全部SUMMARY，max差5.55e-17；执行者原型CPU实际复核222个有效相关max差1.55e-7、234个MSE max差6.52e-8及12个未定义相关。该后置实证与执行前source审查分别记录，详见REVIEW。
