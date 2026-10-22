# SS06 真实公开 EEG 执行补充 — eeg_run01

| 字段 | 当前决定 |
|---|---|
| Task / Session | PUB-01-S03_SS06 / eeg_run01 |
| Parent | PUB-01-S03 / PUB-01 |
| Status | ACTIVE / APPROVED_FOR_EXECUTION |
| 人工依据 | 2026-10-23，在五原型交付后用户明确：“启动吧” |
| 范围 | 既定五框架及任务书对照的真实公开 EEG 数据加载、train/val pilot、方案冻结、训练和合格评估 |
| 执行者 | 原 PUB 会话 01a134e7-c257-7821-838c-b380642787f5；原指定 gpt-6.1-sol / high |
| 计算位置 | 仅 server2203:/home/fanyu/auditory_simulation_m6a；新 eeg_run01 目录 |
| 代码位置 | Windows 独立 SS06 worktree；codex/pub-01-s03-ss06-continuity / PR #6 |
| 完成标准 | 真实指标、预设对照、逐seed/逐record/逐人表、预算/划分/失效说明与审阅；不以 smoke 结束 |
| 患者数据 | FORBIDDEN；不扩大数据/模型下载范围、不在服务器初始化 Git |

## 1. 本次授权与历史的关系

本补充是用户对同一 SS06 的明确下一阶段启动授权，覆盖先前 PROTOTYPE_ONLY、NOT_RUN 与“等待启动”限制。prototype01 的通过记录保持原样，它只证明工程检查通过。当前 SS06 重新 ACTIVE；不修改 SS02/SS03/SS05 历史结论，不启动 SS04、S04 或父 S05。

不重复索要已经给出的启动许可。执行中可自行处理 loader、性能、缓存、训练稳定性和可比较预算问题。未知科学结果不预设阳性；不为了达到旧准确率选择 test、样本或模型。

## 2. 数据和拆分：先保护评估，立即推进可行训练

1. 只读复用已授权 SparrKULee 资产与可核验 metadata。不得重新下载或扩大样本/患者范围；保护旧 raw、normalization、cache、冻结协议和 checkpoint。
2. 先核对 source_family/story/record/subject 及历次 train/val/test 暴露谱系。禁止旧 SS03/SS05 已查看 test 进入 SS06 调参。新的预留组只能在输入身份、可用性及既往用途基础上预先确定，不能看 EEG 解码分数筛选。
3. 优先从 SS05 原 train 的23个 source families 中仅按元数据一次建立 SS06 train/val/audit_holdout，即**本轮前瞻内部审计划分**，但必须明示这些来源曾用于旧模型训练和项目开发，并非从未接触的独立外部数据。不得把旧 test 改名成为新 test。若实际没有足够可解释的预留组，完成全部可行 train/val 比较，标为 EXPLORATORY / HOLD_NEW_TEST；不要整体停工或伪造独立检验。
4. 有可接受内部预留组时，在任何 SS06 拟合和该组信号读取前写入版本化 split manifest、纳排与来源谱系；冻结后不得因成绩改组。所有归一化和预训练只用新 train，重新拟合并保存新文件。旧统计不覆盖。
5. 尚未明确的 test 独立性不能阻断合法 train/val 的 loader、真实 pilot 和训练。先报告可支持哪一种泛化结论，再按对应证据强度执行。

## 3. 连续数据和训练接口

- 输入保持真实连续的 EEG 样本次序，输出逐样本包络，不再压为3秒摘要 token。训练、验证和评分不跨 subject/session/record/split 或数据缺口拼接。
- 同一共同声学 offset / EEG lookahead，正 lag 的时间轴含义写清；只用 train/val 从事先小候选集选共同 lag，或固定既有可审计 lag，不能 test 重对齐。
- 共同目标为拼接的同一连续60秒输出上一次计算 Pearson 加声明的 MSE 项；不能每3秒计算独立 loss 或 optimizer.step 代替整段目标；warmup、mask、短记录处理和段→record→subject 汇总在 pilot 前固定。归一化目标仅 train 拟合，不能逐测试段用真实目标作反向变换。
- 首轮用 batch=1 连续 record 或边界同步批次，避免当前 StreamRunner 任一 lane 变化导致整批 reset；不随机打乱3秒块后声称持续记忆。对 stateful 模型保留跨连续计算块的 state；TBPTT detach 不能变成 reset。任何边界 reset 或随机起点 finite-context burn-in 必须在训练和评估契约说明，不能称其无限连续记忆。不能在尚要反传的计算图上原地更新参数。
- M3 teacher 未来 latent 仅为训练目标；明确无预训练/冻结/微调的权重交接与额外算力；预训练只使用新 train。当前辅助目标仅支持完整连续段，扩展时需验证 teacher 版本与缓存，不能静默重置。
- M4/M5 在既定结构内补全辅助目标和稳定性处理，不增加未批准的骨干或新数据。

## 4. 实际运行顺序与预算

1. 复核 prototype01 关键测试和新 loader 的时间索引/边界/归一化，使用真实 train 的少量输入完成 CPU/GPU 前向与反向；不重新做全库 hash/QC 阻塞推进。
2. 优先启动 M1 与 local 基线的真实 train/val pilot，随后 M2、M3、M4、M5 及必要 GRU/模式对照。每个候选 pilot 初始不超过100更新步；有明确工程原因可一次扩到200步，记录原因。pilot 不用于 test，也不作为正式性能结论。
3. 根据实际每步耗时、内存、train/val 曲线，冻结共同有效样本/训练预算、优化器/学习率、辅助项、验证选择/early-stop规则、至少3个预设seed和对照清单。不要无上限搜索；正式训练从头，不复用挑选过的 pilot checkpoint。
4. 五框架分别完成真实训练和比较，不先把五种叠加。M1为基础，M2/M3优先；M4/M5如主假说条件不满足仍保留明确探索性定位和已执行证据，不冒称确认性改进。报告额外预训练、注册/实际梯度参数、真实 wall time/内存，不用失活参数伪装匹配。
5. 所有训练和验证选择完成、正式协议审阅通过后，才对合格预留组做一次预先声明条件的评估。新内部预留结果如受历史开发暴露限制，必须维持 INTERNAL/EXPLORATORY 标签；不按 test 调参或追加训练。

## 5. 必要对照与判定

- 共同 local-only 与普通 GRU；M1 short/long/multi 并说明公平预算；M2 持久 vs 周期 reset；M3 从头/冻结/微调及预训练成本；M4 fixed vs adaptive；M5 EEG-only/prior-only/full。
- 所有历史机制至少真实/清零/错配历史对照，固定当前 EEG；选择性历史干预还固定 encoder cache。全前缀与状态专用干预分表，排除局部边界效应。
- local+因果EMA与预测包络历史 prior 是输出自相关对照；所有系数只在 train/val 选择。真实过去音频不得成为主推理输入。
- 在正式评估前冻结少量 primary contrasts、统计单位、多重比较和置信区间；同数据同评估尺度才比较，不直接把 SS02/03/05 不同协议分数并列称改进。
- 交付每框架真实指标、相对同协议基线的差异、连续历史增益、逐seed/逐人结果和失败限制；阴性结果同样完成任务。可行训练不得在准备报告/smoke处收工。

## 6. 输出、汇报、停机

代码/配置/轻量报告提交推送 PR #6 分支，不自行合并；执行前后保护主目录 CNN/TRF、stash 和其他 worktree。新报告目录 `reports/pub_01/s03/continuity/eeg_run01/`：EXECUTION_LOG、SPLIT_PROVENANCE、FORMAL_PROTOCOL、RESULTS、REPORT、REVIEW。大型数据、缓存、权重留2203。

启动后给协调会话回报具体入口、服务器目录/进程或调度标识、当前实际阶段；问题与结果向协调会话汇报，授权沿用用户先前“有问题/结果要向你汇报”。不要把内部准备报告反复推给用户。可解决工程问题自行推进；只把不可自行解决且需要用户行动的最小问题上报。

最终完成真实结果和代码/指标复核后停止；若没有可信新test，明确限定为训练/验证探索并列出阻碍，不把合成/验证分数装成独立测试。不得凭阴性自动新增模型、数据或下游任务。
