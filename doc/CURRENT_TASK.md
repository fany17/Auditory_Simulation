# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 状态 | `READY_FOR_COORDINATOR_REVIEW` |
| 执行授权 | `AUTHORIZED_WITH_GATES` |
| 当前方向 | `M6A-PUBLIC：公开声音模型、公开神经数据与通用听觉计算方法` |
| 当前优先任务 | `M6A-PUBLIC-002` |
| 正式任务书 | `doc/tasks/M6A-PUBLIC-002.md` |
| 执行窗口 | `2026-08-13` 至 `2026-08-20` |
| 计算位置 | `server2203:/home/fanyu/auditory_simulation_m6a` |
| Git 边界 | 本地唯一 Git 工作区；远端计算目录不作为第二 Git 工作区 |

本轮 M6A-PUBLIC-002 已完成一次低成本 Stage A-D 复现收口：2203 上 CoNNear、ICNet、PANNs-CNN14、ConvTasNet、SpeechBrain-CRDNN 与 wav2vec2 synthetic inference/probe 有 PASS 证据，共 6/9；Whisper 与 Parakeet 保留阻塞证据，Audio-Mamba 未进入高成本依赖链。因此当前状态为 `M6A-PUBLIC-002_READY_FOR_COORDINATOR_REVIEW`，尚未达到任务书的 7/9 最低目标，不宣称本周 PASS。大模型、缓存、特征和日志仍只在 2203；本地仅保留轻量脚本与报告；不执行训练、微调、downstream probe、患者/STN 数据或 Git。

本轮续跑新增统一 synthetic temporal evidence：CoNNear 已逐级输出 `waveform→BM→IHC→ANF-H/M/L`，ICNet 已输出 `waveform→bottleneck→units_1000`；两者均已覆盖 `tone/regular_clicks/jitter_clicks/omission/phase_shift/speech` 六项输入。Parakeet 按官方 Transformers 路径做了一次 `local_files_only=true`、`trust_remote_code=false` 的六 probe forward 尝试，但现有缓存没有完整模型权重，保留 `BLOCKED_BY_UPSTREAM_DEPENDENCY`；Audio-Mamba 已机器化记录官方权重/依赖未就绪。当前仍为协调审核状态，不升格为 7/9 或任务 PASS。

## 当前决定

本周优先级已由单一 wav2vec2→iEEG encoding 扩展为“预训练声音/时序架构复现周”。目标不是设计新模型，而是先建立足够强的成熟 AI 与 auditory-physiology baseline。

本周硬规则：

- 正式执行对象必须有公开可下载 pretrained checkpoint / trained weights；
- `ZERO_TRAINING`：不从头训练、不微调、不继续预训练；
- 不训练 linear probe、ridge、classifier 或其他 downstream readout；
- 只做 inference、feature extraction、架构审计与无需拟合参数的 temporal representation analysis；
- 不读取、复制或分析 STN 患者数据；
- 本周禁止提出或实现新的 auditory-inspired architecture。

当前正式模型清单：

1. PANNs CNN14；
2. ConvTasNet；
3. SpeechBrain CRDNN；
4. NVIDIA Parakeet-TDT / FastConformer；
5. Audio Mamba / SSAM；
6. `facebook/wav2vec2-base`；
7. OpenAI Whisper turbo；
8. CoNNear periphery；
9. ICNet。

`LEAF` 因正式使用需要任务训练，不满足本周 `ZERO_TRAINING` 边界，降级为 registry-only 历史/候选条目。

## M6A-PUBLIC-002 当前任务

任务书：`doc/tasks/M6A-PUBLIC-002.md`

本周执行顺序固定为：

```text
model registry / checkpoint audit
→ official example smoke
→ unified temporal probes
→ architecture / temporal representation matrix
→ weekly report
```

统一 probe 只使用小型、可控输入：pure tone、regular clicks、jitter、omission、phase shift 和短自然语音。禁止为了本周任务下载大型 benchmark 或训练数据集。

CoNNear 的本周角色是复现人类 auditory periphery surrogate：

```text
sound
→ basilar-membrane vibration
→ IHC receptor potential
→ H/M/L ANF firing-rate representation
```

ICNet 的本周角色是复现正常听力沙鼠 inferior-colliculus population model：

```text
sound
→ shared bottleneck
→ simulated IC neural population
```

两者不能混写：CoNNear 代表外周逐级生理近似，ICNet 代表早期中枢神经编码近似；ICNet 不是人脑 IC 模型，CoNNear 附带的解析 CN/IC backend 也不等于 CoNNear 已学习到中枢听觉层级。

## M6A-PUBLIC-001 历史 checkpoint

`M6A-PUBLIC-001` 不删除、不覆盖，正式任务书仍为 `doc/tasks/M6A-PUBLIC-001.md`。

其当前冻结节点为：

`G4_MINIMAL_SD012_SES02_PRELIMINARY_COMPLETE_AWAITING_COORDINATOR_REVIEW`

已完成同一 `sub-SD012_ses-02_task-PassiveListen` recording 的 40 passages（24/8/8）、36 electrodes、wav2vec2/acoustic layer-wise 最小 G4 preliminary。描述性结果中 wav2vec2 Transformer 09 的最佳中位 test Pearson r 为 0.0387（lag 0.20 s），20-null mechanical p=0.1905；各 feature variant 描述性最佳 lag 的中位 R2 均为负。该结果仍只属于单被试单 recording preliminary，不提供稳定显著性、脑区或泛化结论。

本周不扩大 M6A-PUBLIC-001：

- 不新增 subject / recording；
- 不新增声音模型进入神经拟合；
- 不继续 ridge/null 扩展；
- 不生成新的 M6A→M6B exchange candidate。

已有数据、split、preprocessing、G1–G4 工程审计和报告继续保留，不回滚、不重解释。

## 与 STN_Decoding_Encoding 的接口

跨项目边界不变：

- `Auditory_Simulation` 只处理公开声音、公开模型、公开神经数据和非临床计算方法；
- `STN_Decoding_Encoding` 独立处理患者实验、STN 数据和 STN-specific adaptation；
- 两边只通过冻结、版本化 artifact 接口衔接；
- 患者数据默认不得进入 `Auditory_Simulation`。

现有 M6A→M6B revised DRAFT 状态继续保持：

- `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION`；
- consumer：`READY_WAITING_M6A_CANDIDATE`；
- `CONSUMER_CROSS_TEST=NOT_RUN`；
- contract 仍非 accepted/frozen；
- 本周模型复现任务不自动生成 exchange candidate，也不解锁患者适配。

## 本周 PASS 标准

- 9/9 pretrained source 完成审计；
- 至少 7/9 官方 inference 跑通；
- CoNNear 与 ICNet 两条 auditory-physiology 主线优先跑通；
- 至少 7/9 完成可执行的统一 temporal probe；
- `0` 次训练；
- `0` 次微调；
- `0` 次患者数据读取；
- 输出模型 registry、smoke summary、temporal probe summary、architecture comparison matrix、CoNNear/ICNet 两份复现报告和一份 weekly report。

## 硬停止条件

- 下载接口要求账户本人点击同意、签署新条款、提供个人凭据或付费；
- checkpoint 无法合法公开下载；
- 单个非核心模型因 upstream dependency 占用过高修复成本；
- 预计新增存储超过 500 GB、连续 GPU 运行超过 24 小时或需要重大不可逆操作；
- 任务需要训练、微调、继续预训练或训练 downstream probe；
- 任务需要读取 STN 患者数据；
- 任务试图直接进入新 auditory-inspired architecture 设计。

单个非核心模型被依赖阻断时，记录 `BLOCKED_BY_UPSTREAM_DEPENDENCY` 并继续其他模型，不为了 9/9 耗尽整周。
