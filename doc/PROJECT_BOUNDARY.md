# Auditory_Simulation 项目边界

| 字段 | 内容 |
|---|---|
| 文档性质 | 本项目独立边界；不是跨项目总纲，也不是执行任务书 |
| 版本 | `v1.0` |
| 生效日期 | 2026-08-11 |
| 状态 | `ACTIVE` |
| 适用目录 | `Auditory_Simulation/` |
| 当前任务状态 | 见 `doc/CURRENT_TASK.md` |

## 1. 项目定位

`Auditory_Simulation` 是面向公开声音、公开神经数据和计算仿真的通用方法项目。其核心目标是：

> 建立声音信息与不同层级神经活动之间的可检验表征对齐方法，研究声音中的声学、音位/语音、语言/语义、情绪、音乐和运动相关信息如何在模型与神经活动中被保留、转换和丢失，并为未来人工写入提出可验证的低维候选变量。

本项目首先建设可复用的公开数据与算法基线，不直接承担患者实验、临床同步或 DBS 操作。

## 2. 核心概念

本项目比较的是模型和神经数据的动态表征，不比较异构模型原始参数：

```text
Audio encoder:
    audio(t) -> H_audio(layer, t)

Neural encoder:
    neural(t) -> H_neural(t)

Alignment:
    H_audio(layer, t) <-> H_neural(t)
```

模型参数会受到隐藏维度旋转、神经元排列和等价重参数化影响，通常不存在可解释的逐参数对应关系。因此：

- `Audio -> Brain encoding`：检验某层声音表征能解释多少 held-out 神经活动；
- `Brain -> Audio alignment/decoding`：检验神经活动能否恢复或检索对应的声音 latent；
- `Shared geometry`：使用 CCA、RSA、CKA、contrastive learning 等比较共享几何；
- 任何相关或可预测结果均不自动等同于脑区与模型层功能同一，更不等同于因果机制。

## 3. 本项目负责的工作

### 3.1 声音表征基线

负责冻结、提取和比较公开预训练声音模型的逐层表示，包括：

- 第一阶段语音模型：wav2vec 2.0、HuBERT、WavLM；
- 后续情绪表示：emotion2vec；
- 后续通用声音表示：BEATs；
- 后续声音—文本共享空间：CLAP；
- 后续音乐表示：MERT；
- 传统可解释特征：waveform、envelope、spectrogram、onset、pitch、tempo、phoneme/word timing。

第一阶段优先使用冻结模型建立可审计基线，不把重新预训练大型模型设为进入神经对齐的前置条件。微调、继续预训练或自建基座只有在独立任务书批准后才允许执行。

### 3.2 公开神经数据对齐

负责公开或单独授权数据集上的通用对齐方法，包括：

- 数据、音频、事件、受试者、电极/脑区与许可证清单；
- 音频—神经时间窗对齐；
- 防止 clip、story、speaker、subject 和相邻时间窗泄漏的 grouped split；
- ridge/regularized encoding baseline；
- neural encoder 与 contrastive alignment；
- CCA、RSA、CKA 和 layer-wise comparison；
- 模态、脑区、电极、模型层与信息类型的证据矩阵；
- 阴性结果、失败条件、计算资源和不确定度。

首个候选数据集为 OpenNeuro `ds004703`。当前核验信息为 10 名参与者、11 次 iEEG 记录、约 9.1 小时、自然对话语音被动聆听、CC0。正式下载前仍须冻结 OpenNeuro 版本、文件清单、实际音频与事件可用性、脑区覆盖、许可和防泄漏设计。

后续候选按问题分层，而不是一次性全部下载：

1. Bellier/Pink Floyd ECoG：音乐、声学和节律层级；
2. Narratives `ds002345`：自然故事、词/音位时间标注与语义层 benchmark；
3. JapanEEG `ds007808`：大规模 EEG、面部 EMG 与语音音频；采用前必须核验正式 OpenNeuro 版本、参与者/任务构成和许可；
4. 其他 EEG、MEG、fMRI、iEEG/ECoG 数据：只有在明确问题、版本和许可后加入。

### 3.3 计算神经仿真

负责非临床的声音到神经状态近似：

```text
sound
-> peripheral/auditory approximation
-> firing rate / spike timing
-> population state
-> LFP or macroscopic proxy
-> decoder / simulated intervention
```

这里的 LFP 只能称为 `LFP proxy`，必须具有明确 forward assumption；它不是患者 STN-LFP，也不能被写成临床刺激系统。

## 4. 不属于本项目的工作

以下内容归 `stn-rhythm-pilot` 或另行批准的临床项目：

- PD 患者实验和患者界面；
- STN-LFP 原始数据、EAS、artifact mask、真实 derivatives 和个案结论；
- PC->MR4->LFP 同步、TTL、DBS_OFF anchor 和设备驱动；
- 患者刺激 profile、正式 schedule、伦理、安全和停止规则；
- 患者音乐范式、熟悉度/喜爱度和真实步态验证；
- 人工 DBS 刺激参数、治疗建议或临床有效性；
- STN 项目当前 M5.5 的继续分析或结果改写。

本项目不得直接读取或复制 STN 患者数据。若未来需要去标识衍生数据，必须由单独的伦理、隐私、授权和传输任务明确批准。

## 5. 与 stn-rhythm-pilot 的接口

两个项目保持独立，不共享可变工作目录、当前任务书、数据目录、结果目录或总文档。

`Auditory_Simulation` 可以向 STN 项目输出冻结、可版本化的算法产物：

- 模型 ID、revision、许可证和权重/配置 manifest；
- 音频预处理和逐层 feature schema；
- 经过公开数据验证的 alignment baseline；
- split、null、指标和失败门槛；
- 可安装代码版本或不可变源码快照；
- 明确的适用范围与不能支持的结论。

STN 项目只能导入一个明确版本，并在自己的任务书中记录版本和适配层。STN 不在患者仓库中重复建设通用公开数据训练平台。

反向接口默认只允许：

- 公开的刺激定义、去身份化 schema 或方法需求；
- STN 项目明确发布的非敏感接口；
- 不包含患者原始信号和未授权 derivative 的问题定义。

## 6. M6 拆分

“M6：Auditory-Neural Representation Alignment”作为科学方向拆成两个互不混写的实施节点：

- `M6A-PUBLIC`：位于 `Auditory_Simulation`，负责公开数据、声音模型和通用对齐方法；
- `M6B-STN`：未来位于 `stn-rhythm-pilot`，只负责把冻结的 M6A 产物适配到 STN 数据。

`M6A-PUBLIC` 不因本文生效而自动开始；`M6B-STN` 依赖 M5.5 研究者审阅、独立任务批准和可接受的 M6A artifact。

## 7. 第一阶段建议

首轮正式任务应保持最小：

1. 只冻结 `ds004703` 的版本、许可、结构、音频/事件和可用脑区；
2. 只选择一个主模型家族，建议先用冻结 wav2vec 2.0，并把 HuBERT/WavLM 留作预注册对照；
3. 先完成 acoustic baseline 与逐层 ridge encoding；
4. 再增加 neural->audio latent retrieval 或 contrastive alignment；
5. 以 story/clip/subject 分组，禁止相邻时间点随机切分；
6. 输出 `brain region/electrode x audio layer x information type` 证据表；
7. emotion2vec、BEATs、CLAP、MERT 和其他数据集均由后续任务逐项加入。

## 8. 证据与停止规则

每项正式结果必须回答：

- observed、matched control 和 null 是什么；
- split 是否阻断 stimulus、subject、story 和时间邻接泄漏；
- audio feature 是否冻结且可重建；
- encoding 与 decoding/alignment 是否在 held-out 数据上验证；
- 不同被试、脑区、电极和模型层是否稳定；
- 结果支持“可预测”“共享几何”还是仅“相关”；
- 哪些结果为 `supported / provisional / inconclusive / not_supported / not_estimable`。

出现下列情况应停止或重新设计：

- 数据许可、版本、音频或事件关系无法冻结；
- 训练/测试存在 stimulus 或时间泄漏；
- 音频与神经时间轴不能审计；
- 结果只来自 pooled 数据且个体/脑区方向不一致；
- 必须使用患者数据才能继续但尚无授权；
- 需要把模型层直接命名为某脑区才能维持结论。

## 9. 文档治理

本项目不再维护跨路线、跨项目的“总纲”。当前权威链为：

1. 用户当前明确指令与 `AGENTS.md` 安全边界；
2. 本文件 `doc/PROJECT_BOUNDARY.md`；
3. `doc/CURRENT_TASK.md`；
4. 经人工批准的具体任务书；
5. 实际代码、配置、数据清单、日志、指标、图和审核证据。

旧 `01_听觉时变信息处理项目总纲.md`、`PROJECT_CHARTER.md` 和 `test/` 文档只作历史与设计来源，不再定义当前项目目标或任务优先级。

## 10. 已核验的外部依据

- Défossez et al., 2023：以 contrastive learning 对齐 brain module 与预训练 wav2vec 2.0 speech latent；4 个公开 MEG/EEG 数据集、175 名参与者。<https://www.nature.com/articles/s42256-023-00714-5>
- OpenNeuro `ds004703` 数据集标识：<https://doi.org/10.18112/openneuro.ds004703.v1.1.0>
- Bellier/Pink Floyd ECoG 数据：<https://zenodo.org/records/7876019>
- Narratives 数据与论文：<https://openneuro.org/datasets/ds002345>；<https://pmc.ncbi.nlm.nih.gov/articles/PMC8479122/>
- emotion2vec：<https://aclanthology.org/2024.findings-acl.931/>
- BEATs：<https://arxiv.org/abs/2212.09058>
- CLAP：<https://arxiv.org/abs/2206.04769>
- MERT：<https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html>
- JapanEEG 候选说明：<https://arxiv.org/abs/2606.01264>
