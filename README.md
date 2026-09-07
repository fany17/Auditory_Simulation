# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前核心方向是建立声音表征、时间结构与不同层级神经活动之间可检验的计算关系，并形成可冻结、可迁移到患者侧的 auditory-neural representation artifact。

本项目不负责 PD 患者 STN 实验、临床采集、TTL/MR4 同步、DBS 装置、PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## 当前入口

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/TASKS.md`
4. `doc/CURRENT_TASK.md`
5. `doc/tasks/M6A-PUBLIC-004.md`

## 当前任务

### `M6A-PUBLIC-004：Auditory–Neural Representation Alignment and Transfer`

状态：`READY_TO_EXECUTE`

目标：在公开数据上建立可迁移 auditory-aligned neural representation：

```text
SparrKULee EEG
→ auditory-neural pretraining
→ public sEEG / ds004703 transfer
→ few-shot/domain-gap evaluation
→ ART-AUDREP-v1 candidate
→ STN_Decoding_Encoding consumer cross-test
```

第一轮重点：

1. dataset/timing/leakage audit；
2. acoustic + wav2vec2 + HuBERT layerwise baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject；
6. EEG→public sEEG transfer；
7. frozen artifact candidate。

禁止读取患者/STN 数据；禁止为了 004 再继续扩 pretrained model 名单。

## M6A-PUBLIC-003

`M6A-PUBLIC-003` 已完成执行，状态：`COMPLETED_EXECUTION_READY_FOR_REVIEW`。

45/45 formal runs 和 supplementary 10/20/50 ms probes 已完成。当前结果未支持 early/late downsampling、multiscale/RF growth 或 explicit change branch 在该 benchmark 上解决 localization/discrimination/generalization 问题。阴性结果保留，不继续追加结构变体追结果。

## Frozen baseline

`M6A-PUBLIC-002` 已完成 9/9 pretrained inference、8/9 unified temporal representation probe，覆盖：

- PANNs CNN14；
- ConvTasNet；
- SpeechBrain CRDNN；
- Parakeet-TDT / FastConformer；
- Audio-Mamba / SSAM；
- wav2vec2；
- Whisper；
- CoNNear periphery；
- ICNet。

002 作为 004 的 frozen model bank/reference，不继续补模型。

`M6A-PUBLIC-001` 保留 ds004703/wav2vec2 单被试单 recording preliminary checkpoint；004 将使用新的 held-out、cross-subject 和 transfer 设计建立正式证据。

## 目录边界

```text
Auditory_Simulation/
├── AGENTS.md
├── doc/
│   ├── PROJECT_BOUNDARY.md
│   ├── TASKS.md
│   ├── CURRENT_TASK.md
│   └── tasks/
├── configs/
├── environment/
├── schemas/
├── src/
├── scripts/
├── tests/
├── reports/
└── test/
```

大数据、模型权重、大体积特征和训练输出不进入 Git；计算资产留在 2203，Git 只保留轻量代码、配置、结构化结果和报告。

## 与其他仓库分工

- `Auditory_Simulation`：公开数据、公开模型、公共 EEG/sEEG、通用表征与 M6A。
- `AuditoryReading`：教材、综述、证据整理、学习与全项目非执行型总图。
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation、PINS/SEEG 写入验证与 M6B。

`Auditory_Simulation` 与 `STN_Decoding_Encoding` 仅通过冻结、版本化 artifact 衔接；患者数据默认不得进入本仓库。
