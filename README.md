# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前核心方向是建立声音表征、时间结构与不同层级神经活动之间可检验的计算关系，并研究结构变量如何影响时间信息的保留、转换和丢失。

本项目不负责 PD 患者 STN 实验、临床采集、TTL/MR4 同步、DBS 装置或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## 当前入口

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/TASKS.md`
4. `doc/CURRENT_TASK.md`
5. `doc/tasks/M6A-PUBLIC-003.md`

## 当前任务

### `M6A-PUBLIC-003：Temporal Architecture Perturbation`

状态：`ACTIVE_EXECUTION`

目标：在已完成 pretrained architecture baseline 后，不再继续堆模型，而是系统研究：

- temporal resolution / downsampling timing；
- multiscale receptive field；
- receptive-field growth schedule；
- RF/downsampling decoupling；
- explicit change/onset/event pathway。

第一轮只做：

1. Early vs Late Downsampling；
2. Receptive-Field Growth / Multiscale RF；
3. Explicit Change Branch。

允许训练小型 matched experimental models；禁止微调大型 pretrained backbone；禁止读取患者/STN 数据。

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

CoNNear 已完成 `waveform→BM→IHC→ANF-H/M/L` 六类 probe；ICNet 已完成 `waveform→bottleneck→units_1000` 六类 probe。002 作为 003 的 frozen reference，不继续补模型。

`M6A-PUBLIC-001` 保留 ds004703 / wav2vec2 单被试单 recording preliminary checkpoint，不在当前结构任务中扩展。

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

- `Auditory_Simulation`：公开数据、公开模型、结构实验、通用表征与 M6A。
- `AuditoryReading`：教材、综述、证据整理、学习与汇报知识资产。
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation 与 M6B。

`Auditory_Simulation` 与 `STN_Decoding_Encoding` 仅通过冻结、版本化 artifact 衔接；患者数据默认不得进入本仓库。
