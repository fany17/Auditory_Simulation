# Auditory_Simulation Task Index

## 当前任务

### `M6A-PUBLIC-003` — Temporal Architecture Perturbation

状态：`ACTIVE_EXECUTION`

入口：`doc/tasks/M6A-PUBLIC-003.md`

目标：在既有 pretrained baseline 完成后，系统研究 temporal resolution、downsampling timing、multiscale receptive field、receptive-field growth schedule、RF/downsampling decoupling 与 explicit change/event pathway 的结构因果效应。

第一轮只做：

1. Early vs Late Downsampling；
2. Receptive-Field Growth / Multiscale RF；
3. Explicit Change Branch。

允许训练小型 matched experimental models；禁止微调大型 pretrained backbone；禁止患者/STN 数据。

## 已完成/冻结任务

### `M6A-PUBLIC-002` — Pretrained Audio/Temporal Architecture Reproduction

状态：`COMPLETED_FOR_NEXT_STAGE`

任务书：`doc/tasks/M6A-PUBLIC-002.md`

已完成：9/9 pretrained inference；8/9 unified temporal representation probe；覆盖 PANNs CNN14、ConvTasNet、SpeechBrain CRDNN、Parakeet-TDT/FastConformer、Audio-Mamba/SSAM、wav2vec2、Whisper、CoNNear periphery、ICNet。

该任务结果作为 003 的 frozen baseline，不继续补模型。

### `M6A-PUBLIC-001` — Public Audio→Brain Minimal Alignment

状态：`HISTORICAL_PRELIMINARY_CHECKPOINT`

任务书：`doc/tasks/M6A-PUBLIC-001.md`

保留 ds004703/wav2vec2 单被试单 recording preliminary，不在 003 中扩展。

## 项目边界

- `Auditory_Simulation`：公开数据、公开模型、声音/时序计算、通用方法、M6A。
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation、M6B。
- 只通过冻结、版本化 artifact 衔接；患者数据默认不得进入本仓库。
