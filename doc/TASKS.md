# Auditory_Simulation Task Index

## 当前最高优先任务

### `M6A-PUBLIC-004` — Auditory–Neural Representation Alignment and Transfer

状态：`READY_TO_EXECUTE`

入口：`doc/tasks/M6A-PUBLIC-004.md`

目标：使用公开 EEG（第一主数据：SparrKULee）和公开 intracranial 数据（第一迁移数据：ds004703）建立 auditory-aligned、channel-agnostic neural representation；完成 EEG→sEEG transfer、few-shot 学习曲线，并冻结可由 `STN_Decoding_Encoding` 消费的 `ART-AUDREP-v1` candidate。

第一执行序列：

1. dataset / timing / leakage audit；
2. acoustic + wav2vec2 + HuBERT layerwise baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval benchmark；
5. held-out subject；
6. public EEG→sEEG transfer；
7. frozen artifact candidate。

患者/STN 数据严格禁止进入本仓库。

## 已完成执行、等待审核

### `M6A-PUBLIC-003` — Temporal Architecture Perturbation

状态：`COMPLETED_EXECUTION_READY_FOR_REVIEW`

入口：`doc/tasks/M6A-PUBLIC-003.md`

45/45 formal runs 完成；补充 10/20/50 ms probe 已完成。当前直接结论是：本 benchmark 中 localization shift recovery、regular-vs-jitter discrimination 和 extrapolative jitter magnitude generalization 均失败；early/late downsampling、RF growth/multiscale 和 explicit change branch 未显示可信优势。阴性结果冻结，不继续通过增加结构变体追结果。

## 已完成/冻结任务

### `M6A-PUBLIC-002` — Pretrained Audio/Temporal Architecture Reproduction

状态：`COMPLETED_FOR_NEXT_STAGE`

已完成 9/9 pretrained inference、8/9 unified temporal representation probe。作为 004 的声音模型候选与历史 baseline，不继续补模型。

### `M6A-PUBLIC-001` — Public Audio→Brain Minimal Alignment

状态：`HISTORICAL_PRELIMINARY_CHECKPOINT`

保留 ds004703/wav2vec2 单被试单 recording preliminary；004 将以新的跨数据、held-out、transfer 设计重新执行公共 auditory-neural alignment，不把 001 当作正式证据。

## 项目边界

- `Auditory_Simulation`：公开数据、公开模型、通用 auditory-neural representation、M6A。
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation、PINS/SEEG 写入验证、M6B。
- 两者只通过冻结、版本化 artifact 衔接；患者数据默认不得进入本仓库。
- `AuditoryReading`：证据、教材、研究总图；不执行模型。
