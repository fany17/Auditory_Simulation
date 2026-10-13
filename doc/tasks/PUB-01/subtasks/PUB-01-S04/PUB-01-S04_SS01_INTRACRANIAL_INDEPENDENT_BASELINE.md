# SUBTASK — PUB-01-S04_SS01 — ds004703 Independent Intracranial Direct Baseline

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S04_SS01` |
| Parent Node | `PUB-01-S04` |
| Parent Task | `PUB-01` |
| Title | ds004703 Independent Intracranial Direct Baseline |
| Status | `BACKLOG`（不作为 SparrKULee 主线当前执行） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `PUB-01-S01` ds004703 已授权公开范围/QC，独立数据/时间/许可准入 |
| Inputs | ds004703 公共 EDF/WAV、事件/刺激 pairing、现有319段 candidate virtual index、记录与通道 QC |
| Deliverables | ds004703 独立数据/split/target 定义、最简单 EEG-like neural→声音表征基线、错误/可用性报告 |
| Acceptance | 不依赖 EEG 已训练权重即可得到可复现 sEEG 基线或正式 not_estimable/HOLD；说明音频真实播放对应的限制 |
| Validation | 单位/时轴、样本对应、train/test leak、真正 held-out、独立复核 |
| Go/Hold/Fail | audio pairing / physical sync /许可与可用性不满足→HOLD，不硬配对、不偷偷改对照 |
| Files Allowed to Change | 公共 ds004703 任务/代码/config/轻量报告；既有 raw/virtual 索引只读 |
| Last Updated | 2026-10-13 |

## Goal

ds004703 是**另一条公开颅内验证支线**。先独立回答“在颅内电生理上，能否用已有的简单方法重建一个共同声音目标”，而不是直接把 EEG 训练权重贴到 sEEG 上就宣称 transfer。

已完成的 377 官方对象审计、11 EDF / 270 WAV、319 候选片段的虚拟索引属于先前数据准备阶段，不意味着可直接进入无条件训练。需要重新核查具体使用片段、真实刺激、时轴来源、排除范围与许可证，不能把 Catalan/control 或复制的音频误做独立刺激。

## Completion

`NOT_STARTED`；当前只有任务保留，没有新 sEEG 模型训练。
