# SUBTASK — PUB-01-S04 — ds004703 Independent Intracranial Benchmark and Transfer

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S04` |
| Parent Task | `PUB-01` |
| Title | ds004703 Public Intracranial Baseline and EEG Transfer |
| Status | `BACKLOG`（保留独立支线，本轮不启动） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | S01 对 ds004703 的已授权数据准备；transfer 另需 S03 冻结公共模型 |
| Inputs | 公开 ds004703 EDF/WAV/events、已有虚拟 QC/319 候选片段、S03 frozen EEG representation（仅 transfer 阶段） |
| Deliverables | sEEG 自身的声音特征直接重建基线；适配条件公平比较；zero/few-shot / negative transfer 与 domain-gap report |
| Acceptance | 自身的 intracranial baseline 可单独复现或诚实 HOLD；随后与 from-scratch/frozen/small-adapter 采用统一已定义目标和同等数据预算评估 |
| Validation | audio-event timing、recording/channel QC、split/representation/timebase 验证；独立 review |
| Go/Hold/Fail | 信息配对或许可证不足停止对应分析；无/负迁移为有效结果，不因阴性追架构 |
| Subnodes | S04_SS01 基础研究；S04_SS02 独立 transfer |
| Files Allowed to Change | 公共 sEEG 数据处理/基线/迁移代码、配置、轻量报告与相关任务文件；已有数据/derivative 只读 |
| Last Updated | 2026-10-13 |

## 1. 独立 B 支线定位

S04 对应 ds004703 **公开颅内电生理（sEEG）**。不是 STN 患者实验，更不是先验“EEG 迁移必然有效”。必须先在 ds004703 上独立评估相同定义的声音目标，再检验来自 EEG 的通用表示是否改善低数据量解码。

## 2. 两步且非循环的依赖

- `PUB-01-S04_SS01` — `BACKLOG`：只需数据集自身满足准入，即可独立建立 sEEG baseline；**不依赖 S03 完成或 SOTA**。
- `PUB-01-S04_SS02` — `BACKLOG`：**必须**在 SS01 独立基线可审查、S03 产生冻结模型/接口之后，再做 zero/few-shot, from-scratch/frozen/small-adapter 迁移对照。

因此 B 线的输入审计和独立基线不等待 A 线；**只有迁移验证依赖两条支线交汇**。

## 3. Current evidence / limits

S01 阶段已对账 ds004703 的 377 个公开对象，完成 11 EDF、270 WAV 及辅助对象格式检查，生成 319 个候选虚拟片段索引。上述为**初步数据准备/QC 结果**，不是严格同步确认、不是 sEEG 特征解码结果，必须遵守部分刺激、Catalan 及时间对应不确定性的排除和限制。

## 4. Controlled transfer

比较任务相关声学 target 能否在 EEG→sEEG 两种记录方式下被一致预测，至少含：目标数据零/少量训练、from-scratch、冻结 EEG prior/readout 和小 adapter。在预算公平下报告不确定性和所有无收益/负迁移。禁止以共享时序统计宣称脑区神经机制相同。

## 5. Completion

`NOT_STARTED`。本轮 2026-10-13 只确立并列支线及子任务，未训练 ds004703 模型、未执行 EEG→sEEG 迁移。
