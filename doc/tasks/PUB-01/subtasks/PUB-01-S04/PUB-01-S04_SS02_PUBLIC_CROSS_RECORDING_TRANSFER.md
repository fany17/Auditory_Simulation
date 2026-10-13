# SUBTASK — PUB-01-S04_SS02 — EEG→Public Intracranial Transfer

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S04_SS02` |
| Parent Node | `PUB-01-S04` |
| Parent Task | `PUB-01` |
| Title | Public EEG→Intracranial Transfer with Fixed Data Budgets |
| Status | `BACKLOG` |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `PUB-01-S03` 可复用的 frozen EEG representation + `PUB-01-S04_SS01` 独立颅内 baseline |
| Inputs | accepted public model + same-defined audio target + validated public intracranial data |
| Deliverables | from-scratch / frozen-readout / small-adapter 同数据预算迁移表；正/无/负迁移与 domain-gap |
| Acceptance | 迁移增益符号明确、对照公平、独立测试数据未参与 adapter/layer 选择；阴性可以验收 |
| Validation | feature/shape/timing/contract canary、分割检查、预算控制、独立复核 |
| Go/Hold/Fail | 当目标或脑区/电极覆盖不可比较，报告不可比较而不是强行对齐 |
| Files Allowed to Change | 公共跨记录方式实验代码/config/轻量报告、任务文档 |
| Last Updated | 2026-10-13 |

## Goal

独立 S04_SS01 与 EEG 主线 S03 都有可验证输出之后，比较同一声音任务上 EEG public prior 对 sEEG 的迁移价值。**不是临床 STN 适配**，不是脑区功能同源证明。

固定目标数据量、模型参数/冻结层、train/val/test，与 from-scratch 等价预算进行对照；报告 negative transfer。患者侧 INT-01 将来只接受 PUB-01-S05 生成的冻结、带许可限制的版本化 artifact。

## Completion

`NOT_STARTED`。
