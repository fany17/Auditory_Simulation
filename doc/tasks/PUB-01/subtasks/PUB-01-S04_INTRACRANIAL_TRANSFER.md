# SUBTASK — PUB-01-S04 — Public Intracranial Transfer

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S04` |
| Parent Task | `PUB-01` |
| Title | `Public Intracranial Transfer` |
| Status | `BACKLOG` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `PUB-01-S03` |
| Inputs | accepted public intracranial dataset from S01; frozen public EEG representation from S03 |
| Deliverables | zero/few-shot transfer benchmark, negative-transfer table, domain-gap report |
| Acceptance | from-scratch/frozen-readout/small-adapter conditions compared under fixed data budgets |
| Validation | fixed-budget rerun + split audit + negative-transfer retention |
| Go/Hold/Fail | positive/no/negative transfer all valid; no effect does not trigger architecture chase |
| Files Allowed to Change | public transfer configs/src/reports + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-09-15` |

## 1. Objective
检验 public EEG learned prior 在公开 intracranial data 上的可迁移性和失效边界。

## 2. Scope
至少比较 from-scratch、frozen readout、small adapter；larger adaptation 仅在预先批准时作为上界。

## 3. Inputs / Preconditions
S01 对 intracranial dataset PASS；S03 public EEG benchmark 已完成；数据预算和 split 冻结。

## 4. Execution Steps
1. 冻结 intracranial preprocessing/channel handling。
2. 建立 fixed-minutes/fixed-trials budgets。
3. 跑 from-scratch / frozen / adapter。
4. 计算 zero-shot/few-shot/data-efficiency。
5. 按 participant/session/contact distribution 汇总。
6. 记录 negative transfer/domain mismatch。

## 5. Controls / Failure Modes
禁止弱化 from-scratch comparator、test-driven adapter selection、把不同记录方式当生物学层级。

## 6. Deliverables
condition manifest、learning curves、transfer-gain table、negative-transfer table、domain-gap report、known failures。

## 7. Acceptance Criteria
所有预定义模型条件和预算完成；transfer sign 全报告；domain limitation 明确；不需要 transfer 阳性。

## 8. Validation Procedure
Technical: fixed-budget rerun/shape/split checks。Acceptance: comparator completeness。Parent consistency: 不把 public transfer 写成 STN mechanism。

## 9. Completion Record
`NOT_STARTED`
