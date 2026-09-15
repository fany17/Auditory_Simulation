# SUBTASK — PUB-01-S01 — Dataset / License / Timing Audit

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S01` |
| Parent Task | `PUB-01` |
| Title | `Dataset / License / Timing Audit` |
| Status | `READY` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `NONE` |
| Inputs | SparrKULee + candidate public intracranial metadata/docs/files |
| Deliverables | dataset/license manifest, pairing/timing audit, grouping keys, exclusion inventory |
| Acceptance | data identity, permission, audio-neural pairing, timing and split keys are auditable or explicit NO-GO |
| Validation | independent re-read of manifests + sample pairing/timing checks |
| Go/Hold/Fail | PASS unlocks S02; unresolved license/pairing keeps affected dataset HOLD |
| Files Allowed to Change | `doc/tasks/PUB-01/**`, public dataset inventory/config/report paths; no patient paths |
| Last Updated | `2026-09-15` |

## 1. Objective

冻结并审计 PUB-01 使用的数据版本、许可、样本身份、audio↔neural pairing、timing 与 grouped-split keys。

## 2. Scope

至少覆盖 SparrKULee 和第一 public intracranial candidate。只做 audit，不训练模型。

## 3. Inputs / Preconditions

使用官方 dataset/source metadata、实际可访问目录和许可文本。无法核验的字段保持 UNKNOWN。

## 4. Execution Steps

1. 冻结 dataset source/version/access method。
2. 记录 license/use/redistribution/derivative restrictions。
3. 建立 subject/session/story/recording/audio inventory。
4. 核验 audio-neural pairing、sample rate、event/timing fields。
5. 建立 train/test grouping keys 和 missing/bad recording list。
6. 生成 machine-readable manifest + human-readable audit report。

## 5. Controls / Failure Modes

禁止猜测许可；禁止把文件存在当作 pairing 正确；禁止 random adjacent-time split；发现 audio/timing 不可审计时明确 NO-GO。

## 6. Deliverables

- `dataset_manifest.*`
- `license_use_boundary.*`
- `audio_neural_pairing_audit.*`
- `timing_event_inventory.*`
- `group_split_keys.*`
- `exclusion_inventory.*`

## 7. Acceptance Criteria

每个 candidate dataset 必须得到 `PASS / HOLD / NO_GO`，并明确证据来源、可用范围和 unresolved fields。

## 8. Validation Procedure

### Level 1 — Technical
manifest/schema 可读；抽样 pairing/timing 可复查。

### Level 2 — Acceptance
逐项核对版本、许可、pairing、timing、group keys。

### Level 3 — Parent consistency
确认未接触患者数据，未提前训练，也未放宽 PUB-01 license/leakage 边界。

## 9. Completion Record

`NOT_STARTED`
