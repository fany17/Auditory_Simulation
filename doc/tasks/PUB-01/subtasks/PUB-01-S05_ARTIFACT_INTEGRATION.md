# SUBTASK — PUB-01-S05 — Artifact Candidate and Integration Review

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S05` |
| Parent Task | `PUB-01` |
| Title | `Artifact Candidate and Integration Review` |
| Status | `BACKLOG` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `PUB-01-S01`–`PUB-01-S04` |
| Inputs | accepted manifests, benchmarks, transfer results, known failures |
| Deliverables | `ART-AUDREP-v1` candidate + parent integration review |
| Acceptance | candidate is self-describing, reproducible, licensed, failure-aware, and parent acceptance rechecked |
| Validation | schema/runtime/canary + provenance audit + parent-level integration review |
| Go/Hold/Fail | PASS releases candidate to `INT-01`; HOLD/REWORK stays in PUB-01 |
| Files Allowed to Change | artifact/schema/report docs + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-09-15` |

## 1. Objective
把 S01–S04 的结果收敛为一个可由患者侧消费、但不包含患者信息的 public artifact candidate，并完成 PUB-01 integration review。

## 2. Scope
只做 candidate packaging、validation、known-failure/provenance 汇总和父任务验收，不运行新的 exploratory science。

## 3. Inputs / Preconditions
所有 blocking subtasks 已 COMPLETED/FROZEN，或其 NO-GO 已由父任务正式吸收。

## 4. Execution Steps
1. 冻结 artifact manifest/schema/version。
2. 打包 preprocessing/feature/runtime/model specs。
3. 建立 canary/validator。
4. 汇总 EEG/intracranial benchmark、few-shot 和 negative transfer。
5. 汇总 license/provenance/known failures/unsupported cases。
6. 独立重审 PUB-01 parent Acceptance。
7. 生成 consumer handoff note。

## 5. Controls / Failure Modes
禁止遗漏阴性结果；禁止通过患者侧需求反向修改 artifact 内容；许可不支持分发时必须明确限制。

## 6. Deliverables
`ART-AUDREP-v1` candidate manifest/schema/runtime/canary、benchmark summary、known failures、license/provenance、integration review、handoff note。

## 7. Acceptance Criteria
candidate 能由独立环境按规定入口运行 canary；语义/维度/timing 明确；benchmark 与失败边界完整；父任务 Acceptance 全部重核。

## 8. Validation Procedure
Technical: schema/runtime/canary。Acceptance: artifact completeness。Parent consistency: 对照 PUB-01 TASK.md 全部 Acceptance 做 integration review。

## 9. Completion Record
`NOT_STARTED`
