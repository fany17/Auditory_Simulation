# SUBTASK — <SUBTASK-ID> — <Title>

| Field | Value |
|---|---|
| Subtask ID | `<PUB-01-S01>` |
| Parent Node | `<所属 TASK-ID-S##；主节点填 NONE>` |
| Parent Task | `<PUB-01>` |
| Title | `<semantic title>` |
| Status | `<BACKLOG / READY / ACTIVE / WAITING_EXTERNAL / BLOCKED / REVIEW / FROZEN / COMPLETED / CANCELLED>` |
| Blocking for Parent | `<YES / NO>` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `<IDs / NONE>` |
| Inputs | `<auditable inputs>` |
| Deliverables | `<explicit outputs>` |
| Acceptance | `<decidable criteria>` |
| Validation | `<technical + acceptance + parent consistency>` |
| Go/Hold/Fail | `<effect on parent>` |
| Files Allowed to Change | `<paths>` |
| Last Updated | `<YYYY-MM-DD>` |

## 1. Objective

## 2. Scope

## 3. Inputs / Preconditions

## 4. Execution Steps

## 5. Controls / Failure Modes

## 6. Deliverables

## 7. Acceptance Criteria

## 8. Validation Procedure

### Level 1 — Technical

### Level 2 — Acceptance

Decision: `PASS / PASS_WITH_LIMITATION / REWORK / HOLD / NO_GO`

### Level 3 — Parent consistency

确认未改变 parent scope、claim、patient-data boundary 或 acceptance。

## 9. Completion Record

`NOT_STARTED`

完成后记录 actual inputs/versions、code/config/command、changed files、outputs、validation、failures/deviations、decision、commit/branch、effect on parent、next action。

本模板同时用于 S## 和 S##_SS##。子节点 ID 示例：PUB-01-S01_SS01；任务书位于 subtasks/PUB-01-S01/，状态登记在父任务 SUBTASKS.md 的 S01 子节点表。
