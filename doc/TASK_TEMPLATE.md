# TASK — <PUB-##> — <Title>

Canonical location:

```text
doc/tasks/<TASK-ID>/TASK.md
```

| Field | Value |
|---|---|
| Task ID | `<PUB-##>` |
| Title | `<semantic title>` |
| Track | `<PUBLIC_METHOD / SUPPORT / ...>` |
| Class | `<MAINLINE_EXECUTION / AUDIT / INTERFACE / ...>` |
| Status | `<BACKLOG / READY / ACTIVE / WAITING_EXTERNAL / BLOCKED / REVIEW / FROZEN / COMPLETED / CANCELLED / HISTORICAL_REFERENCE>` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `<YES/NO + downstream>` |
| Patient Data | `FORBIDDEN` |
| Upstream | `<task/artifact/NONE>` |
| Inputs | `<auditable inputs>` |
| Deliverables | `<parent-level outputs>` |
| Acceptance | `<parent-level decidable criteria>` |
| Go/No-go | `<positive/negative/blocked handling>` |
| Downstream | `<consumer task>` |
| Subtask Registry | `doc/tasks/<TASK-ID>/SUBTASKS.md` |
| Last Updated | `<YYYY-MM-DD>` |

## 1. Scientific / Operational Question

## 2. Scope

## 3. Non-goals / Forbidden Actions

## 4. Inputs and Preconditions

## 5. Execution Plan

父任务只写 work-package/stage 结构；独立可验收细节进入 `SUBTASKS.md` 和 subtask spec。

## 6. Controls / Leakage / Confounds

## 7. Deliverables

## 8. Acceptance Criteria

## 9. Go / Hold / Fail Rules

## 10. Known Risks and Negative Outcomes

## 11. Artifact / Handoff Contract

## 12. Amendment Record

`NONE`

改变 question/cohort/primary endpoint/data permission/Acceptance/Go-No-Go/artifact semantics 时必须登记 parent amendment，并重新审查受影响 S##。

## 13. Completion Record

`NOT_STARTED`

父任务完成必须经过 Integration Review；所有 blocking S## 完成/冻结或正式吸收 NO-GO 后，再重核 parent Acceptance。
