# TASK — <SEMANTIC-TASK-ID> — <Title>

| Field | Value |
|---|---|
| Task ID | `<SEMANTIC-TASK-ID>` |
| Pipeline Node | `<same as Task ID unless explicitly split>` |
| Track | `<PUBLIC_METHOD / SUPPORT / ...>` |
| Class | `<MAINLINE_EXECUTION / AUDIT / INTERFACE / ...>` |
| Status | `<BACKLOG / READY / ACTIVE / WAITING_EXTERNAL / BLOCKED / REVIEW / FROZEN / COMPLETED / CANCELLED / HISTORICAL_REFERENCE>` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `<YES/NO + downstream>` |
| Patient Data | `FORBIDDEN` |
| Upstream | `<task/artifact/NONE>` |
| Inputs | `<auditable inputs>` |
| Deliverables | `<paths/artifacts>` |
| Acceptance | `<decidable criteria>` |
| Go/No-go | `<positive/negative/blocked handling>` |
| Downstream | `<consumer/node>` |
| Last Updated | `<YYYY-MM-DD>` |

## Naming check

Task ID must:

- directly describe the work；
- contain no `M<number>` milestone；
- contain no `R0/P0/W0/S0` sequence number；
- contain no date/status/version；
- match filename `doc/tasks/<SEMANTIC-TASK-ID>.md`。

## 1. Scientific / Operational Question

## 2. Scope

## 3. Non-goals / Forbidden Actions

## 4. Inputs and Preconditions

## 5. Execution Plan

### Stage A

### Stage B

## 6. Controls / Leakage / Confounds

## 7. Deliverables

## 8. Acceptance Criteria

## 9. Go / Hold / Fail Rules

## 10. Known Risks and Negative Outcomes

## 11. Artifact / Handoff Contract

## 12. Completion Record

`NOT_STARTED`

> 阴性、no-transfer、negative-transfer 都可以正常完成任务；completion 不以阳性结果为条件。