# Auditory_Simulation Current Task

更新时间：`2026-09-15`

| Field | Value |
|---|---|
| Primary Task | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Status | `READY` |
| Patient Data | `FORBIDDEN` |
| Parent Task | `doc/tasks/PUB-01/TASK.md` |
| Subtask Registry | `doc/tasks/PUB-01/SUBTASKS.md` |
| Current executable Subtask | `PUB-01-S01` |
| Subtask Spec | `doc/tasks/PUB-01/subtasks/PUB-01-S01_DATASET_LICENSE_TIMING_AUDIT.md` |
| Blocking | `NO` for direct STN READ |

## Execute next

只执行 `PUB-01-S01 — Dataset / License / Timing Audit`。

完成后必须进行：

1. technical validation；
2. acceptance validation；
3. parent-consistency validation；
4. 更新 S01 Completion Record；
5. 更新 `PUB-01/SUBTASKS.md`。

只有 S01 被接受后，`PUB-01-S02` 才允许进入 READY。

## Scientific role

PUB-01 是 optional public prior/comparator；positive/no/negative transfer 都是合法结果，不阻塞 direct STN READ。

## Historical review lane

历史 temporal-architecture perturbation 仍为 `REVIEW`，但不是当前 primary task/subtask。

## Stop rule

完成当前 S01 后停止，不自动启动 S02 或患者侧 `INT-01`。
