# Auditory_Simulation Current Task

更新时间：`2026-09-15`

| Field | Value |
|---|---|
| Primary Task | `M6A-PUBLIC-004` |
| Program Node | `R1` |
| Track | `PUBLIC_METHOD` |
| Status | `READY` |
| Class | `MAINLINE_EXECUTION` |
| Patient Data | `FORBIDDEN` |
| Task Spec | `doc/tasks/M6A-PUBLIC-004.md` |
| Blocking | `NO` for STN direct READ `R2–R4` |

## What to execute next

`M6A-PUBLIC-004 — Auditory–Neural Representation Alignment and Transfer`

第一执行 gate：

1. SparrKULee + ds004703 access/license inventory；
2. audio↔neural pairing/timing audit；
3. leak-safe grouped split；
4. interpretable acoustic/onset baseline；
5. held-out benchmark skeleton。

只有 Stage 0/Audit 通过后才进入 neural encoder/contrastive training。

## Scientific role

本任务用于建立公共 auditory-neural prior 和 transfer benchmark。

它**不是**患者 STN 计算问题的必要前置：

- 如果 public pretraining 有 positive transfer，下游可作为 comparator/initialization；
- 如果无增益或 negative transfer，保留结果，患者侧 direct/simple model science 继续。

不得把 public EEG→sEEG 结果直接外推成 STN transfer 已成立。

## Other lane

`M6A-PUBLIC-003`：`REVIEW`。

003 只允许审核、completion record 和必要勘误；不新增 architecture 变体。

## Stop rule

完成本次 primary task 预定义 stage 后：

- 更新 `doc/tasks/M6A-PUBLIC-004.md` Completion Record；
- 更新 `doc/TASKS.md`；
- 更新本文件；
- 停止，不自动启动 M6B 或新 public task。