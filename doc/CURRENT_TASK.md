# Auditory_Simulation Current Task

更新时间：`2026-09-15`

| Field | Value |
|---|---|
| Primary Task | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Status | `READY` |
| Class | `MAINLINE_EXECUTION` |
| Patient Data | `FORBIDDEN` |
| Task Spec | `doc/tasks/PUB-01_PUBLIC_REPRESENTATION.md` |
| Blocking | `NO` for direct STN READ |

## What to execute next

1. SparrKULee + ds004703 access/license inventory；
2. audio↔neural pairing/timing audit；
3. leak-safe grouped split；
4. interpretable acoustic/onset baseline；
5. held-out benchmark skeleton。

只有 audit 通过后才进入 neural encoder / contrastive training。

## Scientific role

PUB-01 建立公共 auditory-neural prior 和 transfer benchmark。它是 optional accelerator/comparator，不是 STN READ 的必要前置。

- positive transfer：下游 `INT-01` 可作为 comparator/initialization；
- no transfer：保留无增益结果；
- negative transfer：保留 domain mismatch；
- direct/simple STN science 继续。

## Historical review lane

旧 temporal-architecture perturbation milestone 保持 `REVIEW`，只允许审核、completion record 和必要勘误。

## Stop rule

完成当前 stage 后更新 task Completion Record、`TASKS.md`、本文件，然后停止；不自动启动患者侧任务或新 public task。
