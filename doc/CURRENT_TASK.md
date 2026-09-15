# Auditory_Simulation Current Task

更新时间：`2026-09-15`

| Field | Value |
|---|---|
| Primary Task | `PUBLIC-REPRESENTATION` |
| Pipeline Node | `PUBLIC-REPRESENTATION` |
| Track | `PUBLIC_METHOD` |
| Status | `READY` |
| Class | `MAINLINE_EXECUTION` |
| Patient Data | `FORBIDDEN` |
| Task Spec | `doc/tasks/PUBLIC-REPRESENTATION.md` |
| Blocking | `NO` for direct STN READ |

## What to execute next

第一执行 gate：

1. SparrKULee + ds004703 access/license inventory；
2. audio↔neural pairing/timing audit；
3. leak-safe grouped split；
4. interpretable acoustic/onset baseline；
5. held-out benchmark skeleton。

只有 dataset/license/timing audit 通过后，才进入 neural encoder / contrastive training。

## Scientific role

`PUBLIC-REPRESENTATION` 用于建立公共 auditory-neural prior 和 transfer benchmark。

它不是患者 STN 计算问题的必要前置：

- positive transfer：下游可作为 comparator/initialization；
- no transfer：保留无增益结果；
- negative transfer：保留 domain mismatch；
- direct/simple STN science 均继续。

不得把 public EEG→intracranial 结果直接外推成 STN transfer 已成立。

## Historical review lane

旧 temporal-architecture perturbation milestone 保持 `REVIEW`，只允许审核、completion record 和必要勘误；不新增 architecture 变体。

## Stop rule

完成本次 primary task 预定义 stage 后：

- 更新 `doc/tasks/PUBLIC-REPRESENTATION.md` Completion Record；
- 更新 `doc/TASKS.md`；
- 更新本文件；
- 停止，不自动启动患者侧任务或新 public task。