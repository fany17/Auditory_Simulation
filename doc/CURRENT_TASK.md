# Auditory_Simulation Current Task

> 当前更新（2026-10-11）：S02既有a063/2203交付已核验，COMPLETED / PASS_WITH_LIMITATION（HuBERT HOLD）。本轮PUB-01-S03_SS01已COMPLETED / PASS_WITH_LIMITATION：10名93条，EEG→包络backward TRF test r=0.1819、R²=0.0178，13秒非循环null r=−0.0033；zero-lag r=0.0737、R²=−0.0098。同支持test、首被试rerun与独立review已通过。报告：reports/pub_01/s03/envelope_decode_20261011/REPORT.md；周报WEEKLY_REPORT.md。本轮计算已结束；S03和PUB-01保持ACTIVE，完整neural/retrieval/held-out-subject未完成，不启动S04。以下9月正文为历史，冲突状态以本段和SUBTASKS最新登记为准。

更新时间：`2026-10-11`

| Field | Value |
|---|---|
| Primary Task | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Status | `ACTIVE` |
| Patient Data | `FORBIDDEN` |
| Parent Task | `doc/tasks/PUB-01/TASK.md` |
| Subtask Registry | `doc/tasks/PUB-01/SUBTASKS.md` |
| Current executable Subtask | 本轮S03_SS01已完成并停止；S03整体ACTIVE，无正在运行的本轮计算 |
| Last completed Subtask | PUB-01-S03_SS01（2026-10-11 PASS_WITH_LIMITATION） |
| Subtask Spec | doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS01_ENVELOPE_DECODING.md |
| Blocking | `NO` for direct STN READ |

## Current result and next review

`PUB-01-S01_SS01 — Server Dataset Download and Initial Cleaning` 已完成并通过协调者独立验收（获准公开范围，PASS_WITH_LIMITATION）。用户本次server2203下载、初步清洗和文档归档goal已达到；S01完整配对/时序及模型准入仍为REVIEW/HOLD，父任务PUB-01整体尚未完成。

已完成技术、交付及父任务边界三层验证，详见 [最终独立验收](../reports/pub_01/COORDINATOR_REVIEW.md) 和 [执行报告](../reports/pub_01/s06/EXECUTION_AND_REVIEW.md)。

后续审阅入口是S01的物理同步、26个大尾差、来源版本/访问限制及grouped split；这些科学准入问题没有因下载完成而自动解决。

`PUB-01-S02` 保持BACKLOG，未启动模型训练。

## Scientific role

PUB-01 是 optional public prior/comparator；positive/no/negative transfer 都是合法结果，不阻塞 direct STN READ。

## Historical review lane

历史 temporal-architecture perturbation 仍为 `REVIEW`，但不是当前 primary task/subtask。

## Stop rule

本次数据准备执行和独立验收结束；不继续轮询下载或自动启动S02/INT-01。受限对象及科学限制保留在正式清单中，后续工作从新的明确任务或S01审阅决定接续。

## 当前执行修订（2026-09-24）
按用户最新要求，SS02先输出两套公开数据的实际波形、频谱、音频/事件和异常示例图及中文概览；正式准入的未决项不阻断描述性探索。旧Stop rule不阻止本轮已授权工作。S02训练尚未启动。

## 用户阶段验收与后续安排

用户明确接受S01作为“公开数据准备、内容理解及初步探索”阶段完成。SS01已完成公开范围下载/读取核验，SS02已提交真实信号、PSD、音频/事件和尾差图。现将S01及SS02标记COMPLETED / PASS_WITH_LIMITATION，转入S02。

这是阶段范围与验收安排的显式修订，不是声称全部QC或模型准入已经解决。原准入报告中的640/26划分、ds004703 HOLD保留为历史判断，不再作为整体停止条件。仍有科学价值的问题随具体分析处理；3秒尾差提示不作为整条记录自动排除阈值。不能读取的对象跳过，不能配对的区间单独排除。受限载荷不请求，不读取私有患者/STN资料。

S02先完成SparrKULee声学基线：复用发布方64Hz EEG/envelope，按实际共同区间分析；最小必要清洗、对齐和分组检查与实现同时完成。拟合前固定首次实验范围、时间滞后、正则化候选、分组和指标；选择只使用训练/验证数据，测试集不参与。分别说明新故事和新被试的评估目标，不能笼统要求所有分组维度同时不交叉。

首个交付是可复现的ridge/TRF预测结果、时间错配对照、逐被试性能表和实际图文解释。先小规模跑通，再执行预定首轮范围；阴性或失败照实交付，不反复改方案追阳性。ds004703保留已有探索，不阻塞EEG首轮；其信号预处理在颅内分析时按实际需要处理。

完成第一版声学基线后汇报；这不等于S02全部完成。深度特征与S03以后不自动启动。所有数据计算继续在2203，报告可回传静态图与轻量结果。无需补修上一版图形排版或重做全量审计作为本轮前置条件。
