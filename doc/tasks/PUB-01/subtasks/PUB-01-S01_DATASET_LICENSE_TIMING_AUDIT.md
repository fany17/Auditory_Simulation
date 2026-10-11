# SUBTASK — PUB-01-S01 — Dataset / License / Timing Audit

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S01` |
| Parent Task | `PUB-01` |
| Title | `Dataset / License / Timing Audit` |
| Status | COMPLETED |
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

2026-09-19 用户确认启动本 S01，并指定所有数据集下载、续传、解压、文件读取核验及处理在 `server2203` 完成。先盘点 `/home/fanyu/auditory_simulation_m6a` 中已有数据与环境；不得在本地下载数据后再上传。官方文档/许可网页可在本地查阅，本地只保存轻量代码、配置、来源/许可/远端路径清单、汇总结果与报告；不回传数据载荷、音频/神经信号、权重或特征张量。连接或访问受阻时记录 HOLD/阻塞，不改为本地执行。

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

2026-09-19：完成服务器实际盘点、SparrKULee 官方 V3.1 清单/许可/公开访问阶段核验；协调者已复核并授权公开分支转S01_SS01。完整 S01 尚未完成：ds004703 已新读11录音头/首秒、270音频头和438事件块，初版多候选 HOLD 表保留，正在按实际刺激角色纠正汇总副本歧义。证据及失败记录见 `reports/pub_01/s01/PROGRESS.md`、`LICENSE_IDENTITY_GATE.md`。模型准入尚未通过；当前 primary 为S01_SS01，不把阶段门当成完整PASS。

## 子节点与当前状态（2026-09-24）
[SS01下载与初步清洗](PUB-01-S01/PUB-01-S01_SS01_DOWNLOAD_INITIAL_CLEANING.md)已COMPLETED；状态登记见父任务SUBTASKS.md的S01子节点表。上文09-19记录是历史状态。当前S01仍REVIEW，无正在执行的子节点；剩余工作是复用现有证据作出配对/时序、纳入排除及实验准入决定，不重复下载或全量QC。SS01以前置许可门为依赖，不依赖S01最终验收。

2026-09-24收尾登记：[SS02数据子集与实验准入](PUB-01-S01/PUB-01-S01_SS02_BENCHMARK_ADMISSION.md)为READY，尚未执行；集中处理配对/时间轴、异常去向、分组可行性与逐数据集准入。SS01既有证据直接复用。

## 用户阶段验收与后续安排

用户明确接受S01作为“公开数据准备、内容理解及初步探索”阶段完成。SS01已完成公开范围下载/读取核验，SS02已提交真实信号、PSD、音频/事件和尾差图。现将S01及SS02标记COMPLETED / PASS_WITH_LIMITATION，转入S02。

这是阶段范围与验收安排的显式修订，不是声称全部QC或模型准入已经解决。原准入报告中的640/26划分、ds004703 HOLD保留为历史判断，不再作为整体停止条件。仍有科学价值的问题随具体分析处理；3秒尾差提示不作为整条记录自动排除阈值。不能读取的对象跳过，不能配对的区间单独排除。受限载荷不请求，不读取私有患者/STN资料。

S02先完成SparrKULee声学基线：复用发布方64Hz EEG/envelope，按实际共同区间分析；最小必要清洗、对齐和分组检查与实现同时完成。拟合前固定首次实验范围、时间滞后、正则化候选、分组和指标；选择只使用训练/验证数据，测试集不参与。分别说明新故事和新被试的评估目标，不能笼统要求所有分组维度同时不交叉。

首个交付是可复现的ridge/TRF预测结果、时间错配对照、逐被试性能表和实际图文解释。先小规模跑通，再执行预定首轮范围；阴性或失败照实交付，不反复改方案追阳性。ds004703保留已有探索，不阻塞EEG首轮；其信号预处理在颅内分析时按实际需要处理。

完成第一版声学基线后汇报；这不等于S02全部完成。深度特征与S03以后不自动启动。所有数据计算继续在2203，报告可回传静态图与轻量结果。无需补修上一版图形排版或重做全量审计作为本轮前置条件。