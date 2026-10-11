# SUBTASK — PUB-01-S02 — Feature Bank and Leak-safe Baseline

> 2026-10-11状态核对：主目录旧正文保留provenance；S02当前为COMPLETED / PASS_WITH_LIMITATION。a063工作区 reports/pub_01/s02/acoustic_baseline_20260924 与 feature_compare_20250925 报告和2203对应真实输出已读取核验，83 subjects/659 records，方向是声音特征→EEG encoding。HuBERT identity/cache分支HOLD。不是EEG decoding或新被试泛化，详见SUBTASKS最新登记。本轮仅接续S03_SS01，不用整份较新文档覆盖主目录。

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S02` |
| Parent Task | `PUB-01` |
| Title | `Feature Bank and Leak-safe Baseline` |
| Status | `COMPLETED` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | PUB-01-S01（含SS01/SS02） |
| Inputs | accepted dataset manifest/split keys, approved frozen audio models |
| Deliverables | interpretable feature bank, frozen split manifest, ridge/TRF baseline |
| Acceptance | features reproducible, split leak-safe, baseline runs without test-driven selection |
| Validation | feature canary + split audit + baseline reproduction |
| Go/Hold/Fail | PASS unlocks S03; leakage or unreproducible features → REWORK/HOLD |
| Files Allowed to Change | public configs/src/scripts/reports + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-09-15` |

## 1. Objective
建立可解释声学/时间特征、冻结 SSL layer bank 和 leak-safe baseline。

## 2. Scope
至少包含 envelope/onset/log-mel or cochleagram、适用的 F0/spectral-flux/timing variables、wav2vec2/HuBERT layerwise frozen features，以及 ridge/TRF baseline。

## 3. Inputs / Preconditions
S01 对相应 dataset 为 PASS；S01_SS01 下载与初步清洗已验收；group split keys 已冻结。

## 4. Execution Steps
1. 冻结 feature extraction spec/version。
2. 生成可重建 acoustic/timing features。
3. 提取批准的 frozen SSL layers。
4. 固化 train/val/test grouped split。
5. 运行 ridge/TRF baseline 与 null。
6. 输出 canary/shape/timing/lag checks。

## 5. Controls / Failure Modes
所有 normalization/PCA/layer choice 只在 train fold；禁止相邻时间泄漏和 test-driven layer selection。

## 6. Deliverables
feature manifest、split manifest、baseline metrics/null、canary、known-failure report。

## 7. Acceptance Criteria
特征可复现；时间轴可解释；split 审计通过；baseline 与 null 完整；失败 layer/recording 保留。

## 8. Validation Procedure
Technical: canary/shape/timing/split tests。Acceptance: leak-safe baseline checklist。Parent consistency: 不增加未批准 backbone，不接触患者数据。

## 9. Completion Record
`NOT_STARTED`

## 用户阶段验收与后续安排

用户明确接受S01作为“公开数据准备、内容理解及初步探索”阶段完成。SS01已完成公开范围下载/读取核验，SS02已提交真实信号、PSD、音频/事件和尾差图。现将S01及SS02标记COMPLETED / PASS_WITH_LIMITATION，转入S02。

这是阶段范围与验收安排的显式修订，不是声称全部QC或模型准入已经解决。原准入报告中的640/26划分、ds004703 HOLD保留为历史判断，不再作为整体停止条件。仍有科学价值的问题随具体分析处理；3秒尾差提示不作为整条记录自动排除阈值。不能读取的对象跳过，不能配对的区间单独排除。受限载荷不请求，不读取私有患者/STN资料。

S02先完成SparrKULee声学基线：复用发布方64Hz EEG/envelope，按实际共同区间分析；最小必要清洗、对齐和分组检查与实现同时完成。拟合前固定首次实验范围、时间滞后、正则化候选、分组和指标；选择只使用训练/验证数据，测试集不参与。分别说明新故事和新被试的评估目标，不能笼统要求所有分组维度同时不交叉。

首个交付是可复现的ridge/TRF预测结果、时间错配对照、逐被试性能表和实际图文解释。先小规模跑通，再执行预定首轮范围；阴性或失败照实交付，不反复改方案追阳性。ds004703保留已有探索，不阻塞EEG首轮；其信号预处理在颅内分析时按实际需要处理。

完成第一版声学基线后汇报；这不等于S02全部完成。深度特征与S03以后不自动启动。所有数据计算继续在2203，报告可回传静态图与轻量结果。无需补修上一版图形排版或重做全量审计作为本轮前置条件。

## 2026-09-25 用户确认：继续S02特征比较

首轮声学基线已在执行worktree交付（83被试、659记录），不重复重跑。用户再次明确启动S02，现继续其既有特征bank与基线比较范围，不新增任务编号，不启动S03。

核心问题：在相同的被试内跨故事source-family评估条件下，声音特征能否预测EEG，冻结的深度音频特征相对简单声学特征是否增加预测能力？

执行顺序：
1. 对已有首轮结果做一次直接的代码/配置核对，重点是X/Y时间索引、同条件比较和train/validation/test隔离；只有发现影响结论的实际错误才另存修正版，不重开全库QC。
2. 复用现有划分和输入记录，补充log-mel声学特征；从已有批准的wav2vec2/HuBERT模型资产清单中固定精确checkpoint与版本，优先复用2203已有权重。不得扩展模型名单或以测试表现选择模型/层。
3. 把原始音频、发布方envelope及EEG的时间基准对应关系落实到提取配置；提取后只保留同一可配对区间，不能直接按数组索引混用不同采样率。深度特征保持冻结，降维/归一化和层选择只在训练/验证内进行。
4. 先用训练侧小样本评估资源，再按事先固定的首轮范围运行；如不能覆盖原83被试，预先确定子集并在同一子集/窗口重算所有比较条件，不以不同数据范围的平均分宣称增益。
5. 输出逐被试声学与深度特征预测性能、错配对照、配对差值与实际图文解释。保持原划分；明确正lag的定义、各特征时间分辨率和解释边界。无增益/负增益照实报告，不追阳性改划分。

许可和既有模型身份确实缺失时只停止该模型分支，其余已具备条件的特征分析继续，不编造具体checkpoint。所有载荷、特征和计算在2203，Windows仅轻量结果和静态图。不使用患者/STN数据，不新建会话，不提交/推送。完成此轮S02比较后汇报，不自动进入S03。
