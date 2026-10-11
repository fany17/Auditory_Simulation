# SUBTASK — PUB-01-S03 — Public EEG Representation Benchmark

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | `Public EEG Representation Benchmark` |
| Status | `ACTIVE` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream Subtasks | `PUB-01-S02` |
| Inputs | frozen features/splits/baselines from S02 |
| Deliverables | held-out EEG encoding/retrieval benchmarks + neural encoder comparison |
| Acceptance | within-subject and held-out-subject evidence reported with predefined nulls and failures |
| Validation | rerun selected folds + leakage audit + metric consistency |
| Go/Hold/Fail | complete regardless of effect sign; output feeds S04/S05 |
| Files Allowed to Change | public model/config/report paths + `doc/tasks/PUB-01/**` |
| Last Updated | `2026-10-11` |

## 1. Objective
在公开 EEG 上建立可审计 auditory-neural representation benchmark，而不是只追求最高准确率。

## 2. Scope
至少比较 interpretable baseline、simple temporal neural model、channel-agnostic neural encoder；alignment 同时保留 encoding 与 contrastive/retrieval 视角。

## 3. Inputs / Preconditions
S02 PASS；split、feature、lag policy 已冻结。

## 4. Execution Steps
1. 固化 model conditions 和 seeds。
2. 运行 within-subject held-out clip/story。
3. 运行 held-out subject/session。
4. 运行 encoding/null 和 contrastive/retrieval/null。
5. 汇总 representation family/layer 稳定性与失败分布。
6. 输出 data-efficiency 基线，为 S04 transfer 做参照。

## 5. Controls / Failure Modes
禁止 test leakage、recording-ID shortcut、只报告 pooled result、只保留最佳 seed。

## 6. Deliverables
benchmark manifest、metrics tables、retrieval/encoding results、subject distribution、negative/failure table、review report。

## 7. Acceptance Criteria
所有预定义 split/metric/null 完成；held-out subject 结果完整；失败/阴性不删；结论限定在 public EEG。

## 8. Validation Procedure
Technical: rerun subset + metric/unit checks。Acceptance: predefined conditions completeness。Parent consistency: 不把 EEG 成功外推成 intracranial/STN。

## 9. Completion Record
`IN_PROGRESS`：2026-10-11 SS01已COMPLETED / PASS_WITH_LIMITATION，10名93条EEG→包络实际test/null及验证见reports/pub_01/s03/envelope_decode_20261011/REPORT.md。本轮backward TRF test r=0.1819、R²=0.0178，null r=−0.0033；zero-lag R²=−0.0098保留。首被试重现、独立指标及图表review通过。完整S03的neural model/retrieval/held-out-subject等仍未执行，保持ACTIVE；不自动进入S04。

## 2026-10-11 本轮初步解码授权与范围
统筹转达用户要求本周继续PUB并由对应项目交实际初步解码结果。本轮在S03内登记SS01，执行公开EEG→发布方声音包络的backward ridge/TRF与zero-lag基线；精确复用S02的被试内source-family划分，先冻结10名，不自动扩展。任务书见 `PUB-01-S03/PUB-01-S03_SS01_ENVELOPE_DECODING.md`。
这是现有public neural benchmark内的线性重建方向补充，沿用ridge/TRF方法及同一公开cohort；不替代S03原定的encoding/retrieval、neural model和held-out-subject验收，S03/PUB-01继续ACTIVE。只完成SS01后停止，不进入S04。
