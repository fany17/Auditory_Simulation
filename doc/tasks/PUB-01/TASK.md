# TASK — PUB-01 — Public Auditory–Neural Representation

| Field | Value |
|---|---|
| Task ID | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Class | `MAINLINE_EXECUTION` |
| Status | `READY` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `NO` for direct STN READ |
| Patient Data | `FORBIDDEN` |
| Upstream | historical public baselines/model bank |
| Inputs | SparrKULee, public intracranial dataset candidates, frozen audio models |
| Deliverables | leak-safe public benchmark + `ART-AUDREP-v1` candidate |
| Acceptance | public EEG and public intracranial audit/benchmark/transfer evidence completed regardless of transfer sign |
| Go/No-go | positive/no/negative transfer all valid; never blocks direct STN science |
| Downstream | `INT-01` optional comparator |
| Subtask Registry | `doc/tasks/PUB-01/SUBTASKS.md` |
| Last Updated | `2026-09-15` |

## 1. Scientific question

> 在完全不读取患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial 数据建立一个可审计、可测试迁移、可用于少样本比较的 auditory-neural representation？

本任务提供 public prior/comparator，不承担 STN 生物学 claim，也不要求 transfer 必须阳性。

## 2. Scope

Primary public EEG：SparrKULee。  
Primary public intracranial candidate：OpenNeuro `ds004703`，使用前必须完成 license/use-condition audit。

父任务覆盖：

- dataset/license/timing/pairing audit；
- interpretable acoustic/timing baseline；
- frozen wav2vec2/HuBERT layer bank；
- channel-agnostic neural encoder；
- encoding + contrastive/retrieval benchmark；
- held-out subject/session/story；
- EEG→public intracranial transfer；
- few-shot / negative-transfer characterization；
- `ART-AUDREP-v1` candidate packaging。

## 3. Non-goals / forbidden actions

- 不读取患者/STN 数据或患者派生 embedding；
- 不把 EEG→intracranial→STN 写成生物学层级；
- 不继续无限扩 pretrained model 名单；
- 不用 test set 选 layer/hyperparameter；
- 不因阴性反复改 split/window/negative sampling；
- 不把 shared geometry 解释为功能同源；
- 不把 public result 写成 stimulation/write-in evidence。

## 4. Inputs and Preconditions

任何正式训练/benchmark 前必须冻结：dataset source/version/license、audio-neural pairing、subject/session/story keys、sampling/channel/reference、timing/event metadata、bad/missing recordings 和 split policy。

患者数据始终 `FORBIDDEN`。

## 5. Execution Plan

父任务按 `SUBTASKS.md` 分解：

1. `PUB-01-S01` Dataset / License / Timing Audit；
2. `PUB-01-S02` Feature Bank and Leak-safe Baseline；
3. `PUB-01-S03` Public EEG Representation Benchmark；
4. `PUB-01-S04` Public Intracranial Transfer；
5. `PUB-01-S05` Artifact Candidate and Parent Integration Review。

任何一个 detail 若形成独立 deliverable/acceptance，按 `AGENTS.md` 分配新的 S##；不得直接堆入父任务。

## 6. Controls / leakage / confounds

至少阻断：相邻时间泄漏、同 story/clip 重复、test-subject normalization leakage、test-driven layer selection、recording identity shortcut、trivial negative sampling。

所有 normalization/PCA/feature/layer selection 只在 train fold 内。

## 7. Deliverables

父任务最终必须具备：

- auditable dataset/license manifests；
- frozen split manifest；
- acoustic/timing baseline；
- public EEG held-out benchmark；
- public intracranial transfer/few-shot evidence；
- negative-transfer/known-failures report；
- `ART-AUDREP-v1` candidate manifest/schema/runtime/canary/provenance。

## 8. Acceptance Criteria

1. license/timing/pairing 可审计；
2. split leak-safe；
3. interpretable baseline 可复现；
4. held-out subject/session/story 结果完整报告；
5. transfer 按预定义条件执行；
6. positive/no/negative transfer 全保留；
7. candidate artifact 记录 provenance、license 与 known failures；
8. patient/STN data boundary 未被突破。

## 9. Go / Hold / Fail

### GO

形成可由 `INT-01` 消费的 `ART-AUDREP-v1` candidate。

### HOLD

数据许可、audio-neural pairing 或 timing 不可审计时，相应分支停止等待。

### NO-GO

某 dataset/model 不支持可靠 benchmark时记录 failure，停止该分支；不自动新增替代模型追阳性。

`PUB-01` 的无迁移结果不构成 direct STN READ 的 no-go。

## 10. Known risks

held-out subject weak、EEG→intracranial negative transfer、layer instability、representation 只复现 envelope/onset、license 限制 artifact distribution。

## 11. Artifact / Handoff Contract

`ART-AUDREP-v1` 至少包含：semantic version、source task/subtasks/commit/config/seed、model/feature spec、preprocessing/timing、neural encoder（若许可）、runtime/transform、canary、benchmark/few-shot evidence、known failures、schema/validator、license/provenance manifest。

不得包含患者/STN 数据。

## 12. Amendment Record

`NONE`

## 13. Completion Record

`NOT_STARTED`
