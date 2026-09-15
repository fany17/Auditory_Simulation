# TASK — PUBLIC-REPRESENTATION — Auditory–Neural Representation Alignment and Transfer

| Field | Value |
|---|---|
| Task ID | `PUBLIC-REPRESENTATION` |
| Pipeline Node | `PUBLIC-REPRESENTATION` |
| Track | `PUBLIC_METHOD` |
| Class | `MAINLINE_EXECUTION` |
| Status | `READY` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `NO` for direct STN READ; optional input to `STN-READ-TRANSFER` |
| Patient Data | `FORBIDDEN` |
| Upstream | historical public model bank / frozen public baselines |
| Inputs | SparrKULee, public intracranial dataset candidates, frozen audio models |
| Deliverables | leak-safe benchmarks + `ART-AUDREP-v1` candidate |
| Acceptance | public EEG + public intracranial audit/benchmark/transfer evidence completed regardless of transfer sign |
| Go/No-go | positive/no/negative transfer all valid outcomes; never blocks direct STN science |
| Downstream | `STN-READ-TRANSFER` optional comparator |
| Last Updated | `2026-09-15` |

## 1. Scientific question

> 在不读取患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial 数据中学习一个对声音时间结构和事件信息敏感、跨被试/跨记录模态可测试迁移、且能以小样本适配新神经记录的 auditory-aligned representation？

本任务的作用是提供 public prior/comparator，不承担 STN 生物学 claim，也不要求 transfer 必须成功。

## 2. Scope

Primary EEG dataset：`SparrKULee`。

Primary public intracranial transfer candidate：`OpenNeuro ds004703`，使用前必须完成 license/use-condition audit。

第一轮包括：

- dataset/license/timing audit；
- interpretable acoustic/timing baseline；
- wav2vec2 / HuBERT layerwise features；
- channel-agnostic neural encoder；
- encoding + contrastive/retrieval benchmark；
- held-out subject/recording；
- public EEG→public intracranial transfer；
- few-shot learning curve；
- artifact candidate freeze。

## 3. Non-goals / forbidden actions

- 不读取患者/STN 数据；
- 不把 EEG→sEEG→STN 写成生物学层级；
- 不为了本任务继续扩 pretrained model 名单；
- 不在少量 public intracranial 数据上端到端训练超大 backbone；
- 不通过反复换 split/window/negative sampling 追阳性；
- 不把 alignment improvement 解释为脑区功能同源；
- 不把某模型层预先命名为“节律层”“STN 层”。

## 4. Inputs and preconditions

### Dataset audit

每个数据集必须冻结：

- source/version；
- license / use restrictions；
- subject/session/recording inventory；
- audio-neural pairing；
- sampling rate/channel/reference；
- timing/event metadata；
- speaker/story/clip grouping keys；
- known missing/bad recordings；
- train/validation/test split policy。

禁止 random adjacent-time split。

### Audio representation bank

至少：

1. envelope / onset / log-mel or cochleagram；
2. F0 / spectral flux / modulation features（适用时）；
3. wav2vec2 layerwise；
4. HuBERT layerwise；
5. explicit event interval / local rate / phase-deviation style timing regressors（可审计定义时）。

### Neural baselines

至少：

- ridge/TRF；
- simple temporal CNN/TCN；
- channel-agnostic neural encoder。

Generic EEG foundation model 只能作为 comparator/initialization ablation。

## 5. Execution plan

### Stage A — Dataset / license / timing audit

完成数据与许可 manifest、pairing audit、group split keys。

### Stage B — Audio feature baseline

构建可重建的 acoustic/timing features 与 frozen SSL layer bank。

### Stage C — Public EEG benchmark

至少报告：

- within-subject held-out clip/story；
- held-out subject；
- held-out recording/session；
- acoustic baseline；
- layerwise SSL；
- simple neural baselines；
- permutation/null。

### Stage D — Auditory-neural alignment

同时保留：

- encoding objective；
- contrastive/retrieval objective。

神经延迟显式处理，不假设 audio/neural 同时刻等价。

### Stage E — Public intracranial transfer

比较：

- from scratch；
- public EEG-pretrained frozen readout；
- small adapter；
- larger adaptation（仅作上界，若合理）。

报告 zero-shot、few-shot、negative transfer。

### Stage F — Artifact candidate

形成 `ART-AUDREP-v1` candidate，等待患者侧 consumer validation。

## 6. Controls / leakage / confounds

必须阻断：

- 相邻时间泄漏；
- 同 story/clip 在 train/test 重复；
- test subject normalization leakage；
- 用 test 结果选 layer；
- recording/block identity shortcut；
- negative samples 被 trivial metadata 区分。

所有 normalization、PCA、feature/layer selection 只在 training folds 内完成。

## 7. Deliverables

至少：

- dataset/license manifest；
- split manifest；
- acoustic/timing baseline table；
- layerwise public EEG benchmark；
- neural encoder benchmark；
- transfer/few-shot curves；
- negative-transfer table；
- known-failures report；
- `ART-AUDREP-v1` candidate manifest + schema + runtime/canary。

## 8. Acceptance criteria

任务完成不要求 transfer 阳性。

必须满足：

1. dataset/license/timing 可审计；
2. split leak-safe；
3. interpretable acoustic baseline 可复现；
4. held-out subject 结果完整报告；
5. public intracranial transfer 按预定条件执行；
6. positive/no/negative transfer 全保留；
7. few-shot/data-efficiency evidence 完整；
8. candidate artifact 包含 provenance、known failures 和 unsupported cases。

## 9. Go / Hold / Fail

### GO

形成可消费 `ART-AUDREP-v1` candidate，供 `STN-READ-TRANSFER` 测试。

### HOLD

若 license、audio-neural pairing 或 timing 无法审计，相关数据集停止，不以猜测补齐。

### NO-GO for a dataset/model

若某 public dataset/model 不支持可靠 benchmark，记录 failure 并停止该分支；不自动新增替代模型追阳性。

`PUBLIC-REPRESENTATION` 的失败永远不构成 `STN-READ-COMPUTATION` 的 no-go。

## 10. Known risks and negative outcomes

- held-out subject performance weak；
- scalp EEG→intracranial negative transfer；
- audio layer instability；
- channel-agnostic architecture 无优势；
- public representation 只复现 envelope/onset；
- license/use restrictions 限制 downstream artifact。

以上均为合法结果。

## 11. Artifact / handoff contract

`ART-AUDREP-v1` 至少包含：

- semantic version；
- source commit/config/seed；
- audio feature/model specification；
- neural encoder architecture/weights；
- preprocessing/timing specification；
- embedding dimensions/frame step；
- portable transform/runtime；
- canary；
- benchmark/few-shot evidence；
- known failures；
- schema/validator；
- license/use-condition manifest。

不得包含患者/STN 数据或患者派生 embedding。

## 12. Completion record

`NOT_STARTED`