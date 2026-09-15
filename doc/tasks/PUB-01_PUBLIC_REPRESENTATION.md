# TASK — PUB-01 — Public Auditory–Neural Representation

| Field | Value |
|---|---|
| Task ID | `PUB-01` |
| Title | `Public Auditory–Neural Representation` |
| Track | `PUBLIC_METHOD` |
| Class | `MAINLINE_EXECUTION` |
| Status | `READY` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `NO` for direct STN READ; optional input to `INT-01` |
| Patient Data | `FORBIDDEN` |
| Upstream | historical public model bank / frozen public baselines |
| Inputs | SparrKULee, public intracranial dataset candidates, frozen audio models |
| Deliverables | leak-safe benchmarks + `ART-AUDREP-v1` candidate |
| Acceptance | public EEG + public intracranial audit/benchmark/transfer evidence completed regardless of transfer sign |
| Go/No-go | positive/no/negative transfer all valid outcomes; never blocks direct STN science |
| Downstream | `INT-01` optional comparator |
| Last Updated | `2026-09-15` |

## 1. Scientific question

> 在不读取患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial 数据中学习一个对声音时间结构和事件信息敏感、跨被试/跨记录模态可测试迁移、且能以小样本适配新神经记录的 auditory-aligned representation？

本任务提供 public prior/comparator，不承担 STN 生物学 claim，也不要求 transfer 必须成功。

## 2. Scope

Primary EEG dataset：`SparrKULee`。  
Primary public intracranial transfer candidate：`OpenNeuro ds004703`，使用前必须完成 license/use-condition audit。

第一轮包括：dataset/license/timing audit、interpretable acoustic/timing baseline、wav2vec2/HuBERT layerwise features、channel-agnostic neural encoder、encoding + contrastive/retrieval benchmark、held-out subject/recording、public EEG→intracranial transfer、few-shot learning curve、artifact candidate freeze。

## 3. Non-goals / forbidden actions

- 不读取患者/STN 数据；
- 不把 EEG→sEEG→STN 写成生物学层级；
- 不继续扩 pretrained model 名单；
- 不在少量 public intracranial 数据上端到端训练超大 backbone；
- 不反复换 split/window/negative sampling 追阳性；
- 不把 alignment improvement 解释为脑区功能同源；
- 不把某模型层先验命名为“节律层/STN 层”。

## 4. Inputs and preconditions

每个数据集必须冻结：source/version、license/use restrictions、subject/session/recording inventory、audio-neural pairing、sampling/channel/reference、timing/event metadata、speaker/story/clip grouping keys、known missing/bad recordings、train/validation/test split policy。

禁止 random adjacent-time split。

至少保留 envelope/onset/log-mel or cochleagram、F0/spectral flux/modulation（适用时）、wav2vec2/HuBERT layerwise、explicit timing regressors。

Neural baselines 至少包括 ridge/TRF、simple temporal CNN/TCN、channel-agnostic neural encoder。

## 5. Execution plan

### Stage A — Dataset / license / timing audit
完成数据与许可 manifest、pairing audit、group split keys。

### Stage B — Audio feature baseline
构建可重建 acoustic/timing features 与 frozen SSL layer bank。

### Stage C — Public EEG benchmark
报告 within-subject held-out clip/story、held-out subject、held-out recording/session、acoustic baseline、layerwise SSL、simple neural baselines、permutation/null。

### Stage D — Auditory-neural alignment
同时保留 encoding objective 与 contrastive/retrieval objective；神经延迟显式处理。

### Stage E — Public intracranial transfer
比较 from scratch、EEG-pretrained frozen readout、small adapter、larger adaptation（仅作合理上界）；报告 zero/few-shot 和 negative transfer。

### Stage F — Artifact candidate
形成 `ART-AUDREP-v1` candidate，等待 `INT-01` consumer validation。

## 6. Controls / leakage / confounds

阻断相邻时间泄漏、同 story/clip train-test 重复、test subject normalization leakage、用 test 选 layer、recording/block identity shortcut、trivial negative samples。

所有 normalization/PCA/feature/layer selection 仅在 training folds 内完成。

## 7. Deliverables

- dataset/license manifest；
- split manifest；
- acoustic/timing baseline table；
- layerwise EEG benchmark；
- neural encoder benchmark；
- transfer/few-shot curves；
- negative-transfer table；
- known-failures report；
- `ART-AUDREP-v1` candidate manifest + schema + runtime/canary。

## 8. Acceptance criteria

不要求 transfer 阳性。必须完成 dataset/license/timing audit、leak-safe split、可解释 baseline、held-out subject、public intracranial transfer、few-shot/data-efficiency、negative conditions、artifact provenance/known failures。

## 9. Go / Hold / Fail

### GO
形成可消费 `ART-AUDREP-v1` candidate，供 `INT-01` 测试。

### HOLD
license、audio-neural pairing 或 timing 不可审计时暂停相应 dataset。

### NO-GO for a branch
某 dataset/model 不支持可靠 benchmark 时记录 failure 并停止该分支；不自动新增模型追阳性。

PUB-01 的失败永远不构成 READ-03 的 no-go。

## 10. Known risks

held-out subject weak、EEG→intracranial negative transfer、layer instability、channel-agnostic architecture 无优势、只复现 envelope/onset、license 限制 downstream artifact，均为合法结果。

## 11. Artifact / handoff

`ART-AUDREP-v1` 至少包含 version、source task/commit/config/seed、feature/model spec、neural encoder、preprocessing/timing、embedding dimensions/frame step、portable runtime、canary、benchmark/few-shot evidence、known failures、schema/validator、license manifest。

不得包含患者/STN 数据或患者派生 embedding。

## 12. Completion record

`NOT_STARTED`
