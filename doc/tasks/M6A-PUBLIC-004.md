# TASK — M6A-PUBLIC-004 — Auditory–Neural Representation Alignment and Transfer

| Field | Value |
|---|---|
| Task ID | `M6A-PUBLIC-004` |
| Program Node | `R1` |
| Track | `PUBLIC_METHOD` |
| Class | `MAINLINE_EXECUTION` |
| Status | `READY` |
| Owner Repo | `Auditory_Simulation` |
| Blocking | `NO` for STN direct READ `R2–R4`; optional accelerator/comparator |
| Patient Data | `FORBIDDEN` |
| Upstream | `M6A-PUBLIC-002` completed model bank; `M6A-PUBLIC-003` review evidence; public datasets |
| Inputs | SparrKULee, ds004703, acoustic/timing features, frozen audio models |
| Deliverables | public benchmarks + `ART-AUDREP-v1` candidate |
| Acceptance | leak-safe public EEG benchmark + intracranial transfer evaluation + few-shot/negative-transfer report |
| Go/No-go | positive/no/negative transfer all valid; do not block STN direct science |
| Downstream | optional comparator/initialization for STN patient-side work |
| Last Updated | `2026-09-15` |

## 1. Scientific / Operational Question

> 在完全不读取患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial recordings 学到一个对声音时间结构/事件信息敏感、具有可审计跨被试与跨记录方式迁移性质的 auditory-neural representation？这种公共 pretraining 在 few-shot 场景中何时有用、何时无用或产生 negative transfer？

本任务的科学输出是**公共方法和迁移边界**，不是“证明 EEG、sEEG、STN 是同一表征层级”。

## 2. Scope

Primary EEG dataset：`SparrKULee`。  
Primary public intracranial dataset：`OpenNeuro ds004703`。

第一轮 feature/model bank：

- waveform envelope；
- onset/event features；
- log-mel / cochleagram；
- F0 / spectral flux（适用时）；
- explicit interval/rate/phase/deviation features（可定义时）；
- frozen wav2vec2 layerwise representations；
- frozen HuBERT layerwise representations。

Neural models 至少包含：

- ridge/TRF encoding baseline；
- simple temporal CNN/TCN baseline；
- channel-agnostic neural encoder。

## 3. Non-goals / Forbidden Actions

- 不读取患者/STN 数据或患者派生 embedding；
- 不使用 STN 结果选择 audio layer/model；
- 不继续扩 pretrained architecture 名单；
- 不为了阳性结果反复更换 split、window 或负样本；
- 不把 EEG→sEEG transfer 自动外推为 EEG→STN transfer；
- 不把 representation correlation 解释成脑区同源；
- 不在少量 public intracranial 数据上默认端到端训练超大 backbone；
- 不要求本任务成功后患者主线才能继续。

## 4. Inputs and Preconditions

Stage 0 必须先完成：

- dataset license/access manifest；
- subject/session/recording inventory；
- audio identity 与 neural recording 对应；
- sampling/channel/reference metadata；
- timing/event metadata；
- story/clip/speaker identity；
- split grouping keys；
- known missing/bad channels；
- audio-neural pairing 是否可审计。

禁止相邻时间点随机切分。

## 5. Execution Plan

### Stage A — Dataset / license / timing audit

输出：

- dataset inventory；
- license/use boundary；
- timing/pairing audit；
- grouped split manifest；
- leakage checklist。

Gate A：若真实 audio-neural pairing 无法可靠确定，先 `BLOCKED`，不得用近似标签直接训练后宣称 alignment。

### Stage B — Interpretable baseline

先运行：

- envelope/onset；
- log-mel/cochleagram；
- timing/rhythm features；
- ridge/TRF。

目标：得到可解释、可复现的 held-out baseline。

### Stage C — Layerwise frozen audio representations

逐层评估 wav2vec2/HuBERT。

所有 standardization、PCA、layer selection 只在 train fold 内完成。

不得预先把某层命名为“节律层”“高级层”或“STN 层”。

### Stage D — Channel-agnostic neural encoder

推荐结构：

```text
per-channel signal
→ shared temporal stem
→ per-channel temporal tokens
→ channel pooling/attention
→ shared temporal module
→ neural embedding
```

设计目标是适应不同 channel count，不是假设 scalp topography 与 intracranial recording 同构。

### Stage E — Alignment objectives

至少两类：

1. encoding：audio features/latent → neural signal/latent；
2. contrastive/retrieval：真实配对 audio-neural window vs leak-safe negatives。

负样本不能让 model 仅靠 story/session/block identity 做分类。

### Stage F — Generalization and transfer

必须报告：

- within-subject held-out clip/story；
- held-out subject；
- held-out recording/session；
- EEG→public intracranial zero/few-shot；
- fixed-data-budget learning curves；
- negative transfer。

### Stage G — Artifact candidate

通过人工审核后形成 `ART-AUDREP-v1` candidate。

## 6. Controls / Leakage / Confounds

- random adjacent-time split：禁止；
- same story neighboring windows across train/test：禁止；
- test subject normalization fitting：禁止；
- test result-driven layer selection：禁止；
- story/block/recording identity shortcut：必须检查；
- acoustic/onset baseline：必须保留；
- public EEG 与 intracranial modality/channel mismatch：必须作为结果报告，不得隐藏。

## 7. Deliverables

至少：

```text
reports/m6a_public_004/...
public_dataset_inventory.*
license_manifest.*
split_manifest.*
acoustic_baseline.*
layerwise_audio_representation.*
neural_encoder_benchmark.*
transfer_learning_curves.*
negative_transfer_report.*
ART-AUDREP-v1-candidate/
```

Git 只保存轻量代码、配置、结构化结果和报告；大型数据/权重/cache 留在计算环境。

## 8. Acceptance Criteria

### Gate A — Public EEG pipeline valid

- timing/pairing audit complete；
- leak-safe split；
- interpretable baseline reproducible；
- held-out-subject result必须报告，无论正负。

### Gate B — Pretraining value characterized

必须明确得到以下之一：

- positive transfer；
- no transfer；
- negative transfer。

不要求阳性才能验收。

### Gate C — Public intracranial transfer characterized

必须报告：

- zero/few-shot；
- modality/channel mismatch；
- adapter parameter count；
- transfer vs from-scratch under fixed data budget。

不得用一个 subject/recording 代表跨模态成功。

### Gate D — Candidate artifact review

只有 A–C 完成且人工 review 通过，才形成 candidate。

## 9. Go / Hold / Fail Rules

### GO

公共 benchmark/transfer 可以被稳定复现，并形成明确边界 → 发布 `ART-AUDREP-v1` candidate。

### HOLD

数据 access/license/timing/pairing 无法确认 → `WAITING_EXTERNAL/BLOCKED`。

### FAIL / NEGATIVE

若 public pretraining 无增益或 negative transfer：

- 保留结果；
- 仍可完成本任务；
- artifact claim 相应限制；
- STN direct/simple model path 不受阻塞。

## 10. Known Risks and Negative Outcomes

- held-out-subject performance collapse；
- EEG→intracranial negative transfer；
- audio-layer preference unstable；
- task/story identity shortcut；
- public datasets 与 STN 后续任务内容差异过大；
- license 限制后续 artifact 使用方式。

上述均属于应报告结果，不是理由去无限改模型。

## 11. Artifact / Handoff Contract

`ART-AUDREP-v1` candidate 至少包含：

- semantic version；
- source commit/config/seed；
- frozen audio feature/model spec；
- neural encoder architecture + weights（若允许分发）；
- preprocessing/timing specification；
- embedding dimension/frame step；
- portable transform/runtime；
- tiny public/synthetic canary；
- benchmark/few-shot/negative-transfer evidence；
- known failures；
- schema + validator；
- license/provenance manifest。

不得包含患者/STN 数据或依据患者结果挑选的 model/layer。

Consumer cross-test 通过前，只能称 `candidate`。

## 12. Completion Record

`NOT_STARTED`
