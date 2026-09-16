# Auditory_Simulation 项目边界

| 字段 | 内容 |
|---|---|
| 文档性质 | 本项目独立边界；不是执行 task spec |
| 版本 | `v2.1` |
| 生效日期 | `2026-09-15` |
| 状态 | `ACTIVE` |
| 当前任务入口 | `doc/CURRENT_TASK.md` |
| Task Registry | `doc/TASKS.md` |

## 1. 项目定位

`Auditory_Simulation` 负责公开数据、公开声音模型和通用 auditory-neural representation 方法。

当前 canonical task：`PUB-01 — Public Auditory–Neural Representation`。

> 使用公开声音—神经数据建立可审计的 representation、迁移和 few-shot benchmark，并向患者侧提供冻结、版本化公共方法 artifact。

PUB-01 是患者侧 optional accelerator/comparator，不是 STN READ 的硬前置。

## 2. 本项目负责

- interpretable acoustic/onset/timing features；
- frozen wav2vec2/HuBERT 等已批准模型；
- SparrKULee 等公开 EEG；
- ds004703 等公开 intracranial data；
- dataset/license/timing/pairing/grouped split audit；
- ridge/TRF、channel-agnostic neural encoder、contrastive/retrieval；
- held-out subject/session/story；
- public EEG→public intracranial transfer；
- zero/few-shot 和 negative transfer；
- frozen/versioned artifact packaging。

历史 pretrained model bank 已足够支持 PUB-01；不因新 architecture 自动扩模型名单。

## 3. 不属于本项目

严格属于 `STN_Decoding_Encoding`：

- 患者实验与患者数据；
- STN-LFP raw/intermediate/derivative；
- MR4/TTL/clinical synchronization；
- `READ-##` patient-specific science；
- `INT-01` patient-side transfer evaluation；
- `WRITE-##` 刺激、system ID、target、pilot、validation；
- `SEEG-##` 临床合作与刺激；
- patient-specific adapter result；
- clinical/behavioral write-in claims。

本仓库不得读取患者/STN 原始数据、患者派生 embedding 或私有临床 metadata。

## 4. 与患者项目的关系

- `PUB-01`：公共 prior/comparator；
- `READ-01 → READ-04`：患者 direct READ science，可独立推进；
- `ART-AUDREP-v1` 到达后，由患者侧 `INT-01` 做 transfer benchmark；
- positive/no/negative transfer 均可正常完成；
- transfer 失败不得阻塞 READ-03 direct/simple models。

`EEG → public intracranial → STN` 不是预设生物学层级，也不是必须成功的串行技术路线。

## 5. Current task and historical provenance

Current：`PUB-01 = READY`。

Historical：

- `M6A-PUBLIC-001`：HISTORICAL_REFERENCE；
- `M6A-PUBLIC-002`：COMPLETED frozen reference bank；
- `M6A-PUBLIC-003`：REVIEW only。

旧 M6A 名称只保留 provenance，不继续编号。

## 6. PUB-01 first-stage boundary

Primary public EEG：SparrKULee。  
Primary public intracranial transfer candidate：ds004703。

必须先完成 access/license、subject/session/story/audio inventory、true pairing/timing、leak-safe grouped split、interpretable acoustic/onset baseline、held-out benchmark skeleton，再进入 neural encoder/contrastive/transfer。

## 7. Artifact boundary

`ART-AUDREP-v1` 至少包含 semantic version、source task/commit/config/seed、model/feature spec、preprocessing/timing、neural encoder、portable runtime、canary、public benchmark、few-shot/negative-transfer evidence、known failures、schema/validator、license/provenance manifest。

患者结果不得用于选择 artifact 内容；INT-01 consumer validation 通过前只能称 candidate。

## 8. 证据解释边界

可以写 public held-out predictive value、positive/no/negative transfer、layer/feature stability、data efficiency。

不能据此写 EEG/sEEG/STN 表征同源、DNN layer 等于脑区、public pretraining 证明 STN 机制、shared geometry 等于因果机制或 public model 已具备 write-in 能力。

## 9. Stop / redesign rules

license 不清、timing/pairing 不可审计、split 泄漏、held-out subject collapse、negative transfer、layer instability、需要患者数据才能继续但无授权时，必须停止/重设计，而不是追阳性。

## 10. Document governance

权威读取链：`AGENTS.md` → `PROJECT_BOUNDARY.md` → `DOCUMENT_INDEX.md` → `TASK_EXECUTION_STANDARD.md` → `TASKS.md` → `CURRENT_TASK.md` → `tasks/PUB-01_PUBLIC_REPRESENTATION.md` → config/code/log/report evidence。

旧 charter/总纲、historical M6A tasks、exchange drafts 和 reports 仅作 provenance/背景，不能覆盖 current Registry。
