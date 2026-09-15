# Auditory_Simulation 项目边界

| 字段 | 内容 |
|---|---|
| 文档性质 | 本项目独立边界；不是执行 task spec |
| 版本 | `v2.0` |
| 生效日期 | `2026-09-15` |
| 状态 | `ACTIVE` |
| 适用目录 | `Auditory_Simulation/` |
| 当前任务入口 | `doc/CURRENT_TASK.md` |
| Task Registry | `doc/TASKS.md` |

## 1. 项目定位

`Auditory_Simulation` 是**公开数据、公开声音模型与通用 auditory-neural representation 方法项目**。

当前 pipeline 角色：`PUBLIC-REPRESENTATION / PUBLIC_METHOD`。

> 使用公开声音—神经数据建立可审计的 representation、迁移和 few-shot benchmark，并向患者侧提供冻结、版本化的公共方法 artifact。

本项目不负责证明 STN 患者内神经机制。public pretraining 是患者侧 optional accelerator / comparator，不是 STN READ 科学的硬前置。

## 2. 本项目负责

### Public audio representation

- interpretable acoustic/onset/timing features；
- frozen wav2vec2 / HuBERT 等已批准模型；
- layerwise representation；
- model source/revision/license/preprocessing provenance。

历史 pretrained model bank 已足够支持当前任务；不因出现新 architecture 自动扩模型名单。

### Public neural data

- SparrKULee 等公开 EEG；
- ds004703 等公开 intracranial data；
- subject/session/story/clip/channel inventory；
- license/use boundary；
- audio↔neural timing/pairing；
- grouped split 与 leakage audit。

### General methods

- ridge/TRF encoding；
- channel-agnostic neural encoder；
- contrastive/retrieval；
- held-out subject/session/story；
- public EEG→public intracranial transfer；
- zero/few-shot and negative transfer；
- frozen/versioned artifact packaging。

## 3. 不属于本项目

以下严格属于 `STN_Decoding_Encoding`：

- 患者实验与患者数据；
- STN-LFP raw/intermediate/derivative；
- MR4/TTL/clinical synchronization；
- `STN-READ-*` patient-specific scientific claims；
- expectation/error/update protocol；
- `STN-WRITE-*` 刺激能力、system ID、target、pilot、validation；
- `SEEG-*` 临床合作与刺激；
- patient-specific adapter result；
- clinical/behavioral write-in claims。

本仓库不得读取患者/STN 原始数据、患者派生 embedding 或私有临床 metadata。

## 4. 与患者项目的关系

当前结构：

- `PUBLIC-REPRESENTATION`：公共 prior/comparator；
- `STN-READ-PROTOCOL → STN-DATA-QC → STN-READ-COMPUTATION → STN-READ-VALIDATION`：患者 direct READ science，可独立推进；
- `ART-AUDREP-v1` 到达后，患者侧可运行 `STN-READ-TRANSFER`；
- public transfer 为 positive/no/negative 均可正常完成；
- public transfer 失败不得阻塞 direct/simple STN models。

`EEG → public intracranial → STN` 不是预设生物学层级，也不是必须成功的技术路线。

## 5. Current task and historical provenance

Current task：

- `PUBLIC-REPRESENTATION`：`READY`。

Historical milestone evidence：

- historical temporal-architecture perturbation：`REVIEW`；只允许审核/收口；
- historical pretrained baseline：`COMPLETED`；frozen reference bank；
- historical preliminary public alignment：`HISTORICAL_REFERENCE`。

旧 `M6A-*` task 文件中的命名和状态只保留 provenance，不覆盖 current Registry，也不再继续编号。

## 6. `PUBLIC-REPRESENTATION` first-stage boundary

Primary public EEG：SparrKULee。  
Primary public intracranial transfer candidate：ds004703。

第一阶段必须先完成：

1. access/license audit；
2. subject/session/story/audio inventory；
3. true pairing/timing audit；
4. leak-safe grouped split；
5. interpretable acoustic/onset baseline；
6. held-out benchmark skeleton。

然后才进入 neural encoder/contrastive/transfer。

不能为了“先跑起来”随机切相邻时间点或忽略 story/subject leakage。

## 7. Artifact boundary

`ART-AUDREP-v1` 至少包含：

- semantic version；
- source task/commit/config/seed；
- model/feature specification；
- preprocessing/timing spec；
- neural encoder architecture/weights（若许可允许）；
- portable transform/runtime；
- canary；
- public benchmark；
- few-shot/negative-transfer evidence；
- known failures；
- schema/validator；
- license/provenance manifest。

患者结果不得用于选择 artifact 内容。

患者侧 consumer validation 通过前只能称 `candidate`。

## 8. 证据解释边界

可以写：

- public audio representation 在 held-out neural data 有预测力；
- public prior 有 positive/no/negative transfer；
- 某 layer/feature 在预定义 benchmark 中更稳定；
- 某 neural encoder 提高 data efficiency。

不能仅凭这些写：

- EEG、sEEG 和 STN 表征同源；
- DNN layer 等于某脑区；
- public pretraining 证明 STN 机制；
- shared geometry 等于因果机制；
- public model 已具备 write-in 能力。

## 9. Stop / redesign rules

必须停止/重设计而不是追阳性的情况：

- license/use boundary 不清；
- audio-neural timing/pairing 不可审计；
- split 泄漏；
- held-out subject collapse 但只想保留 within-subject；
- negative transfer；
- layer preference unstable；
- 需要患者数据才能继续但无授权。

Negative/no-transfer 均可正常完成任务。

## 10. Document governance

当前权威读取链：

1. `AGENTS.md` 与安全/数据边界；
2. `doc/PROJECT_BOUNDARY.md`；
3. `doc/DOCUMENT_INDEX.md`；
4. `doc/TASK_EXECUTION_STANDARD.md`；
5. `doc/TASKS.md`；
6. `doc/CURRENT_TASK.md`；
7. `doc/tasks/PUBLIC-REPRESENTATION.md`；
8. config/code/log/report evidence。

旧 charter/总纲、historical milestone tasks、exchange drafts 和 reports 可作为设计/证据来源，但不能覆盖 current Registry。