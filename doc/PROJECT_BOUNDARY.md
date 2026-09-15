# Auditory_Simulation 项目边界

| 字段 | 内容 |
|---|---|
| 文档性质 | 本项目独立边界；不是执行 task spec |
| 版本 | `v1.2` |
| 生效日期 | `2026-09-15` |
| 状态 | `ACTIVE` |
| 适用目录 | `Auditory_Simulation/` |
| 当前任务入口 | `doc/CURRENT_TASK.md` |
| Task Registry | `doc/TASKS.md` |

## 1. 项目定位

`Auditory_Simulation` 是**公开数据、公开声音模型与通用 auditory-neural representation 方法项目**。

当前 Program 角色为 `R1 / PUBLIC_METHOD`：

> 使用公开声音—神经数据建立可审计的表示、迁移和 few-shot benchmark，并向患者侧提供冻结、版本化的公共方法 artifact。

本项目不负责证明 STN 的患者内神经机制；public pretraining 是患者侧的 **optional accelerator / comparator**，不是 STN READ 科学的硬前置。

## 2. 本项目负责

### 2.1 Public audio representation

- interpretable acoustic/onset/timing features；
- frozen wav2vec2 / HuBERT 等已批准模型；
- layerwise representation；
- 模型来源、revision、license、preprocessing provenance。

现有 M6A-002 model bank 已足以支持当前 004；不再因为出现新 architecture 自动扩模型名单。

### 2.2 Public neural data

- SparrKULee 等公开 EEG；
- ds004703 等公开 intracranial data；
- subject/session/story/clip/channel inventory；
- license/use boundary；
- audio↔neural timing/pairing；
- grouped split 与 leakage audit。

### 2.3 General methods

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
- STN patient-specific scientific claims；
- Protocol v2 expectation/error/update experiments；
- PINS/DBS stimulation capability 或患者刺激；
- SEEG 临床合作/刺激；
- patient-specific adapter result；
- clinical/behavioral write-in claims。

本仓库不得读取患者/STN 原始数据、患者派生 embedding 或私有临床 metadata。

## 4. 与患者项目的关系

旧结构曾把 M6B-STN 写成必须等待完整 M6A artifact 才能继续。现更新为：

- `M6A-PUBLIC-004 / R1`：公共 prior/comparator；
- `STN R0/R2/R3/R4`：患者 direct READ science，可以独立推进；
- `ART-AUDREP-v1` 到达后，患者侧可运行可选 transfer/comparator；
- public transfer 为 zero/no/negative 均可正常完成任务；
- public transfer 失败不得阻塞患者 direct/simple models。

`EEG → public intracranial → STN` 不是预设的生物学层级链，也不是必须成功的技术路线。

## 5. 当前任务和历史任务

状态以 `doc/TASKS.md` 为准。

- `M6A-PUBLIC-004`：`READY`；Program `R1`；
- `M6A-PUBLIC-003`：`REVIEW`；执行已完成，不继续扩 architecture；
- `M6A-PUBLIC-002`：`COMPLETED`；frozen model/reference bank；
- `M6A-PUBLIC-001`：`HISTORICAL_REFERENCE`。

旧 task 文件中的自定义状态只保留历史 provenance，不覆盖 Registry。

## 6. M6A-PUBLIC-004 第一阶段边界

Primary public EEG：SparrKULee。  
Primary public intracranial transfer dataset：ds004703。

第一阶段必须先完成：

1. access/license audit；
2. subject/session/story/audio inventory；
3. true pairing/timing audit；
4. leak-safe grouped split；
5. interpretable acoustic/onset baseline；
6. held-out benchmark。

然后才进入 neural encoder/contrastive/transfer。

不能为了“先跑起来”随机切相邻时间点或忽略 story/subject leakage。

## 7. Artifact 边界

未来 `ART-AUDREP-v1` 至少包含：

- semantic version；
- source commit/config/seed；
- model/feature specification；
- preprocessing/timing spec；
- neural encoder architecture/weights（若许可允许）；
- portable transform/runtime；
- canary；
- public benchmarks；
- few-shot/negative-transfer evidence；
- known failures；
- schema/validator；
- license/provenance manifest。

患者结果不得用于选择 artifact 内容。

患者侧 consumer cross-test 通过前只能称 `candidate`。

## 8. 证据解释边界

可以写：

- 某 public audio representation 在 held-out neural data 有预测力；
- 某 public prior 有 positive/no/negative transfer；
- 某 layer/feature 在预定义 benchmark 中更稳定；
- 某 neural encoder 提高 data efficiency。

不能仅凭这些写：

- EEG、sEEG 和 STN 表征同源；
- 某 DNN layer 等于某脑区；
- public pretraining 证明 STN 机制；
- shared geometry 等于因果机制；
- public model 已具备人工 write-in 能力。

## 9. 停止规则

必须停止/重设计而不是追阳性的情况：

- license/use boundary 不清；
- audio-neural timing/pairing 不可审计；
- split 泄漏；
- held-out subject collapse 但只想保留 within-subject；
- negative transfer；
- layer preference unstable；
- 需要患者数据才能继续但无授权。

Negative/no-transfer 均可正常完成任务。

## 10. 文档治理

当前权威读取链：

1. `AGENTS.md` 与安全/数据边界；
2. `doc/PROJECT_BOUNDARY.md`；
3. `doc/DOCUMENT_INDEX.md`；
4. `doc/TASK_EXECUTION_STANDARD.md`；
5. `doc/TASKS.md`；
6. `doc/CURRENT_TASK.md`；
7. 当前 `doc/tasks/<TASK-ID>.md`；
8. config/code/log/report evidence。

旧 `PROJECT_CHARTER.md`、`01_听觉时变信息处理项目总纲.md`、historical exchange drafts 和 reports 均可作为设计/证据来源，但不能覆盖当前 Registry。