# Auditory_Simulation Task Registry

执行规范：`doc/TASK_EXECUTION_STANDARD.md`  
文档索引：`doc/DOCUMENT_INDEX.md`  
Pipeline：`AuditoryReading/RESEARCH_PIPELINE.md`

`TASKS.md` 是本仓库 current/future task 状态 source of truth。

## Current / future tasks

| Task ID | Title | Track | Class | Status | Blocking | Patient Data | Task Spec | Primary Output |
|---|---|---|---|---|---|---|---|---|
| `PUB-01` | Public Auditory–Neural Representation | PUBLIC_METHOD | MAINLINE_EXECUTION | READY | NO for direct STN READ | FORBIDDEN | `doc/tasks/PUB-01_PUBLIC_REPRESENTATION.md` | `ART-AUDREP-v1` candidate + public benchmark |

## Historical provenance

| Historical ID | Status | Role |
|---|---|---|
| `M6A-PUBLIC-001` | HISTORICAL_REFERENCE | preliminary public alignment checkpoint |
| `M6A-PUBLIC-002` | COMPLETED | frozen pretrained model/reference bank |
| `M6A-PUBLIC-003` | REVIEW | temporal-architecture perturbation evidence; review/close only |

这些历史 ID 不再作为 current/future task 命名来源，不继续创建 `M6A-PUBLIC-005`。

## PUB-01 — READY

目标：在公开数据中建立可审计 auditory-neural prior，并测试跨记录方式迁移和 data efficiency。

第一阶段：

1. SparrKULee / ds004703 access、license、timing、identity、split audit；
2. acoustic/onset/timing + wav2vec2/HuBERT baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject/session/story；
6. public EEG→public intracranial transfer；
7. few-shot / negative-transfer evidence；
8. `ART-AUDREP-v1` candidate。

PUB-01 是 optional accelerator/comparator。positive/no/negative transfer 均可验收，不阻塞 READ-01/02/03 的 direct science。

## Hard boundary

- 患者/STN 数据不得进入本仓库；
- 不读取患者派生 embedding；
- 不使用 STN 结果选择 public model/layer；
- 不继续堆 pretrained model 名单；
- downstream handoff 必须是冻结、版本化 artifact。
