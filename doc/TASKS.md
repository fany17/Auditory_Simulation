# Auditory_Simulation Task Registry

执行规范：`doc/TASK_EXECUTION_STANDARD.md`  
文档索引：`doc/DOCUMENT_INDEX.md`  
Pipeline：`AuditoryReading/RESEARCH_PIPELINE.md`

`TASKS.md` 是本仓库 current/future task 状态 source of truth。

## Current / future tasks

| Task ID | Pipeline Node | Track | Class | Status | Blocking | Patient Data | Task Spec | Primary Output |
|---|---|---|---|---|---|---|---|---|
| `PUBLIC-REPRESENTATION` | same ID | PUBLIC_METHOD | MAINLINE_EXECUTION | READY | NO for direct STN science | FORBIDDEN | `doc/tasks/PUBLIC-REPRESENTATION.md` | `ART-AUDREP-v1` candidate + public benchmark |

## Historical provenance

旧 public milestone 只作为历史证据：

| Historical ID | Status | Role |
|---|---|---|
| `M6A-PUBLIC-001` | HISTORICAL_REFERENCE | preliminary public alignment checkpoint |
| `M6A-PUBLIC-002` | COMPLETED | frozen pretrained model/reference bank |
| `M6A-PUBLIC-003` | REVIEW | temporal-architecture perturbation evidence; review/close only |

这些历史 ID 不再作为 current/future task 命名来源，不继续创建 `M6A-PUBLIC-005`。

## `PUBLIC-REPRESENTATION` — READY

目标：在公开数据中建立可审计 auditory-neural prior，并测试跨记录方式迁移和 data efficiency。

第一阶段顺序：

1. SparrKULee / ds004703 access、license、timing、identity、split audit；
2. acoustic/onset/timing + wav2vec2/HuBERT baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject/session/story；
6. public EEG→public intracranial transfer；
7. few-shot / negative-transfer evidence；
8. `ART-AUDREP-v1` candidate。

### Pipeline boundary

`PUBLIC-REPRESENTATION` 是 optional accelerator/comparator：

- positive transfer：患者侧可作为预训练条件；
- no transfer：保留结果，direct/simple STN models 继续；
- negative transfer：保留 domain mismatch；
- public EEG→intracranial 成功不能自动外推 STN。

因此该任务不阻塞 `STN-READ-PROTOCOL`、`STN-DATA-QC` 或 direct `STN-READ-COMPUTATION`。

## Historical perturbation review rule

历史 temporal-architecture perturbation 已完成执行，当前只允许：

- independent/human review；
- completion/frozen record；
- provenance/bug correction。

不允许因为阴性结果继续追加 architecture 变体。

## Hard boundary

- 患者/STN 数据不得进入本仓库；
- 不读取患者派生 embedding；
- 不使用 STN 结果选择 public model/layer；
- 不继续堆 pretrained model 名单；
- downstream handoff 必须是冻结、版本化 artifact。