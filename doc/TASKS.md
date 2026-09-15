# Auditory_Simulation Task Registry

执行规范：`doc/TASK_EXECUTION_STANDARD.md`  
文档索引：`doc/DOCUMENT_INDEX.md`  
Program mapping：`AuditoryReading/PROGRAM_RESEARCH_GRAPH.md`

`TASKS.md` 是本仓库 Local Task 状态 source of truth。

| Task ID | Program Node | Track | Class | Status | Blocking | Patient Data | Task Spec | Primary Output |
|---|---|---|---|---|---|---|---|---|
| `M6A-PUBLIC-001` | historical support | PUBLIC_METHOD | MAINLINE_EXECUTION | HISTORICAL_REFERENCE | NO | FORBIDDEN | `doc/tasks/M6A-PUBLIC-001.md` | preliminary ds004703/wav2vec2 checkpoint |
| `M6A-PUBLIC-002` | R1 support bank | PUBLIC_METHOD | MAINLINE_EXECUTION | COMPLETED | NO | FORBIDDEN | `doc/tasks/M6A-PUBLIC-002.md` | frozen pretrained model/reference bank |
| `M6A-PUBLIC-003` | R1 support evidence | PUBLIC_METHOD | MAINLINE_EXECUTION | REVIEW | NO | FORBIDDEN | `doc/tasks/M6A-PUBLIC-003.md` | temporal architecture perturbation negative/boundary evidence |
| `M6A-PUBLIC-004` | `R1` | PUBLIC_METHOD | MAINLINE_EXECUTION | READY | NO for STN direct R2–R4 | FORBIDDEN | `doc/tasks/M6A-PUBLIC-004.md` | `ART-AUDREP-v1` candidate + public benchmark |

## Current priority

### `M6A-PUBLIC-004` — READY

目标：在公开数据中建立可审计的 auditory-neural prior，并测试跨记录方式迁移和 data-efficiency。

第一阶段顺序：

1. SparrKULee/ds004703 access、license、timing、identity、split audit；
2. acoustic/onset/timing + wav2vec2/HuBERT baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject/session/story；
6. public EEG→public intracranial transfer；
7. few-shot/negative-transfer evidence；
8. `ART-AUDREP-v1` candidate。

### Program boundary

`R1` 是 **optional accelerator / comparator**。

- positive transfer：下游 STN 可以作为预训练条件；
- no transfer：保留结果，STN direct/simple models 继续；
- negative transfer：保留结果并限制 artifact claim；
- `EEG→sEEG` 成功不能自动写成 `EEG→STN` 已成功。

因此 `M6A-PUBLIC-004` **不阻塞** STN 的 Protocol v2、患者持续采集或 direct computational-variable analysis。

## `M6A-PUBLIC-003` review rule

003 已完成执行，当前只允许：

- independent/human review；
- completion/frozen record；
- provenance/bug correction。

不允许因为阴性结果继续追加 architecture 变体。

## Hard boundary

- 患者/STN 数据不得进入本仓库；
- 不读取患者派生 embedding；
- 不使用 STN 结果选择 public model/layer；
- 不继续堆 pretrained model 名单；
- 所有 downstream handoff 必须是冻结、版本化 artifact。