# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前核心任务是 `PUBLIC-REPRESENTATION`：建立可审计的 auditory-neural representation 与 transfer benchmark，并向患者侧输出冻结、版本化的公共方法 artifact。

本项目不负责 PD 患者 STN 实验、临床采集、TTL/MR4 同步、DBS/PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## Current entry

按顺序读取：

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/DOCUMENT_INDEX.md`
4. `doc/TASK_EXECUTION_STANDARD.md`
5. `doc/TASKS.md`
6. `doc/CURRENT_TASK.md`
7. `doc/tasks/PUBLIC-REPRESENTATION.md`

## Current task — `PUBLIC-REPRESENTATION`

Track：`PUBLIC_METHOD`  
Status：`READY`  
Role：optional accelerator / comparator for patient-side READ。

```text
SparrKULee EEG
→ public auditory-neural representation
→ public intracranial transfer
→ few-shot / negative-transfer evaluation
→ ART-AUDREP-v1 candidate
```

第一轮重点：

1. dataset/license/timing/leakage audit；
2. acoustic/onset/timing + wav2vec2 + HuBERT layerwise baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval；
5. held-out subject/session/story；
6. public EEG→intracranial transfer；
7. frozen artifact candidate。

### Scientific boundary

`PUBLIC-REPRESENTATION` 不是患者 STN READ 的硬前置：

- positive transfer：患者侧可作为初始化/比较条件；
- no transfer：保留无增益结果；
- negative transfer：保留 domain mismatch；
- direct/simple STN science 继续；
- public EEG→intracranial 成功不能自动外推成 STN transfer 成功。

## Historical milestones

旧 public milestone 文件保留 provenance，但不再定义 current/future pipeline：

- historical public alignment preliminary：`HISTORICAL_REFERENCE`；
- historical pretrained model baseline：`COMPLETED`；
- historical temporal-architecture perturbation：`REVIEW`，只允许审核/收口/勘误。

旧文件名中的 `M6A-PUBLIC-*` 不做批量重命名，以避免破坏已完成证据链；但不再创建新的 `M6A-*` task。

## Directory boundary

```text
Auditory_Simulation/
├── AGENTS.md
├── doc/
│   ├── PROJECT_BOUNDARY.md
│   ├── DOCUMENT_INDEX.md
│   ├── TASK_EXECUTION_STANDARD.md
│   ├── TASK_TEMPLATE.md
│   ├── TASKS.md
│   ├── CURRENT_TASK.md
│   └── tasks/
│       └── PUBLIC-REPRESENTATION.md
├── configs/
├── environment/
├── schemas/
├── src/
├── scripts/
├── tests/
├── reports/
└── test/
```

大数据、模型权重、大体积特征和训练输出不进入 Git；计算资产留在计算环境，Git 只保留轻量代码、配置、结构化结果和报告。

## Repository split

- `Auditory_Simulation`：`PUBLIC-REPRESENTATION` 与其他未来公共方法任务。
- `AuditoryReading`：`RESEARCH_PIPELINE.md`、证据、教材、综述、调研和统一治理规范。
- `STN_Decoding_Encoding`：`STN-READ-*`、`STN-WRITE-*`、`SEEG-*` 与 patient-specific analysis。

公共项目与患者项目只通过冻结、版本化 artifact 衔接；患者数据不得进入本仓库。

## Naming and execution rules

- current/future task 只使用语义型 ID；
- 状态、日期、版本不写进 Task ID；
- `doc/TASKS.md` 是当前状态 source of truth；
- 历史 task/report/draft contract 不会因文件仍存在而自动恢复执行；
- 阴性、no-transfer、negative-transfer 可以正常完成任务；
- 不因结果阴性自动扩模型、改 split、换窗口追阳性；
- task 完成后更新 registry 并停止，不自动启动患者侧任务。