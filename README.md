# Auditory_Simulation

本目录是独立的公开听觉模型、公开神经数据与计算方法项目。当前核心方向是建立声音表征、时间结构与不同层级神经活动之间可检验的计算关系，并形成可冻结、可迁移到患者侧的 auditory-neural representation artifact。

本项目不负责 PD 患者 STN 实验、临床采集、TTL/MR4 同步、DBS 装置、PINS/SEEG 人体刺激或真实患者 STN-LFP 专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## 当前入口

按顺序读取：

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/DOCUMENT_INDEX.md`
4. `doc/TASK_EXECUTION_STANDARD.md`
5. `doc/TASKS.md`
6. `doc/CURRENT_TASK.md`
7. 当前 `doc/tasks/<TASK-ID>.md`

## 当前任务

### `M6A-PUBLIC-004 — Auditory–Neural Representation Alignment and Transfer`

Program Node：`R1`  
状态：`READY`  
角色：**PUBLIC_METHOD / optional accelerator + comparator**。

目标：在公开数据上建立可审计的 auditory-neural representation 与 transfer benchmark：

```text
SparrKULee EEG
→ public auditory-neural representation
→ public intracranial / ds004703 transfer
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

### 关键 Program 边界

`M6A-PUBLIC-004` **不是**患者 STN READ 科学的硬前置：

- positive transfer：患者侧可作为初始化/比较条件；
- no transfer：保留结果，STN direct/simple model science 继续；
- negative transfer：保留结果并限制 claim；
- EEG→public intracranial 成功不能自动外推成 EEG→STN 成功。

## 历史任务状态

### `M6A-PUBLIC-003 — Temporal Architecture Perturbation`

当前 Registry 状态：`REVIEW`。

45/45 formal runs 和 supplementary 10/20/50 ms probes 已完成。现有结果未支持 early/late downsampling、multiscale/RF growth 或 explicit change branch 在该 benchmark 上解决 localization/discrimination/generalization 问题。

当前只允许审核、completion record 和必要勘误；不继续追加结构变体追结果。

### `M6A-PUBLIC-002 — Pretrained Audio/Temporal Architecture Reproduction`

状态：`COMPLETED`。

已完成 9/9 pretrained inference、8/9 unified temporal representation probe，覆盖：

- PANNs CNN14；
- ConvTasNet；
- SpeechBrain CRDNN；
- Parakeet-TDT / FastConformer；
- Audio-Mamba / SSAM；
- wav2vec2；
- Whisper；
- CoNNear periphery；
- ICNet。

002 作为 004 的 frozen model/reference bank，不继续补模型。

### `M6A-PUBLIC-001 — Public Audio→Brain Minimal Alignment`

状态：`HISTORICAL_REFERENCE`。

保留 ds004703/wav2vec2 单被试单 recording preliminary；004 以新的 held-out、cross-subject 和 transfer 设计重新建立正式公共证据。

## 目录边界

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
├── configs/
├── environment/
├── schemas/
├── src/
├── scripts/
├── tests/
├── reports/
└── test/
```

大数据、模型权重、大体积特征和训练输出不进入 Git；计算资产留在 2203，Git 只保留轻量代码、配置、结构化结果和报告。

## 与其他仓库分工

- `Auditory_Simulation`：公开数据、公开模型、公共 EEG/intracranial、通用 auditory-neural representation 与 M6A。
- `AuditoryReading`：Program 总图、证据、教材、综述、调研和统一任务治理规范。
- `STN_Decoding_Encoding`：患者实验、STN 数据、自然计算变量、patient-specific analysis、PINS/SEEG 写入验证与 M6B。

`Auditory_Simulation` 与 `STN_Decoding_Encoding` 仅通过冻结、版本化 artifact 衔接；患者数据默认不得进入本仓库。

## 文档与执行规则

- `doc/TASKS.md` 是当前状态 source of truth；
- 历史 task/report/draft contract 不会因文件仍存在而自动恢复执行；
- 新 task 必须按 `doc/TASK_TEMPLATE.md` 创建；
- 阴性、no-transfer、negative-transfer 可以正常完成任务；
- 不因结果阴性自动扩模型、改 split、换窗口追阳性；
- task 完成后更新 registry 并停止，不自动启动 downstream M6B。