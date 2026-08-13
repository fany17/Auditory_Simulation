# Auditory_Simulation

本目录是一个独立的公开听觉模型、公开神经数据与计算方法项目。当前核心方向是建立声音表征与不同层级神经活动之间可检验的对齐关系，研究声学、语音、语言、语义、情绪、音乐和运动相关信息的保留、转换与丢失。

本项目不负责 PD 患者 STN 实验、临床采集、TTL/MR4 同步、DBS 装置或真实患者 STN-LFP 的专属分析；这些工作独立属于 `STN_Decoding_Encoding`。

## 目录边界

```text
Auditory_Simulation/
├── AGENTS.md       # 本项目工作规则与证据门禁
├── doc/            # 当前边界、任务书、当前状态及历史规划材料
├── configs/        # 机器可读的正式任务配置
├── environment/    # 2203 专用环境定义与非哈希版本记录
├── schemas/        # split 与冻结 artifact schema
├── src/            # M6A 正式代码
├── scripts/        # 2203 启动与自检入口
├── tests/          # 正式单元测试；不等同于历史 test/
├── reports/        # 轻量阶段报告
└── test/           # 历史测试、验证、演示和执行参考
```

### `doc/`：本项目独立文档

- `PROJECT_BOUNDARY.md`：当前有效的项目目标、所有权、排除项及与 STN 项目的接口。
- `CURRENT_TASK.md`：当前阶段、候选任务和执行授权状态。
- `01_听觉时变信息处理项目总纲.md`：历史总纲，仅保留规划来源，不再作为当前权威。
- `PROJECT_CHARTER.md`：历史章程草案，仅保留规划来源，不再作为当前权威。

本项目不再维护跨项目“总文档”。项目边界只在本目录内定义；跨项目协作通过有版本、只读、带 manifest 的冻结 artifact 完成。

### `test/`：测试与验证材料

- `02_从零启动算法研究操作手册.md`：历史测试、执行与审核流程参考。
- `03_*`：独立测试、wav2vec2 验证、Demo 材料和历史尝试。
- `deliverables/`：测试和演示形成的交付物。

`test/` 中的文件不自动成为正式任务或科学结论。只有经人工批准、建立明确任务书并完成运行与审核后，相关证据才能进入正式项目链。

## 当前状态

- 当前边界：声音模型、公开神经数据和计算神经仿真的方法平台。
- 当前方向：`M6A-PUBLIC`，建立公开数据上的声音—神经表征对齐。
- 当前唯一任务：`M6A-PUBLIC-001`，冻结 OpenNeuro `ds004703 v1.1.0` 并建立最小逐层对齐基线。
- 首轮模型策略：冻结 `facebook/wav2vec2-base`，不微调声音基座；HuBERT/WavLM 留待后续对照任务。
- 执行状态：`ACTIVE_EXECUTION`；专用环境、公开数据、公开模型和轻量分析均获授权在 2203 执行。
- 当前节点：最小 G4 已在 2203 完成同一 `sub-SD012_ses-02_task-PassiveListen` recording 的 40 passages（24/8/8），形成单被试 preliminary held-out 指标与 20-null 机械检查，状态为 `G4_MINIMAL_SD012_SES02_PRELIMINARY_COMPLETE_AWAITING_COORDINATOR_REVIEW`。这不是协调接受的正式 G4 科学结果；当前仍无 M6A→M6B exchange candidate 或 consumer cross-test。
- 许可边界：CC0 元数据与 README 更严格限制共同执行；仅限非商业学术研究、禁止再识别、禁止原始数据外传。
- 完整性边界：只使用文件名、字节数、时间戳、数量、schema 与抽样可读性；不进行任何哈希或校验和验证。
- `test/` 中既有 Demo 不自动升级为 M6A 正式结果。
- Git 管理：本目录是唯一 Git 工作区；大数据、模型权重、大体积特征与训练输出不进入本地仓库。

## 与 STN_Decoding_Encoding 的接口

- `M6A-PUBLIC` 在本项目内建立并外部验证公开数据方法与冻结 artifact。
- 未来可能的 `M6B-STN` 只在 `STN_Decoding_Encoding` 内、经独立任务书和数据治理批准后运行。
- 两项目不共享可变工作目录、当前任务文档、结果状态或项目总纲。
- Auditory 的公开数据结果不能替代 STN 真实数据证据；STN 单被试结果也不能反向充当公开模型的通用验证。

## 阅读顺序

1. `AGENTS.md`
2. `doc/PROJECT_BOUNDARY.md`
3. `doc/CURRENT_TASK.md`
4. 经人工批准的具体任务书与实际运行证据
5. 只有追溯历史决策时，才阅读旧总纲、旧章程和 `test/`
