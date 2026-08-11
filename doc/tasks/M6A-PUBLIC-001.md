# M6A-PUBLIC-001 正式任务书

| 字段 | 内容 |
|---|---|
| 任务名称 | `ds004703 数据冻结与最小 layer-wise Audio->Brain encoding` |
| 状态 | `ACTIVE_EXECUTION` |
| 授权日期 | 2026-08-11 |
| 数据集 | OpenNeuro `ds004703 v1.1.0` |
| 主模型 | `facebook/wav2vec2-base`，冻结参数 |
| 主终点 | held-out layer-wise ridge encoding |
| 执行位置 | 2203 专用目录和专用 Conda 环境 |
| 配置 | `configs/m6a_public_001.json` |

## 1. 目标与可证伪问题

本任务建立首个可复现、防泄漏、可审计的公开声音—神经对齐入口，并回答：在严格 held-out 刺激条件下，冻结 wav2vec 2.0 各层表征是否比传统声学基线更好地预测 iEEG 神经目标；该优势是否超过保持时间结构的 null，且是否在被试、脑区和电极层面具有可报告的稳定性。

这是 encoding/predictability 问题，不是脑区与模型层功能同一、因果机制或临床有效性问题。

## 2. 授权范围

允许：

- 在 2203 创建独立环境、数据、缓存、模型和输出目录；
- 获取 `ds004703 v1.1.0` 与 `facebook/wav2vec2-base`；
- 读取公开 BIDS 元数据、音频与 iEEG，做轻量检查和正式预处理；
- 提取冻结模型逐层表征，建立 acoustic baseline 和 ridge encoding；
- 保存失败、阴性结果、日志、轻量 manifest 和阶段报告；
- 在本地唯一 Git 工作区修改、测试、提交和推送轻量资产。

不允许：

- 商业用途、任何再识别尝试、原始数据外传或把原始数据提交 Git；
- 读取或分析 STN 患者数据；
- 微调声音模型、引入第二数据集或扩展为患者特异结论；
- 主动生成、读取、比较或验证任何 SHA、MD5、checksum、文件哈希或 Git 对象哈希。

## 3. 输入冻结

### 3.1 数据集

- ID：`ds004703`
- 版本：`1.1.0`
- DOI：`10.18112/openneuro.ds004703.v1.1.0`
- `dataset_description.json`：声明 `CC0`
- README 更严格限制：不得商业使用；不得以任何方式消歧参与者身份
- 项目状态：`ACCEPTED_WITH_STRICTER_README_BOUNDARY`
- 远端根：`/home/fanyu/auditory_simulation_m6a/data/ds004703/v1.1.0`

若实际下载接口要求本人交互同意、签署新条款或个人凭据，立即停止，不代替用户接受。

### 3.2 声音模型

- 模型 ID：`facebook/wav2vec2-base`
- 角色：纯预训练 speech SSL encoder；避免首轮受 ASR 微调目标混杂
- 许可：模型卡声明 `Apache-2.0`
- 输入：16 kHz mono waveform，显式传入采样率
- 输出：卷积投影后的初始表示与 12 个 Transformer block 的 hidden states
- 参数：完全冻结，不训练、不微调
- revision：使用 `main` 在 2026-08-11 的 2203 解析快照；因 no-hash 政策不记录 Git 对象 ID，使用文件名、字节数、时间戳、配置内容与库版本形成非密码学 manifest，并明确其不可提供密码学不可变性保证

## 4. 数据与时间轴门禁

正式提取前必须生成轻量元数据表，至少包含：

- participant、session、task、run/recording；
- iEEG 文件、channels/electrodes、采样率、通道类型和状态；
- events 的 onset、duration、trial/stimulus 标识；
- 每个 stimulus 的 language；无法从正式 metadata 或文件结构确定时必须标记 `UNKNOWN` 并阻止对应跨语言声称；
- 音频文件名、字节数、时间戳、采样率、声道、时长和抽样可读性；
- 脑区/电极元数据可用性、缺失项与排除原因；
- 每个 recording 的音频—events—iEEG 时间关系。

不得用文件存在替代可读性，不得用事件行号猜测刺激身份。README 指示名称以 `C` 开头的通道未使用，正式排除规则须由 channels/electrodes 实际字段复核后执行。

## 5. 防泄漏 split

主分析定义为 `stimulus_generalization`：

- train/validation/test 比例目标为 70/15/15；
- 同一 `stimulus_id` 不跨 split；
- 真实 manifest 证明 `stimulus_id + recording_id` 联合分组形成单一 connected component，无法形成 train/validation/test；该原始方案保留为 `INFEASIBLE_SINGLE_CONNECTED_COMPONENT`；
- 主 split 采用确定性的 group-aware segment-count 比例优化：目标 70/15/15，只使用 block 的 eligible segment 数，不读取神经信号、模型性能或结果指标；固定 tie-break 后 block 01/02/05/06 为 train，block 03 为 validation，block 04 为 test，对应 223/48/48（69.9%/15.0%/15.0%）；
- 旧的随机种子 4/1/1 block 分配得到 208/32/79（65.2%/10.0%/24.8%），已由协调审核否决并作为失败历史保存，不得恢复为正式 split；
- recording 可跨 split，但 passage 窗口不得重叠。当前 2 s 只定义为 `preliminary_minimum_embargo`，因此 split 状态为 `PRELIMINARY_NOT_BASELINE_FINAL`；正式窗口化前必须计算 `final_embargo = max(2 s, maximum encoding lag, filter/padding edge, audio model receptive field/context overlap)` 并重新运行 guard；
- speaker_id 单独作为 advisory 审计；新 split 有 5 个 speaker 跨 split（`s1303a`、`s1401a`、`s2102a`、`s3301a`、`s3903b`），主结果不得声称 speaker-held-out 或 speaker-generalization；
- language 必须进入 manifest 并接受 split 覆盖审计；Catalan 音频存在 provenance 冲突，当前排除于 primary baseline 并标记 `LANGUAGE_GENERALIZATION_NOT_ESTIMABLE`；
- subject 可跨主 split，因此当前只支持 `within-subject unseen-stimulus/block generalization`；不支持 subject-held-out、speaker-held-out 或 cross-language 声称。任何新增 generalization 分析必须另立并审核独立 split manifest。

任何 sample_id 重复、必需 group key 缺失、stimulus/block 跨 split、language 缺失或时间邻域泄漏均为 hard fail。

## 6. 特征、目标与 baseline

首轮 acoustic baseline：

- broadband amplitude envelope；
- log-mel spectrogram 的低维 PCA 表征，PCA 只在 train 拟合。

首轮模型特征：

- wav2vec 2.0 initial projected state + 12 Transformer layers；
- 统一映射到预注册的神经时间网格；
- 任何标准化、PCA、特征选择只在 train 拟合并应用到 validation/test。

首轮 neural target 候选原为 `70-150 Hz high-gamma power`。G2 已确认采样率为 512/1024 Hz、工频为 60 Hz，因此 120 Hz 二次谐波位于候选带内；目标状态已设为 `REDESIGN_REQUIRED_BEFORE_G3`。在正式冻结明确的 60/120 Hz rejection/filter edge 或重新设计频带前，禁止神经特征提取，不得静默沿用原目标。

主模型为 ridge encoding：

- alpha 仅在 grouped train/validation 内选择；
- 主指标：电极级 held-out Pearson r；
- 辅指标：held-out R2 与被试汇总；
- anatomy gate 当前为 `ANATOMY_MAPPING_NOT_READY`：标准 `electrodes.tsv`/`coordsystem.json` 均缺失，9 份非标准 contact RAS CSV 没有脑区标签。因此 `region_summary=NOT_ESTIMABLE`；该门禁不单独阻塞电极级 smoke，但禁止从 contact 名称猜脑区；
- 不能只报告 pooled 最佳层。

## 7. null 与统计边界

至少保存：

1. `circular_time_shift`：在 recording/stimulus 内循环移位，移位量必须超过 lag、滤波边缘和特征感受野；
2. `stimulus_derangement`：在时长匹配桶内打乱 stimulus—neural 配对，保持 split 不变；
3. smoke 阶段至少 20 次，正式阶段默认 1000 次；随机种子和失败 permutation 全部保留；
4. 多层、多 lag、多电极比较必须使用预注册的 FDR 或 max-statistic 校正，不得事后选择更有利方法。

## 8. 内部 manifest 与跨项目 exchange contract

当前运行产物只使用 `schemas/m6a_public_internal_manifest.schema.json`。它是 Auditory 项目内部 run manifest，不是 M6A→M6B 合同，也不得作为已冻结跨项目 artifact 发布。

跨项目草案见 `doc/M6A_TO_M6B_EXCHANGE_CONTRACT_DRAFT.md` 与 `schemas/m6a_to_m6b_exchange_manifest_v1.schema.json`。STN consumer 二审与协调独立复跑已接受 revised DRAFT，当前 review 状态为 `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION`，consumer 状态为 `READY_WAITING_M6A_CANDIDATE`。这不是 release accepted/frozen：未生成真实 manifest、method package 和 canary 前不得升为 candidate；只有真实 candidate 完成 STN consumer cross-test 且协调接受后，才可另行建立不带 draft 的 frozen/accepted v1。

内部运行 manifest 必须包含：

- task/artifact 版本和状态；
- 数据集、模型、split、特征、目标、null、指标和失败边界；
- 文件名、字节数、时间戳、数量、schema 和 remote-only 状态；
- `can_claim` 与 `cannot_claim`；
- 原始数据与权重只记录 `REMOTE_ONLY`，不得复制进 artifact。

M6A→M6B 正式交付目标是冻结 method package、schema、公开 benchmark evidence 和 tiny synthetic/canary fixture。`ds004703` 原始数据、模型缓存和全量特征不得进入 STN 仓库。高维 feature payload 不得使用 TSV；正式数组格式须在 2203 用真实 shape 和读写访问模式完成 G3 基准后选择。tiny TSV 仅可作为 canary。

## 9. 阶段门禁

- `G0`：任务书、配置、schema、代码和单元测试通过；
- `G1`：2203 专用环境创建，自检记录 Python/包/GPU/路径，且不污染其他环境；
- `G2`：版本/许可/文件结构/抽样可读性审计通过，split guard 对真实轻量 manifest 通过；
- `G3`：单个 recording 的音频与神经时间轴可审计，acoustic 和 wav2vec2 特征 shape/时间网格一致；
- `G4`：至少一个被试的 ridge smoke run、null 和 held-out 指标可复现；
- `G5`：扩展到正式首轮样本并形成审核报告。

每个门禁失败均保留日志和状态，不自动提升后续结果。

## 10. 停止与资源规则

硬停止条件沿用 `doc/CURRENT_TASK.md`。此外：

- 单次 smoke GPU 运行上限 2 小时；
- 首轮数据与缓存预计达到 500 GB 前必须重新报告；
- 连续正式 GPU 运行预计超过 24 小时前必须重新报告；
- 磁盘可用空间低于 500 GB 时不启动新增大体积步骤；
- 下载失败、超时、缺文件和 no-go 均属于证据，不删除、不覆盖。

## 11. 可声称与不可声称

门禁通过后可声称：环境、数据结构、split 和 baseline 的工程可复现性；held-out 数据上的统计预测性能；相对于预注册 null 的证据强度。

不可声称：模型层等于脑区、相关即因果、可读即生物 decoder、跨数据集/跨语言/跨任务泛化、STN 适用性、实时因果系统或临床有效性。
