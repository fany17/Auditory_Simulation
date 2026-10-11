# PUB-01-S01_SS02 — 数据子集与实验准入复核

日期：2026-09-24  
执行主机：`server2203`；项目根：`/home/fanyu/auditory_simulation_m6a`  
本地交付：`reports/pub_01/s01/ss02/`  
结论状态：SS02 `COMPLETED`；本报告交人审状态为 `READY_FOR_REVIEW`；S01 整体仍为 `REVIEW`。

## 1. 先给决定

| 数据集/子集 | 候选记录 | 逐记录去向 | 本节点决定 | 可交接用途 | 主要限制 |
|---|---:|---|---|---|---|
| SparrKULee 主候选 | 640 / 666 | `INCLUDE` | `PASS_WITH_LIMITATION` | 可交 S02 冻结 EEG benchmark 的数据范围 | 64 Hz publisher common-overlap；物理同步 UNKNOWN；49 条 raw 受限；历史 generator revision UNKNOWN |
| SparrKULee 大尾差 | 26 / 666 | `HOLD` | `HOLD` | 不进入主 fold；后续若使用须单独声明 sensitivity/secondary 分析 | 每条均为 `audiobook_2_2`；192 samples 仅为历史审阅标记，不得为扩样放宽 |
| ds004703 英语 passage | 319 / 319 候选片段 | `HOLD` | `HOLD` | 暂不交 S02 或 intracranial transfer | README/CC0 使用边界冲突、immutable snapshot 身份不足、physical sync UNKNOWN |
| ds004703 control | 78 / 438 事件块 | `EXCLUDE` | `NO_GO`（对 benchmark） | 仅保留排除 provenance | `CONTROL_NOT_PASSAGE` |
| ds004703 Catalan | 41 / 438 事件块 | `EXCLUDE` | `NO_GO`（对 benchmark） | 仅保留排除 provenance | played-audio provenance conflict；不得静默换音频 |

候选清单共 1,104 条唯一记录键：SparrKULee 666 条 + ds004703 438 个事件块。最终逐记录计数为 `INCLUDE=640`、`HOLD=345`、`EXCLUDE=119`；三类互斥且无漏项。319 条 ds004703 英语片段虽然配对/样本窗口可审计，但因数据集级 HOLD 不得写成 PASS。

## 2. 来源、配对和异常处置

### SparrKULee

- 官方 RDR V3.1 清单冻结 4,338 个对象：4,142 个公开、196 个受限；本节点不请求受限对象。CC-BY-NC-4.0 允许当前非商业研究范围，但要求署名、来源、许可和修改说明；不默认允许商业或无条件再分发。
- 666 个 published EEG derivative 与对应 envelope 通过 publisher mapping/common overlap 闭合。服务器目标路径逐条存在；64 Hz、起点 0、共同前缀是输入约定，不是物理同步测量。
- 617 条 raw header 已核验，49 条 raw 受限；受限 raw 不访问，但 derivative 仍可在“derivative-only、无 raw header/physical-sync claim”的边界内使用。另有 1 条 raw 公开但 `stimulation.tsv` 受限，因此共 50 条记录的 raw trigger/story match 为 UNKNOWN；这 1 条不因 stimulation 说明受限而自动排除，但禁止相应 raw trigger/physical-sync claim。逐记录清单中 640 条主候选含 596 条公开/alias-resolved 来源和 44 条受限来源；26 条尾差 HOLD 中含 21 条公开、5 条受限来源。
- 26 条大尾差逐条写入 `admission_inventory.csv`，没有相减估算最终样本数，也没有把历史 192-sample 提示改写成验收阈值。孤立截断 `podcast_35-1.data_dict` 无对应 EEG/envelope 引用，保留在既有 exclusion provenance，不伪造配对。

### ds004703

- 当前现有 v1.1.0 资产完成 377 对象大小/格式盘点；11 EDF、270 WAV 和 438 事件块均有轻量/服务器证据。319 条英语片段与 S06 `clean_segment_index.csv` 一一闭合，服务器虚拟窗口验证 319/319 PASS；48 个实际英语音频路径均在 270 条音频 QC 中。
- 原生 EDF 率为 512/1024 Hz；319 条窗口使用 `floor(onset*sfreq):ceil((onset+audio_duration)*sfreq)` 半开区间，保持原生样本。33 次直接数值比较最大差为 0，但这仍只证明索引/读取一致，不证明 physical sync。
- 78 个 control block 和 41 个 Catalan block 排除。6 个 Catalan 播放标记音频与 `pleasePressSpace` 波形一致，不能以另一个文件替代。72/70/62 的 raw stimulus 角色统计分别是 published stimulus、filename trigger 和 auxiliary unverified；filename role 不是播放证明。
- README 的非商业/禁止重识别限制与 CC0 元数据并存；本节点不为此补造许可解释，也不批准再分发。不可变内容身份仅能追溯到 v1.1.0 路径与已有大小对账，保留 UNKNOWN/HOLD。

## 3. 分组可行性结论

`grouping_feasibility.csv` 是可行性与约束表，不是正式 fold manifest。

- SparrKULee 主候选有 85 subjects、11 sessions、640 records、71 个 raw story labels；subject-holdout 结构可行，用于回答 unseen-listener generalization。story-holdout 可行但受 alias-family 未冻结和 4 个单被试 story label 限制；必须先冻结 canonical story family，再把重复/shifted/artefact 关系放在同一分组中。subject×story 联合留出可设计，但稀疏性需要 S02 检查，不能由随机切片补齐。
- SparrKULee 26 条尾差不参加主 split；它们全部落在 `audiobook_2_2` raw label，保留单独 HOLD。
- ds004703 英语候选有 10 subjects、2 sessions、11 recordings、48 个 eligible story labels；每个 story 覆盖 4–10 个 subject，subject-holdout、story-holdout 和 subject×story 结构上均可表达，但整体继续 HOLD。319 条记录对应 317 个非空 subject-story cells，说明存在重复 trial，不能把 record count 当作独立 story count。
- 两数据集均禁止 adjacent-time random split；所有重复 story/clip、recording identity、subject/session nested records 和 audio aliases 必须在同一 split。所有 normalization/PCA/feature/layer selection 只能在训练分区拟合。

## 4. 预处理与时序契约

完整约定见 [`timing_preprocessing_contract.md`](timing_preprocessing_contract.md)。核心交接是：SparrKULee 使用 publisher derivative/common-overlap，不重新滤波或重采样；ds004703 使用只读 EDF 的索引型虚拟 derivative，不做滤波、重参考、重采样、ICA 或 imputation。任何 `offset=0`、constant onset offset 或模板时长一致都不能升级为硬件/声学同步证明。

## 5. 定向检查与证据

### 已复用证据

- `reports/pub_01/s01/current_audit_20260920_v2/` 六类清单：配对、时序、分组、许可、数据集 manifest 和排除。
- `reports/pub_01/s06/final_handoff_20260920_v1/`：公开范围交付、recording QC、边界和验证摘要。
- `reports/pub_01/s06/ds004703_qc_v2/`：事件/EDF 配对、音频 QC、segment index、virtual validation 和 channel index。
- `reports/pub_01/s06/sparrkulee_channel_provenance_20260920_v3/` 与 `raw_stimulus_roles_20260920_v2/`：raw access/header 与刺激角色限制。

### 仅针对缺口的一次批量定向检查

通过 SSH（`ClearAllForwardings=yes`、`ForwardAgent=no`）在 2203 使用 `/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python`，只读取既有 CSV/JSON、索引路径和 derivative 路径存在性；没有下载、解压、原始信号重读、全量 QC、模型训练或新派生物写入。结果：

1. `pair/timing/group` 三表均 1,104 行，`dataset+sample_key` 唯一且完全对齐。
2. SparrKULee 666 条；ds004703 438 条，其中 319 PASS_WITH_LIMITATION、119 EXCLUDE。
3. S01 的 319 条 ds 英语 `sample_key+story` 与 S06 eligible segment index 完全闭合：缺失 0、额外 0。
4. 319 条 virtual segment validation 全部 PASS；48 个 eligible 音频路径均在 270 条音频 QC 中，音频非有限计数为 0。
5. SparrKULee 666 条 derivative EEG/envelope 路径均存在；640 条 `WITHIN_3S_ADVISORY`、26 条 `LARGE_TAIL_REVIEW`，physical sync 及 split 均保持 UNKNOWN/UNASSIGNED。
6. raw provenance 统计为 617 条 header verified、49 条 restricted raw；另有 1 条公开 raw 的 `stimulation.tsv` 受限，逐记录清单分别记录 `source_access` 与 `stimulation_access`；未访问受限对象。

第一次调用因服务器虚拟清单目录名是 `sparrkulee_virtual_20260919_v4` 而非本地副本的扁平名称，在读取前即因路径不存在退出；修正为服务器现有目录后按同一批检查完成，前一次没有信号读取或写入。

## 6. 独立准入复核与三层验收

本节点已重新读取四项最终产物，并抽查正常记录、26 条 tail HOLD、ds 英语 HOLD、control 排除和 Catalan 排除；补入 stimulation access/story-match 字段后再次复核，确认 schema 可读、记录 ID 唯一、`INCLUDE/HOLD/EXCLUDE` 互斥、关键时序字段无未解释空值、四项文件之间计数一致。独立复核输出为 `INDEPENDENT_REVIEW=PASS_AFTER_STIMULATION_FIELDS`，inventory 1,104 行、unique ID 1,104、grouping feasibility 8 行；49 条 raw restricted 与额外 1 条 restricted `stimulation.tsv` 的交叉计数分别为 49 和 1。

- Technical validation：`PASS`；见上述 1,104 行闭合检查和最终产物复读。
- Acceptance validation：`PASS_WITH_LIMITATION`（SparrKULee 640 条主候选）；`HOLD`（26 条大尾差与 ds004703 319 条）；`NO_GO`（119 条明确排除事件块）。
- Parent consistency：`PASS`；未读取患者/STN 数据，未训练模型，未改变 PUB-01 scope/claim/license/leakage boundary，未生成 hash/checksum 审计。

## 7. S01 状态建议与交接

建议 S01 继续保持 `REVIEW`，原因是存在两个实质未决范围：SparrKULee 26 条 tail HOLD，以及 ds004703 数据集级许可/immutable identity/physical sync HOLD。S01 不应标记整体 COMPLETED，也不应把混合决定压缩成一个笼统总 HOLD。

同时，SparrKULee 640 条主候选已经具备交给 S02 的数据范围、时序/预处理边界和分组可行性依据；ds004703 HOLD 不连带阻塞这条 EEG 分支。S02 若获明确启动授权，仍须自行冻结正式 fold、种子、训练/验证/测试分配和模型超参数；本 SS02 不替代该冻结，也没有启动 S02。

SS02 停止于本审阅门，不自动启动 S02、S03、S04 或患者侧任务。
