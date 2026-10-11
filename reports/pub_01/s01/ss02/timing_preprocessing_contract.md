# PUB-01-S01_SS02 — 时序与预处理契约

日期：2026-09-24  
范围：SparrKULee 3.1；OpenNeuro ds004703 1.1.0   
状态：仅冻结准入边界和可执行约定；不冻结 S02 的 fold、随机种子或模型超参数。

## 1. 来源、版本与使用范围

| 数据集 | 当前输入 | 许可/访问 | 本节点允许的范围 | 未决边界 |
|---|---|---|---|---|
| SparrKULee | v3.1；发布方 preprocessed EEG derivative + paired envelope | DOI `10.48804/K3VSND`；CC-BY-NC-4.0；4,142 个公开对象，196 个受限对象未请求 | 仅非商业、带署名/来源/许可/修改说明的 derivative 使用；640 条主候选可交 S02 继续冻结实验设计 | 49 条 raw 来源受限；历史生成 revision UNKNOWN；物理声学同步 UNKNOWN；26 条大尾差 HOLD |
| ds004703 | OpenNeuro v1.1.0；现有 11 EDF、270 WAV 及已生成索引 | 元数据 CC0 与 README 的非商业/禁止重识别限制并存；本节点保守按项目内部非商业使用 | 319 条英语 passage 只作为可追溯候选保留，不进入本轮 benchmark | immutable snapshot 身份、无条件再分发、物理播放同步均未充分核验；整体 HOLD |

SparrKULee 的两个 `.bdf`/`.bdf.gz` 记录名是同一来源的 alias，不计为额外录音。ds004703 的 `excerpts/Block *` 与根目录重复音频按 story/事件身份处理，不按路径增加样本；6 个 Catalan block 与 `pleasePressSpace` 波形相同，但这不能证明真实播放来源，因此不静默替换。

## 2. SparrKULee 配对与时间轴

- 666 条 EEG derivative 均能由发布方文件名映射到对应 envelope；服务器上 666 条 EEG/envelope derivative 路径均存在，配对状态为 `PUBLISHER_MAPPING_AND_COMMON_OVERLAP`。
- 首轮输入使用发布方已经处理的 EEG 与 envelope，采用发布方消费约定的共同前缀：起点 `0`，终点 `min(eeg_samples, envelope_samples)`，不重新滤波、重参考、重采样、归一化或截断到自选时窗。
- 当前 derivative 的配对率为 64 Hz；`ENVELOPE_64` 只表示 envelope/发布 EEG 的输入时间基准。raw stimulus NPZ 的采样率另行记录，不能用来替代 derivative 时间轴。
- 公开 raw header 核验结果为 617 条、raw 受限为 49 条。公开来源的实际 header 为 1024 Hz，而 dataset sidecar 为 8192 Hz；二者差异被保留，当前输入使用发布方 derivative 64 Hz，不把 sidecar 8192 Hz 当作 raw header 事实。
- 另有 1 条 raw 本身公开、但对应的 `stimulation.tsv` 受限；连同 49 条 raw 受限记录，共 50 条记录的 `stimulation_story_match=UNKNOWN`。现有 derivative/envelope 配对证据足以支持发布方 common-overlap 输入，但不足以支持该记录的 raw trigger/story 匹配或 physical-sync claim。
- `neural_offset=0` 是数组共同起点约定，不是 loopback、TTL 或声学硬件延迟的测量。当前没有证据支持硬件延迟、物理播放时间或精确声学同步的 claim。
- 640 条记录的 EEG 尾差在历史 192-sample（3 秒）审阅标记内，进入 `PASS_WITH_LIMITATION` 主候选；26 条 `eeg_tail_samples > 192` 的记录逐条保留为 `HOLD`，不通过修改阈值或邻近切片来增加准入。192 只是历史提示，不是本节点新拟合的科学合格阈值。
- 72 个可读 `data_dict` 与 published stimulus provenance 一致；另有 1 个 `podcast_35-1` 截断孤立对象，无同名 envelope/EEG 引用，作为外部对象排除，不改变 666 条记录的准入计数。

允许的 EEG 主候选用途是发布 derivative/common-overlap 上的非商业 encoding/retrieval/representation benchmark。不得把该输入解释为测得的硬件同步，不得把 derivative-only 结果扩展成 raw header、物理播放或跨数据集生物学同源证据。

## 3. ds004703 配对与时间轴

- 服务器现有 v1.1.0 输入包含 11 个 EDF 记录，原生采样率为 512 或 1024 Hz；音频 QC 覆盖 270 个 WAV，音频率按各路径记录（现有清单为 44.1 kHz）。不为本节点重采样到共同神经率。
- 438 个事件块逐条保留；其中 319 个英语 passage 通过现有事件模板的 word/POS/phone、duration 与 constant-offset 审计，进入可执行 segment index。78 个 control block 和 41 个 Catalan block 不进入 benchmark。
- 对 319 个英语片段，虚拟 segment 的样本窗口约定为：

  `start = floor(onset_s * native_sfreq)`；`stop_exclusive = ceil((onset_s + audio_duration_s) * native_sfreq)`。

  窗口为半开区间，保留原 EDF 原生样本；服务器已对 319 条索引读取验证，11 个 EDF 中每个抽查 3 个位置，共 33 次数值比较，最大绝对差为 0。
- `offset_s`、`audio_duration_s`、`template_margin_s` 是事件/模板层的时间字段。它们支持当前 metadata-defined segment mapping，但不证明真实扬声器播放、触发线或神经硬件之间的 physical sync；`physical_sync=UNKNOWN` 保持不变。
- 6 个 Catalan block 的标记音频与 `pleasePressSpace` 波形相同，当前只记录冲突并排除；不使用另一版本文件“修复”播放来源。

首轮 ds 输入是原始 EDF 的只读、索引型虚拟 derivative：只应用已审计的通道/片段排除索引，不做滤波、重采样、重参考、ICA、插值、归一化或 imputation。通道排除沿用官方 bad 标记、README 的 C-prefix 规则、非神经类型、nonfinite 和精确平直规则；虚拟 derivative 依赖原 EDF，不能被描述成已经完成科学预处理或硬件同步校正。

ds004703 当前不能交给 S02 或 intracranial transfer：319 条记录的配对/索引证据是可审计的，但许可信号、不可变版本身份及 physical sync 尚不足以把它们升级为 PASS。

## 4. 跨数据集禁止事项与交接边界

1. 不从模型效果、相关系数或任何 benchmark 表现反推 audio-neural pairing 或 physical sync 正确。
2. 不跨 subject/session/story 随机切邻近时间片；重复 story、音频副本、`.bdf`/`.bdf.gz` alias 必须共用分组键。
3. 不在 test 上拟合 normalization、PCA、feature/layer selection 或任何统计量。
4. 不把 raw source 受限等同于 derivative 必然不可用；但受限 raw 记录不得声称完成 raw header、硬件同步或原始预处理复现。
5. 不在 SS02 冻结具体 fold、种子、模型超参数；这些交给 S02，在本契约和 `grouping_feasibility.csv` 的约束内完成。

本节点可交接的范围只有 SparrKULee 640 条主候选及其限制。ds004703 319 条候选继续 HOLD；该 HOLD 不连带阻塞已通过范围的 EEG 分支。
