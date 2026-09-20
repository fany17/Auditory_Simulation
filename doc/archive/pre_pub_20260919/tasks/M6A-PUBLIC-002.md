# M6A-PUBLIC-002：预训练声音/时序架构一周复现

| 字段 | 内容 |
|---|---|
| 状态 | `ACTIVE_EXECUTION` |
| 周期 | `2026-08-13` 至 `2026-08-20` |
| 执行位置 | `server2203:/home/fanyu/auditory_simulation_m6a` |
| 训练 | `ZERO_TRAINING` |
| 患者数据 | `FORBIDDEN` |

## 1. 本周目标

本周禁止设计新架构。只复现可公开下载、已有预训练权重、无需任何训练即可 inference / feature extraction 的成熟声音与时序模型，以及 auditory-physiology 模型。

要回答：

1. 成熟 AI 已怎样处理局部声音结构、长时间上下文、状态记忆和时间压缩；
2. 已有 auditory-inspired / auditory-physiology 模型已经模拟听觉系统到什么程度；
3. 哪些所谓“听觉启发”其实已被成熟架构覆盖，哪些仍可能构成真实缺口。

`M6A-PUBLIC-001` 保留为 ds004703 / wav2vec2 已有 checkpoint，本周不继续扩大其 subject、recording、模型或神经拟合范围。

## 2. 硬约束

正式模型必须：

- 有公开可下载 pretrained checkpoint / trained weights；
- 能通过官方/作者 inference 路径执行；
- 不从头训练、不微调、不继续预训练；
- 不训练 linear probe、ridge、classifier 或任何 downstream readout；
- 不使用患者 STN 数据；
- 不用不同原始任务的 accuracy/WER/SDR 直接横向判定模型优劣。

允许：下载、环境安装、冻结 inference、中间表示提取、shape/frame-rate/context 审计，以及无需拟合参数的 representation distance、autocorrelation、RSA/CKA 等分析。

## 3. 正式模型清单

| ID | 架构角色 | 具体 pretrained 实现 | 原始任务 |
|---|---|---|---|
| `PANNs-CNN14` | CNN | `Cnn14_16k_mAP=0.438.pth` | AudioSet audio tagging |
| `ConvTasNet` | temporal convolution | Asteroid 16 kHz pretrained ConvTasNet，优先 `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k` | speech enhancement |
| `SpeechBrain-CRDNN` | CNN + BiLSTM | `speechbrain/asr-crdnn-rnnlm-librispeech` | LibriSpeech ASR |
| `Parakeet-TDT` | FastConformer | `nvidia/parakeet-tdt-0.6b-v3` | multilingual ASR |
| `Audio-Mamba` | selective SSM | 官方 SSAM pretrained weight，优先 tiny 配置 | self-supervised audio representation |
| `wav2vec2` | speech SSL Transformer | `facebook/wav2vec2-base` | speech SSL |
| `Whisper` | encoder-decoder Transformer | OpenAI `turbo` / `large-v3-turbo` | multilingual ASR |
| `CoNNear-periphery` | auditory periphery | `HearingTechnology/CoNNear_periphery` | cochlea/IHC/ANF physiological surrogate |
| `ICNet` | inferior colliculus | `fotisdr/ICNet` | simulated IC population neural responses |

`LEAF` 本周排除：重要但需要任务训练，不满足 `ZERO_TRAINING`。HuBERT、WavLM、BEATs、MERT、AST、Branchformer、Zipformer、S4 等仅进候选 registry，不执行。

## 4. CoNNear：原任务与本周任务

### 原任务

CoNNear 不是声音分类器。其 pretrained CNN 模块近似人类听觉外周解析/生物物理模型：

```text
waveform
→ cochlea
→ basilar-membrane vibration
→ IHC receptor potential
→ ANF-H / ANF-M / ANF-L firing rate
```

官方 `CoNNear_periphery` 包含 5 个 trained AECNN：cochlea、IHC、三类不同 spontaneous-rate 的 ANF。默认沿 201 个 cochlear characteristic-frequency channels 输出。

### 本周必须完成

1. 下载官方仓库与 trained weights，禁止重训；
2. 跑通官方 `connear_example.py`；
3. 原样复现官方 pure-tone multi-level 示例；
4. 导出/汇总 BM、IHC、ANF-H、ANF-M、ANF-L；
5. 输入统一 temporal probes，观察 onset、周期、jitter、omission、phase shift 在外周各级如何变化；
6. 输出 `sound → BM → IHC → ANF` 逐级时间表示图和 201-CF summary。

不要把仓库附带的解析 CN/IC backend 当成 CoNNear 已学习到的中枢模型。

## 5. ICNet：原任务与本周任务

### 原任务

ICNet 不是普通 audio classifier，也不是人脑模型。它是 convolutional encoder-decoder，用 9 只正常听力、麻醉沙鼠的 inferior colliculus 大规模 multi-unit recordings 建模 shared sound coding：

```text
sound
→ encoder
→ shared bottleneck latent
→ decoder
→ simulated IC population activity
```

官方 pretrained 模型可模拟特定动物、跨 9 只动物汇总的 neural units（如 `units_1000`），或直接输出共享 `bottleneck`。

### 本周必须完成

1. 下载 `fotisdr/ICNet` 的 pretrained architecture/weights/configuration，禁止重训；
2. 跑通官方 `ICNet_example.ipynb` 或 `ICNet_example.py`；
3. 用仓库自带 `scribe_male_talker.wav` 跑通原始示例；
4. 固定导出 `bottleneck`、`units_1000` 与 CF/tonotopic summary；
5. 输入统一 temporal probes，比较 regular/jitter/omission/phase-shift 在 bottleneck 和 simulated IC population 中的变化；
6. 至少复现一个 neurophysiology sanity check，优先 `forward masking`；若成本很低再做 `dynamic range adaptation`；
7. 输出 `sound → bottleneck → simulated IC population` 时间表示图和 tonotopic population summary。

不得把沙鼠 IC 外推成人类 IC，也不得把 bottleneck 预先命名为认知/语义变量。

## 6. 统一 temporal probes

使用同一组轻量输入，采样率不同时只做确定性 resampling 并记录：

1. 单纯音 `tone`；
2. 规则 click train `regular_clicks`；
3. 相同事件数/平均速率的 `jitter_clicks`；
4. `omission`；
5. 单次 `phase_shift`；
6. 一段统一短自然语音 `speech`。

synthetic probe 全部脚本生成，不下载大型 benchmark。

## 7. 零训练统一分析

每个模型/关键 stage 至少记录：

- 输入预处理与 sample rate；
- tensor shape；
- 时间 frame rate / stride；
- temporal downsampling factor；
- convolution / recurrent state / attention / SSM 机制；
- parameter count（可靠可得时）；
- inference wall time；
- peak GPU memory（可靠可测时）；
- event-triggered activation；
- baseline vs perturbation representation distance；
- autocorrelation / temporal persistence；
- 无需拟合参数的 within-model RSA/CKA。

禁止直接比较异构 embedding 的原始幅值。

## 8. 执行顺序

### Stage A — registry / download gate
为 9 个模型记录 repo/model id、论文/DOI、architecture、原始训练任务、checkpoint、大小、license、framework、sample rate、download status、inference status。

### Stage B — official example first
每个模型必须先跑作者/官方 example。失败先记录原因，不立即重写模型。

### Stage C — unified temporal probe
官方 example 通过后才进入统一 probe。

### Stage D — comparison
输出 architecture/temporal representation matrix，回答哪些生物启发概念已被成熟 AI 覆盖，以及 CoNNear/ICNet 额外实现了哪些有明确生理定义的变换。本周不得实现新架构。

## 9. 最小交付物

1. `reports/model_registry_pretrained_week.*`
2. `reports/pretrained_model_smoke_summary.*`
3. `reports/temporal_probe_summary.*`
4. `reports/architecture_comparison_matrix.*`
5. `reports/CONNEAR_REPRODUCTION.md`
6. `reports/ICNET_REPRODUCTION.md`
7. `reports/M6A-PUBLIC-002_WEEKLY_REPORT.md`

报告优先写清楚“输入是什么 → 模型做什么 → 输出是什么 → 哪些时间信息保留/改变”，不要堆安装日志。

## 10. 一周 PASS 标准

- 9/9 pretrained source 完成审计；
- 至少 7/9 官方 inference 跑通；
- CoNNear 与 ICNet 两条 auditory-physiology 主线必须优先跑通；若被官方依赖阻断，给出可复现失败证据；
- 至少 7/9 完成可执行的统一 temporal probe；
- `0` 次训练；
- `0` 次微调；
- `0` 次患者数据读取；
- 不提出未经 baseline 支持的新架构。

## 11. 停止/降级规则

- 单个非核心模型环境修复成本过高时，标记 `BLOCKED_BY_UPSTREAM_DEPENDENCY` 后继续；
- 权重要求本人接受新条款、凭据或付费时立即停止；
- checkpoint 不可合法公开下载则降级为 registry-only；
- 不为追求 9/9 而训练替代模型；
- 不修改 `STN_Decoding_Encoding`，不导入患者数据。
