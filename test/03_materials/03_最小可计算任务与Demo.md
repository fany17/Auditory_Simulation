# wav2vec 2.0 本地复现与层级交互 Demo：独立执行规范

> 文档性质：第一项最小可运行 Demo 的**独立执行规范**；完整发送本文件即可作为单 Agent 的执行边界，不依赖原项目中的 `01`、`02`、README、AGENTS 或其他上下文。它不是已经实现的软件包，也不是正式任务书或 Agent 状态文件。  
> 建议归属：`TB001-DEMO001`；该标识不占用正式 TA/TB 任务编号。  
> 核心交付：一个本地推理流程和一个无需联网即可浏览结果的 HTML 控制台。  
> 默认执行方式：一个 Agent 在同一会话中完成审计、环境、实现、推理、HTML、测试、自检和汇报；不要求创建 Planner/Executor 两个会话。  
> 当前状态：只完成方案设计和官方资料核验；尚未下载权重、安装环境、运行模型或生成实验结果。  
> 独立交付版：`2026-08-06-r1`；核验日期：2026-08-06。

# 0. 先确认交付物性质和启动方式

## 0.1 本文件可以独立发送，但它不是现成代码

本文件面向**具备文件、命令行和 Python 操作能力的编码 Agent**。它已经包含目标、授权边界、输入、模型、干预公式、目录、输出、测试、失败条件、报告要求和停止点。收件人无需取得原项目的其他文档，也不得把原项目中的隐含约定当作前置条件。

必须完整发送本文件，不能只截取第 17 章。第 17 章是启动 Prompt；第 0—16 章是它引用的完整技术规范。若收件人只是普通使用者、不会编写和测试 Python，则本文件本身不足以“一键运行”，应先由编码 Agent 生成并验收第 12 章规定的软件交付物，再把软件包交给普通使用者。

## 0.2 中性工作根目录

收件人应新建一个专用空目录作为 `DEMO_ROOT`，把本文件原样保存为：

```text
DEMO_ROOT/
└── docs/
    └── STANDALONE_EXECUTION_SPEC.md
```

后文所有相对路径均相对于 `DEMO_ROOT`。允许使用任意绝对路径；不得假设目录名是 `Auditory_Simulation`，不得访问原发送者的上级目录，也不得要求 `01_听觉时变信息处理项目总纲.md` 或 `02_从零启动算法研究操作手册.md` 存在。

如果目标目录不是空目录，Agent 必须先列出现有文件，只能新增或修改本规范明确列出的 Demo 文件，不得覆盖无关内容。若它不是 Git 仓库，不得初始化 Git；若它已经是 Git 仓库，本任务也不授权提交、推送、切换分支、重置或清理工作区。

## 0.3 最低运行前提

开始前必须确认并记录：

- Python 3.10 或 3.11；
- 可导入的 PyTorch、Transformers、NumPy 和一种能读取/写入 WAV 的后端；
- 至少 10 GiB 可用磁盘和 8 GiB 可用内存；
- CPU 可以运行；CUDA 只是可选加速，不得成为正确性的前提；
- 首次获取模型和公开语音时可访问官方来源；模型与输入落盘后，推理和 HTML 浏览必须可断网进行。

优先复用已经通过 import/CUDA smoke test 的隔离环境。若需新建环境，只能在 `DEMO_ROOT/.venv`、`DEMO_ROOT/.conda` 或管理员明确指定的位置创建，不得修改系统/base 环境。依赖安装后必须保存 `python --version`、`pip freeze`、CUDA 可用性和实际设备；不得只在文档里预写一个未经运行的“理想版本”。

## 0.4 可复现性与来源固定

首次联网阶段必须把 `facebook/wav2vec2-base-960h` 的 `main` 解析为完整 commit SHA，后续 `from_pretrained` 对模型和 processor 使用同一个完整 SHA。记录实际下载的配置、词表、processor 和权重文件清单、文件大小与 SHA256；不得只记录可漂移的 `main`。

输入优先使用参与者自录且有书面授权的英语语音。若没有自录音，可从 Hugging Face 的 `hf-internal-testing/librispeech_asr_demo` **只取一个** 2–6 秒样本；先解析该数据集的完整 revision 和具体 `row_idx/file/id`，再只下载该行的音频，不下载 9 MiB demo 子集或完整 LibriSpeech。来源须追溯到 [OpenSLR SLR12](https://www.openslr.org/12)，并在 manifest 中记录其 CC BY 4.0 许可、说话者/章节/utterance ID、原始转写、下载 URL、revision 与 SHA256。如果无法确认具体样本的来源和许可，检查点 S1 必须记为 `BLOCKED`，不能改用来历不明的音频。

## 0.5 三类“token”必须分开

本 Demo 的 wav2vec 2.0 推理不调用文本生成 API，因此没有 OpenAI/Hugging Face **计费 token**。CTC frame 数、greedy token ID 数和转写字符数是模型内部统计，不得与 Agent 的 LLM token 混为一谈。

如果本规范由外层编码 Agent 通过 SSH 驱动服务器执行，服务器只运行 Python、模型、测试和静态服务，不在服务器上启动第二个 Codex Agent。Agent token 只能由**控制端会话**的运行器统计：

```text
input_tokens
cached_input_tokens
output_tokens
total_tokens = input_tokens + output_tokens
```

`cached_input_tokens` 是输入 token 的子集，不得再次加到 total。若控制端运行器提供 usage，最终只把脱敏摘要写入 `reports/TB001-DEMO001_TOKEN_USAGE.json`；若控制端没有返回 usage 字段，只能标记 `UNAVAILABLE`，不得用字符数伪装成精确 token。SSH 命令、服务器 Python 和 wav2vec 2.0 推理不会产生另一份 Agent token；不得为了统计 token 在服务器上额外启动 `codex exec`。

# 1. Demo 要回答什么

本 Demo 不再使用手算事件序列，而是直接复现一个公开的预训练语音模型：

> 使用 `facebook/wav2vec2-base-960h` 对一条本地英语语音进行推理，观察并干预 12 个 Transformer 层，展示不同层被旁路或减弱后，转写输出和中间表示怎样变化。

最小闭环是：

```text
本地16 kHz英语语音
→ wav2vec 2.0卷积前端
→ 12层Transformer
→ CTC输出
→ 保存基线转写与各层hidden state
→ 选择一个层并调节其残差贡献
→ 重新运行后续层与CTC头
→ 比较转写、CTC分布和表示变化
→ 在离线HTML控制台中交互展示
```

核心不是追求识别准确率，也不是重新训练模型，而是让使用者直观看到：

1. 模型由哪些阶段组成；
2. 选择不同层时，中间表示有什么差异；
3. 减弱或旁路某一层后，下游输出是否改变；
4. 哪些改变只影响表示，哪些会传递到最终转写；
5. 这些现象能支持什么结论，不能支持什么结论。

# 2. 为什么选择这个模型

## 2.1 建议模型

```text
模型ID：facebook/wav2vec2-base-960h
任务头：Wav2Vec2ForCTC
语言：英语
输入：16 kHz单声道语音
```

选择理由：

- 模型已经预训练并在 LibriSpeech 960 小时英语语音上微调，可直接进行 CTC 转写；
- 官方模型卡提供 Transformers 加载示例；
- 配置为 12 个 Transformer hidden layers，规模足以展示层级变化，又比 large 模型更适合作为本地第一轮；
- 官方 Transformers 接口支持 `output_hidden_states=True`；
- hidden state 的最后一维为 768，各 Transformer 层输入输出形状一致，适合做残差贡献缩放；
- 模型卡标注许可证为 Apache-2.0。

## 2.2 已核验的官方信息

| 项目 | 当前核验结果 | 来源 |
|---|---|---|
| 模型当前可访问 | 是 | [官方模型卡](https://huggingface.co/facebook/wav2vec2-base-960h) |
| 模型用途 | wav2vec 2.0 Base，预训练并在 LibriSpeech 960 h 上微调的英语 CTC 模型 | [官方模型卡](https://huggingface.co/facebook/wav2vec2-base-960h) |
| 输入采样率 | 16 kHz | [官方模型卡](https://huggingface.co/facebook/wav2vec2-base-960h) |
| 许可证 | Apache-2.0 | [官方模型卡](https://huggingface.co/facebook/wav2vec2-base-960h) |
| 卷积特征层 | 7 层 | [官方 config](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json) |
| Transformer 层 | 12 层 | [官方 config](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json) |
| hidden size | 768 | [官方 config](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json) |
| attention heads | 12 | [官方 config](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json) |
| hidden state 接口 | embeddings 输出加每层输出 | [Transformers 官方文档](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) |
| 原始实现状态 | fairseq 官方仓库截至核验日已归档、只读 | [官方 fairseq 仓库](https://github.com/facebookresearch/fairseq) |

工程建议：第一轮使用当前 Transformers 接口复现，不以已归档的 fairseq 仓库作为新环境主入口。实际执行时仍需冻结 `transformers`、`torch`、模型 revision、权重文件哈希和处理器版本。

# 3. 最小任务边界

## 3.1 只做这些

1. 准备一条 2–6 秒的本地英语单声道语音；
2. 转换或核验为 16 kHz；
3. 用原模型完成一次 baseline CTC 推理；
4. 提取 embedding 输出和 12 层 Transformer hidden states；
5. 对每个 Transformer 层做两种非基线强度的残差贡献调节；
6. 保存转写、CTC 统计、中间表示差异、运行时间和配置；
7. 用本地静态 HTML 展示所有结果。

## 3.2 第一版明确不做

- 不重新预训练或微调 wav2vec 2.0；
- 不同时比较 HuBERT、WavLM、Whisper 或其他大型模型；
- 不做大数据集 WER benchmark；
- 不调整卷积前端的 7 层；
- 不训练复杂 probe；
- 不把层编号命名为“语义层”“节律层”或某个脑区；
- 不把单条语音的结果写成普遍机制；
- 不要求模型在浏览器中直接运行；
- 不要求实时 streaming；
- 不把 HTML 手工填入的示例值冒充模型输出。

卷积前端在 HTML 架构图中显示，但第一版只读，不提供开关。原因是卷积层具有不同 kernel、stride 和中间形状，直接旁路会同时改变时间分辨率和张量接口，不适合作为第一项最小实验。

## 3.3 单 Agent 执行边界

本 Demo 有意小到可以由一个 Agent 完成。人类把**本文件全文**作为上下文，并发送第 17 章的启动 Prompt 后，同一个 Agent 可以：

1. 阅读本文件全文并确认 `DEMO_ROOT`；
2. 审计运行环境和 `DEMO_ROOT` 现有文件；
3. 在授权范围内复用或建立隔离环境并安装最少依赖；
4. 下载且只下载指定模型、processor 和一条合法来源的演示语音；
5. 实现 baseline、hidden-state 提取和单层干预；
6. 完成 25 次唯一推理；
7. 生成 JSON、图和离线 HTML；
8. 运行测试并自行核对页面；
9. 编写一次完整执行报告后停止。

这个单 Agent 不需要先创建 `PROJECT_CHARTER.md`、`CURRENT_TASK.md`、`AGENT_PROTOCOL.md` 或双会话交接文件。本独立规范本身给出固定范围和验收条件。若目标机器的全局规则强制要求交接日志，应创建并追加该日志，但不得因此扩大科学任务或强行初始化 Git。

单 Agent 仍不得：

- 擅自把任务扩展成正式 TB001；
- 换用其他大型模型；
- 下载完整大数据集；
- 降低测试标准来宣布成功；
- 隐藏失败 run；
- 初始化、提交或推送 Git，除非人类另行明确授权；
- 删除已有项目材料。

如果后续进入多模型、多人审核、正式 TB001 或长期实验，应另行提供协作规范；任何原项目中的双 Agent 工作流都不是当前 Demo 的前置条件。

# 4. “调节一层”具体是什么意思

必须把“观察一层”和“改变一层”分开。

## 4.1 观察模式

使用官方 `output_hidden_states=True`，读取：

```text
H0：Transformer之前的embedding/projection输出
H1：Transformer第1层输出
...
H12：Transformer第12层输出
```

官方文档说明 hidden states 包含初始 embedding 输出和每个 Transformer 层的输出，因此这里预期得到 13 组表示。实际数量、shape 和时间长度必须在本地运行后验证，不能只依据文档写死。

观察模式不改变模型，只用于展示：

- 每层表示的 shape；
- 每层平均范数；
- 相邻层 cosine distance；
- 沿时间的表示变化强度；
- 低维 PCA 投影或热图；
- 原始 baseline CTC 输出。

## 4.2 干预模式

对第 `k` 个 Transformer block，记录：

```text
h_in  = 进入第k层的状态
h_out = 原第k层的输出
```

定义工程干预：

```text
h_adjusted = h_in + α × (h_out - h_in)
```

其中：

- `α = 1.0`：原模型，不改变该层；
- `α = 0.5`：只保留一半“该层相对输入产生的变化”；
- `α = 0.0`：完全旁路该层，后续层直接接收 `h_in`。

然后继续运行第 `k+1` 层到第 12 层及原 CTC head。

实现时优先在一次完整 `model(...)` 前向中，对 `model.wav2vec2.encoder.layers[k-1]` 注册**临时 forward hook**：从该 block 的实际输入取得 `h_in`，从其实际输出取得 `h_out`，只替换输出中的 hidden-state 主张量，并原样保留 attention 等其余返回项；前向结束后在 `finally` 中移除 hook。不得复制一份简化 encoder 循环而遗漏 positional convolution、layer norm、attention mask 或版本特定逻辑。开始批量运行前必须通过运行时 introspection 确认模块路径、层数、输入/输出类型和 shape；若当前 Transformers 版本的结构不匹配，应明确失败并修复兼容层，不能静默改用另一种干预。

这是一种可解释的工程消融，不是模型训练时的正常工作方式，也不代表神经系统中的“关闭某脑区”。它会改变后续层接收到的分布，因此效果只能解释为“当前干预下的模型敏感性”。

第一版把 `α` 限制在 `[0, 1]`，不做负值或大于 1 的放大。

# 5. 实际需要运行多少次

第一版只用一条语音、12 个 Transformer 层和三个展示强度：

```text
α ∈ {0.0, 0.5, 1.0}
```

`α=1.0` 对所有层都等价于同一个 baseline，不需要重复计算 12 次。因此唯一推理数为：

```text
1次baseline
+ 12层 × 2个非基线强度
= 25次唯一推理
```

HTML 可以显示一个 `12 × 3` 的选择矩阵，但其中 12 个 `α=1.0` 单元格共同引用 baseline 结果。

如果以后增加第二条音频，唯一推理数变为 `25 × 2 = 50`；第一轮不扩展。

# 6. 输入设计

## 6.1 第一轮只用一条语音

建议由执行者或参与者自行录制一句 2–6 秒的清晰英语，例如：

```text
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
```

使用自录音可以避免为了 Demo 下载完整数据集。必须保存：

- 原始 WAV；
- 语句文本；
- 录制者授权说明；
- 原始采样率；
- 转换后的 16 kHz 单声道 WAV；
- duration；
- SHA256；
- 是否做归一化及其参数。

如果使用公开样本，必须遵守第 0.4 节，只获取一个具体 item，并记录数据集 revision、row/item、许可证和来源。模型是英语 CTC 模型，第一轮不使用中文语音判断“层效果”。

## 6.2 输入控制

第一版只设一个输入选择：

```text
clean：原始16 kHz语音
```

HTML 可预留但禁用以下按钮：

```text
+ silence insertion
+ speed change
+ noise
+ local omission
```

只有单条 clean 语音闭环稳定后才加入输入扰动，避免把“输入改变”和“层干预”混在一起。

# 7. 每次推理保存什么

每个 baseline 或 intervention run 保存：

```text
run_id
created_at_utc
audio_id
audio_sha256
model_id
model_revision
processor_revision
transformers_version
torch_version
layer_id
alpha
input_sample_rate
input_duration
hidden_state_shapes
baseline_transcript
adjusted_transcript
ctc_frame_count
ctc_blank_ratio
mean_frame_max_probability
transcript_edit_distance_from_baseline
final_hidden_cosine_distance_from_baseline
ctc_logit_divergence_from_baseline
inference_time_ms
device
dtype
status
error
```

需要特别说明：

- `mean_frame_max_probability` 只是模型内部置信度代理，不是校准概率；
- `transcript_edit_distance_from_baseline` 说明干预后文本改变多少，不等于识别正确率；
- 如果有人工 reference，可另外计算 CER/WER；
- 表示距离和 CTC divergence 不能单独证明某层“负责语义”。

第一版统一采用以下定义，避免不同接收者得到同名异义的指标：

- CTC blank ID 取 `model.config.pad_token_id`，`ctc_blank_ratio` 是逐 frame argmax 等于 blank ID 的比例；
- `mean_frame_max_probability` 是对每个 frame 的 softmax 最大值再取均值；
- `transcript_edit_distance_from_baseline` 是标准 Levenshtein 字符编辑距离，同时保存是否完全相同；
- `final_hidden_cosine_distance_from_baseline` 是在时间轴和特征维逐位置计算 cosine distance 后取有限值均值；
- `ctc_logit_divergence_from_baseline` 使用两个逐 frame softmax 分布的 Jensen–Shannon divergence，再对 frame 取均值；自然对数、概率下限 `1e-12`；
- 所有聚合指标都保存计算公式版本 `metrics_schema_version: 1`，不得把 NaN/Inf 替换为 0。

# 8. HTML 控制台

## 8.1 展示方式

第一版采用：

```text
Python离线推理
→ 保存JSON/CSV和压缩后的可视化数据
→ 静态HTML读取本地结果
→ 浏览器交互比较
```

HTML 不负责下载模型，也不在浏览器内运行 PyTorch。这样可以：

- 把模型依赖与展示依赖分开；
- 在推理机器上一次生成结果；
- 之后断网浏览；
- 保证每个图都来自可追溯结果文件；
- 避免 CDN 和远程服务失效。

如果浏览器对本地文件读取有限制，可使用最小 localhost 静态服务器；这不改变结果内容。

## 8.2 页面布局

```text
┌─────────────────────────────────────────────────────────────┐
│ Wav2Vec 2.0 Layer Lab                                      │
│ 模型revision | 音频 | 设备 | 运行状态 | 恢复baseline        │
├─────────────────────────────────────────────────────────────┤
│ 音频播放器 + waveform + CTC时间轴                           │
├─────────────────────────────────────────────────────────────┤
│ 模型结构：Conv1…Conv7（只读）→ H0 → L1 → ... → L12 → CTC   │
│                         [选择层滑块：1—12]                   │
├─────────────────────────────────────────────────────────────┤
│ 干预模式：观察 / 残差调节                                   │
│ α：0.0 ─────────●───────── 1.0                              │
├───────────────────────┬─────────────────────────────────────┤
│ Baseline              │ 当前层与α                            │
│ transcript            │ adjusted transcript                 │
│ CTC统计               │ edit distance / blank ratio / latency│
├───────────────────────┴─────────────────────────────────────┤
│ 层级热图 | 相邻层距离 | 时间变化曲线 | CTC分布差异           │
├─────────────────────────────────────────────────────────────┤
│ 结论边界、run_id、配置、JSON路径、失败状态                   │
└─────────────────────────────────────────────────────────────┘
```

## 8.3 必须有的控件

1. 音频播放和暂停；
2. Transformer 层选择：`1–12`；
3. 模式选择：`观察` / `残差调节`；
4. `α` 选择：`0.0`、`0.5`、`1.0`；
5. baseline/adjusted 并排比较；
6. “恢复 baseline”按钮；
7. 指标解释开关；
8. 显示 run id、模型 revision 和状态；
9. 失败 run 可见，不能只展示有变化的层；
10. 导出当前视图或复制配置。

## 8.4 必须有的 panel

### Panel A：模型结构

显示卷积前端、H0、12 个 Transformer block 和 CTC head。被选层高亮，α=0 时显示“bypass”，α=0.5 时显示“attenuated”。

### Panel B：声音与时间轴

显示 waveform、播放位置和 CTC frame 时间轴。第一版不把 frame 直接解释为单个音素。

### Panel C：转写变化

并排显示：

```text
baseline transcript
adjusted transcript
字符级差异
如果有reference：CER/WER
```

### Panel D：层级表示

至少显示：

- 选定层表示的时间×特征热图或降维图；
- 相邻层表示距离；
- baseline 与 adjusted 最终 hidden state 距离；
- 沿时间的表示变化强度。

### Panel E：CTC 输出

显示 blank ratio、平均 frame 最大概率和 baseline/adjusted logit 差异。不要把 softmax 最大值称为临床或统计意义上的可信度。

### Panel F：证据与边界

显示当前选择对应的：

- run id；
- 模型与 processor revision；
- layer、α、设备、dtype；
- 配置和结果文件；
- 是否成功；
- 当前观察能支持和不能支持什么。

# 9. 用户应该能够看到什么效果

HTML 不预先宣称“第几层负责什么”。实际运行后，用户可能看到四类情况：

| 类型 | 页面现象 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| 表示变、转写不变 | hidden distance 增加，文本相同 | 后续层或 CTC 输出对该次变化仍保持稳定 | 该层没有作用 |
| 表示与 CTC 都变、转写不变 | logits/blank ratio 变化，文本相同 | 输出分布改变但 greedy transcript 保持 | 语义完全不受影响 |
| 转写局部改变 | 若干字符或词变化 | 当前语音与干预下，输出对该层敏感 | 该层专门编码这些词 |
| 大范围失效 | blank、重复字符或错误明显增加 | 当前干预破坏了后续处理 | 该层等同某脑区或是唯一关键层 |

结果也可能是 12 层的旁路都不改变这条语音的最终文本。只要表示和 CTC 指标如实保存，这仍是有效结果；下一步可以换第二条语音或更严格指标，但不能隐藏阴性结果。

# 10. 数据流与文件建议

本节只规划路径，不在编写 03 时创建这些文件：

```text
README.md
requirements.lock.txt

docs/
├── STANDALONE_EXECUTION_SPEC.md
└── CODEX_PROJECT_LOG.md          # 仅当运行环境规则要求；只追加，不写 token

src/route_b_temporal_models/
└── wav2vec2_layer_demo/
    ├── infer_baseline.py
    ├── run_layer_interventions.py
    ├── extract_hidden_states.py
    ├── metrics.py
    └── export_demo_data.py

tests/
└── test_wav2vec2_layer_demo.py

configs/
└── TB001-DEMO001.yaml

scripts/
├── preflight.py
├── run_demo.py
├── serve_demo.py
└── verify_delivery.py

stimuli/
└── TB001-DEMO001/
    ├── input_16k_mono.wav
    └── manifest.json

outputs/TB001-DEMO001/
├── baseline/
├── interventions/
├── metrics/
└── demo_data/

demo/TB001-DEMO001/
├── index.html
├── styles.css
├── app.js
└── data/
    ├── manifest.json
    ├── runs.json
    └── visualization.json

reports/
├── TB001-DEMO001_EXECUTION_REPORT.md
└── TB001-DEMO001_TOKEN_USAGE.json  # 仅在运行器提供 usage 时生成
```

HTML 的所有数值必须来自 `outputs/` 导出的 JSON，不允许在 `app.js` 里手工写“示范结果”。

最终软件包必须提供相对于 `DEMO_ROOT` 的三条稳定入口，并在 `README.md` 写出实际命令：

```text
预检：python scripts/preflight.py
运行：python scripts/run_demo.py --config configs/TB001-DEMO001.yaml
验收：python scripts/verify_delivery.py
浏览：python scripts/serve_demo.py --port 8000
```

脚本不得依赖调用者的当前目录：必须由脚本自身解析 `DEMO_ROOT`，或明确要求 `--root` 并校验。`run_demo.py` 必须可重入：已完成且哈希匹配的 run 可以复用，失败或配置变化的 run 写新目录，不得覆盖旧结果。任何下载都只发生在显式 prepare 阶段；运行和浏览阶段应支持离线模式。

# 11. 最低测试

## 11.1 模型与输入

- [ ] 模型 ID、revision 和许可证已保存；
- [ ] 模型和 processor 使用同一个完整 commit SHA，关键文件 SHA256 已保存；
- [ ] `from_pretrained` 的 missing/unexpected/mismatched keys 和全部加载 warning 已保存；任何会在 eval 前向中实际使用的随机初始化参数都必须先解释或视为 BLOCKED；
- [ ] 输入确为单声道 16 kHz；
- [ ] 音频来源、许可、utterance ID 和 SHA256 已保存；
- [ ] 模型处于 `eval` 模式；
- [ ] 推理使用 `no_grad` 或 inference mode；
- [ ] 相同输入、配置和设备重复运行结果一致；
- [ ] baseline 可以得到非空 logits 和可解码文本；
- [ ] hidden state 实际数量和 shape 已记录；
- [ ] 所有结果无 NaN/Inf。

## 11.2 层干预

- [ ] `α=1.0` 与 baseline 的 transcript、token IDs、hidden shapes 完全一致；float32 logits 的最大绝对误差 `≤1e-5`，若使用其他 dtype，必须先在配置中给出有依据的更宽容差；
- [ ] `α=0.0` 确实把第 k 层输出替换为输入；
- [ ] `α=0.5` 使用规定公式，不是任意缩放整个 hidden state；
- [ ] 每次只干预一个 Transformer 层；
- [ ] 下游第 `k+1–12` 层和 CTC head 仍正常运行；
- [ ] 12 层×2 个非基线干预全部保存，失败也留档；
- [ ] 层编号与官方模型实际模块对应，无 off-by-one。
- [ ] 每次 forward 后 hook 均已移除；连续选择不同层不会叠加旧 hook。

## 11.3 HTML

- [ ] 断网后页面仍能打开；
- [ ] 不使用 CDN；
- [ ] 音频可播放；
- [ ] layer 和 α 控件只选择真实存在的 run；
- [ ] baseline 恢复有效；
- [ ] 页面数值与 JSON 一致；
- [ ] 切换层不会把旧结果残留到新视图；
- [ ] 失败 run 显示明确状态；
- [ ] 每个图能追溯到 run id；
- [ ] 页面不出现预设“节律层”“语义层”等结论。

## 11.4 独立交付与重跑

- [ ] 在不提供原项目 `01`、`02` 或上级目录的条件下，预检、运行和验收入口仍能定位全部文件；
- [ ] README 不含发送者机器的绝对路径；配置默认值均落在 `DEMO_ROOT` 内；
- [ ] 断开网络并启用本地模型缓存后，baseline、验收和 HTML 浏览仍能运行；
- [ ] 第二次运行不会覆盖首轮日志或失败记录；
- [ ] `verify_delivery.py` 对缺文件、哈希不符、run 数不足、NaN/Inf、HTML/JSON 错位返回非零退出码；
- [ ] 交付清单不包含 Hugging Face cache、模型权重、原始 Codex JSONL、凭据或与 Demo 无关的文件。

# 12. 最低交付

只有以下内容同时存在，Demo 才算完成：

1. 一条有来源记录的 16 kHz 英语语音；
2. 冻结的模型 ID、revision、processor 和环境；
3. 一次 baseline 本地推理；
4. 13 组实际 hidden states 的 shape 记录；
5. 24 个非基线层干预结果；
6. baseline 与各干预的转写和 CTC 指标；
7. 所有配置、日志、失败 run 和运行时间；
8. 可离线打开的 HTML 控制台；
9. 层选择、α 选择、baseline 比较和恢复功能；
10. 至少一张层级总览图和一张表示/CTC 变化图；
11. 单元测试与 HTML 静态数据核对；
12. 一页结论边界；
13. 一份由同一 Agent 编写、能回到实际文件和日志的执行报告。
14. `README.md`、冻结依赖和预检/运行/验收/浏览四个稳定入口；
15. 一份文件清单与 SHA256 manifest，证明交付包不依赖原项目路径；
16. 若由 Codex CLI 执行且 usage 可用，一份符合第 0.5 节口径的 token 摘要。

# 13. 单 Agent 最小实施流程

## 13.1 单 Agent 的职责

同一个 Agent 对本 Demo 的端到端闭环负责：

```text
只读审计
→ 环境与资源核验
→ 最小依赖和模型获取
→ baseline
→ 单层干预原型
→ identity test
→ 12层批量推理
→ JSON与图
→ 离线HTML
→ 自动测试与人工可视检查
→ 执行报告
```

它既执行也自检，但不能把“自检”伪装成独立科学审核。Demo 完成后，最终是否接受仍由人类检查 HTML、日志和结论边界。

## 13.2 实施顺序

```text
第1步：确认DEMO_ROOT并核验Python、CPU/GPU、内存、磁盘、网络和现有文件
第2步：解析并冻结Transformers/PyTorch版本、模型与processor完整revision
第3步：准备一条16 kHz英语语音
第4步：跑通原始baseline CTC转写
第5步：提取并检查13组hidden states
第6步：实现一个层的α=0和α=0.5
第7步：验证α=1严格回到baseline
第8步：扩展到12层，共得到24个非基线run
第9步：导出统一JSON
第10步：构建离线HTML控制台
第11步：核对页面与原始JSON
第12步：写结果和不能支持的结论
第13步：从一个不含原项目文件的路径执行独立交付验收
```

必须先完成第 4–7 步的小闭环，再批量运行 12 层。若第一个层的干预不能稳定回到 baseline，不进入批量 sweep。

## 13.3 单 Agent 检查点

Agent 不需要每一步都等待人类，但必须保留以下检查点：

| 检查点 | Agent 必须确认 | 可以继续的条件 | 必须停止的情况 |
|---|---|---|---|
| S0：审计结束 | DEMO_ROOT、现有文件、硬件、磁盘、网络、全局规则 | 路径与资源明确且不依赖原项目 | 写入范围不清、磁盘明显不足 |
| S1：模型就绪 | 模型ID、revision、许可、processor、SHA256 | 指定模型可本地加载 | 来源或许可无法确认 |
| S2：baseline | 输入16 kHz、logits、转写、13组表示 | 两次结果一致 | baseline为空、NaN或不可重复 |
| S3：单层原型 | α=0/0.5/1公式与层编号 | α=1回到baseline | identity test失败或off-by-one |
| S4：批量完成 | 24个非基线run和失败状态 | 每个run可追溯 | 结果被覆盖或层配置混乱 |
| S5：HTML完成 | JSON一致、断网可用、控件可复位 | 静态检查和交互检查通过 | 页面使用手填结果或数据错位 |
| S6：报告完成 | 文件、命令、测试、图、阴性结果 | 人工可据此审核 | 只写“已完成”而无证据 |

非阻塞问题由 Agent 采用最小、可回退的默认方案并写入报告；只有涉及权限、许可、不可恢复写入、缺少语音输入或改变科学目标时才向人类提问。

## 13.4 单 Agent 执行报告

建议写入：

```text
reports/TB001-DEMO001_EXECUTION_REPORT.md
```

至少包含：

- 实际环境和模型 revision；
- 新建与修改文件；
- 下载来源、大小和哈希；
- 完整运行命令及退出码；
- baseline 和 24 个干预 run 状态；
- 测试与 HTML 检查；
- 主要变化、阴性结果和失败层；
- 当前能支持与不能支持的结论；
- 未完成内容；
- 是否建议进入第二条语音或正式 TB001。
- 独立交付验收命令、退出码和 manifest；
- token 统计口径及 `AVAILABLE`/`UNAVAILABLE` 状态；若可用只引用脱敏摘要，不嵌入原始 Codex JSONL。

# 14. Go / No-go

## 14.1 Go

满足以下条件，才值得继续扩展第二条语音或输入扰动：

- baseline 可重复；
- hidden state 接口稳定；
- α 干预实现通过 identity test；
- 25 次唯一推理均有可追踪状态；
- HTML 能断网展示并准确切换结果；
- 至少一个指标能够区分部分层或强度，或者可靠得到“未发现差异”的阴性结果；
- 结论没有把单语音敏感性扩大成普遍层功能。

## 14.2 No-go 或返工

出现以下情况应停在当前 Demo：

- 模型或权重许可、来源或 revision 无法固定；
- 本机资源不足且没有更小的安全运行方案；
- baseline 本身不稳定；
- α=1 不能恢复 baseline；
- 层编号或 hidden state 对齐不清楚；
- HTML 数值与 JSON 不一致；
- 结果只能通过手工修改页面展示；
- 使用者看见变化，却无法追溯到具体 run。

# 15. 本 Demo 能支持的结论

完成后最多可以说：

> 在指定模型 revision、指定本地英语语音和工程定义的残差调节下，不同 Transformer 层的干预对中间表示、CTC 分布和最终转写产生了可重复的不同影响。

不能直接说：

- 某一层就是“声音层”“节律层”或“语义层”；
- 某层对应听神经、听觉皮层或其他脑区；
- 某层对所有语音都最重要；
- 转写不变表示该层没有信息；
- 单层旁路证明了真实模型训练机制；
- wav2vec 2.0 已经复制人类听觉系统。

# 16. 后续可选扩展

只有第一版完成后，按一次一个变量扩展：

1. 加入第二条英语语音，检查层效果是否跨输入稳定；
2. 加入 silence insertion，检查时间边界；
3. 加入 speed change，检查 tempo/时间伸缩；
4. 加入 noise，检查层敏感性与鲁棒性；
5. 为 onset、phoneme boundary 或局部 omission 建立简单 probe；
6. 比较原模型、单层旁路和残差减弱；
7. 再考虑 HuBERT 或 WavLM 作为第二模型。

这些都不属于第一版最低交付。

# 17. 单 Agent 完整执行 Prompt

本节 Prompt 可以与**本文件全文**一起直接发给一个新的 Agent 会话。发送它表示人类允许该 Agent 在一个专用 `DEMO_ROOT` 内完成 Demo 所需的最小环境、指定模型下载、代码、推理、HTML、测试和报告；不表示允许 Git 提交、推送、删除任何材料或扩大为正式 TB001。

```text
你是TB001-DEMO001的单一执行Agent。你收到的STANDALONE_EXECUTION_SPEC.md就是完整且唯一的任务规范；不得要求原项目的01、02或上级目录。请在同一个会话中端到端完成其中定义的wav2vec 2.0本地复现和层级交互HTML，不创建Planner/Executor两个会话，也不创建正式任务书或coordination工作流。

开始前必须：
1. 把包含本规范的专用目录确认成DEMO_ROOT，并完整阅读本文件；
2. 检查DEMO_ROOT现有文件、Python、CPU/GPU、内存、磁盘、网络、Git状态和适用的全局规则；
3. 保留所有已有内容，不覆盖无关修改；所有新产物留在DEMO_ROOT或明确的外部缓存/私有审计目录；
4. 如果DEMO_ROOT不是Git仓库，不得初始化Git；本任务无论如何都不授权提交、推送、切换分支、重置或清理；
5. 若全局规则要求项目交接日志，在任何实质修改前创建docs/CODEX_PROJECT_LOG.md并只追加；不得把token或原始事件写入该日志；
6. 输出一段简短执行摘要，然后直接按本规范继续，不为非阻塞问题反复等待确认。

本任务授权你：
- 使用现有合适环境，或在第0.3节允许的位置创建一个隔离环境；
- 安装完成本Demo确实需要的最少依赖；
- 从官方来源下载facebook/wav2vec2-base-960h及processor；
- 把main解析成完整commit SHA，并用同一个SHA加载模型与processor；冻结依赖版本、许可证信息、文件清单和权重哈希；
- 如果没有可用且获授权的英语语音，严格按第0.4节只获取一个2—6秒LibriSpeech demo样本，不下载完整数据集；
- 创建本规范规划中列出的最小代码、配置、测试、outputs、demo和report文件；
- 运行baseline、hidden-state提取、层干预、HTML生成和必要测试。

严格执行以下范围：
1. 输入只使用一条2—6秒、16 kHz、单声道英语语音；
2. 模型固定为facebook/wav2vec2-base-960h，不切换其他模型；
3. 模型设为eval并使用inference/no_grad模式；
4. 完成一次原始CTC baseline并提取H0和12层Transformer输出；
5. 对每个Transformer层使用h_adjusted = h_in + alpha × (h_out - h_in)；
6. alpha只使用0.0、0.5、1.0，其中1.0引用或校验baseline；
7. 第一版不调节7层卷积前端，不训练模型或probe；
8. 先在一个层上验证alpha=1 identity test，再扩展12层；
9. 运行1次baseline和24个非基线干预，共25次唯一推理；
10. 保存每个run的配置、日志、转写、CTC统计、表示距离、耗时、状态和错误；
11. 阴性结果和失败run必须保留，不能只选变化最大的层。

构建一个无需CDN、可断网浏览的静态HTML控制台。页面至少包含：
- 音频播放器与waveform；
- Conv1—7只读结构、H0和L1—L12；
- layer选择、观察/干预模式、alpha=0/0.5/1控件和恢复baseline；
- baseline与adjusted转写对比；
- blank ratio、frame最大概率代理、文本edit distance、hidden distance和logit difference；
- 层级热图或降维图、相邻层距离和时间变化曲线；
- run_id、模型revision、配置路径、状态和结论边界；
- 失败run可见。

HTML中的数值必须从推理导出的JSON或CSV读取，不得手工填写结果。优先使用原生HTML/CSS/JavaScript或本地静态资源；如果file协议限制数据加载，可提供最小localhost启动方式并测试。

至少完成本规范第11章列出的模型、层干预、HTML和独立交付测试。特别检查：
- 相同输入和配置可复现；
- hidden-state数量和shape来自实际运行；
- alpha=1与baseline在预定义容差内一致；
- 层编号无off-by-one；
- 24个非基线run全部有状态；
- 页面与JSON数值一致；
- 断网或本地服务器模式能够完整浏览；
- 没有把任何层命名为语义层、节律层或脑区。

如果遇到模型来源/许可无法确认、磁盘不足、缺少合法语音、baseline不可重复、alpha=1无法回到baseline、可能破坏已有文件或需要改变科学目标，请停止并明确报告BLOCKED。普通依赖兼容或实现错误应在当前范围内诊断和修复，并保留失败日志。

完成后编写reports/TB001-DEMO001_EXECUTION_REPORT.md，汇报：
- 实际文件；
- 环境、模型revision、下载来源和哈希；
- 全部关键命令与退出码；
- baseline和25次唯一推理的实际状态；
- 测试；
- HTML入口和本地打开方式；
- 哪些层和alpha产生了哪些实际变化；
- 阴性结果与失败case；
- 能支持和不能支持的结论；
- 是否达到本规范第14章的Go条件。

最终停止在Demo交付，不创建下一任务，不提交或推送Git。
```

如果只希望 Agent 先审计而不安装或运行，应在启动 Prompt 后追加“本轮只读审计”；否则以上 Prompt 的默认含义是允许一个 Agent 在限定范围内完成整个 Demo。

# 18. SSH 初步部署模式

需要先验证一台新机器、但暂不开始 24 个非基线干预和 HTML 时，在第 17 章 Prompt 后追加以下覆盖指令：

```text
本轮是初步部署，只执行S0、S1，并在资源和依赖允许时执行S2 baseline smoke test。服务器只作为计算节点，由当前控制端Agent通过SSH执行命令；不得在服务器上启动第二个Agent或codex exec。必须创建可接手的README、预检结果、环境冻结、模型/processor完整revision与哈希、一个合法输入的manifest、baseline转写和13组hidden-state shape记录。不要执行S3—S6，不要运行24个非基线干预，不要构建最终HTML。把未执行项明确标记PENDING，不得宣称Demo完成。写reports/TB001-DEMO001_INITIAL_DEPLOYMENT.md后停止。
```

Linux 服务器只作为计算节点，由外层操作者或当前 Agent 通过 SSH 驱动。先在服务器创建新的隔离 `DEMO_ROOT`，再上传本规范和已审计脚本：

```bash
SSH_TARGET=server2203
REMOTE_ROOT=/absolute/path/to/new/demo_root
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_ROOT/docs' '$REMOTE_ROOT/scripts'"
scp docs/STANDALONE_EXECUTION_SPEC.md \
  "$SSH_TARGET:$REMOTE_ROOT/docs/STANDALONE_EXECUTION_SPEC.md"
scp scripts/preflight.py scripts/run_demo.py scripts/verify_delivery.py \
  "$SSH_TARGET:$REMOTE_ROOT/scripts/"
ssh "$SSH_TARGET" \
  "python '$REMOTE_ROOT/scripts/preflight.py' && \
   python '$REMOTE_ROOT/scripts/run_demo.py' --config '$REMOTE_ROOT/configs/TB001-DEMO001.yaml' && \
   python '$REMOTE_ROOT/scripts/verify_delivery.py'"
```

外层操作者必须记录每条 SSH/远端命令、开始/结束 UTC、退出码、stdout/stderr 日志路径和产物 SHA256。网络下载、环境准备、baseline 和测试应拆成可定位失败的独立步骤；失败日志必须保留，修复后写新 attempt，不得覆盖。

必须核对**产物门槛**，不能只看最后一条 SSH 命令是否退出 `0`。至少确认 INITIAL_DEPLOYMENT 报告状态、S0/S1/S2 必需文件、模型与音频哈希、baseline 重复性和离线复核。服务器无需安装、配置或修复 Codex CLI、`bwrap`、远程 Agent 或 Agent sandbox；这些都不属于本 Demo 的服务器依赖。
