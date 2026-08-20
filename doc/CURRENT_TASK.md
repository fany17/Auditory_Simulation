# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | `2026-08-21` |
| 状态 | `ACTIVE_EXECUTION` |
| 当前方向 | `M6A-PUBLIC：公开声音模型、公开神经数据与通用听觉计算方法` |
| 当前优先任务 | `M6A-PUBLIC-003` |
| 正式任务书 | `doc/tasks/M6A-PUBLIC-003.md` |
| Task index | `doc/TASKS.md` |
| 患者/STN 数据 | `FORBIDDEN` |

## 当前任务

`M6A-PUBLIC-003：Temporal Architecture Perturbation`

目标不再是继续补 pretrained baseline，而是研究明确结构变量如何影响快速声音时间信息的保留、整合与丢失。

第一轮只做三个主题：

1. **Early vs Late Downsampling**：保持总压缩尽量一致，只改变下采样发生层级；
2. **Multiscale Receptive Field / RF Growth**：kernel、dilation、RF growth schedule、parallel RF，以及 RF/downsampling decoupling；
3. **Explicit Change/Event Branch**：baseline vs 显式变化支路 vs 参数匹配普通第二支路。

允许训练小型、matched experimental models 来做结构因果比较；禁止微调 wav2vec2、Whisper、Parakeet、Audio-Mamba 等大型 pretrained backbone。

详细执行要求见 `doc/tasks/M6A-PUBLIC-003.md`。

## Frozen baseline：M6A-PUBLIC-002

`M6A-PUBLIC-002` 已满足进入下一阶段所需的 baseline 条件：

- 9/9 pretrained inference；
- 8/9 unified temporal representation probe；
- 覆盖 PANNs CNN14、ConvTasNet、SpeechBrain CRDNN、Parakeet-TDT/FastConformer、Audio-Mamba/SSAM、wav2vec2、Whisper、CoNNear periphery、ICNet；
- CoNNear 已覆盖 `waveform→BM→IHC→ANF-H/M/L` 六类 probe；
- ICNet 已覆盖 `waveform→bottleneck→units_1000` 六类 probe；
- 0 training / 0 finetuning / 0 patient data。

002 作为 003 的 frozen reference，不继续为了补更多模型或 9/9 hidden representation 消耗时间。

旧失败记录继续保留，不删除 historical blocker。

## Historical checkpoint：M6A-PUBLIC-001

`M6A-PUBLIC-001` 保留 ds004703 / wav2vec2 的单被试单 recording preliminary checkpoint，不在 003 中扩 subject、recording 或神经拟合，也不自动生成 M6A→M6B exchange candidate。

## 项目边界

- `Auditory_Simulation`：公开数据、声音模型、通用表征/时间结构方法、M6A；
- `STN_Decoding_Encoding`：患者实验、STN 数据、STN-specific adaptation、M6B；
- 两者只通过冻结、版本化 artifact 衔接；
- 患者数据默认不得进入 `Auditory_Simulation`；
- `AuditoryReading` 独立负责教材、综述和知识整理，不作为第二实验运行目录。

## 当前硬规则

- 不读取患者/STN 数据；
- 不修改 `STN_Decoding_Encoding`；
- 不继续堆 pretrained model；
- 不微调大型 pretrained backbone；
- 阴性结构结果必须保留；
- 参数无法 matched 时必须标记 confounded；
- 禁止按脑区名字机械搭建“仿听觉系统网络”；
- 所有 auditory-inspired 修改必须有明确计算原则、baseline deficit、minimal implementation、matched control、ablation 和 falsifiable endpoint。
