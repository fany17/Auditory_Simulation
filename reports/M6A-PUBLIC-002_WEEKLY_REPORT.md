# M6A-PUBLIC-002 周期阶段报告

状态：`M6A-PUBLIC-002_READY_FOR_COORDINATOR_REVIEW`。

CoNNear 与 ICNet 的统一 probe 当前均为 6/6：`tone`、`regular_clicks`、`jitter_clicks`、`omission`、`phase_shift`、`speech`。其中 `speech` 是与其他缓存模型相同的确定性 synthetic waveform；报告中的距离仍仅是 no-fit 表征扰动描述。

本轮在 `server2203:/home/fanyu/auditory_simulation_m6a/m6a_public_002` 续跑，未重跑已确认的 wav2vec2 synthetic inference 与 CoNNear 既有 smoke；完成 ICNet 官方 smoke/temporal probe、PANNs 同源重取后的真实加载，以及低成本 ConvTasNet、SpeechBrain CRDNN 的 synthetic inference/probe。

## 核心结果

| 项目 | 结果 |
|---|---|
| inference PASS | 7/9：wav2vec2、PANNs、ConvTasNet、SpeechBrain、CoNNear、ICNet、Parakeet |
| temporal probe PASS | 6/9；均为 no-fit synthetic representation probe |
| CoNNear | BM/IHC/ANF 五级输出，201 CF，finite |
| ICNet | bottleneck `[1,1,762,64]`；units_1000 `[1,762,1000]`，finite |
| PANNs | embedding `[1,2048]`，finite；同源重取后加载 PASS |
| ConvTasNet | 每个 synthetic probe 输出 `[1,1,16000]`，finite |
| SpeechBrain CRDNN | synthetic decoder smoke PASS |
| wav2vec2 | projected + 12 hidden layers，tone `[1,49,768]`，finite |
| Whisper turbo | 官方加载未形成 inference 结果；失败证据保留，不绕过 |
| Parakeet-TDT | 官方 Transformers six-probe generate smoke PASS；token 序列 finite，但未提取 hidden representation |
| Audio-Mamba/SSAM | 官方 tiny 权重已在 2203 正式目录；`mamba_ssm`/`causal_conv1d` 依赖链仍阻塞 |

## 已完成交付

- `model_registry_pretrained_week.csv/json`
- `pretrained_model_smoke_summary.json`
- `temporal_probe_summary.json`
- `architecture_comparison_matrix.csv`
- `CONNEAR_REPRODUCTION.md`
- `ICNET_REPRODUCTION.md`
- 本报告

所有大模型、缓存、日志和任何高维输出仍只在 2203；本地只保留轻量报告与执行脚本。未训练、未微调、未运行 ridge/classifier/downstream probe，未读取患者或 STN 数据。

## 科学可声称与不可声称

可声称：这些公开 pretrained/physiology-surrogate 路径在 synthetic 输入上完成了有限的加载、shape、finite 与时间扰动 smoke；CoNNear 和 ICNet 的输出级别及机制边界可被审计。

不可声称：模型优劣、脑区对应、神经编码等价、因果听觉机制、跨被试泛化或科学显著性。当前 temporal probe 仍为 6/9，尚未达到任务书 7/9 最低目标，因此不写本周 PASS。

## 失败与下一门禁

PANNs 原截断文件失败记录保留，同源重取后已通过；Whisper 与 Audio-Mamba 的失败/未运行证据保留，Parakeet 已完成官方 generate smoke。协调者需决定是接受当前 7/9 inference、6/9 temporal probe 的阶段性结果，还是另行授权补齐 temporal probe；在此之前停止，不启动 G3、exchange candidate、患者数据或任何训练流程。

## 本轮续跑补充

Parakeet 本轮从 HF mirror 下载完整 snapshot 到 2203 专用 cache，并按官方 Transformers 路径做六 probe `generate` smoke；输出为 finite decoder token sequences，未将其当作 hidden-layer temporal representation。Audio-Mamba 官方 tiny 权重已按 `weights/checkpoints/checkpoint-99.pth` 落盘 2203；官方 Mamba 依赖构建受 CUDA/PyTorch 工具链不匹配阻塞，未绕过官方路径，也未启动 inference。

CoNNear 的统一 probe 已真正输出 `waveform→BM→IHC→ANF-H/M/L`。六类输入均已记录，其中 regular/jitter/omission/phase 为重点生理比较；stage distance、lag-1 persistence 和事件后 10 ms 平均响应写入 2203 `reports/m6a_public_002_auditory_physio_probes.json`。ICNet 对六类输入输出 `waveform→bottleneck→units_1000`；bottleneck 与 units distance 已记录，单帧跨 probe persistence 明确为 `null`。

## 架构结论与研究缺口

- CNN/TCN 能在有限卷积上下文中保留局部时间结构；池化或 separator 输出会压缩或重整时间信息，不能直接当作神经时间码。
- BiLSTM/CRDNN 和 Transformer/Conformer 通过 recurrent state 或 attention 处理长上下文；wav2vec2 的 passage-global attention 与 Whisper/Parakeet 的 encoder-decoder 语义状态不能被简化为有限生理 receptive field。
- Mamba/SSAM 代表 state-space 压缩，但本轮无可运行官方权重证据。
- CoNNear 增加了有明确生理命名的 BM、IHC、ANF-H/M/L 变换；ICNet 增加了 bottleneck 到 1000-unit animal IC surrogate 的 population transform。二者都不等于人脑皮层、患者神经信号或跨物种功能等价。
- 成熟模型已覆盖局部卷积、循环状态、全局注意力、时序压缩和部分 auditory-physiology surrogate；真正缺口仍是可审计的时间参考、跨模型表征语义、有限/全局上下文边界，以及与公开神经数据的防泄漏 alignment 证据。本轮没有运行 Audio↔Brain 或 downstream readout。

本轮状态仍为 `M6A-PUBLIC-002_READY_FOR_COORDINATOR_REVIEW`；inference 已 7/9，但 temporal probe 为 6/9，不宣称任务 PASS 或科学结果。
