# M6A-PUBLIC-002 周期阶段报告

状态：`M6A-PUBLIC-002_READY_FOR_COORDINATOR_REVIEW`。

本轮在 `server2203:/home/fanyu/auditory_simulation_m6a/m6a_public_002` 续跑，未重跑已确认的 wav2vec2 synthetic inference 与 CoNNear 既有 smoke；完成 ICNet 官方 smoke/temporal probe、PANNs 同源重取后的真实加载，以及低成本 ConvTasNet、SpeechBrain CRDNN 的 synthetic inference/probe。

## 核心结果

| 项目 | 结果 |
|---|---|
| inference PASS | 6/9：wav2vec2、PANNs、ConvTasNet、SpeechBrain、CoNNear、ICNet |
| temporal probe PASS | 6/9；均为 no-fit synthetic representation probe |
| CoNNear | BM/IHC/ANF 五级输出，201 CF，finite |
| ICNet | bottleneck `[1,1,762,64]`；units_1000 `[1,762,1000]`，finite |
| PANNs | embedding `[1,2048]`，finite；同源重取后加载 PASS |
| ConvTasNet | 每个 synthetic probe 输出 `[1,1,16000]`，finite |
| SpeechBrain CRDNN | synthetic decoder smoke PASS |
| wav2vec2 | projected + 12 hidden layers，tone `[1,49,768]`，finite |
| Whisper turbo | 官方加载未形成 inference 结果；失败证据保留，不绕过 |
| Parakeet-TDT | NeMo 依赖缺失；Transformers GPU synthetic generation 超过 180 s，保留阻塞 |
| Audio-Mamba/SSAM | 官方权重与 Mamba 依赖链本轮未进入 |

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

不可声称：模型优劣、脑区对应、神经编码等价、因果听觉机制、跨被试泛化或科学显著性。当前 6/9 未达到任务书 7/9 最低目标，因此不写本周 PASS。

## 失败与下一门禁

PANNs 原截断文件失败记录保留，同源重取后已通过；Whisper、Parakeet、Audio-Mamba 的失败/未运行证据保留。协调者需决定是接受当前 6/9 的阶段性结果，还是另行授权补齐至少 7/9；在此之前停止，不启动 G3、exchange candidate、患者数据或任何训练流程。
