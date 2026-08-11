# M6A-PUBLIC-001 neural target redesign 草案

| 字段 | 内容 |
|---|---|
| 状态 | `DRAFT_AWAITING_COORDINATOR_REVIEW` |
| 触发证据 | 所有 iEEG sidecar 为 60 Hz 工频；原 70-150 Hz 候选带包含 120 Hz 二次谐波 |
| 当前门禁 | `REDESIGN_REQUIRED_BEFORE_G3` |
| 神经提取授权 | `false` |

## 不可直接沿用的方案

不能在未声明 120 Hz 处理的情况下，把单一 70-150 Hz band-pass + Hilbert envelope 写成已冻结 target。这样会把工频二次谐波混入 target，并使不同 recording 的噪声差异可能被模型或 split 利用。

## 对照决策表

两种方案都只使用 channels.tsv 中 `status=good` 且 type 为 SEEG/ECOG 的通道，并严格排除 README 指定的 C-prefix channel。

| 项目 | 方案 A：70-150 Hz + line rejection | 方案 B：避开 120 Hz 的分带 target |
|---|---|---|
| 频率定义 | 在宽带信号上显式抑制 60/120 Hz，再提取 70-150 Hz | 分别提取 70-110 与 130-150 Hz，不跨 120 Hz |
| 优点 | 保留原任务书 high-gamma 频带口径 | 不依赖窄 notch 跨过候选带中心 |
| 主要风险 | notch/band-stop ringing、transition 与 edge 可能较长 | 两子带带宽不等，聚合尺度和生理解释需重新冻结 |
| 必须实测 | 60/120 残余、passband 扭曲、impulse response、padding | 两子带噪声、尺度、edge、聚合稳定性 |
| 当前状态 | `PROPOSED_NOT_FROZEN` | `ALTERNATIVE_NOT_FROZEN` |

Producer 当前不自行选定。协调冻结前，两者都不能进入神经提取。

## 两方案共用的待冻结步骤

1. 明确滤波实现、阶数/transition、相位、padding 和有效 impulse-response edge；
2. 解析信号后计算 amplitude 还是 power；若为 power，冻结 `log(power + epsilon)` 的 epsilon 与单位；
3. 冻结到 50 Hz 神经网格的 anti-alias 与降采样实现；
4. 冻结 lag grid（当前最大 0.5 s）及 lag 的正负时间语义；
5. 若使用双向/非因果滤波，先测量真实 edge，并禁止跨 split 邻近窗口共享 padding；
6. 依据 target/filter edge、lag 与音频模型 context 实测值计算 final embargo，再重跑 split guard；
7. 该 target 仍是工程 neural proxy，不代表完整生理 high-gamma，也不提供脑区定位。

## G3 冻结前验收

- 512 Hz 与 1024 Hz recording 各至少一个 smoke；
- 报告处理前/后 60、120 Hz 邻域 PSD，不只展示最佳通道；
- 验证分析候选通道名真实存在于 EDF，C-prefix 与 bad channel 不进入输出；
- 记录滤波器 impulse/step response、padding 与有效 edge；
- 保存 NaN、常数、饱和、异常噪声和读取失败通道；
- target frame time 与音频/事件时间零点一致；
- final embargo gate 重新计算并通过后，split 才能成为 baseline-final。

## Embargo 草案

`final_embargo = max(preliminary 2 s, maximum encoding lag, filter/padding edge, audio-model receptive-field/context overlap)`。

当前 maximum encoding lag 为 0.5 s；filter/padding edge 与 audio-model context overlap 尚未实测，因此 final embargo 保持 `PENDING`。若 wav2vec2 每个 stimulus 独立提取且不共享跨 stimulus chunk context，跨 split 的 audio context overlap 可候选为 0，但必须通过 G3 frame-time/canary 实测后才能写入。

## 声称边界

本草案只提出可审核的 target 与验证路线。当前不能声称 neural target 已冻结、G3 可启动、high-gamma 已提取、脑区标签可用或任何 layer-wise baseline 已运行。
