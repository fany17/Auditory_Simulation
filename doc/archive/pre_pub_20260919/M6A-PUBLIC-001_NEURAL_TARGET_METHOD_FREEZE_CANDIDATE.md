# M6A-PUBLIC-001 neural target frozen method

| 字段 | 内容 |
|---|---|
| 状态 | `METHOD_FROZEN_COORDINATOR_ACCEPTED` |
| Coordinator review | `ACCEPT`（2026-08-11） |
| 主目标 | `LINE_HARMONIC_EXCLUDED_MULTIBAND_HIGH_GAMMA_LOG_POWER` |
| 主 reference | `AS_RECORDED_SCALP_REFERENCE` |
| 真实神经提取 | 全数据 `NOT_ALLOWED`；仅 G3 scoped single-passage smoke 获授权 |
| 模型下载 / baseline | `NOT_ALLOWED` |
| 配置 | `configs/m6a_neural_target_method_candidate.json` |
| schema | `schemas/m6a_neural_target_method_candidate.schema.json` |
| 语义门禁 | `python -m m6a_public.neural_target_gate ...` |

本文件由既有 candidate 原位升级为唯一正式方法文档；文件名保留 candidate provenance，不另建重复 frozen 配置或文档。

## 1. 预声明主目标

主目标固定为六个等宽 10 Hz 子带：`70-80`、`80-90`、`90-100`、`100-110`、`130-140`、`140-150 Hz`。`110-130 Hz` 因包含 120 Hz 工频二次谐波而排除。六个子带各自在 train valid frames 上拟合变换，再以 `1/6` 等权平均；不得按 encoding 表现改变频带、权重、顺序或主目标。

原方案 `70-150 Hz + 60/120 Hz rejection` 仅保留为 `PREDECLARED_SENSITIVITY_NOT_PRIMARY`。它不能替换主目标、不能按结果择优，且执行前须另行批准。

## 2. Reference 冻结

11 个 iEEG sidecar 的 `iEEGReference` 均应由机器审计为 `scalp electrode, not included with data`。主分析保持数据的 `AS_RECORDED_SCALP_REFERENCE`：

- 不猜测或重建未随数据提供的 scalp reference；
- 不从 contact 名称构造 bipolar pair；
- CAR、CMR、bipolar 只能作为未来单独批准的 sensitivity；
- 该限制逐 recording 写入 G2 metadata audit，任何缺失或不一致均 fail closed。

## 3. 有限支持滤波与功率

每个子带使用 `scipy.signal.firwin` 设计对称奇数阶 Kaiser FIR，formal backend 为 `scipy.signal.oaconvolve` overlap-add：

- nominal passband 为各子带 low/high；
- 两侧 transition 均为 2 Hz，cutoff midpoint 为 `low-1` 与 `high+1 Hz`；
- stopband attenuation 60 dB；
- taps 规则为 `odd(kaiserord(60 dB, 2/(fs/2)))`；
- 输入短于 bandpass FIR 时直接失败，禁止 `same` 模式静默改变长度；
- bandpassed signal 先平方，再用有限 Kaiser low-pass 平滑；low-pass passband edge 10 Hz、stopband edge 20 Hz、cutoff 15 Hz、attenuation 60 dB；
- 禁止对整段 recording 使用 FFT Hilbert；连续 recording 可处理，但完整支持越过 event/split 允许区间的输出帧必须 mask，padding 或卷积不得跨 split 共享。

实际 profile：

| fs | bandpass taps / radius | power taps / radius | FIR chain radius | linear interpolation radius | total edge |
|---:|---:|---:|---:|---:|---:|
| 512 Hz | 931 / 465 samples | 187 / 93 samples | 558 samples | 1 sample | 559 samples = 1.091796875 s |
| 1024 Hz | 1859 / 929 samples | 373 / 186 samples | 1115 samples | 1 sample | 1116 samples = 1.08984375 s |

## 4. Train-only 变换与时间语义

对每个 channel 的每个子带：

1. 仅用 train valid frames 的正功率拟合 `epsilon = max(1e-30, 1e-6 * median(positive power))`；
2. 计算自然对数 `log(power + epsilon)`；
3. 仅用 train valid frames 拟合 mean 与 population standard deviation；
4. epsilon/center/scale 均须 shape `(6,)`、全 finite，且 epsilon、scale 严格大于 0；失败时拒绝该 channel/subband，不静默广播或除零；
5. 冻结参数后应用到 validation/test，最后六子带等权平均。

输出采用 50 Hz 网格 `k/50 s`，timestamp 表示 output frame center 的 recording time。先有限低通得到 power，再以两个最近源采样点线性插值，不外推。正 lag 定义为 `audio at t predicts neural at t + lag`，预声明 lag 为 0 至 0.5 s、步长 0.05 s。

## 5. Synthetic 设计验收

512/1024 Hz 均须覆盖 impulse、step、60 Hz、120 Hz、代表性 passband sine，并验证：

- coefficient symmetry absolute tolerance `1e-12`；
- overlap-add 与 tiny/bounded direct convolution：absolute/relative tolerance 均 `1e-12`；
- impulse 在声明 FIR-chain support 外绝对值不超过 `1e-12`；
- passband center gain 位于 `[-0.1, 0.1] dB`；
- 60/120 Hz frequency-response gain 均不高于 `-55 dB`；
- time-domain harmonic/passband power RMS ratio 不高于 `0.01`；
- short input、NaN/±Infinity、退化 transform、错误 support mask 均 fail closed。

这些测试只验证数学实现和门禁，不是对真实神经 target 的科学验证。

## 6. Final embargo 与停止点

已计算 filter/resampling edge 最大值为 1.091796875 s。结合实测 0.0 s audio cross-split overlap 与 0.0006349206349206349 s 音频重采样边缘，协调已接受 `final_embargo=2.0 s` 和 baseline-final split。

该接受不开放全数据神经提取。当前只有 `configs/m6a_g3_single_recording_candidate.json` 明确列出的单 recording、单 passage、36 channels 与有限支持范围可用于工程 smoke；其他 recording/segment、正式 train-only transform、baseline/G4 仍被阻塞。

## 7. 声称边界

现在可声称：神经目标的有限支持方法、参数、时间语义与 synthetic fail-closed 门禁已冻结；G3 单 passage 工程 smoke candidate 已生成。不能声称正式 train-only high-gamma target、目标生理特异性、brain region、ridge/null/指标、M6A exchange candidate、整条 M6A accepted/frozen 或科学结果已经完成。
