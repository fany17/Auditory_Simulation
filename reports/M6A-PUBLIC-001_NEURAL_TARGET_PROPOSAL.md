# M6A-PUBLIC-001 neural target / final embargo proposal

| 字段 | 内容 |
|---|---|
| 状态 | `DRAFT_AWAITING_COORDINATOR_REVIEW` |
| Target gate | `NEURAL_TARGET_REDESIGN_REQUIRED_BEFORE_G3` |
| Anatomy gate | `ANATOMY_MAPPING_NOT_READY` |
| Neural extraction | `NOT_ALLOWED` |
| Baseline final | `false` |

## 已冻结的输入证据

- 11 份 iEEG sidecar 均声明 60 Hz 工频；sampling rate 为 512 或 1024 Hz；
- 原 70-150 Hz target 穿过 120 Hz 二次谐波；
- channels.tsv 的 good SEEG/ECOG 需排除 C-prefix；
- 标准 electrodes.tsv/coordsystem.json 均为 0，region summary 为 `NOT_ESTIMABLE`。

## 待协调选择

| 方案 | Target | 核心验证 | 当前状态 |
|---|---|---|---|
| A | 宽带信号显式 60/120 Hz rejection 后提取 70-150 Hz | 残余谐波、passband 扭曲、ringing、filter edge | `PROPOSED_NOT_FROZEN` |
| B | 70-110 与 130-150 Hz 分带解析功率后聚合 | 子带尺度、聚合稳定性、edge、跨 sampling-rate 一致性 | `ALTERNATIVE_NOT_FROZEN` |

两方案都必须在冻结前明确：滤波实现/transition/相位、瞬态 padding、Hilbert amplitude 或 power、`log(power+epsilon)`、50 Hz anti-alias/downsample、最大 0.5 s lag 的方向语义，以及真实 filter edge。

## Final embargo gate

已实现 fail-closed 计算器：

`max(preliminary 2 s, maximum lag 0.5 s, measured filter/padding edge, measured wav2vec2 receptive-field/context overlap)`。

后两项仍为 null，因此 gate 为 `PENDING_MEASUREMENT`、`baseline_final=false`。负值与 NaN/±Infinity 均 fail closed。target/filter 与 wav2vec2 isolated-stimulus context 未实测冻结前，不重跑 final split guard，也不进入 G3。

## 运行证据

```text
53 passed, 22 subtests passed in 0.17s
All checks passed!
Success: no issues found in 19 source files
config gate: PASS
```

本 proposal 不下载模型、不提取神经特征、不运行 baseline，不把 contact 名称推断为脑区，也不执行任何哈希或校验和验证。
