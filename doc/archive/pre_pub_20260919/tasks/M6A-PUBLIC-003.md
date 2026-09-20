# M6A-PUBLIC-003：Temporal Architecture Perturbation

| 字段 | 内容 |
|---|---|
| 状态 | `ACTIVE_EXECUTION` |
| 起始日期 | `2026-08-21` |
| 所属项目 | `M6A-PUBLIC / Auditory_Simulation` |
| 前置任务 | `M6A-PUBLIC-002` 已完成 pretrained architecture baseline |
| 患者/STN 数据 | `FORBIDDEN` |
| 大型 pretrained backbone 微调 | `FORBIDDEN` |

## 1. 任务定位

`M6A-PUBLIC-002` 已完成 9/9 pretrained inference、8/9 unified temporal representation probe，并覆盖 CNN、temporal convolution、BiLSTM、FastConformer、Transformer、Mamba、CoNNear auditory periphery 与 ICNet auditory midbrain surrogate。本任务不再补 baseline，也不再下载新旗舰模型。

本阶段只回答一个问题：**现有声音/时序网络中，哪些明确的结构变量会造成快速时间信息的保留、整合或丢失？**

禁止按 `cochlea→CN→IC→MGB→A1` 机械映射网络模块。任何 auditory-inspired 修改必须先被表述为可计算、可消融、可证伪的机制。

## 2. 第一阶段核心科学问题

优先研究三组问题：

1. **Temporal resolution / downsampling timing**：过早 stride、pooling 或 patching 是否导致 onset、jitter、omission、phase shift 等快速事件过早丢失？
2. **Multiscale receptive field**：receptive field 应该多快随层级增长；同层并行多个时间窗口是否优于单一尺度；扩大 RF 是否必须牺牲 temporal resolution？
3. **Explicit change/event pathway**：显式变化通路是否提供超出“多一条普通支路”本身的收益？

`fast/slow pathway`、auditory-periphery frontend、HSR/MSR/LSR-like dynamic-range channels、event-gated Mamba/LSTM state update 均先进入候选 registry，不与第一轮三个主题同时叠加。

## 3. Structural Modification Registry

先生成 `reports/STRUCTURAL_MODIFICATION_REGISTRY.md` 与轻量 CSV。每个候选修改必须记录：

- 修改位置；
- 原始结构；
- 修改结构；
- 计算意义；
- 对 temporal resolution 的影响；
- 对 theoretical receptive field 的影响；
- 参数量/计算量变化；
- 预期改善的信息维度；
- 潜在代价；
- matched control；
- 可证伪判据。

## 4. Temporal resolution / downsampling

必须显式审计：

- stride；
- pooling；
- patch size；
- downsampling 所在层级；
- 总 downsampling factor；
- 每层 output frame step / temporal resolution。

第一轮至少比较：

### E1-A Early downsampling
较早进行 stride/pooling。

### E1-B Late downsampling
保持最终总下采样倍率尽量一致，但把下采样推迟到后层。

核心判据：在参数量、深度和最终压缩倍率接近时，early vs late downsampling 对快速时间事件定位/分类的影响。

## 5. Multiscale receptive field：必须拆开研究

禁止只写“加入多尺度感受野”。至少把以下旋钮分开。

### 5.1 Kernel size

比较例如 `k=3/7/15/31`。解释单层直接观察的时间范围如何变化，并换算到毫秒。

### 5.2 Dilation

固定或近似固定 kernel，比较 dilation。例如 `d=1/2/4/8/16`。

至少比较三种 schedule：

- `uniform/local`: `1,1,1,1,...`
- `exponential`: `1,2,4,8,...`
- `delayed-growth`: `1,1,1,2,4,8,...`

核心问题：网络前部是否应更长时间维持小 RF，而不是迅速整合大窗口。

### 5.3 Parallel multiscale RF

同层并行多个 RF，例如：

```text
Input
├─ k=3
├─ k=7
├─ k=15
└─ k=31
   ↓
 fusion
```

或固定 kernel、并行不同 dilation。必须与 parameter-matched 单尺度模型比较，不能把收益归因于参数变多。

### 5.4 Receptive-field growth schedule

至少比较：

- fast growth；
- delayed growth；
- parallel multiscale。

必须逐层计算 RF，而不是只描述“浅层/深层”。

### 5.5 RF 与 downsampling 解耦

建立 matched comparison：

- A：主要通过 stride/pooling 扩大 effective RF，同时降低 temporal resolution；
- B：主要通过 dilation 扩大 RF，尽量保持 temporal resolution。

尽量匹配 depth、parameter count 与 theoretical RF。核心问题是：快速时间信息的损失来自“大 RF 本身”，还是来自“获得大 RF 时伴随的下采样”。

### 5.6 必须生成 RF 表

生成 `reports/receptive_field_by_layer.csv`，至少包含：

- model/variant；
- layer；
- kernel；
- stride；
- dilation；
- jump/frame-step；
- theoretical_RF_samples；
- theoretical_RF_ms；
- output_time_resolution_ms。

同时在报告中解释 theoretical RF 与 effective RF 不等价。

## 6. Explicit change/event pathway

第一版必须最小化，不允许直接发明复杂模块。

示意：

```text
Input
├─ content branch
└─ change/event branch: Δx(t) / onset-energy / simple difference feature
      ↓
    fusion
```

至少设置：

1. baseline single branch；
2. explicit change branch；
3. parameter-matched ordinary second branch。

只有 2 优于 1 且相对 3 仍有稳定收益，才能支持“显式变化表示”而非“多加参数/多加支路”。

## 7. 小型结构 benchmark

本任务允许训练**小型 matched experimental models**，因为结构修改后必须重新学习参数；但禁止训练/微调 wav2vec2、Whisper、Parakeet、Audio-Mamba 等大型 pretrained backbone。

要求：

- 使用统一小型 1D CNN/TCN backbone；
- 参数量尽量匹配，目标 ±10%；
- 相同 optimizer、epoch、batch、split；
- 至少 3 seeds；
- 禁止为每个结构单独 hyperparameter search；
- synthetic 数据训练/验证/测试按独立生成 seed 与参数范围隔离。

至少包含：

- rate discrimination；
- regular vs jitter；
- omission detection；
- phase-shift detection；
- onset/event timing；
- unseen-rate / unseen-jitter / unseen-phase magnitude generalization。

## 8. 第一轮正式实验

### Experiment 1 — Early vs Late Downsampling
只改变 downsampling timing，尽量保持总压缩和参数量一致。

### Experiment 2 — Receptive-Field Growth
至少比较 uniform/local、exponential、delayed-growth、parallel multiscale。

### Experiment 3 — Explicit Change Branch
baseline vs change branch vs parameter-matched ordinary branch。

第一轮禁止把 cochlear frontend、fast/slow branch、Mamba gating 同时混进上述实验。

## 9. 指标

不能只给 overall accuracy。至少记录：

- balanced accuracy；
- omission detection；
- phase-shift detection；
- onset/timing error (ms)；
- perturbation magnitude generalization；
- unseen-rate generalization；
- parameter count；
- approximate FLOPs/compute；
- inference latency；
- theoretical RF；
- temporal resolution by layer；
- no-fit representation distance；
- event-triggered response；
- temporal persistence/autocorrelation。

重点回答：10/20/50 ms 级扰动经过不同结构后还能否定位、区分和泛化。

## 10. 解释模板

每个实验必须回答：

1. 改了哪个结构旋钮？
2. 计算上具体改变什么？
3. RF 如何变化？
4. temporal resolution 如何变化？
5. 参数量/计算量是否变化？
6. 哪些任务改善？
7. 哪些任务恶化？
8. matched control 是否支持结构因果解释？
9. 是否支持下一阶段继续？

禁止仅用“更像大脑”“更符合听觉层级”解释阳性结果。

## 11. 后续候选，仅登记不混跑

- fast/slow parallel pathways：不同 frame/update rate；
- cochlear-like filterbank/compression/IHC nonlinearity；
- HSR/MSR/LSR-like sensitivity/dynamic-range paths；
- event-gated recurrent/Mamba state update。

这些必须等第一轮得到明确结构缺口后再单独立项。

## 12. 交付物

至少：

- `reports/STRUCTURAL_MODIFICATION_REGISTRY.md`
- `reports/receptive_field_by_layer.csv`
- `reports/TEMPORAL_BENCHMARK_DESIGN.md`
- `reports/DOWNSAMPLING_ABLATION.md`
- `reports/RECEPTIVE_FIELD_ABLATION.md`
- `reports/EVENT_BRANCH_ABLATION.md`
- `reports/M6A-PUBLIC-003_SUMMARY.md`

图优先展示：RF(ms) 随 depth、temporal resolution 随 depth、performance vs perturbation magnitude、matched architecture comparison。

## 13. 硬边界与停止规则

- 不读取患者/STN 数据；
- 不修改 `STN_Decoding_Encoding`；
- 不微调大型 pretrained backbone；
- 不为了阳性结果反复调参；
- 阴性结果必须保留；
- 多个结构同时变化时不得归因到单一机制；
- 参数匹配失败时明确标记 confounded；
- 本任务仍属于公开模型与通用计算方法的 `M6A-PUBLIC`。
