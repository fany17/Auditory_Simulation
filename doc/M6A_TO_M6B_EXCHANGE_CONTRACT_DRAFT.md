# M6A→M6B Exchange Contract v1 草案

| 字段 | 内容 |
|---|---|
| 发布端 | `Auditory_Simulation / M6A-PUBLIC` |
| 消费端 | `STN_Decoding_Encoding / future M6B-STN` |
| 状态 | `DRAFT_PROPOSED_BY_M6A` |
| Schema | `schemas/m6a_to_m6b_exchange_manifest_v1.schema.json` |
| 冻结状态 | `NOT_FROZEN` |
| 接受条件 | 协调审核 + STN consumer cross-test |

## 1. 目的

M6A 向 M6B 发布可在患者声音上独立重放的冻结提取方法、变换规范、公开 benchmark 证据和 tiny synthetic/canary fixture。接口不传输 `ds004703` 原始数据，不要求 STN 仓库接收公开数据全量特征，也不允许 M6A 读取 STN 患者数据。

Auditory 内部运行 manifest `m6a-public-internal-manifest-v1` 只服务于公开数据实验；它与本 exchange manifest 的身份、字段和发布状态不同。

## 2. 必需交付组成

### 2.1 Identity 与发布状态

- contract、release、method package 和 manifest 的可读版本；
- `DRAFT / CANDIDATE / FROZEN / ACCEPTED / RETIRED` 状态；
- producer review、consumer cross-test 和协调决定分开记录；
- 当前必须保持 `DRAFT_PROPOSED_BY_M6A`，不得提前写成 frozen/accepted。

### 2.2 公开数据与许可边界

- dataset ID、版本、DOI、公开 benchmark 用途和完整许可边界；
- `ds004703` 执行 CC0 元数据与 README 非商业/禁止再识别的更严格组合；
- redistribution 明确为“不随 exchange package 分发原始数据或全量公开特征”。

### 2.3 模型与运行时

- model ID、可读 revision label、解析日期、许可、frozen 状态；
- Python、PyTorch、Transformers、NumPy 等直接软件版本；
- no-hash 政策导致 revision label 不具备密码学不可变性时，必须写入 limitation。

### 2.4 音频预处理与时间参考

- 输入采样率、声道、幅度归一化、重采样实现、padding/chunk 规则；
- frame center/left-edge 定义、模型感受野、输出 frame rate；
- 输入音频零点与输出第一个 frame 的对应关系；
- 任意裁剪、上下文重叠和边缘丢弃规则。

### 2.5 Layer inventory 与 extraction spec

- 每层可读 layer key、ordinal、来源、frame rate、dtype、feature dimension 和时间轴语义；
- 冻结提取入口、配置 schema、batch/chunk 行为、设备与 deterministic 限制；
- 不把层号直接命名为脑区。

### 2.6 可迁移变换

- train-only normalizer、PCA 或 basis 的类型、输入层、输出维度、拟合数据/split 和适用范围；
- 变换参数作为轻量/中等规模文件列入 inventory；
- 不允许在 STN 数据上重新拟合后仍声称使用同一冻结变换；若需适配，必须由 M6B 独立任务记录。

### 2.7 公开 benchmark 证据

- split、泄漏检查、null、指标、benchmark summary、失败与 not-estimable 项；
- can_claim/cannot_claim 分离；
- M6A 的公开 benchmark 只支持方法性能边界，不支持患者特异结论。

### 2.8 文件 inventory 与 canary

- 文件路径、角色、字节数、时间戳、schema 与 remote-only 状态；
- tiny fixture/canary 必须是合成或可再分发数据，用于验证 shape、frame time 和 layer ordering；
- tiny TSV 可作为人工可读 canary，但不得作为高维 feature 主载荷。

## 3. 高维载荷策略

正式高维特征格式当前为 `PENDING_G3_ARRAY_FORMAT_BENCHMARK`。候选限定为 Zarr、HDF5、Parquet 或 NPY bundle，选择依据包括：

- 真实层数、时间帧、feature dimension 和 dtype；
- 单 recording/单层切片速度；
- 顺序写入、随机读取、压缩、元数据表达与部分读取；
- Python 依赖稳定性和 STN consumer 的只读实现成本。

TSV 被禁止作为高维主载荷。reference payload 是可选 remote-only 资产，不是 STN 接口的必需输入；正式合同必须能仅凭 method package + transforms + canary 在消费端重放提取。

## 4. Consumer cross-test

冻结前，STN consumer 必须在不读取 `ds004703` 原始数据、不访问 M6A 可变目录的条件下完成：

1. 安装或导入冻结 method package；
2. 读取 exchange manifest 与 canary；
3. 对 tiny synthetic audio 重放提取；
4. 验证 layer ordering、shape、dtype、frame time 和预处理口径；
5. 验证缺文件、版本不兼容和不支持输入能显式失败；
6. 证明无需 TSV 高维全量载荷。

未完成 cross-test 时状态保持 `DRAFT_PROPOSED_BY_M6A` 或 `CANDIDATE_FOR_CROSS_TEST`。

## 5. 当前结论

- `CROSS_PROJECT_INTERFACE=REDESIGN_REQUIRED`
- `FROZEN_M6A_ARTIFACT=NO`
- `EXCHANGE_CONTRACT_STATUS=DRAFT_PROPOSED_BY_M6A`
- 该状态不阻塞 M6A G0-G2 环境、数据读取、split 与 baseline 门禁。
