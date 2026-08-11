# M6A→M6B Exchange Contract v1 修订草案

| 字段 | 内容 |
|---|---|
| 发布端 | `Auditory_Simulation / M6A-PUBLIC` |
| 消费端 | `STN_Decoding_Encoding / M6B-STN` |
| 状态 | `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW` |
| Schema | `schemas/m6a_to_m6b_exchange_manifest_v1.schema.json` |
| Validator | `src/m6a_public/exchange_validator.py` |
| 冻结状态 | `NOT_FROZEN` |
| Consumer cross-test | `NOT_RUN` |
| 下一接受条件 | STN consumer schema/cross-test 复核 + 协调决定 |

## 1. 身份与边界

本草案定义 M6A 向 M6B 发布的可重放方法接口。正式目标交付由冻结提取方法、运行时规范、可迁移 train-only 变换、公开 benchmark 证据和 tiny synthetic canary 组成。它不传输 `ds004703` 原始数据，不要求 STN 仓库接收公开数据全量特征，也不授权 M6A 读取 STN 患者数据。

Auditory 内部运行 manifest 的 `$id` 为 `m6a-public-internal-manifest-v1`，只服务公开数据实验。它与本 exchange manifest 的身份、字段和发布状态不同，不能作为跨项目合同发布。

当前不存在真实 exchange manifest、method package、runtime spec、canary 或冻结 M6A artifact。本轮仅修订 schema、validator、测试和文档，不得把测试 fixture 当作候选 bundle。

## 2. 草案状态机与 acceptance

本 draft schema 只允许：

1. `DRAFT_PROPOSED_BY_M6A`；
2. `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW`；
3. `CANDIDATE_FOR_CROSS_TEST`。

`FROZEN`、`ACCEPTED` 和 `RETIRED` 不属于 draft schema。只有真实 candidate 完成 consumer cross-test 且协调审核接受后，Auditory 才能另行发布不带 `draft` 的 v1 schema。

合法状态跃迁为：

| 前态 | 后态 | 条件 |
|---|---|---|
| 初始 | `DRAFT_PROPOSED_BY_M6A` | 首次提出字段草案 |
| `DRAFT_PROPOSED_BY_M6A` | `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW` | producer 完成 16 项返工与本地/远端测试 |
| `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW` | `CANDIDATE_FOR_CROSS_TEST` | 真实 method、runtime、canary、inventory 与 benchmark evidence 全部存在且 validator 通过 |
| 任一草案状态 | 同一状态 | 只更新该状态内的审计证据，不降低门禁 |

禁止跳过 revised draft 直接进入 candidate。状态历史必须从最初 draft 开始，按顺序记录，并以当前 `release_status` 结束。

`producer_review`、`consumer_cross_test`、`coordinator_decision` 只能取 `PENDING / PASS / FAIL / NOT_APPLICABLE`。当前修订草案必须为 `PASS / PENDING / PENDING`。Candidate 可记录 consumer 的 `PENDING / PASS / FAIL`，但 draft schema 中 coordinator 始终保持 `PENDING`；协调接受后应转入独立正式 v1 schema。

## 3. 模型、运行时与 revision 限制

Manifest 必须记录：

- model ID、可读 revision label、解析日期、许可和 `frozen=true`；
- `python / torch / transformers / numpy` 最小直接版本 profile；
- 实际 model config、preprocessor 和权重的文件名、字节数、时间戳、缓存边界和是否进入 exchange；
- revision 是否不可变；若不是不可变 revision，必须给出 `revision_limitation`。

本项目禁止密码学校验，因此 `main + 解析日期 + 文件名/字节数/时间戳/配置/软件版本` 只能形成受限的非密码学复现边界，不能声称模型权重得到密码学固定。模型缓存和权重保持 2203 remote-only，不进入 STN 仓库。

## 4. 音频、时间轴、layer 与提取方法

音频预处理必须明确输入采样率、声道、幅度规则、重采样实现、chunk/padding、时间零点和边缘处理。Layer inventory 必须包含稳定的 `layer_key`、从 0 连续递增的 ordinal、来源、frame rate、dtype、feature dimension 与 frame-time 语义。

Validator 会拒绝：

- 重复 layer object、layer key 或 ordinal；
- 未按 ordinal 顺序且不从 0 连续的 layer；
- canary layer order 与 layer inventory 不一致；
- canary shape、dtype、feature dimension 或 frame-time shape 与声明不一致。

Extraction spec 与 method package 必须显式引用 entrypoint、runtime spec、config schema 和 output schema；这些路径均须列入 inventory，并具有匹配角色。

## 5. 可迁移变换

Normalizer、PCA 或 basis 必须声明 train-only fit scope、输入层、输出维度、适用范围和 artifact 文件。Artifact 必须列入 inventory，candidate 中必须本地包含且可读。若 STN 在患者数据上重新拟合，不能继续声称使用同一冻结变换，必须由 M6B 独立记录适配边界。

## 6. Benchmark evidence 最小结构

`validation` 必须具有非空 split、leakage checks、nulls、metrics 和 benchmark summary。每条 benchmark evidence 至少包括：

- scope；
- dataset ID/version；
- split；
- target；
- model or baseline；
- metric 与 value；
- null method/status/value/permutation/limitations；
- evidence status；
- limitations。

状态允许 `SUPPORTED / PROVISIONAL / FAILED / NOT_ESTIMABLE`。`FAILED` 与 `NOT_ESTIMABLE` 的 value 必须为 null；`SUPPORTED` 必须为数值。草案阶段不能用空数组掩盖未运行，应使用显式 `NOT_ESTIMABLE` 和 limitations。Candidate 必须使用 `CANDIDATE_EVIDENCE` profile。

公开 benchmark 仅支持方法性能及失败边界，不支持患者特异、临床或模型层等同脑区的结论。

## 7. Inventory 与安全路径

Inventory 路径必须是唯一、规范化的 POSIX 相对路径。绝对路径、Windows drive、反斜杠、`.`/`..` 路径段和 bundle root 逃逸均 fail closed。

受控角色包括 method entrypoint、runtime、extraction schemas、canary input/output、transform、benchmark evidence、model metadata、documentation 与可选 reference payload。Candidate 对 method/runtime/schema/canary 的单例角色必须各恰好出现一次；所有交叉测试所需文件必须为 `INCLUDED_LOCAL`。Validator 使用文件存在性、可读性和字节数核对，不做任何哈希或校验和验证。

## 8. Canary cross-test contract

Canary 必须是可再分发的合成单声道音频，并定义：

- input file/format、采样率、声道与 sample count；
- expected output file/format；
- 完整 expected layer order；
- 每层 shape、dtype、frame count 与 feature dimension；
- frame-time array key、单位、shape 与参考零点；
- absolute、relative 和 frame-time 数值容差。

Candidate validation 需要显式 `bundle_root`；缺少本地 method/runtime/schema/canary 文件、字节数不符或关联角色错误时必须失败。Canary 用于数值、shape、dtype、frame-time 和 layer-order 重放，不提供模型权重的密码学固定证据。

Tiny TSV 只可作为 canary 输出；高维主载荷仍禁止 TSV。正式高维格式将在 2203 基于真实 shape、切片、顺序写入、随机读取和依赖成本，从 Zarr、HDF5、Parquet 或 NPY bundle 中选择。

## 9. Validator 与失败语义

验证入口：

```text
python -m m6a_public.exchange_validator MANIFEST.json \
  --schema schemas/m6a_to_m6b_exchange_manifest_v1.schema.json \
  --bundle-root BUNDLE_ROOT
```

Draft schema 先拒绝未知字段和无效类型，语义 validator 再核对状态跃迁、路径、inventory 集合关系、角色、layer 唯一性、canary 元数据与 candidate 本地文件。任一错误均返回非零退出码；不允许仅记录 warning 后继续。

## 10. 当前结论

- `CROSSWALK_REVIEW=PASS_WITH_REWORK`（STN 首轮只读审核结论）
- `REWORK_DISPOSITION=16/16_IMPLEMENTED_BY_M6A`
- `EXCHANGE_CONTRACT_STATUS=REVISED_DRAFT_AWAITING_CONSUMER_REVIEW`
- `CONSUMER_CROSS_TEST=NOT_RUN`
- `CANDIDATE_BUNDLE=NO`
- `FROZEN_M6A_ARTIFACT=NO`

该状态不阻塞 M6A G2 数据读取、语言/刺激 manifest、split 与 baseline 门禁。真实 candidate 只能在 G2 完成并依据真实音频与模型行为生成后提出。
