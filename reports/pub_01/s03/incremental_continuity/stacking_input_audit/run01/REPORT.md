# PUB-01-S03_SS07 后处理训练预测来源核查

2026-11-06 · `stacking_input_audit/run01` · 只读证据汇总

**D19、D20、D22、D23 的 426 条 TRAIN 记录全部使用记录内训练预测。** 三个 seed 都先用同一批 426 条记录联合微调 EEG 编码器和 head，再用这批记录拟合 Ccap 与其变换。每个 seed 用自己的同一个 FT checkpoint 生成 TRAIN 和 VAL 的缓存。随后各候选共享相同 FT+Ccap 基线。当前 49 条 VAL 记录与拟合集的记录 ID 交集为 0，VAL 用于 checkpoint 或候选选择，仍属于开发评估。已查范围内未找到可复用的折外预测。

## 训练来源与覆盖

| 上游环节 | 当前 TRAIN 交集 | 当前 VAL 交集 | 实际证据和限制 |
|---|---:|---:|---|
| EEG 全局通道均值和标准差 | 426 | 0 | `prepared/manifest_frozen.json.normalization_fit_phase=train_only`。准备源码累积原始共同支持的 TRAIN 样点，清单合计 24,297,126 点/通道。两份训练协议与 intake 记录同一 `train_only_normalization.npz` 标识。 |
| EEG/XLSR 对比学习 | 426 可采 | 0 可采 | `prepared/train_windows.json`、训练源码和完成收据可复原候选集合及 820 batches/epoch。TRAIN 126,341 条索引中 126,156 个记录×时间窗可进入实际采样，每轮呈现 104,960 窗。采样先选同一 story 的时间，再从该时间可用被试中抽样。现有收据没有逐记录访问日志，无法据此确认选中 checkpoint 前每个 seed 的精确访问集合。VAL 的 49 条记录另走固定种子的验证采样。 |
| Native envelope head | 426 | 0 | 每 seed 的 `cache_manifest.json.records` 将 252,454 TRAIN 窗和 30,500 VAL 窗定位到原记录。源码逐批遍历全 TRAIN，保留尾批。每 seed 完成收据记载 250 轮、63,113,500 次 TRAIN 窗呈现，选中 epoch 249。 |
| Native encoder + head 联合 FT | 426 | 0 | `resolved_config.json` 的初始 encoder/head 标识与冻结 checkpoint 清单一致。各 seed 完成收据为 5 轮、19,725 次更新、1,262,270 次 TRAIN 窗呈现，选中 epoch 4。训练源码每轮断言完整遍历 252,454 窗，VAL 单独评分。 |
| `P_FT` 仿射校准 | 426 | 0 | D8 `P_FT_BASELINE.json.calibration.TRAIN_physical_points` 每 seed 为 24,275,628 个正 Hann 支持点。 |
| Ccap 变换及 ridge 系数 | 426 | 0 | D8 `fit_seed` 只从 TRAIN 收集潜变量、校准后的 P、标准化潜变量的平方及残差。每记录前 4 窗留作校准后的 P 回退。变换与系数拟合共 250,750 TRAIN 窗。另有 30,304 个 VAL 窗可用 Ccap 评分，这四轮固定使用 `λ=0.01`。 |

EEG 的全局通道统计在对比学习、native head 缓存和 FT 中复用。对比学习的向量 L2 归一化按样本计算，模型的 LayerNorm 随模型训练，无另外拟合的全局均值表。Head 与 FT 直接切片已发布的 envelope 目标，负 Pearson 损失按窗中心化。D22 与 D23 进一步复用同 seed 的 D20 `P` TRAIN 标量统计。

## 缓存与评分的绑定

- D8 `PROTOCOL.json.records.{train,val}` 与 prepared、envelope、D19 的清单在记录顺序、被试、story 和共同样本数上相同。D8 的 `cache_seed` 对每个 seed 先验证并加载同一个选中 FT checkpoint，设置 eval 后依次生成 VAL 和 TRAIN。各 seed 的 `CACHE_COMPLETED.json.checkpoint_sha` 与 D8 协议一致。当前核验引用了既有 checkpoint 摘要并比较文件大小，未重新读取 checkpoint 字节。
- D19 `PREPARATION_METADATA.json.inputs` 指向 D8 各 seed 的 `{train,val}_{latent,P}.npy`、`TRAIN_TRANSFORMS.npz`、`Ccap_lambda0.01_COEFFICIENTS.npz` 及 `P_FT_BASELINE.json`。21 个文件的当前大小和 mtime 均与记录相符。D20 `d19_pins` 及 D22/D23 `frozen_pins` 固定同一 D19 元数据与协议。逐 seed 对比 12 份 `fit/seed*_ANCHOR.json`，其中保存的 `anchor` 对象在四轮间完全相同。
- 三 seed 等权 anchor 相关从 TRAIN 的 **0.246589611** 降到 VAL 的 **0.182339940**。Active MSE 从 **0.191850855** 升至 **0.250262038**，绝对差 **0.058411183**，相对约 **30.4%**。全覆盖 MSE 为 **0.204884700 / 0.242378889**。这些是已有标量的描述性差距，无法将差距归因于记录内训练预测，也无法由此判断后处理训练退化的原因。

## 音频来源与折外资产

`SOURCE_FAMILY_JOIN.json` 的真实 `source_family` 把同源刺激的不同 story ID 归并。TRAIN 有 41 个 story、37 个源族和 70 名被试，VAL 有 5 个 story、5 个源族和 49 名被试。精确 story ID 交集为 0。两边共享 `audiobook_13`（TRAIN 6、VAL 6）和 `audiobook_5`（TRAIN 48、VAL 23），覆盖 **54/426 TRAIN** 与 **29/49 VAL** 记录。全部 49 名 VAL 被试在 TRAIN 出现。TRAIN 源族记录数最少 1、中位数 6、最多 65，17 个源族各不超过 5 条。完整族级分组与窗口计数见 [AUDIT.json](AUDIT.json)。

OOF 搜索覆盖 26 个相关项目根下 1,500 个目录，按文件名/目录名匹配 `oof`、`out-of-fold`、`crossfit`、`fold<N>`、`kfold` 等，跳过旧测试和外部数据目录。无候选名称。已知 D8 缓存的协议和记录清单明确指向每 seed 一份全 TRAIN checkpoint。不同 seed 复用同一划分。该结论限于所列根目录及核对过的协议和文件名，未知名称或其他路径下的缓存无法由这次盘点排除。

供协调审阅的最小对照是先冻结仅 TRAIN 的 `source_family` 分折清单，按各族窗数平衡，单 seed 固定一个既有标量后处理，用相同训练行比较记录内与折外输入。每折需要从对比学习、通道统计、head、FT、P 校准和 Ccap 中一并排除该折记录，保留同一全 TRAIN 基线在原 VAL 上的评分口径。若成本只允许少量整族折外预测，两臂都限制到同一批预定训练行，并将结果标为局部可行性诊断。该对照尚未运行。

## 复核入口

[AUDIT.json](AUDIT.json) 保存远端绝对路径、协议字段、各 seed checkpoint 既有标识、样点/窗口计数、37 个 TRAIN 源族与 VAL 的分组及查找范围。[生成脚本](../../../../../../scripts/pub_01_s03_ss07_stacking_input_audit.py) 仅在 server2203 处理 JSON 元数据与文件属性。证据源码为旧 baseline 的 `input/pub_01_s03_transformer_inputs_attempt02.py`、`pub_01_s03_transformer_train_v2.py`、`pub_01_s03_transformer_envelope.py`，SS07 的 `input/pub_01_s03_ss07_baseline.py`、`pub_01_s03_ss07_finetune.py`，以及 D8 的 `input/pub_01_s03_ss07_ft_readout.py`。此次核查读取的信号数组、权重字节、模型调用、拟合、下载均为 0，未读取旧测试与外部评估的载荷或指标。
