# DECAF 公开实现协议链审计

2026-11-06 · PUB-01-S03_SS07 · 源码与轻量配置审计

**最接近论文的公开命令已定位。锁定实现的普通推理路径从过去 EEG 重新预测包络，再生成先验；调用间没有把上一条最终融合输出存下来反馈。** 论文叙述的是上一最终输出回授。现有公开记录也无法确定论文所述 500 ms EEG 延时在磁盘数据的哪个生成环节实现。由此可以界定可复用组件和协议差异，但无法用该提交认定论文旧 run 的确切行为。[论文 Methods](https://arxiv.org/html/2602.19395v1#S3) · [静态回执](STATIC_CHECK.json)

## 最接近论文的入口

JHU-LCAP 的 README 指向 carankt/DECAF。审计锁定后者 Git commit `f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332`，逐字节核对 49 份源或配置的 Git blob。21 份核心文本与 MIT 许可存于 `upstream/`。读取范围包含注册、训练、数据、两分支模型、出图和作者补充配置，无数据集载荷、模型权重、作者模块导入、运行、拟合或打分。[作者入口](https://github.com/JHU-LCAP/DECAF/blob/main/README.md) · [固定源码索引](SOURCE_INDEX.csv) · [上游许可](upstream/LICENSE)

| 环节 | 固定提交的可追溯证据 |
| --- | --- |
| 命令与模型 ID | `train_models_3.sh:1` 调用 `train_model.py --model two_branch_recursive_context_3.0sec_dynamic --loss_function pearson_l1`。README 的通用 `--config` 示例在该入口的 parser 里没有对应参数。注册器用模型名选 `TwoBranchRecursiveConfig`，动态门、3.0 s 历史，继承 3.5 s 预测，递归与 contextual 标志开启，GRU。factory 用 `config.model_params()` 构造 `TwoBranchModelRecursive`。 |
| 普通 DECAF 与 Oracle | `paper_figures/generate_fig2.py:18–19` 及 `branch_analysis.py:14–15` 为两种标签分别读普通 `Old_…20250910_104156/…test_results.csv` 与 `…20250910_170516/…test_results_teacher_forcing…/test_results.csv`。作者运行时间笔记引用后者时间戳。路径证据不绑定共同 checkpoint 或确切历史源码，普通 run 名含 `Old_`。 |
| 指标 | 默认测试逐条保存窗口 r 及两分支 r；论文图表脚本将 CSV 先按受试者等权窗口平均，再比较受试者结果。公开 CSV 路径字符串和汇总程序可查，未读取实际 CSV 或复算论文数值。 |

证据定位见[训练脚本](upstream/train_models_3.sh) · [注册器](upstream/src/models/registry.py) · [模型工厂](upstream/src/models/model_factory.py) · [论文图表脚本](upstream/paper_figures/generate_fig2.py) · [复现说明](upstream/paper_figures/decaf_reproducibility_notes.md)。

## loader、500 ms 与样本支持

递归清单生成器对已经切好的 EEG/包络文件检查样本数相同，然后生成同文件局部半开区间。64 Hz 下，过去 EEG 与过去真包络使用 `[k,k+224)`，当前 EEG 与当前真包络使用 `[k+224,k+448)`。DataLoader 对两种模态执行这些同索引切片。模型取过去包络的末 192 点 `[k+32,k+224)`。在预测历史路径里，过去 HappyQuoka 全窗注意力、窗内标准化允许该尾部使用整个过去 224 点 EEG。当前端直接预测与评分的数组长度均为 224。清单生成器的 80% 训练重叠按整个 448 点样本计算，步长为 89 点（1.390625 s）；VAL/test 步长为 448 点（7 s）。同一行过去与当前不交叠，训练相邻行的整体区间交叠 359 点，当前段彼此交叠 135 点。VAL/test 因每行预留 224 点过去段，当前评分段之间相隔 224 点。训练 shuffle 并丢弃最后不完整 batch，VAL/test 不 shuffle。[配置](upstream/configs/two_branch_recursive_config.py) · [清单生成](upstream/create_filelist_recursive.py) · [loader](upstream/src/data/eeg_dataloader.py) · [DataLoader 装配](upstream/src/data/data_utils.py)

论文 §3.1 写明 500 ms EEG 延时与 3 s 评分窗口，而上述主链没有 EEG 专属的 +32/-32 样本操作。配置中的 `filter_length_ms=500` 没有进入已查递归路径。可视化脚本的 `CROP_END=3*64` 同时截取预测和真值的前 192 点，不是 EEG 相对包络位移，也不是普通图表 CSV 的可核验生成证据。过去段从 224 点截末 192 点同样不能证明 EEG 延时。由于文件已经预处理、磁盘资产未取，真实上游时间原点、是否有预移、方向、边界裁剪和与旧 run 的关系都尚未确定。若另设 `EEG[y+32]` 对齐包络 `y`，两段 EEG 的原始坐标都要加 32，且当前预测必须等到整当前输入块可得；这里只是延时坐标的条件解释。[论文 Methods](https://arxiv.org/html/2602.19395v1#S3.SS1) · [基础配置](upstream/configs/baseline_config.py) · [可视化裁剪](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/visualize_model_predictions_eeg_context.py#L1337-L1348)

## Forecaster、融合与训练/验证/推理

- **实际 forecaster。** 1→128、核 7 的 Conv1d 和 LayerNorm 输入两层单向 GRU。每次调用用新零初态。GRU 路径计算 output LayerNorm 与四头 self-attention，但其结果无消费者。输出头实际从未精炼的 `h_n[-1]` 取值，经 128→128、ReLU、dropout、128→224 一次生成整段 prior。当前调用是直接 `self.envelope_branch(...)`。未调用其 `predict_autoregressive` helper，后者将历史与预测长度的占位拼接，再交给要求输入等于历史长度的 forward，正历史长度配置下静态可见长度契约冲突。独立文件里的 autoregressive forecaster 不是这条模型调用链。论文或补充表的名义参数包含声明的模块，不能当作当前输出活跃路径的参数量。[Forecaster](upstream/src/models/envelope_forecaster.py) · [递归模型](upstream/src/models/two_branch_model_recursive.py)
- **实际融合。** 输入通道顺序是 prior、当前 EEG。门用居中 padding 的 Conv1d 2→16→8→1，核 5/3/1、ReLU、sigmoid。以 prior 权重 `w` 表达，输出是 `w·prior+(1−w)·EEG`，与论文用 EEG 权重的写法互补。当前 HappyQuoka 注意力、窗内归一化和居中门都依赖当前窗口其他时刻，支持的是块内离线计算，无法据代码声明样本级因果。[融合和 forward](upstream/src/models/two_branch_model_recursive.py) · [HappyQuoka](upstream/src/models/happyquoka.py)
- **实际训练。** recursive trainer 的两段比例都为 1，注释里的 90%/40% 没进入控制值。train 仍走普通 forward，输入真过去包络作为上下文，当前 EEG 分支原 sigmoid 后再做窗内 min–max（分母加 1e−6）。loader 对过去和当前 EEG 各自按通道、沿时间标准化，两段真包络各自 min–max 到 0–1，再由模型取过去尾部；默认 envelope filter 为 none，真包络 min–max 实现没有常量窗 epsilon。预测 prior 有 sigmoid。公开命令的损失为 `(1−r)+0.2·L1`，与复现笔记相符，和论文式 (5) 的 `L1−0.2r` 权重不同。Adam 及训练默认配置见固定源码。[训练分支](upstream/src/train/trainer.py) · [损失](upstream/src/train/losses.py) · [递归配置](upstream/configs/two_branch_recursive_config.py)
- **实际验证和选点。** 每轮验证调用独立的 `forward_teacher_forcing`，仍给真过去包络。该路径对当前 EEG 原 sigmoid 输出再做一次 sigmoid，和普通 train/test 当前分支的 min–max 不同。训练器的 best 和 patience 由 VAL correlation 更新；作者补充表的 validation loss 文字与此处实现不同。允许 teacher-forcing 可以是协议选择，不能用 teacher VAL 证明正常预测历史性能。[训练器](upstream/src/train/trainer.py) · [补充表](upstream/paper_figures/decaf_reproducibility_tables.tex)
- **实际测试分支。** `train_model.py` 结尾会自动调用普通测试。普通默认测试用过去 EEG 经同一 EEG 分支重新解出包络，保留该分支原 sigmoid 输出后取尾 192 点，当前分支仍窗内 min–max。Oracle flag 用 GT teacher 分支。另一 `test_model_recursive.py` 入口导入名与 data_utils 定义不一致，且 trainer 的递归调用指向模型类里没有的 `forward_recursive`。这些静态断点和普通路径中无持久融合状态的事实约束本提交的可确认范围，不能推翻论文旧 run 或断言测试泄漏。[训练入口](upstream/train_model.py) · [普通测试](upstream/test_model.py) · [独立递归测试脚本](upstream/test_model_recursive.py)

## 与强锚点及 D19 的关系

若以后复用作者活跃 forecaster 和动态凸融合，最小接口是现成 FT+Ccap 连续预测提供当前观测与同记录近期预测历史，在原 TRAIN/VAL 内对齐历史和目标，以同一固定训练尺度处理预测与目标。要保留 D19 的单点未来权限，门只能使用当前及更早的观测，直接复制居中卷积可能跨一个原 Hann 相位并多读原始 EEG。使用离线块权限则应明确整个块的原始支持、可用时间和评分协议。使用冻结历史与训练/推理一致的预测输入也不会复刻作者的 teacher-forcing/Oracle、过去 HappyQuoka 重解或 3.5 s 协议。该适配只用于界定可比范围，**本轮未选它作为执行候选**。

唯一下一准备由协调固定为 D20。保留 D19 的 P 解码器、当前 `g(t)`、冻结强锚点、起点 1344、物理 MSE 和所有 split、掩码、选点、优化规则，只将 384 点预测上下文从 `[t−671,t−287)` 改为紧邻的 `[t−384,t)`，仍按 96 个四点箱。同一 seed 的新 P 上下文统计按其新输入仅用 TRAIN 重估。历史最后一个样本与当前仅差 1 点，而 D19 相差 288 点。对任一历史点 `u<t`，归一化 Hann 最后正贡献窗 `min(w−1, floor((u−1)/96))` 不超过当前点的对应值，故紧邻历史不扩大当前锚点的原始 EEG 末端；两者原始支持可以大量重叠。D20 检验紧邻预测上下文，无法识别纯额外过去 EEG、上一最终输出反馈或在线效能。现有 D19 P 和 D20 都没有独立预报分支、作者式动态凸融合或回授。D19 科学结果保持冻结。[D19 方案](../../../physical_stream/run01/PLAN.md) · [协调原始支持核验](../../../recent_context_review/run01/COORDINATOR_SUPPORT_CHECK.json)
