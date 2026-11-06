# D24 独立核验与验收

## 训练和来源

- 九套折上游的独立复核重算了三折的 TRAIN 通道统计，并按原正式 cuDNN 精度用固定真实 EEG 点重放保存的 FT+Ccap+Hann 输出。V03 共 4,068 窗、78 批、5,760 个物理点，最大差 2.22×10⁻¹⁶。三个 seed 的九套真实生成链均通过。V01 的 CPU 与正式 GPU cuDNN TF32 数值路径不一致，后续 GPU 开关对照定位了差异。V02 的三组复核与调度中断也分别留档，V03 完成后已退出，GPU 空闲才启动下游。[正式恢复总账](UPSTREAM_RECOVERY_TERMINAL.json) · [V03 独立回执](coordinator_upstream_audit_v03/COMPLETED.json) · [精度修订](COORDINATOR_REPLAY_PRECISION_AMENDMENT_V02.json)
- 正式下游准入将九套输出的 54 个载荷和 74 个来源绑定到通过的独立重放。每 seed 正式前重新核实实际输入的六个预测文件和冻结历史来源。原 VAL 锚点评分与历史保存值相符。正式完成 12 个拟合，三 seed 合计 347,280 次更新和 5,688,842,400 个物理反传点。[正式门](DOWNSTREAM_FORMAL_GATE_V06.json) · [下游完成](DOWNSTREAM_TERMINAL.json)
- 下游保存审计独立重算了全 426 条记录两种 lane 的训练归一化，检查三 seed 的 12 臂完整轮次、真实选点或锚点回退、每臂 752 块和 49 人的汇总、保存权重与 Adam 状态。计 145,656,324 个 TRAIN anchor 点和 13,653,221,760 个按箱重复次数表示的上下文值。0 新神经网络前向、0 拟合和 0 标签数组读取。[下游保存审计](COORDINATOR_DOWNSTREAM_AUDIT_V02_COMPLETED.json)
- 最终功能重放覆盖全部 12 臂的末轮及实际存在的选中权重，组成 42 个权重与历史条件组合。独立 NumPy 和生产路径各重放 10,752 个物理点与 3,756 个既有冻结读出窗，输入与锚点差均为 0，预测最大差 9.69×10⁻⁸。原 VAL 加载器打开 6 条标签数组，重放没有新科学评分、raw EEG 编码或拟合。回退臂只有实际的末轮权重用于重放。[功能重放](COORDINATOR_DOWNSTREAM_FUNCTIONAL_REPLAY.json)

- 独立科学复算逐项重算全部三 seed 的选点、末轮、逐人方向及预设配对，另核对仅在 20 个已训练轮次内求得的最佳值和完整成本。该步没有新神经网络前向、预测或标签数组读取或优化更新。[科学复算](COORDINATOR_SCIENTIFIC_REVIEW.json) · [已训练最佳与成本](COORDINATOR_TRAINED_AND_COST_CHECK.json)

## 判定

- **Technical PASS_WITH_LIMITATIONS**。全矩阵、实际计算和关键输入来源通过核验。开发集仍为反复使用的 VAL，涉及旧精度核验与调度修订，原始记录保留。
- **Acceptance PASS_WITH_LIMITATION**。完成预设端到端来源对照。历史 B 在 OOF lane 三 seed 的 60 个已训练轮次全部低于共同锚点。三个选中值均为锚点回退，最佳已训练均值 0.176480208，末轮 0.169019619。S_OOF 的选中小幅相关改善伴随活动区 MSE 升高。四臂对照保留全部结果。
- **Parent consistency PASS**。结果限于 PUB-01 的公开 SparrKULee 开发数据。原测试与外部确认未重开。SS07 持续科研目标仍为 ACTIVE。
