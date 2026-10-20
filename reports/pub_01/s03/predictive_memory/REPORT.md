# PUB-01-S03_SS05 — Prediction-Error Multi-rate Associative Memory (PI-MRAM)

2026-10-19。**只完成结构原型、10 项本地测试与两项合成序列实验；公开 EEG 数据加载/训练/测试尚未执行。** 实际代码、机器结果见本目录 `SYNTHETIC_RESULTS.json`；不涉及患者或 STN。

## 核心目的

检验连续时序网络能否保存、选择性修改并检索远处历史，再判断历史预测误差是否为必要机制。设计将三采样点因果局部特征提取、多速率递归状态、历史预测残差、门控更新、关联矩阵 delta-rule 读写结合。门控状态与 delta-rule 都已有明确先例，属于**探索性组合**，不能凭形式称新算子。

供 EEG 用的 `WindowPredictiveDecoder` 输入 `[B,W,192,64]`，输出 `[B,W,192,1]`。3 秒窗口被编码为摘要 token；token 之间递归更新并将记忆反馈给当前逐采样预测。**跨窗口因果，但窗口内部使用完整3秒 EEG，是离线处理。**

## 合成实验 A：单一远距线索回忆

第 2–6 时刻输入 ±1，第 25–30 时刻查询，局部3点无原线索。1024 train、128 validation、128 test；单 seed 1024，Adam 0.007，160 optimizer steps，validation MSE选检查点。

| 结构 | Test 符号准确率 | Test MSE |
|---|---:|---:|
| 局部输入 | 48.44% | 1.00018 |
| 三速率递归状态 | 100% | 0.001873 |
| 状态＋预测误差 | 100% | 0.002325 |
| 关联记忆（无预测误差） | 100% | 0.000132 |
| 完整 PI-MRAM | 100% | 0.000184 |
| 标准 GRU | 100% | 0.000213 |

反转遥远历史线索而保持当前数据不变时，长期记忆模型的预测也发生相应反转；局部模型不受影响。**GRU 同样满分**，因此此实验只证明远程依赖的计算通路可用。

## 合成实验 B：干扰事件中的键值回忆

依次写入两个不同 key 的 ±1 值，再随机覆盖一个 key；相隔较长历史后，查询指定 key 的最近写入值。序列48步，query在39–46步，样本量与训练预算同上。

| 结构 | Test 准确率 | Test MSE |
|---|---:|---:|
| 局部输入 | 50.78% | 1.02265 |
| 三速率递归状态 | 71.88% | 0.73424 |
| 状态＋预测误差 | 70.31% | 0.72308 |
| 关联记忆（无预测误差） | **100%** | 0.001729 |
| 完整 PI-MRAM | **100%** | 0.001478 |
| GRU（160步） | 62.50% | 0.92307 |

额外诊断：GRU 若延长训练到 **600步** 也达到 **100%**（MSE 0.001568）。600步并非公平的160步比较，说明此任务不具备“只有我们的结构能够解决”的性质。更合理的初步线索是关联记忆可能提供学习效率优势，需要更多 seed、预算和任务复现。

**负面结论必须保留：** `assoc_no_pred` 在两项任务都达到和完整模型相同的准确率，其中第一项 MSE 甚至略低。现阶段**没有证据**认为预测误差门控是这次成绩的关键贡献。

## 工程验证、容量与限制

本地 PyTorch 2.10.0+cpu 完成 **10 项 PASS**：参数结构/有限梯度、流式分块与整段等价、无未来 EEG 泄漏、局部历史隔离、历史干预有效、不同录音重置有效、窗口级 GRU 流式处理等。合成用 PI-MRAM 5消融配置均实例化6,014参数，GRU 5,801参数；关闭分支后活跃参数/FLOPs 不一定等量。

64通道窗口级 EEG 模型实际实例化 **68,546** 参数，含同窗口前端的 GRU 对照 **70,449** 参数（+2.78%）。这仅是参数总数接近，不能代表计算时间、活跃模块、训练预算都等价。

实验为单seed的特定合成分布，不代表真实EEG、真实声学预测、长序列外推、神经机制或发表级算法创新。现有公开数据 `physical_sync=UNKNOWN` 限制不变。PI-MRAM结合现有 *selective state propagation*、*gated delta-rule memory*、*long-term memory* 思想，需要与原工作严肃对标，例如 [Mamba](https://arxiv.org/abs/2312.00752)、[Gated Delta Networks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)、[Titans](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)。

## 进一步的正确问题

优先验证 `assoc_no_pred` 相对普通递归状态是否在真实连续数据上获得稳定增益，以及 `full` 相对 `assoc_no_pred` 是否仍然有效。若第二点阴性，应该删去无效的预测误差组件，而非堆加更多机制。

**下一步仅可使用全新冻结的公开 EEG 内部协议进行 train/validation 探索。** SS03 已打开的 `test_seen`/`test_heldout` 不得用于SS05调参或结构选择；历史只在同一真实录音连续窗口间传递，跨录音/被试强制重置。新实验详情见 `NEXT_PUBLIC_EEG_PROTOCOL.md`。SS04/S04及 ART-AUDREP-v1 不自动启动。

### 本地可重跑的合成实验

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_pub_01_s03_ss05*' -v
PYTHONPATH=scripts python scripts/pub_01_s03_ss05_synthetic_probe.py --steps 160 --output /tmp/ss05_cue.json
PYTHONPATH=scripts python scripts/pub_01_s03_ss05_keyed_probe.py --steps 160 --output /tmp/ss05_keys.json
# 仅作非等预算的附加容量诊断
PYTHONPATH=scripts python scripts/pub_01_s03_ss05_keyed_probe.py --steps 600 --variants gru --output /tmp/ss05_gru600.json
```
