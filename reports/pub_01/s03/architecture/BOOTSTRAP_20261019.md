# SS03 多时间尺度原型启动记录 — 2026-10-19

本文件只记录**结构代码启动**。真实 SparrKULee 的 SS03 数据加载、GPU训练、validation、held-out test 均 **NOT_RUN**；此处**不存在任何模型性能结果**。

## 版本

- SS02基线锁定在 `c7b23648b5a443ac9d4cba534493c5ae7d3a4c74`，其 `PASS_WITH_LIMITATION` 和局限不变。
- 新原型：`scripts/pub_01_s03_temporal_scales.py`；配置：`configs/pub_01_s03_ss03_multiscale_20261019.json`；可证伪假设与Codex执行步骤：`doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS03_ARCHITECTURE_HYPOTHESIS.md`。
- 四个固定变体 short/middle/long/multiscale，仅改变 dilation；输入维度 `[B,192,64]`，输出 `[B,192,1]`；均为离线非因果预测。模型没有声音候选输入。

## 当前可确认的 CPU smoke

在本地独立Python镜像（PyTorch `2.10.0+cpu`）使用与提交的结构模块相同的源码运行直接 forward/backward 合约测试，而非2203训练：

| variant | dilations | parameters | output | finite gradient |
|---|---|---:|---|---|
| short | (1,1,1) | 6001 | (2,192,1) | PASS |
| middle | (4,4,4) | 6001 | (2,192,1) | PASS |
| long | (16,16,16) | 6001 | (2,192,1) | PASS |
| multiscale | (1,4,16) | 6001 | (2,192,1) | PASS |

此外检查了9-tap temporal receptive fields：`9/33/129`采样点，且位于最长感受野以外的输入扰动不会改变被测中心时点的输出。所有判定均为**工程烟雾测试**；没有宣称将GitHub测试在目标服务器执行、没有统计显著性或新的神经模型性能。

## 待Codex执行的第一关（可继续）

1. 在独立Git worktree核对本分支代码（本地旧CNN/TRF未提交修改、stash不得动）。
2. 只读核对2203现有SS02可用公开fold4输入、readout `manifest_frozen`、96样本hop、64Hz EEG和原包络。
3. 建立同一个 EEG→包络真实窗口对；完成实际数据forward/backprop smoke与训练/验证pilot（不读test科研结果）。
4. 固定正式训练预算、seed与模型验证规则，再启动4种结构 × 3种子同预算实验。任何输入缺失和对齐限制均明确标注，不用人为拼补。
5. 单独验收结果后才可能进入SS04。**当前不启动SS04，不产生ART-AUDREP-v1，不读取STN/患者数据。**
