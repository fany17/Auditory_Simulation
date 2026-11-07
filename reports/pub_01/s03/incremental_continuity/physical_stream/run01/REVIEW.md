# D19 验收与独立复核

| 验收项 | 结论 | 范围 |
| --- | --- | --- |
| Technical validation | PASS_WITH_LIMITATIONS | 冻结工程与正式运行通过，实际计数吻合，独立复核覆盖保存证据及有限非零输出重放 |
| Acceptance validation | PASS_WITH_LIMITATION | 一个预先固定的候选已完整检验，历史收益门未通过，确认性科学目标未达成 |
| Parent consistency | PASS | 属于现有 PUB-01-S03_SS07 开发诊断。仅用既有公开 TRAIN/VAL，未扩大或改变父任务数据边界 |

## 冻结身份和实际验证

- 协议 133b60eeb6e937def9f25bdb8303ab80d9a2cf00d0f86ea3f6768c0f56a7b078 绑定三份源及元数据。有限工程回执 87225694f4a3716a52e8ae248c8341eb887d6ad3dfbd0989144aa6d34679d578 通过后才有独立正式准入。真实工程复现全部锚点块、正 Hann 贡献、分箱与跨记录坐标，检查零头、实际梯度、尾批等权及切批。[正式审阅](EXECUTION_REVIEW.json) · [工程回执](ENGINEERING_COMPLETED.json)
- 正式进程 PID 1237393，UID 1012，Linux startticks 787679082，完整九组共 180 轮恢复文件和 9 个末轮权重。真实首轮恢复状态确认有 Adam、CPU/CUDA 随机状态及原截止时间。没有执行恢复或重启。运行身份、成本和科学结果可由正式回执核对。[正式标量回执](SUMMARY.json)
- 协调三批独立检查覆盖九组已保存的选点、逐人聚合、参数、Adam 与 RNG。所选均为 −1，所有九组已训练最好轮均低于同 seed 原锚点，H 已训练最好在各 seed 低于 P。固定 TRAIN MSE 改善和非零残差、梯度支持通路在实际训练。[批次 01](COORDINATOR_FIT_AUDIT_BATCH01.json) · [批次 02](COORDINATOR_FIT_AUDIT_BATCH02.json) · [批次 03](COORDINATOR_FIT_AUDIT_BATCH03.json)
- 独立 NumPy/GELU 重放覆盖每 seed 的 VAL 首末两记录，各记录取活动区首末各 64 点，对九组实际末轮检查点每条 256 点。实际原实现与独立实现各 2,304 物理前向点，归一化输入与锚点最大差均为 0，最终预测最大绝对差为 6.184×10⁻⁷。独立和原实现各重建 3,756 个固定读出窗；未重新运行原 EEG encoder、评分或选择模型。[独立功能重放](COORDINATOR_FUNCTIONAL_REPLAY.json)

## 科学边界

独立重放验证有限输入点的实现一致性。正式历史干预只有锚点回退，九组的已训练模型没有 selected VAL near/older 比较。VAL 反复使用且与 TRAIN 共享 49 人，已训练最佳只作描述性证据。模型各臂的有效输入不同，也无法将 D19 与此前结果之差归因于单个组件。该固定候选封存为阴性，SS07 总体研究目标保持 ACTIVE。[完整中文结果](REPORT.md)
