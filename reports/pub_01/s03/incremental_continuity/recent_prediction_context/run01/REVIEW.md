# D20 验收与独立复核

| 验收项 | 结论 | 范围 |
| --- | --- | --- |
| Technical validation | PASS_WITH_LIMITATIONS | 三组预算、选点、恢复、原锚点和逐人聚合通过。独立数值重放仅限原 VAL 固定点 |
| Acceptance validation | PASS_WITH_LIMITATION | lag1 相对冻结锚点和原 D19 P 获得同向的重复开发 VAL 增益。最强已有开发参考与确认性科研目标仍未超越 |
| Parent consistency | PASS | 既有 PUB-01-S03_SS07 的训练/验证增量问题与原数据边界保持一致 |

- 真实工程与正式执行的源码、D19 引用和输入印记在拟合前独立审阅。正式进程 PID 1278512、UID 1012、Linux startticks 787942986 正常退出。所有配对与 86,820 次更新达到预定计数，无重启。[协议](PROTOCOL.json) · [准入](EXECUTION_REVIEW.json) · [正式回执](COMPLETED.json)
- 独立保存审计核对同 seed 初始化和当前统计、60 个逐轮恢复文件的数量与命名、最早严格最优、49 人和 752 块聚合。实际加载末轮及选中轮的恢复状态与 last/selected 权重，核对相应 Adam 与 RNG。最大聚合误差 1.11×10⁻¹⁶，0 新网络前向。[保存审计](COORDINATOR_FIT_AUDIT_BATCH01.json)
- 原实现与独立 NumPy 的有界重放覆盖六个实际末轮或选中权重，各 1,536 个物理点，固定读出各重建 3,756 窗。输入和锚点最大绝对差为 0，预测差上限 2.086×10⁻⁷。独立重放未产生新科学评分、反传或更新。[功能重放](COORDINATOR_FUNCTIONAL_REPLAY.json)
- 协调独立读全文并目视曲线六面板，另对保存标量复算选中 r、近远分差、60 个同轮结果、逐人方向与末轮回落。该标量复算用 0.063829841 CPU 秒和 0.064012205 秒墙钟，单列于两项终验，无模型前向或新拟合。[科学审阅](COORDINATOR_SCIENTIFIC_REVIEW.json)
- 三个末轮仍低于锚点。固定模型的近远替换改变输入分布。紧邻历史与当前原始 EEG 支持重叠，VAL 重复使用且均为 TRAIN 已见受试者。结果支持该单因素在本开发协议有效，D18 最强均值仍更高。[完整结果](REPORT.md)
