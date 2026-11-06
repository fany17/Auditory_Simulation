# D21 独立复核与验收

| 验收项 | 状态 | 范围 |
| --- | --- | --- |
| Technical validation | PASS_WITH_LIMITATIONS | 固定输入、18 个精确解、原评分端点、完整预算与有界独立重放通过 |
| Acceptance validation | PASS_WITH_LIMITATION | 完整交付冻结输出端 A/F 对照。F 超过锚点与 A，未追平 D20，属于开发证据 |
| Parent consistency | PASS | 保持既有 SS07 与公开 TRAIN/VAL 边界，未重开外部和患者侧 |

- 正式审阅绑定实际源、协议与工程回执。正式 PID 1304200、UID 1012、startticks 788106508 正常终止。冻结锚点先在每 seed 完整 752 块桥接，工程成本与正式成本分列。[正式准入](EXECUTION_REVIEW.json) · [正式终态](COMPLETED.json)
- 协调独立从三个已保存充分统计和真实系数核验 18 个正规方程，最大绝对误差 2.01×10⁻¹⁸。对 15,792 个保存 VAL 块重聚合，最大差 1.11×10⁻¹⁶，逐人、选点和预算匹配。[独立终验](COORDINATOR_TERMINAL_AUDIT.json)
- 固定原 VAL 首末记录各首尾 64 活动点，独立显式分箱与生产实现各重放 4,608 个候选线性输出。分箱差为 0，生产与手写输出最大差 1.11×10⁻¹⁶，名义物理核与真实 FP32 路径最大差 6.73×10⁻⁹。固定读出另重建 3,756 窗，无新神经前向或拟合。核验范围是保存 Gram 和固定样本，未全量重算 TRAIN Gram、所有原始 VAL 预测或完整记录 Pearson r。[独立终验](COORDINATOR_TERMINAL_AUDIT.json)
- 协调独立核读全文、目视两张图，并按每人总块数从保存块计算前缀与纯活动贡献。A 的 r 差集中于前缀块，F 与 D20 在纯活动块仍有增益。D20 的纯活动贡献约占其主指标增益的 90%。这一复核约 0.056 秒墙钟、0.056 CPU 秒，0 新模型计算。[独立科学汇总](COORDINATOR_SCIENTIFIC_REVIEW.json)
- 选中 F 与 D20 的开发 r 提高和活动 MSE 升高同时存在。两者的差距尚不足以定位非线性、当前 latent 或学习过程各自的贡献。SS07 仍为 ACTIVE。[分块与远端数组索引](ARRAYS_AND_PREFIX.json) · [完整结果](REPORT.md)
