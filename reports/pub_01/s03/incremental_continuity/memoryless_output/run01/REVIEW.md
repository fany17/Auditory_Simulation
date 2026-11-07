# D23 独立验收与复核

| 验收项 | 状态 | 范围 |
| --- | --- | --- |
| Technical validation | PASS_WITH_LIMITATIONS | 真实历史扰动与原锚点工程、3×20 正式计量、独立保存和固定样本重放通过 |
| Acceptance validation | PASS_WITH_LIMITATION | 无历史对照取得持续开发增益，B 三 seed 最佳值更高，确认性目标尚未达到 |
| Parent consistency | PASS | 保持既有 SS07 及公开 TRAIN/VAL 边界 |

- 工程读取实际原记录，改变所检点之前的全部 384 点并保留当前 b，再重新建表。三 seed 所检输出完全一致。改动当前 b 则 q/context 都改变。共享初始化、原顺序、零头/非零头梯度及 84 个固定 TRAIN 锚点块核对通过。正式进程 PID 1337533、UID 1012、startticks 788318955 正常结束。[工程回执](ENGINEERING_COMPLETED.json) · [正式终态](COMPLETED.json)
- 独立保存审计核对三组 3,361 参数、两次初始化列缩放、实际复制统计、预算和全评分选点。恢复文件的数量和名称全部核对，实际加载末轮与选中恢复及参数，确认 Adam 和 RNG 状态。旧执行器的两个槽使用同一当前值，保存指标完全一致。[保存审计](COORDINATOR_FIT_AUDIT_BATCH01.json)
- 对固定原 VAL 首末记录的首尾 64 活动点重放三个末轮、三个选中权重。生产实现和独立 NumPy 各 1,536 个物理前向点、各重建 3,756 个冻结读出窗。双输入及锚点误差为 0，预测最大绝对误差 5.59×10⁻⁹。没有新增科学评分、反传或拟合。[功能重放](COORDINATOR_FUNCTIONAL_REPLAY.json)
- 独立科学汇总重核保存指标。S 的 60 个训练轮次均高于锚点，在 52/60 个同轮次高于 B；B 各自最优三组都更高，其优势来自纯活动块。两函数类的输入维数、历史和初始化变化限制机制解释。[科学汇总](COORDINATOR_SCIENTIFIC_REVIEW.json) · [完整结果](REPORT.md)
