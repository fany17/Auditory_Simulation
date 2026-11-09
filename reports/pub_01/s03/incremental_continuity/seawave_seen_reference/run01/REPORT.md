# D27 Sea-Wave 已见被试结果（2026-11-09）

3 个共享 initial 和 147 个 personal 微调全部完成。三种子个人模型的开发集 Pearson r 均值为 **0.216585948**，高于历史 D22 B 的 **0.184580904**，绝对差 **+0.032005044**；这是已有 Sea-Wave 方法的个体适配结果，不是本项目新框架已达到 SOTA 的证据。共享 initial 的均值为 **0.162819585**，仍低于原 Transformer FT+Ccap。

本次按用户“看一眼情况，然后 commit and push”整理实际结果。持续研究 goal 保持此前暂停状态；没有新训练、改预算、重开测试或选择下一方法。D27 状态为 **训练与保存证据核验完成，完整数值复算待执行**，不标记科研目标完成。

## 实测结果

使用冻结 PROTOCOL_V04、原 TRAIN 统计、426 条 TRAIN 和 49 条 VAL；49 个 VAL 被试均见于 TRAIN。评分固定为每个种子的原 752 块，先按被试聚合，再作三种子均值。这是反复使用的开发 VAL，并非独立测试。

| seed | initial r | personal r | personal − initial r | initial MSE | personal MSE |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.163759739 | 0.217441511 | +0.053681772 | 0.244976986 | 0.243629950 |
| 2026 | 0.162963160 | 0.217616964 | +0.054653803 | 0.245463225 | 0.234893029 |
| 31415 | 0.161735856 | 0.214699371 | +0.052963515 | 0.246416164 | 0.235913018 |
| 均值 | **0.162819585** | **0.216585948** | **+0.053766363** | **0.245618792** | **0.238145332** |

原 Transformer FT+Ccap 的同支持均值为 r=0.182339940、MSE=0.242378889。personal 三个种子的 r 均高于它，但 seed 1024 的 MSE 反而高 0.001353490。相对各自 initial，三个种子分别有 47/49、48/49、48/49 人的 r 提高，36/49、38/49、40/49 人的 MSE 降低。147 个个人选点有 4 个回退到 initial（epoch −1）。不能概括为所有人、所有指标均改善。

详见 [逐种子指标](report01/SEED_SCORES.csv)、[逐人指标](report01/PEOPLE.csv)、[个人配对差](report01/PAIRS.csv)、[逐块指标](report01/BLOCKS.csv)、[分层结果](report01/STRATA.csv)和[完整汇总](report01/RESULT_SUMMARY.json)。

## 已核验的范围

2203 上独立运行冻结的保存证据审计，354,032 项断言通过：覆盖 150 个实际拟合与退出凭据、4512 行块评分、294 行逐人指标、147 行个人配对、全部 3205 行训练/初始点记录、选点与成本，以及 150 个选中权重的字节 SHA。审计未导入训练或报告实现；没有解码权重、执行模型前向或读取信号载荷。

[独立审计原件](COORDINATOR_SAVED_EVIDENCE_AUDIT.json)的 SHA256 为 `69d8256c14dc8595d6f6a4ede05d2fefcc6cad991e0f3958618a3420372630cb`，`saved_evidence_audit_passed=true`，但 `raw_to_prediction_replay_performed=false`、`scientific_goal_achieved=false`。预备的[全模型 CPU 数值复算](COORDINATOR_FULL_REPLAY_PROTOCOL.json)尚未执行，不能把保存分数聚合通过写成原始 EEG→预测已独立复算通过。

[完整训练终态](COMPLETED.json)、[新调度终态](resource_concurrency_v01/TERMINAL.json)、[旧队列退休凭据](resource_concurrency_v01/OLD_QUEUE_RETIRED.json)及[报告完成凭据](resource_concurrency_v01/REPORT_WAITER_COMPLETED.json)均已取得。2026-11-09 检查时原登记训练、调度和报告进程均已退出，未重启任何作业。服务器权重和信号仍只留在 2203，本地仅镜像轻量报告。

## 训练成本与可比性

| 阶段 | 拟合数 | 实际训练 epoch 合计 | optimizer updates | CPU 秒合计 | 各拟合 elapsed 秒之和 |
|---|---:|---:|---:|---:|---:|
| initial | 3 | 603 | 14,230,800 | 201,164.748 | 204,459.611 |
| personal | 147 | 2,455 | 18,290,061 | 278,796.853 | 282,897.005 |

总计 32,520,861 次优化更新。elapsed 列是各拟合用时相加，并非并行训练的日历时长。完整重复样本呈现、选中点之前的成本、验证次数、共享 initial 复用及工程成本见 [FIT_COSTS](report01/FIT_COSTS.json)与独立审计。

网络为 150,433 参数的 Sea-Wave Small 适配，感受野 249 点（±124）；共享 initial 未做 population global FT，不能称为作者完整 S2。个人微调使用个人 TRAIN 数据，并按各自 VAL 选点。个人/共享参数、时间上下文、预算和选点方式不同，因此同评分支持不等于严格公平的同预算架构比较。当前结果支持“个体适配提高本开发任务的 r”，尚不能识别提升是否来自网络、更多训练或个人适配本身。

## 三张实际图的核对

已逐张查看自动报告生成的 PNG，并与保存指标核对：

- [训练与选点](report01/training_selection.png)：三个共享 initial 均完成 201 轮，VAL 轨迹明显波动；个人轨迹普遍上升，也保留下降后回退的个体。选中点不等于末轮。
- [同支持对照](report01/val_same_support.png)：个人 Sea-Wave 的 r 在三个种子均高于共享 Transformer；MSE 的种子差异较大。图中共享与个人方法必须结合上述训练条件阅读，不能当等预算排名。
- [个人适配](report01/personal_adaptation.png)：r 的改善分布于五个故事；MSE 的方向具有故事差异，audiobook_13_2 的均值恶化。多数 r 改善不代表幅度误差全面改善，也不证明跨被试泛化。

本次没有查看新测试成绩。训练完成与开发集提升均有实际证据；独立数值复算、可比的个体训练强对照及独立泛化证据仍未完成，不据此宣称新方法、论文同条件复现或 SOTA。
