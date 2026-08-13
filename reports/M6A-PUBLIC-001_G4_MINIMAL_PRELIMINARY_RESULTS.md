# M6A-PUBLIC-001 G4 minimal preliminary results

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 状态 | `G4_MINIMAL_SD012_SES02_PRELIMINARY_COMPLETE_AWAITING_COORDINATOR_REVIEW` |
| 范围 | `sub-SD012_ses-02_task-PassiveListen`，单被试、单 recording |
| split | 24 train / 8 validation / 8 test passages |
| 神经目标 | 36 个 eligible electrodes，六子带 high-gamma target |
| 实际运行时间 | 161.36149486806244 s |
| 核心机器报告 | `reports/g4_preliminary_report_20260813.json` |
| 远端输出 | `/home/fanyu/auditory_simulation_m6a/outputs/g4_minimal_sd012_ses02_preliminary_20260813` |

## 初期 held-out test 结果

下表的“最佳 lag”仅用于描述本次 test 结果，不能作为正式的 test-driven 模型选择。

| 表征 | 描述性最佳 lag | 中位 Pearson r | IQR | 中位 R2 |
|---|---:|---:|---:|---:|
| Amplitude envelope | 0.05 s | 0.0157 | 0.0057–0.0206 | -0.0427 |
| Log-mel PCA20 | 0.50 s | 0.0176 | 0.0068–0.0295 | -0.0493 |
| 最佳 wav2vec2：Transformer 09 | 0.20 s | 0.0387 | 0.0202–0.0553 | -0.1071 |

Transformer 12 的描述性中位 Pearson r 也较高（0.0377，lag 0.25 s），但本轮不能据此声称稳定层优势。所有 15 个 feature variant 的描述性最佳 lag 对应中位 R2 均为负，说明当前绝对预测误差表现较弱。

## 20-null 机械检查

- wav2vec2 family：observed max Pearson r=0.0916；20 个预声明 max-null 中 3 个大于或等于 observed，机械性 `p=0.1905`；
- acoustic family：observed max Pearson r=0.0615；20 个预声明 max-null 中 10 个大于或等于 observed，机械性 `p=0.5238`；
- 20 次置换只用于验证 stimulus-derangement 和 max-statistic 机械流程，不提供稳定显著性结论。

## 工程门禁与失败保留

40/40 passages、71,253 个 common-support frames、15/15 feature variants 完成；5,148 个 wav2vec2 family cells 与 792 个 acoustic cells 均可估计。输入可读和范围、tensor shape/finite、split/stimulus 无泄漏三项必要门禁通过，最终 active partial 为 0。

首次运行因把必然被 mask 的卷积无效边缘纳入负 power 容差而 fail closed；失败 partial 保留在远端输出根的 `interrupted/`。修正仅将数值检查限定到完整 FIR support，随后完成运行。

## 科学边界

这是单被试、单 session、单 recording 的 preliminary encoding 结果，不是协调接受的正式 G4 科学结论。不能声称稳定 layer-wise brain alignment、显著优于 acoustic baseline、脑区对应、subject-held-out、speaker-held-out、跨语言泛化或临床/STN 适用性。`region_summary=NOT_ESTIMABLE`。

高维 tensor 约 3.70 GB，仅保留在 2203；本地只同步轻量核心 JSON 与本报告。当前尚未建立 M6A→M6B exchange candidate，也未运行 consumer cross-test。
