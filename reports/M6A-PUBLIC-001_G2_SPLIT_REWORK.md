# M6A-PUBLIC-001 G2 primary split 返工报告

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 审核输入 | `PRIMARY_SPLIT=REWORK` |
| 处置状态 | `SPLIT_REWORK_COMPLETED` |
| Split 行数 | 319 |
| 当前门禁 | `PRELIMINARY_NOT_BASELINE_FINAL` |
| G2 总状态 | `PENDING_FULL_DATASET_AUDIT` |

## 被否决方案与失败保留

旧 block 分配得到 train/validation/test=208/32/79（65.2%/10.0%/24.8%），明显偏离任务书 70/15/15，已被协调审核否决。失败证据没有覆盖：

- `reports/ds004703_primary_split_rejected_v1.csv`；
- `reports/ds004703_manifest_summary_rejected_v1.json`；
- `reports/ds004703_split_guard_rejected_v1.json`；
- 2203 对应备份目录：`metadata/rejected_split_v1/`。

## 确定性比例优化

优化器只读取 analysis-eligible segment 的 block 计数：01=79、02=72、03=48、04=48、05=40、06=32。它枚举完整 block assignment，以 segment counts 距 70/15/15 的整数平方误差最小为主目标，再以绝对误差、held-out block 数和 validation/test/train block ID 字典序作固定 tie-break。禁止输入神经信号、模型性能或结果指标。

最终分配：

- train：block 01/02/05/06，共 223（69.9%）；
- validation：block 03，共 48（15.0%）；
- test：block 04，共 48（15.0%）。

## 泄漏与范围

- stimulus 与 block 均不跨 split；
- recording 允许跨 split，新 manifest 中 6 个 recording 跨 split，但 preliminary 2 s 时间邻域无冲突；
- speaker 仅 advisory，5 个 speaker 跨 split，不支持 speaker-held-out；
- Catalan 显式标注并因音频 provenance 冲突排除，不能声称 cross-language；
- subject 跨 split，因此当前只支持 within-subject unseen-stimulus/block generalization，不支持 subject-held-out。

2 s 仅为 `preliminary_minimum_embargo`。正式窗口化前必须计算：

`final_embargo = max(2 s, maximum encoding lag, filter/padding edge, audio model receptive field/context overlap)`

随后必须重新运行 guard；在此之前 split 不是 baseline-final。

## 真实运行证据

2203 专用环境：

```text
44 passed, 18 subtests passed in 0.19s
All checks passed!
Success: no issues found in 14 source files
```

319 行 split guard：`PASS`，issues 为空，`baseline_final=false`。

后续首次 full audit 发现下载状态盘点缺陷：下载器临时文件名为 `.partial-<id>`，原先只检查 `*.part`，因此曾把 active partial 误计为最终对象。该 377/377 表述已撤回；G2 改为按官方 inventory 的精确相对路径与字节数 fail closed，在所有 final path 到齐前保持 `PENDING_FULL_DATASET_AUDIT`。这项失败作为审计证据保留。

本轮只使用文件名、字节数、时间戳、数量、schema 与抽样可读性，不进行任何哈希或校验和验证。
