# M6A-PUBLIC-001 G2 full audit 未完成证据

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 状态 | `PENDING_FULL_DATASET_AUDIT` |
| 首次 full audit | `FAIL_DOWNLOAD_INCOMPLETE` |
| 执行位置 | 2203 专用目录/环境 |
| 完整性策略 | `NON_HASH_AUDIT` |

## 下载状态纠正

下载器使用 `.partial-<id>` 临时名。早先只检查 `*.part`，曾把 active partial 误计为最终对象；该 377/377 表述已经撤回并作为失败历史保留。

改用官方 S3 inventory 的精确相对路径与字节数核对后，审计时点结果为：

- expected：377 files、14,173,350,514 bytes；
- final paths：374 files、8,306,097,426 bytes；
- missing final paths：SD011、SD019、SD022 三个 iEEG EDF；
- active `.partial-<id>`：3；
- unexpected paths：0；已完成 final path 的 byte mismatch：0。

后台下载未中断。完整 G2 必须等待三个临时 EDF 以预期文件名落盘，再重新执行 exact path/byte reconciliation 和 11/11 EDF header audit。

## 已完成的 metadata 证据

- `dataset_description.json`：DOI/version 与 CC0 声明符合任务书；README 更严格的非商业和禁止再识别限制继续生效；
- 10 participants、11 recordings；11 份 iEEG sidecar、channels 与 events 可读；
- PyBIDS `validate=True` 可建立布局，识别 10 subjects；
- 采样率为 512/1024 Hz，工频统一为 60 Hz；
- README 的 C-prefix channel 排除已转为机器审计，当前 metadata 中共有 727 个 C-prefix 行，排除后候选 good SEEG/ECOG 为 1,346 channel-recording entries；
- 标准 `electrodes.tsv` 与 `coordsystem.json` 数量均为 0；9 份 contact RAS CSV 没有解剖脑区标签，SD012 解剖数据按 README 缺失；
- 已完成的 EDF 中，分析候选 SEEG/ECOG 通道名均存在；3 个 recording 的 EDF 总通道数与 channels.tsv 相差 1–3 行，仅记录 warning，不据此删除神经通道。

## Target no-go

70-150 Hz high-gamma 候选带包含 120 Hz 工频二次谐波，因此 target 状态为 `REDESIGN_REQUIRED_BEFORE_G3`。在冻结明确的 60/120 Hz rejection 与 filter edge，或重设频带之前，`neural_extraction_allowed=false`。

## 运行证据

```text
48 passed, 18 subtests passed in 0.16s
All checks passed!
Success: no issues found in 17 source files
```

失败运行保存在：

- `reports/ds004703_full_audit_download_incomplete_v1.json`；
- `reports/ds004703_full_audit_download_incomplete_v2.json`；
- `reports/ds004703_neural_metadata_download_incomplete_v1.json`；
- `reports/ds004703_neural_recordings_download_incomplete_v1.csv`。

这些文件是失败/进行中证据，不是 G2 PASS。全程未生成、读取、比较或验证任何哈希或校验和。
