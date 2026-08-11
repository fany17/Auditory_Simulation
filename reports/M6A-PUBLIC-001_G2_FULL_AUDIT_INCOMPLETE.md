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

官方 inventory 由 `scripts/public_s3_inventory.py` 通过 OpenNeuro public S3 `ListObjectsV2` 取得，source 为 `https://s3.amazonaws.com/openneuro.org`，取得时间为 `2026-08-11T11:14:08.907738+00:00`；原始轻量记录保存为 `reports/ds004703_s3_inventory_summary.json` 与 `reports/ds004703_s3_inventory.csv`。

原单连接 downloader 后续经保存的 161 s 同窗观测为 0.09938 MiB/s；补强门禁后的 8×16 MiB Range benchmark 为 0.23246 MiB/s、0 失败。原进程族已确认停止，恰好 3 个旧 partial 移入 `/home/fanyu/auditory_simulation_m6a/log/interrupted_downloads/20260811T133028Z/`，full range 仅对 SD011/SD019/SD022 active。完整 G2 仍必须等待三个 EDF 以预期文件名落盘，再重新执行 exact path/byte reconciliation 和 11/11 EDF header audit；切换本身不是 G2 PASS。

## 已完成的 metadata 证据

- `dataset_description.json`：DOI/version 与 CC0 声明符合任务书；README 更严格的非商业和禁止再识别限制继续生效；
- 10 participants、11 recordings；11 份 iEEG sidecar、channels 与 events 可读；
- PyBIDS `validate=True` 可建立布局，识别 10 subjects；
- 采样率为 512/1024 Hz，工频统一为 60 Hz；
- 11 个 sidecar 的 `iEEGReference` 已逐 recording 机器记录，唯一值均为 `scalp electrode, not included with data`，一致性 gate 为 PASS；主分析 reference 因此限定为 `AS_RECORDED_SCALP_REFERENCE`；
- README 的 C-prefix channel 排除已转为机器审计，当前 metadata 中共有 727 个 C-prefix 行，排除后候选 good SEEG/ECOG 为 1,346 channel-recording entries；
- 标准 `electrodes.tsv` 与 `coordsystem.json` 数量均为 0；9 份 contact RAS CSV 没有解剖脑区标签，SD012 解剖数据按 README 缺失；
- 已完成的 EDF 中，分析候选 SEEG/ECOG 通道名均存在；3 个 recording 的 EDF 总通道数与 channels.tsv 相差 1–3 行，仅记录 warning，不据此删除神经通道。

逐名称差异已写入最新 recording report：SD010 的 TSV-only 为 `Trigger Event`、`Patient Event`、`STI 014`（均 MISC/bad）；SD018 与 SD021 的 TSV-only 为 `STI 014`；SD012 两次 recording、SD013、SD015、SD017 各为 TSV-only `STI 014` 与 EDF-only `Pleth`。这些差异均不属于 analysis-eligible SEEG/ECOG；8 个已完成 EDF 的 events 最大 offset 均未越 EDF 时间轴。其余三个 recording 仍因 EDF 下载未完成而 fail closed。

## Target method candidate gate

旧 70-150 Hz high-gamma 候选带包含 120 Hz 工频二次谐波，已被替换为协调接受的 frozen method：主目标为排除 110-130 Hz 的六个等宽 10 Hz 子带；旧宽带方法仅为 sensitivity。reference 保持 as-recorded，不猜缺失 scalp reference、不按 contact 名构造 bipolar。当前 `neural_extraction_allowed=false`；方法冻结不构成真实神经 target、G3 启动或 G2 PASS。

## 运行证据

```text
53 passed, 22 subtests passed in 0.17s
All checks passed!
Success: no issues found in 19 source files
```

上述是首次 incomplete audit checkpoint。method candidate checkpoint 另为：

```text
66 passed, 76 subtests passed in 1.36s
All checks passed!
Success: no issues found in 22 source files
neural target method gate: PASS
main config gate: PASS
formal src direct convolution scan: PASS
```

method freeze + Range switch 最终提交前 checkpoint 为：

```text
83 passed, 93 subtests passed in 1.41s
All checks passed!
Success: no issues found in 26 source files
neural target method gate: PASS
main config gate: PASS
formal src direct convolution scan: PASS
```

失败运行保存在：

- `reports/ds004703_full_audit_download_incomplete_v1.json`；
- `reports/ds004703_full_audit_download_incomplete_v2.json`；
- `reports/ds004703_neural_metadata_download_incomplete_v1.json`；
- `reports/ds004703_neural_recordings_download_incomplete_v1.csv`。
- `reports/ds004703_neural_metadata_download_incomplete_v3.json`；
- `reports/ds004703_neural_recordings_download_incomplete_v3.csv`；
- `reports/ds004703_neural_metadata_download_incomplete_v4.json`；
- `reports/ds004703_neural_recordings_download_incomplete_v4.csv`。

这些文件是失败/进行中证据，不是 G2 PASS。全程未生成、读取、比较或验证任何哈希或校验和。
