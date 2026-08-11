# M6A-PUBLIC-001 G2 resumable Range 下载切换报告

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 决定 | `SWITCHED_TO_RESUMABLE_MULTI_RANGE` |
| Full range | `ACTIVE_SD011_SD019_SD022_ONLY` |
| G2 | `PENDING_3_EDF_FULL_RANGE_AND_FINAL_AUDIT` |
| 并发 / chunk | 总并发 8 / 固定 16 MiB |
| staging | `/home/fanyu/auditory_simulation_m6a/logs/range_staging/ds004703_range_v1` |
| 完整性口径 | start/end、HTTP 206、Content-Range、预期字节数和最终官方总字节；非密码学 completeness |

## 1. Range smoke

在不停止原 downloader 的情况下，对 SD011 做 1 MiB smoke：

```text
HTTP status: 206
Content-Range: bytes 0-1048575/1833841408
expected/actual bytes: 1048576 / 1048576
elapsed: 2.1427 s
throughput: 0.4667 MiB/s
```

入口仅使用 `Content-Range` 响应字段。证据见 `reports/range_smoke_sd011_1mib_v1.json`。

## 2. Fail-closed 实现

`scripts/public_s3_range_download.py` 与 `scripts/move_interrupted_partials.py` 当前保证：

- inventory 拒绝 duplicate/empty/unsafe path、空 source、非正 size；source 必须与官方 ds004703 S3 HTTPS 路径逐字对应；
- chunk 固定 start/end；HTTP 必须为 206，Content-Range 的 start/end/object-total 和实际字节必须全部匹配；
- staging 严格位于 dataset 外；workers 强制在 1–8；
- chunk 续用只依据文件名所声明 range 与预期字节数，明确不证明内容身份；
- wrong-size chunk、已有 wrong-size final、stale assembly partial 只移动到 staging backup，不删除；
- 所有 move 前 resolve，并分别验证 source/destination 严格位于 dataset、staging 或人工指定 interrupted backup 根内；
- 只有全部 chunk exact-size 到齐后，才按 range 顺序装配到 dataset 内 `.partial-range-<run_id>`；仅在 assembled bytes 等于官方 object bytes 时原子改名；
- elapsed 小于等于 0 时速率返回 null，不做除零或虚假速度声明。

受控临时目录覆盖顺序多 chunk 装配、缺 chunk 阻塞、wrong-size final/partial 备份、wrong-size chunk 备份后重下、路径/移动边界和原子 final。切换前 2203 专用环境证据为：

```text
80 passed, 92 subtests passed in 1.40s
Ruff: PASS
mypy: PASS (25 source files)
neural target gate: PASS
main config gate: PASS
formal src direct-convolution scan: PASS
range safety subset after exact interrupted-move entry: 16 passed, 16 subtests
range subset Ruff: PASS
range subset mypy: PASS (4 source files)
```

全部新文件同步后的最终 2203 专用环境记录：

```text
83 passed, 93 subtests passed in 1.41s
Ruff: PASS
mypy: PASS (26 source files)
neural target gate: PASS
main config gate: PASS
formal src direct-convolution scan: PASS
```

## 3. 同窗吞吐与切换判定

原 downloader 的保存观测：

- t1 `2026-08-11T13:20:40Z`，3 partial 合计 613,416,960 bytes；
- t2 `2026-08-11T13:23:21Z`，3 partial 合计 630,194,176 bytes；
- 161 s 增加 16 MiB，即 0.09938 MiB/s、5.96 MiB/min。

Range benchmark：

| 运行 | 数据 | elapsed | aggregate | 相对原 downloader | 失败 |
|---|---:|---:|---:|---:|---:|
| v1 | 8 × 16 MiB | 625.89 s | 0.20451 MiB/s | 2.06× | 0 |
| v2（补强门禁后的当前代码） | 8 × 16 MiB | 550.62 s | 0.23246 MiB/s | 2.34× | 0 |

v2 为 13.95 MiB/min，明显高于原路线 5.96 MiB/min，因此满足切换条件。原始证据见 `reports/range_direct_observation_t1.tsv`、`reports/range_direct_observation_t2.tsv` 与两份 benchmark JSON。

## 4. 停止、备份与启动

原 downloader 已按 PGID 核对：会话 SIGINT 只终止外层 wrapper，内部 Python 一度成为 orphan；随后仅对同一确认 PGID 发送 SIGTERM，复查 `public_s3_download.py` 全部成员消失后才移动文件。

精确发现集合等于明确列出的 3 个 dataset partial。移动后 dataset partial=0，backup files=3，没有其他文件被移动：

| recording | interrupted bytes | backup |
|---|---:|---|
| SD011 | 225,443,840 | `/home/fanyu/auditory_simulation_m6a/log/interrupted_downloads/20260811T133028Z/sub-SD011/...partial-0ec96c78dfd843129a6229cf6432695f` |
| SD019 | 225,443,840 | `/home/fanyu/auditory_simulation_m6a/log/interrupted_downloads/20260811T133028Z/sub-SD019/...partial-442d997a49584d7fb938f1b247493187` |
| SD022 | 212,860,928 | `/home/fanyu/auditory_simulation_m6a/log/interrupted_downloads/20260811T133028Z/sub-SD022/...partial-5e0f3ded3d5c4392910cded7e6f556a2` |

完整相对路径和字节见 `reports/range_interrupted_move_20260811T133028Z.json`。Full range 只选择 SD011/SD019/SD022；三对象共 352 个固定 chunks，其中 benchmark 已提供 16 个可按字节数续用的 chunk。启动后首个 checkpoint 为 22 complete chunks、8 active chunk partials、dataset 内 0 个 assembly partial；正式结果日志将在完成时写入远端 `full_range_download_v1.json`。

## 5. Snapshot 恢复收口

- 远端 snapshot 根 README 已恢复为 3,998 bytes；
- 误平铺的 3 个轻量文件保存在 `/home/fanyu/auditory_simulation_m6a/log/interrupted/sync_layout_20260811_2058/`；
- snapshot 根旧 `m6a_public_001.json` 为 4,046 bytes，已移动到 `/home/fanyu/auditory_simulation_m6a/log/interrupted/sync_layout_20260811_2058/older_root_config_20260811_1944/m6a_public_001.json`；
- 唯一正式路径保持 `code_snapshot/configs/m6a_public_001.json`，当时为 6,930 bytes；数据目录未受 snapshot 恢复操作影响。

## 6. 声称边界

本报告只支持“Range/bytes 路线经安全门禁后切换且 full download 已启动”。chunk size match 属于非密码学 completeness，不证明内容身份。当前仍不能声称 3 个 EDF 下载完成、377/377、G2 PASS、神经 target 已生成、G3 已启动、baseline 已运行或整条 M6A/exchange artifact 已 frozen。
