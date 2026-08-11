# M6A-PUBLIC-001 G2 manifest/split checkpoint

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| Manifest gate | `PASS_WITH_CATALAN_EXCLUSION_AND_SPLIT_REDESIGN` |
| Full-data gate | `PENDING_FULL_DATASET_AUDIT` |
| Dataset | `ds004703 v1.1.0`，仅在 2203 |
| Primary split | English passage，stimulus+block held-out |
| Catalan | 显式 `ca`，因 audio provenance 冲突排除于 primary baseline |

## 完成内容

- 从 11 份 phoneme-level events 构建 438 行轻量 segment manifest，覆盖 11 recordings、58 stimuli；
- 识别 319 个 English passage segments 与 41 个 Catalan passage segments；
- English 的 319 行全部通过 event-template offset、音频可读性与 timeline margin 门禁；
- 生成固定 block split：02/03/04/05=train，06=validation，01=test；
- 生成 language、speaker、stimulus、block、recording 与时间窗口审计；
- 生成原 `stimulus+recording` connected-component no-go 证据。

## 运行证据

- 测试：`40 passed in 0.12s`；
- Ruff：`All checks passed!`；
- mypy：`Success: no issues found in 14 source files`；
- 真实 split guard：319 rows，`PASS`，issues 为空；
- stimulus split conflicts：0；block split conflicts：0；2 s temporal cross-split violations：0；
- language `en` 覆盖 train/validation/test；
- 原联合分组：59 nodes、1 component，不能形成三个互斥 split；
- speaker advisory conflicts：6 个，故不声称 speaker-generalization。

## Catalan provenance 失败边界

README/sidecar 写为 6 blocks × 7 passages，但每个 block 目录实际包含 8 个 English passage 和 1 个 Catalan-labeled 文件。六个 block Catalan 文件均为 3.443 s，并与 `pleasePressSpace_normed.wav` 波形相关 1.0、RMS difference 0.0；events template 与 `catalan_v2` 文件约为 46 s。由于 exact played-waveform provenance 尚未闭环，Catalan 当前只进入语言/刺激 manifest，不进入 primary baseline，也不支持跨语言结论。

上述是波形/时间轴分析，不是哈希、校验和或密码学完整性验证。

## 下载状态与阻塞

最近非哈希盘点为 370/377 个对象、8,037,660,562 bytes 完整匹配；剩余 3 个 EDF 与 4 个 CT NIfTI 仍在后台下载。G2 尚不能标记 PASS，完整对象盘点、EDF 抽样读取和全数据 audit 必须等待下载完成。

## 科研表述

可声称：English primary split 的工程门禁与 leakage guard 已建立；原 recording-disjoint 方案被真实 connected-component 证据否定并完成可审计重冻结。

不可声称：G2 全部完成、Catalan exact playback 已确认、speaker/跨语言泛化、神经 encoding baseline 已运行、任何模型层对应脑区、M6A artifact 已冻结。
