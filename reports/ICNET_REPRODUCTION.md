# ICNet reproduction

## Probe scope correction (2026-08-14)

The existing remote runner was extended with the same deterministic synthetic `speech` waveform used by the cached model probes. The current remote JSON now contains six inputs: `tone`, `regular_clicks`, `jitter_clicks`, `omission`, `phase_shift`, and `speech`. The speech relative distances to tone are bottleneck 0.5692 and units_1000 0.9459. Persistence remains `null` because the synthetic output is a one-frame population summary.

状态：`PASS`，证据位于 2203 专用目录的 `reports/m6a_public_002_stage_bc_smoke.json` 与 `reports/m6a_public_002_panns_icnet_retry.json`。

使用官方 `fotisdr/ICNet` v1.0 代码、配置和随附权重，在 legacy 专用环境运行官方示例路径并对 deterministic synthetic tone 做 smoke/probe；未读取 IC recordings、患者数据或 STN 数据，未训练或微调。

实测输出：输入 `[1, 26432, 1]`；`bottleneck` `[1, 1, 762, 64]`；`units_1000` `[1, 762, 1000]`；均 finite，CF 数组存在。一次运行约 1.78 s。

这里的 ICNet 是正常听力、麻醉沙鼠 inferior-colliculus population 的模型 surrogate，不是人脑 IC，也不支持脑区或跨物种功能等价结论。

## 统一六类时间 probe

在相同 synthetic 输入上覆盖 tone、speech、regular clicks、jitter clicks、omission 和 phase shift；其中 regular/jitter/omission/phase 是重点生理比较。speech 的 bottleneck/`units_1000` distance 为 0.5692/0.9459；所有输出 shape/finite 均通过；跨 probe 的 lag-1 persistence 因输出为单帧时间摘要而记为 `null`，不插值。

这只证明输入扰动可传播到模型输出，不提供人类 IC、脑区或科学优越性结论。完整轻量 JSON 在 2203 的 `reports/m6a_public_002_auditory_physio_probes.json`。
