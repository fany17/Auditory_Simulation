# CoNNear reproduction

## Probe scope correction (2026-08-14)

The existing remote runner was extended with the same deterministic synthetic `speech` waveform used by the cached model probes. The current remote JSON now contains six inputs: `tone`, `regular_clicks`, `jitter_clicks`, `omission`, `phase_shift`, and `speech`. The speech relative distances to tone are BM 1.0890, IHC 0.0738, ANF-H 0.9951, ANF-M 1.0801, and ANF-L 1.0441. These remain no-fit synthetic representation distances, not physiological or scientific results.

状态：`PASS`，证据位于 2203 专用目录的 `reports/m6a_public_002_stage_bc_smoke.json` 与 legacy 运行日志。

使用官方 `HearingTechnology/CoNNear_periphery` v1.0 代码与随附 trained weights；只运行官方 pure-tone 示例，并使用运行时兼容 shim 处理当前 Keras 导入差异，未修改官方源码、未训练或微调。

输出均 finite：`vbm`、`vihc`、`anf_hsr`、`anf_msr`、`anf_lsr` 均为 `[3, 8192, 201]`，分别保留 BM、IHC 与三类 ANF stage。该结果是听觉外周 surrogate 的工程复现，不是人脑听觉皮层或临床证据。

统一 probe 只使用 synthetic 输入；未读取 benchmark、患者或 STN 数据。CoNNear 与 ICNet 的生理语义不可混写。

## 统一六类时间 probe

同一批 20 kHz synthetic 输入包含 tone、speech、regular clicks、jitter clicks、omission 和 phase shift，逐级保留 `BM → IHC → ANF-H/M/L`。其中 regular/jitter/omission/phase 是重点生理比较；相对 tone 的 distance 与 lag-1 persistence、事件后 10 ms 平均绝对响应均按 stage 记录。

这些是无拟合的输入扰动描述，不能解释为听觉神经机制或人类生理效应；完整轻量 JSON 在 2203 的 `reports/m6a_public_002_auditory_physio_probes.json`。
