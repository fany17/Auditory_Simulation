# ICNet reproduction

状态：`PASS`，证据位于 2203 专用目录的 `reports/m6a_public_002_stage_bc_smoke.json` 与 `reports/m6a_public_002_panns_icnet_retry.json`。

使用官方 `fotisdr/ICNet` v1.0 代码、配置和随附权重，在 legacy 专用环境运行官方示例路径并对 deterministic synthetic tone 做 smoke/probe；未读取 IC recordings、患者数据或 STN 数据，未训练或微调。

实测输出：输入 `[1, 26432, 1]`；`bottleneck` `[1, 1, 762, 64]`；`units_1000` `[1, 762, 1000]`；均 finite，CF 数组存在。一次运行约 1.78 s。

这里的 ICNet 是正常听力、麻醉沙鼠 inferior-colliculus population 的模型 surrogate，不是人脑 IC，也不支持脑区或跨物种功能等价结论。
