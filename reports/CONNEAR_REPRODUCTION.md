# CoNNear reproduction

状态：`PASS`，证据位于 2203 专用目录的 `reports/m6a_public_002_stage_bc_smoke.json` 与 legacy 运行日志。

使用官方 `HearingTechnology/CoNNear_periphery` v1.0 代码与随附 trained weights；只运行官方 pure-tone 示例，并使用运行时兼容 shim 处理当前 Keras 导入差异，未修改官方源码、未训练或微调。

输出均 finite：`vbm`、`vihc`、`anf_hsr`、`anf_msr`、`anf_lsr` 均为 `[3, 8192, 201]`，分别保留 BM、IHC 与三类 ANF stage。该结果是听觉外周 surrogate 的工程复现，不是人脑听觉皮层或临床证据。

统一 probe 只使用 synthetic 输入；未读取 benchmark、患者或 STN 数据。CoNNear 与 ICNet 的生理语义不可混写。
