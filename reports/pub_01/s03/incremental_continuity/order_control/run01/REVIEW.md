# D14独立三级验收

2026-11-02；PUB-01-S03_SS07。

## Technical：PASS_WITH_LIMITATIONS

协调会话在2203实际运行预声明独立实现：source `746380c40be3d8a7e301d830f1164370745956af10ede8d27f96b957d9df9d8c`，手写GRU/Hann helper `0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64`，绑定v2 protocol83b05eaf…d0172。实际 [REVIEW.json](coordinator_review/REVIEW.json) 原样保存。

- 九selected、九last19、三H selected older分别CPU功能实现回放；本次无selected/last状态相同复用，合计631239残差NN窗。48624比较无失败，最大r/其他比较绝对差1.1807e−7，在预声明容差内。它独立于生产Conv/GRU调用和Hann/聚合代码。
- 全180条实际排列/epoch/update日志一致，每fit20epoch/78900更新，selected含−1/最早tie规则复算；九fit/710100更新/61710828正式NN窗一致。
- 全54份离线TRAIN（45新检查点+9旧last参考）逐块→人→phase聚合通过，18项旧last物理指标与D13核对通过。
- 未做第二次完整TRAIN网络回放以独立重算windowMSE，也未回放未保留的中间epoch模型；这些限制显式保留，不能称完全独立TRAIN loss复现。
- 独立成本额外CPU72.036s/wall23.907s、RSS1174516KiB、GPU0，新encoder/head/optimizer/fit0。正式训练与成功/失败工程另计。
- 四张真实PNG已逐张目视，报告逐图包含组别、坐标、方向、意义与限制；固定TRAIN离线、在线轨迹、VAL选点/last比较未混淆。生产源码未在运行中修改，failed attempt01 source/protocol/log保留。

## Acceptance：PASS_WITH_LIMITATION（诊断），历史收益门FALSE

九组固定last的TRAIN windowMSE/r相对anchor和旧last都改善，原共同训练退化被预设顺序干预修正；VAL训练后曲线多数下降，H selected只在一个seed微小增加。H−P方向混合、near−older主r近0；未满足H对anchor/C/P三seed均正及一致历史干预收益。该诊断完成不代表方法创新/独立确认/科研goal达到。

协调另核唯一非零H2026的near/older人/块差在实现数值一致性尺度（人max7.95e−9、块max1.58e−7），原样证据见 [HISTORY_INTERVENTION_RESOLUTION](coordinator_review/HISTORY_INTERVENTION_RESOLUTION.json)。报告按此明确不使用极小负号声称历史有害，也不以older结果推出所有历史未使用。协调已实际阅读全文并逐张查看四PNG，数字、解释及阴性边界通过，无新增阻断项。

## Parent consistency：PASS

只原TRAIN/VAL，旧D12/D13/D9与旧test/DTU/ds004408冻结，无患者/新数据/新权重/新encoder/head；唯一正式变化是固定窗口顺序，未更改loss/LR/结构/预算。九组全部保留，阴性不触发新增模型。只读LOSS_SOURCE_AUDIT无新NN或执行批准；后续科学必要性由协调审定，PR6仍Draft不合并。
