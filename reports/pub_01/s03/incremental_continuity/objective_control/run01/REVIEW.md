# D15三级验收与证据边界

2026-11-02；PUB-01-S03_SS07，整体科研goal未达到。

## Technical：PASS_WITH_LIMITATIONS

协调实际独立review source `36e9eebda70a7b60d8bb1cbc1ac27785cf36cd29fe403f260b2cd94d8731385b`，helper `0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64`，绑定v2 protocolb80f9aef…6506f。[实际JSON](coordinator_review/REVIEW.json) SHA46486c0d…bb35c，预登记依赖runner PID1781796仅执行一次，完成return0。

- 手写functional GRU与独立pairwise Hann，九selected+九last19+三H selected older共631239残差NN窗，无同状态复用；48624比较无失败，最大r/其他差1.1348755e−7。
- 原60排列的180实际日志、每fit20epoch/78900更新、全部选点/最早tie和共710100更新/59473248正式NN窗一致。
- 全180条既有反传首批梯度日志及45个离线TRAIN逐块→人→phase/幅度/overlap聚合通过。未第二次完整TRAIN NN复算目标/窗口MSE/幅度，未回放未保留中间epoch，不能称完整独立TRAIN loss复现。
- 新目标数学与独立NumPy/有限差分吻合，float64/eps语义、12个TRAIN低范数目标保留；fresh非零输出探针通路有效，旧late小批退化未被抹去。工程288+192=480NN、0fit/update，IO mock/AST0NN，原v1/source/protocol保留。
- 四实际PNG已逐张目视，逐图解释目标、物理MSE/r与幅度/overlap；工程、正式、独立和存储预检成本分行，v2复制的v1计时不当作累计480计时。最终source/数值及资源见REPORT/COST_LEDGER。

## Acceptance：PASS_WITH_LIMITATION（诊断），历史收益门FALSE

训练负cosine改善，TRAIN物理r也增长；相对D14 MSE，窗口MSE较差、物理r没有H特异稳定收益。H三seed选回anchor，只有C2026有极小开发增益；H−C方向非正、H−P及near/older均结构相等，不能支持额外历史贡献。晚期抽查token/GRU零/极小梯度意味着不能声称已解决该支路优化退化；抽查不是全数据/全部更新或因果消融证据。已知PCC目标本身不是创新，反复开发VAL不是独立确认。

## Parent consistency：PASS

唯一科学变化为同原完整FT+Ccap/三臂/三seed/初始化/排列/预算下的float64 centered negative cosine。执行并发/存储调整与日志均预审，0额外正式更新，不改主评分或选点。所有组和阴性结果保留；旧D12/D13/D14/D9、test/DTU/ds004408冻结，无患者/新数据/新权重/新encoder/head/额外架构或LR搜索。PR6保持Draft不合并，后续问题另由协调决定。
