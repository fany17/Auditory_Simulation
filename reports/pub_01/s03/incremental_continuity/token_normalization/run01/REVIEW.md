# D16三级验收与证据边界

2026-11-02；PUB-01-S03_SS07，持续科研goal未达到。

## Technical：PASS_WITH_LIMITATIONS

协调预声明独立源SHA `1bd999ec8d1e09bb3f7141f4ee9f8ab54ad8594f8bcf333c1d1c0ccf814e15f2`、helper `0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64`，绑定v2协议56f4be82…70b3a2。依赖runner1834387只运行一次，return0；[实际REVIEW](coordinator_review/REVIEW.json) SHA c6696649…dcc77d。

- 手写normalization（显式population variance）/GRU/独立Hann：九selected、九last19、三H selected older共631239窗，无同状态复用。48624比较无失败，最大r差2.3282e−7、其他量4.3531e−6，不混为同一误差。
- 九fit各20epoch/78900更新，合计710100/59473248正式残差窗；180排列与最早tie/含−1选点、180已有反传及token状态日志、45离线TRAIN聚合通过。
- 没有第二次完整TRAIN NN重算目标/窗口MSE/幅度，没有回放未保留中间epoch，不能称全部训练数值独立复现。
- 原528工程、后到288逐点桥、0NN refreeze分别计账，累计816工程/0fit/update，主四源与正式初始化不变，原v1与桥来源保留。
- 实际逐张目视五PNG，中文完整解释所有图；D15/D16同epoch batch IDs一致但权重不同；symlog保留零、未把first-batch描述为全数据。
- CPU/wall/RSS/GPU及存储分阶段记录，制图7.100CPU秒/0NN，账本0NN，行政未测范围明确；只取回结构化汇总与静态图。

## Acceptance：PASS_WITH_LIMITATION（诊断），历史收益门FALSE

D16抽查H三seed的epoch1…19首批token/GRU无精确零，last GRU L2约.01323/.01087/.02238，较D15的0/0/1.10e−14明显活跃；归一化后token RMS约1，但上游尺度仍增长、部分GRU状态近饱和。通路改善仅限样本，不能声称全部更新有效或因果路径贡献。

TRAIN目标/窗口MSE/物理r改善，H的last物理r均值.262342高于D15 .256368。但H1024/31415所选仍−1，H2026只增+.000116401且低于P2026 .000130032；H对anchor/C/P三seed稳定正及一致近历史收益的门不满足。两个near=older是所选零头结构相等；非零H2026的−1.43e−10也在数值误差尺度内，不能称近历史有益/有害。反复开发VAL不是独立确认，既有归一化不是创新。

## Parent consistency：PASS

唯一科学变化为同D15模型/三seed/初始化/排列/完整预算下的无仿射逐token归一化；它改变输入尺度/平移信息及函数约束，不称纯优化干预或FLOPs相同。诊断不参与选点，所有阴性与旧结果保留。所有科学计算仅2203，无患者/新数据/权重/encoder/head或外部重开。PUB/S03/SS07保持ACTIVE，PR6 Draft不合并；后续具体问题由协调另行界定。
