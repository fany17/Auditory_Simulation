# D18 执行状态（正式拟合前）

2026-11-03：最终源 `49b223239611cfd52a41da185ca5b49705d23090`，最终工程矩阵12行×2次临时更新全部通过；正式源/输入/协议独立审阅待回执，未开始正式fit或新VAL/test。

正式协议SHA256：`71e19d9ed88d1b591994d26cc6a52a7d4a4f45dddbbab86721d8ebd6d5501367`。工程协议：`bdabb49c9ef7eacadffe70e307f187967a437c949282d88f6c2fdb3b7a80181e`。

全部12行zero-memory同raw初始预测最大误差0，CPU/CUDA dropout RNG一致，构造前后CUDA RNG逐tensor一致；固定Ccap值/独立解析VJP、原raw/cache桥、第二步真实Q/K/V及query梯度、P输入内容经K/V的变化、cold fallback/整体bypass、冻结性和参数checkpoint序列化均通过。这里只是工程可用性，不是历史信息收益。

最终工程CPU15.346733秒、wall11.880187秒、RSS峰2701316KiB、GPU分配峰2861372928字节。最终生产current prefix/projection/head各2304窗，冻结prefix1563、attention query1563、P projection1798；原native参考prefix/projection/head各384，纯读出188窗/3次VJP、attention query VJP9次，临时Adam24次。

两次失败完整保留于`attempts/engineering_attempt01`与`attempts/engineering_attempt02`：先是objective漏导入，后是CPU初始化误改CUDA RNG；数值阈值从未放宽。旧生产current prefix/projection/head各512、冻结prefix648、attention query411，参考各256、纯读出126窗/2次VJP、query VJP2，临时Adam4次。最终决策`D18_ENGINEERING_FINAL_03`授权累计28次临时更新（旧4＋最终24），全部丢弃，不是正式fit；原forward上限与正式预算不变。旧attempt01时钟/峰值receipt缺失，明确未知，不记作0。

完整指标保留为gzip JSON，至多87文件×2MiB；元数据阶段尺寸压力case TRAIN5616853→820172字节、VAL696543→102876字节，无损往返相等。这是合成JSON尺寸检查，不是科学结果。按完整checkpoint实测13614175字节、24份，另174MiB大指标、32MiB其他和64MiB在途，正式总预算609855720字节，低于768MiB硬限。

正式预算保持四臂H/C/T/P×三seed、各20epoch，946800更新位置、60588960监督目标窗；原TRAIN/VAL、固定强FT+Ccap基线和首9anchor不变。历史收益需完整结果及独立复核后判断。SS07与科研目标保持ACTIVE；不启动其他子节点或旧外部测试。
