# 当前状态：SATA恢复正式队列已启动

后续报告的划分解释以[协调元数据核验](COORDINATOR_SPLIT_SCOPE.md)及其[0NN receipt](COORDINATOR_SPLIT_SCOPE.json)为依据：49名VAL参与者全部见于TRAIN；29条VAL记录属于重叠的audiobook_13/audiobook_5来源组，20条仅相对当前TRAIN清单来源未见。record/story/EEG路径/包络路径无交集不证明音频波形无重叠，三个相对未见来源也不等于此前开发从未查看。本轮只能称已见参与者、混合来源的反复开发比较，不能称新参与者或全新来源独立泛化。最终保留全部49人的既定60秒块→逐人宏平均主指标及原逐来源描述；不把20条子集改为主结果/选点/验收，不将seed或重复参与者作新增独立样本。训练、预算、选择、验收与正在运行的冻结源均不变，本次仅补报告解释。

恢复首真实更新已核验：同一STARTUP_VERIFIED快照H1024=1000/C1024=1400 Adam、empty0，实际UID/start_ticks/精确argv匹配，SATA writer free178586537984B，home_readonly单列20287533056B；新ROOT62816331B。只读取已有metadata/首batch梯度，0额外NN/backward。旧partial成本另列，尚无完整矩阵结论。

新protocol255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632经独立审阅通过，review e051f5c2d5978d88c6f7c275b9f8ff56bc29c1455bb0c61cd03dbdec95487c63。新队列2050834，H/C1024 workers2050842/2050850；原24h总start保留，未重置。旧FREEZE_RECORD仅历史旧protocol，新恢复冻结由STORAGE_MIGRATION_RECEIPT与新review证明；旧工程不覆盖改动的queue时钟，后者以QUEUE_CLOCK_CHECK单独0NN核验。两中断fit的CPU/完整峰值与H精确终止计数未留存，记未知，不推断0。尚无完整矩阵/科学增益结论。

# 当前入口：D18已迁移SATA，恢复协议审阅中

2026-11-03 D18首轮因home可用空间跌破20GiB触发既定停止；C1024确切23000更新、H1024保守[18400,18500]，无完整Adam/RNG可精确续跑。原131文件/33319215B内容及mtime全等备份后，原homeROOT完整保留在*_home_backup_formal_attempt01，canonical路径现指向SATA；两partial fits/旧队列/源/协议完整归档，未删除、未用于新选点或初始化。SATA真实写盘20GiB阈值，home只读且未声称空间已满足20GiB。仅queue小改沿用原24h绝对deadline1793726963.790547，8项神经相关源及科学配置逐字节不变；没有重复NN工程。新协议255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632、资源/迁移/0NN验证见preprojection_memory/run01；等待协调绑定实际新SHA后，按原四臂三seed各20轮干净重启12fits。保留2中断＋12完成预期=14attempt，原成本界独立列账；SS07/科研目标ACTIVE，不重开旧测试/外部或合并PR6。

# D18 执行状态（正式矩阵运行中）

2026-11-03：最终源 `49b223239611cfd52a41da185ca5b49705d23090`，最终工程矩阵12行×2次临时更新全部通过；正式源/输入/协议独立审阅已通过，冻结12fit队列已启动；当前无完成结果，不作收益判断。未重开旧test或外部。

正式协议SHA256：`71e19d9ed88d1b591994d26cc6a52a7d4a4f45dddbbab86721d8ebd6d5501367`。工程协议：`bdabb49c9ef7eacadffe70e307f187967a437c949282d88f6c2fdb3b7a80181e`。

全部12行zero-memory同raw初始预测最大误差0，CPU/CUDA dropout RNG一致，构造前后CUDA RNG逐tensor一致；固定Ccap值/独立解析VJP、原raw/cache桥、第二步真实Q/K/V及query梯度、P输入内容经K/V的变化、cold fallback/整体bypass、冻结性和参数checkpoint序列化均通过。这里只是工程可用性，不是历史信息收益。

最终工程CPU15.346733秒、wall11.880187秒、RSS峰2701316KiB、GPU分配峰2861372928字节。最终生产current prefix/projection/head各2304窗，冻结prefix1563、attention query1563、P projection1798；原native参考prefix/projection/head各384，纯读出188窗/3次VJP、attention query VJP9次，临时Adam24次。

两次失败完整保留于`attempts/engineering_attempt01`与`attempts/engineering_attempt02`：先是objective漏导入，后是CPU初始化误改CUDA RNG；数值阈值从未放宽。旧生产current prefix/projection/head各512、冻结prefix648、attention query411，参考各256、纯读出126窗/2次VJP、query VJP2，临时Adam4次。最终决策`D18_ENGINEERING_FINAL_03`授权累计28次临时更新（旧4＋最终24），全部丢弃，不是正式fit；原forward上限与正式预算不变。旧attempt01时钟/峰值receipt缺失，明确未知，不记作0。

完整指标保留为gzip JSON，至多87文件×2MiB；元数据阶段尺寸压力case TRAIN5616853→820172字节、VAL696543→102876字节，无损往返相等。这是合成JSON尺寸检查，不是科学结果。按完整checkpoint实测13614175字节、24份，另174MiB大指标、32MiB其他和64MiB在途，正式总预算609855720字节，低于768MiB硬限。

正式预算保持四臂H/C/T/P×三seed、各20epoch，946800更新位置、60588960监督目标窗；原TRAIN/VAL、固定强FT+Ccap基线和首9anchor不变。历史收益需完整结果及独立复核后判断。SS07与科研目标保持ACTIVE；不启动其他子节点或旧外部测试。


2026-11-03 D18 projection前memory正式矩阵已启动：最终12行工程全部通过，累计28次临时更新（原失败4＋最终24）全部丢弃，数值/forward上限不变。协调独立重核486 source pins/547 input identities及12行工程，回执COORDINATOR_EXECUTION_REVIEW.json绑定正式协议71e19d9ed88d1b591994d26cc6a52a7d4a4f45dddbbab86721d8ebd6d5501367。正式H/C/T/P×3seed×20轮，946800更新位置/60588960监督窗，总24h/双GPU各一fit；当前Eθ/head可训、历史E0eval冻结、Ccap固定可微，near5/3、older9/7、首9anchor，原TRAIN/VAL和目标不变。唯一队列PID1986322，首H1024 PID1986330(cuda0)、C1024 PID1986345(cuda1)，UID1012/start_ticks/argv及空卡准入已记录；当前运行中，不称正式完成或历史增益。执行/失败/预算证据见reports/pub_01/s03/incremental_continuity/preprojection_memory/run01/。SS07/科研goal保持ACTIVE，不重开旧test/DTU/ds004408，不新增分支或合并PR6。


正式启动核验：STARTUP_VERIFIED.json的同一实际快照中H1024已700次、C1024已1000次Adam更新，empty均0；按/proc核验UID/start_ticks/argv通过，记录既有首batch梯度，0额外NN/backward。这不是完成或比较结论。
