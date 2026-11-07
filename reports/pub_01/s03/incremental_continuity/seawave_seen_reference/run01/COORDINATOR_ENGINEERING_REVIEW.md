# D27 实际实现与工程审阅

2026-11-06。协调已核验 `input_v04` 的实际服务器源、工程回执和数据身份，**允许按冻结矩阵正式执行**。正式训练须等 D26 全矩阵完成、原精确进程退出和 GPU 空闲；本审阅时 D26 仍在运行，D27 尚无正式训练或性能结果。

生效协议为 [PROTOCOL_V04.json](PROTOCOL_V04.json)，SHA `8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476`。协调直接在服务器核对 16 份源指纹和 522 个输入文件身份，并写入 [执行审阅回执](COORDINATOR_EXECUTION_REVIEW.json)，SHA `e5cfcb770962a322e8f25304621478db63a0fdd1835445c0907064400cef033d`。原 [V03 审阅](COORDINATOR_EXECUTION_REVIEW_V03.json)完整保留。

实际审查覆盖作者模型/weight norm/初始化、自定义 lr 与 weight decay 重启规则、原 TRAIN 统计和目标坐标、记录内顺序与尾批、201 轮 initial、各人独立起点与 21 轮/patience5、所有 seed/逐人选点、完整记录推理和原 752 块评分。当前 49 名 VAL 被试均在 TRAIN，因此执行 3 个 initial 和 147 个 personal fit；initial 群体参考未做 global FT，不称完整 S2 对照。

[V02 工程证据](ENGINEERING.json)来自实际两条 TRAIN：2 次 MSE 更新、1 次个人目标更新，总计 8 次模型前向，0 CUDA、0 VAL 性能试验。实际参数 150,433、感受野 249；非零权重作者桥最大差 0，整段与 halo 最大差 1.4552×10⁻¹⁰，旧共同支持评分 r 差 6.9389×10⁻¹⁸、MSE 差 0。早层和末层梯度为有限非零值。协调读取了服务器原日志及版本差异；V03 只修复正式入口缺失的 `math` 导入，没有重复神经计算。

V04 相对 V03 仅修订调度、终态证据幂等、版本绑定及失败成本说明；数据、模型、工程三模块逐字节不变。[零神经计算调度验证](QUEUE_ENGINEERING_V04.json)确认：某 seed 的个人 fit 可在另一 seed initial 运行时使用空闲卡，但必须等自己的 initial 完成；相同已发布汇总原样复用，冲突内容拒绝。没有改变任何 fit、seed、训练目标、预算或选点。

恢复 canary 中的 Adam moments 是手动填充的合成张量，验证的是非空优化器状态、调度器、RNG 和游标的序列化恢复，不能称 EEG 训练状态的独立重放。故障时未提交更新量记未知或界限，不能按零计成本。队列保留 PID/UID/startticks 检查，不把观察超时当作重启依据。

技术准入不等于性能或 SOTA 结论。个人适配相对共享 Transformer 的自由度、XLSR 教师信息、原生上下文和总预算分别披露；反复开发 VAL 不能充当独立确认。最终保留 [D22 B、D18 P、D23 S 整臂开发参考](../../contrastive_history/run01/SOTA_COMPARISON_SCOPE.md)，不因历史机制未成立删除性能候选，也不跨方法拼选 seed。旧 test、DTU 和 ds004408 继续关闭。
