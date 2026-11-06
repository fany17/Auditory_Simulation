# D27 Sea-Wave-Small 已见被试强对照

复用 [协调决定](COORDINATOR_DECISION.md) 和原 426 TRAIN / 49 VAL。49 名 VAL 被试全在 TRAIN，主对照包括 3 个共享 initial 和 147 个个人 FT。每个人从自己 seed 的同一 initial-best 独立分叉，使用自己的原 TRAIN。群体 initial 行明确标为未做群体 FT。三个 seed 等权，personal 的不同适配自由度、Sea-Wave 的整记录上下文、原 Transformer 的 XLSR 教师信息均披露。

作者 commit 910138cd 的模型、调度器与 Pearson 函数直接导入，保留原 weight norm、初始化和重启数学。结构为 20 层、32 残差/skip、dilation cycle5、感受野249。训练连续记录只做原 TRAIN 通道归一化和共同长度截断，字典序记录、窗内时间递增，无 shuffle。模型接收批级切窗，连续记录按需缓存，重叠窗不物化到磁盘。保留尾批。VAL 使用 core4096 与左右 halo124 的整记录等价路径，按原正 Hann 共同支持计 49 人、752 块。初始 201 轮，个人最多 21 轮、patience 5，epoch −1 是个人候选。

## 先验工程上限

只在 2203 CPU，2 线程、RSS 4 GiB。固定 seed 1024 和排序后的前两条 TRAIN 记录。两次 B2 initial MSE 及一次 B2 personal 更新对应 3 次模型前向。另计划 5 次纯前向用于非零同权重作者桥的 2 次、整段 4,608 点的 1 次、两个带 halo 核心块的 2 次。恢复与评分桥无模型前向。合计最多 8 次模型前向和 3 次优化。实际工程状态不进入正式训练。

## 正式范围与恢复

精确预算由 2203 的冻结清单产生。[协调几何核验](GEOMETRY_RECEIPT.json) · [执行预算](BUDGET.json)

每个作业每 2,048 次更新及轮末保存模型、AdamW、作者调度器、Python/NumPy/Torch RNG 与游标。恢复从已提交边界开始，故障后尚未提交部分单列，不自动重启。每个人直接重新设同一 seed，空优化器、空调度器，严格加载同一 seed 的选中 initial。每 worker CPU 4 线程、RSS 32 GiB、CUDA reserved 40 GiB，SATA 新根 16 GiB、保留 64 GiB。

D27 正式只在 D26 全矩阵完成且原 PID、UID、startticks 所属进程实际退出后允许启动。当前只准备代码与有界 CPU 工程，GPU 仍由 D26 使用。

## 调度与恢复补充

D27 仍等待 D26 原矩阵及进程实际终结。D27 启动后三个 initial 作业优先排入两张卡。一旦某 seed 的 initial 正常完成，其个人作业可使用空闲卡，其他 seed 的 initial 无须结束。每名被试仍严格来自本 seed 的同一选中权重。

跨重启 canary 的 Adam moments 由合成张量手工填充，验证的是序列化恢复。实际 EEG 工程三次更新单列。从已提交的 2,048 步边界恢复，异常发生前未提交的实际更新数记未知，若处于训练区间则每个未提交段不超过 2,048 次已完成更新。已有个人汇总、最终 JSON 和终态权重核对后复用。
