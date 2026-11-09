# D26 固定最终模型的有限原始路径复算

本项落实 D26 任务书已有的独立复算要求，不增加拟合、训练更新、数据范围或模型选择。正式 V05 与 D27 接续门不变。

预先固定原 VAL 清单第一条记录的第一个 60 秒块，重放起点 0、96、…、3744 的 40 个完整 192 点窗，实际 EEG 只取该记录前 3936 点、包络前 3840 点。它同时覆盖前四窗读出回退、前九窗历史未就绪、历史就绪和 960 点后的纯 memory 支持。九个 seed/arm 各只使用该次新训练的 epoch4 FINAL.pt 与 READOUT.npz。HH 另重放既定 far/copy；所有条件共用同一当前 query，不据分数选择记录或窗口。

从原 EEG/原 TRAIN 通道统计直接构造输入；使用已冻结的原生 backbone forward 的投影前 hook 获取 token，不调用 D26 的 prefix/project/decode。memory 的多头 Q/K/V、softmax 和输出投影手工计算，并用同一实际 token 与冻结 memory.forward 做有限数值桥。Ccap 的标准化、平方项、前四窗回退、Hann 及块内 Pearson/MSE 独立计算。对照正式保存的首块 raw/affine/Ccap；HH 另对照 far/copy 和各自 960 点起的纯 memory 块。

固定上限：九个最终模型、15 个路由、1290 个 backbone 窗调用（包含其未使用的原生初始 projection）、600 个最终 projection/head 窗、465 个手工 memory 查询窗及 465 个 library memory 桥查询窗。无音频前向、梯度、优化、读出重拟合或旧测试读取。每次调用计入独立账本；失败也记录已执行计数，不能因观测超时重跑。该有限验证不冒充全量原始信号重放，完整 49 人/752 块另由保存得分汇总审阅。

只在 2203，CPU affinity 44–47、最多四线程、RSS 4 GiB、输出最多 64 MiB，不使用 GPU，允许与原训练/后续 D27 并行。每个模型至多一次成功重放；同一固定案例的实现错误需保留失败证据后明确纠正，不能换样本或训练模型。

预先容差：手工和 library memory 最大绝对差 2e-5；CPU 独立重放与正式 CUDA 首块 r、MSE 绝对差各至多 1e-4。原 CUDA 允许 cuDNN TF32，故不要求跨后端逐位相同；实际误差逐项报告。超过容差时保留失败和原结果，先定位实现/精度原因，不修改模型、选点、预算或按成绩放宽标准。原始文件及 checkpoint、readout、正式 score 的指纹绑定每份回执。
