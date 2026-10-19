# Codex独立执行的SS05源码与合成审计

2026-10-19。旧原型/报告来自协调者，不默认正确。Codex在2203 env01/PyTorch2.11.0重跑原10项tests、cue160/key160及GRU600，另做5项窗口tests、真实EEG CPU/GPU smoke、有标签事件反事实及资源审计。旧环境为PyTorch2.10 CPU，不能声称逐位复现。

## 发现与修正

原sample模块按step因果递归；窗口前端可访问当前完整3s，属于跨窗口因果、窗内离线，不能称sample-causal实时模型。原窗口local仍把prev2/prev1送入local_filter，会读取前两个3s token，不能称无历史。原toy的局部3时刻定义仍成立，不改其历史报告；新FiniteHistoryDecoder对local逐当前窗用空状态单token调用，新增test与真实反事实验证历史影响为0。

每query只读同record前20个连续窗，state在60s上下文左边界清零，完整BPTT21 token。共享编码20past+20current的40连续窗，20个query各自取前20窗；不把随机片段接成长序列，不在record/person/session间传state。

predict_local没有显式next-token forecasting损失，因此f−prediction只能称学习到的历史残差。即便full胜assoc，也不能直接证明真实预测误差或surprise机制；本轮没有为追结果添加辅助损失。关联矩阵按key读出残差做outer-product delta update，含固定.995 decay和学习write gate，属于已有思想组合。

## 旧报告数值复核与额外合成反事实

12个160step accuracy与旧表相同：A local48.4375%、其他含GRU100%；B local50.78125%、state71.875%、predictive_state70.3125%、assoc/full100%、GRU62.5%。GRU600重跑100%、MSE=.00156327，不能与160步作公平比较。

多数MSE差为0至约1.8e-7；A-assoc旧.0001322621、新.0000573186，差.0000749435，accuracy仍100%。新run选step160；旧汇总缺完整checkpoint/log，无法确证差异原因，环境/优化敏感性仅为可能解释。因此计算能力和定性结果可复现，原全部MSE未精确复现。逐行old/new见evidence/intake/CODEX_SYNTHETIC_AUDIT.json。

有ground-truth的key probe中，改变查询key最后写入后assoc/full预测变化约1.959/2.010；改无关key仅.0395/.0412。相关翻转的新标签、无关翻转的原标签accuracy均100%；插入16/32空白间隔后仍100%。state与160步GRU没有同样清晰的选择性。它只证明特定合成分布的能力，不是EEG事件相关性证据；assoc不需要预测残差也成功。

## 容量与计算不能只看实例化数量

| 组 | 实例化 | 有grad张量的参数库存 | 五真实train batch梯度非零标量并集 |
|---|---:|---:|---:|
| local | 68546 | 62209 | 43777 |
| state | 68546 | 62209 | 55041 |
| assoc_no_pred | 68546 | 65442 | 65442 |
| full | 68546 | 68546 | 68546 |
| GRU | 70449 | 70449 | 70449 |

最后一列为经验活动下界，不是可识别自由度证明。有grad张量不等于每列有效：零历史/零retrieved使部分列恒零，关闭预测/关联还会断开整张量，不能用相同68546声称全部容量匹配。

batch1/60s的conv-Linear-GRU名义MACC为local156.59M、state172.97M、assoc/full174.30M（另约.645M关联读写）、GRU174.99M；不计bias、激活和逐元素算术。未使用但仍执行的predict_local已计入。GPU warmup后约local2.25ms、state14.40ms、assoc52.88ms、full52.03ms、GRU4.09ms。full/GRU名义算量接近，Python小算子循环却比cuDNN GRU慢，不能称等墙钟预算或硬件效率优势。精确值见smoke/scalar_parameter_audit证据。

## 已有方法与原创性边界

标准GRU已有reset/update gates控制向量状态，本轮nn.GRU是直接控制。PI模块改成三速率向量和可选矩阵，不因此成为新算子。[GRU原论文](https://arxiv.org/pdf/1406.1078)

Mamba以输入依赖SSM参数选择传播并提供硬件感知算法；本原型固定三retention加非线性门控，未实现其selective SSM/scan，只共享选择性状态思想。[Mamba原论文](https://arxiv.org/abs/2312.00752)

Delta Rule/Gated Delta Networks已有key关联残差定向改写与遗忘gate。本原型采用同类outer-product correction，decay为常数，另有write gate，未实现其训练算法或证明等价。[Gated Delta Networks原论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)

Titans按关联记忆损失的梯度/surprise及动量更新记忆；本原型推理时不做该梯度优化，历史残差也未被显式训练为forecast误差，不能借用其surprise解释。[Titans原论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)

以上为原始来源与本代码的结构对照，不是完整模型性能复现；没有下载或追加这些模型。EEG正式结果另列，合成结果不能代替真实验证。
