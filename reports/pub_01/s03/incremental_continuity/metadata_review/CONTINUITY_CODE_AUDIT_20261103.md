# 连续性方法的一手实现核对（协调端，2026-11-03）

目的：在D17封存后识别尚未检验的学习路径，并核对相邻公开方法。本文无新训练、数值计算、权重或数据下载，不是性能结果或新颖性证明。D17最终提交为`fe70a051548933e8824926f17daa51d76baaf552`；其HTML16个相对链接已独立检查无缺失，已审REPORT/REVIEW哈希保持不变。

## 已有实现的实际边界

原`pub_01_s03_ss07_finetune.py`确实联合微调原EEG encoder和head；其窗口仍为3秒、hop96、64Hz。`pub_01_s03_method_models.py`的原encoder在64维时间token经过10层卷积/attention后，才由最后线性层投影至8维。

随后D12及D14–D16调用缓存FT输出，历史分支处理的是每窗8×192 latent，经卷积/全窗Flatten/线性层压成32维token，再经两步GRU调节当前摘要。训练脚本不构造原encoder，新增历史任务损失不可能回传到原attention。D16增加的token归一化没有改变这一边界。因此这些实验既不等于原始EEG没有历史信息，也不等于已经检验了高维逐时间token的端到端跨窗适配。

这一判断不覆盖全部历史实验：[warm round02](../run02/REPORT.md)已经完成18fit，其中joint-M4更新原encoder，并保留跨更新detach的历史状态；它仍先将过去压为8×6摘要/H16，past对同容量current三seed均负（平均r差−.000360）。因此不能说“此前从未联合训练过历史模型”。待核对的新缺口仅是projection前192×64 token的query-dependent交互，不能重复早期M4 joint方案后改名。

D17仅在单人48窗证明：所选H2026历史响应很小，三个末轮H仍有明显响应。结合D16验证结果，不能把失败统称为全部检查点历史路径死亡；也不能据此断言提高历史响应强度即可改善泛化。

## DECAF：论文入口与已读代码尚未完整对齐

论文提出EEG估计与过去预测包络先验的动态融合，与本项目较简化的output_prior不同。[论文](https://arxiv.org/html/2602.19395v1)

[作者主页README](https://github.com/JHU-LCAP/DECAF/blob/main/README.md)实际将代码指向`carankt/DECAF`。协调通过GitHub连接器读取以下文件，括号为当次Git blob SHA：

- [recursive模型](https://github.com/carankt/DECAF/blob/main/src/models/two_branch_model_recursive.py)（`6a821e9e73ef807b9a69869f7710023fe0c344fa`）：所读eval forward重新用EEG分支预测过去片段包络，再输入预测器；此函数没有持久保存并反馈前次最终融合输出。当前EEG输出采用窗内minmax，而teacher路径采用sigmoid。
- [trainer](https://github.com/carankt/DECAF/blob/main/src/train/trainer.py)（`9ab04f098358cd210bb40938e4dcc913aade700e`）：所读训练递归路径teacher-forcing概率两分支都是1；VAL递归路径调用真实过去包络teacher函数；默认test则调用预测历史路径。另有调用`forward_recursive`的测试函数，但上述模型文件没有定义该方法。
- [data utilities](https://github.com/carankt/DECAF/blob/main/src/data/data_utils.py)（`166d8b23ace1b67ff5f28a53d639efc973c20eff`）及[recursive配置](https://github.com/carankt/DECAF/blob/main/configs/two_branch_recursive_config.py)（`d15840eea894e537d2f5bc500dfbf37e9cb3cdc5`）：训练shuffle/drop_last，配置指向80%重叠的过去/当前CSV，验证和测试另用不重叠CSV。
- [train入口](https://github.com/carankt/DECAF/blob/main/train_model.py)（`6cefa2dbadb7e695a696552b04c3e3f019fcd00f`）：训练后自动调用test；本项目不执行此入口。

这只是已读入口与模型链的审计，不是完整仓库穷举；不能据此断言没有其他paper专用脚本、论文结果无效或出现测试标签泄漏。真实过去包络用于训练本身不是测试泄漏，但teacher式VAL与部署预测历史不同，需要明确。未执行作者程序、下载其Hugging Face权重/数据或复现论文分数。不能将该入口未经核对直接纳入“完整DECAF同协议复现”。

## SHINE与segment memory的先例

[SHINE论文](https://arxiv.org/html/2602.23960v2)使用长窗、多尺度与局部/上下文融合；[仓库README](https://github.com/xuxiran/SHINE/blob/main/README.md)在本次读取时仍写明代码待发表后公开（blob`49203188da940913dcc05f711a454933d0ea1974`）。不能称已经复现作者代码。

[Transformer-XL](https://aclanthology.org/P19-1285/)已提出segment recurrence与相对位置设计。因此给原EEG模型接入跨窗attention只是一种待检验适配，不能把组件组合直接称为新方法贡献。后续需要具体性能、严格当前输入/容量/输出历史对照与独立泛化证据；本次阅读不关闭研究goal。

## 当前决策边界

执行者正在核对高维token适配和物理输出融合的最小实现。尚未冻结D18模型、损失或正式训练矩阵。D9的物理融合增益与单窗误差相消同时保留；不修改旧解，不把重叠视图当作额外纯过去。所有旧test、DTU和ds004408均保持冻结。
