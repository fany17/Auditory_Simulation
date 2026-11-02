# 下一步科学缺口：一手文献与公开实现核验

2026-11-02，协调端只读核验；无新模型、训练、预测、数据或权重下载。上轮外部结果已冻结。本文件补充既有 `run01/RELATED_WORK_COMPARISON.md`，不以文献数字替代同协议实验。

## 可以排除的创新表述

- [DECAF v1，§3](https://arxiv.org/html/2602.19395v1)已有预测包络历史与当前 EEG 的动态融合。其 3 秒窗、500 ms 偏移、训练重叠及联合损失不同；我们的 output_prior 不是完整 DECAF。仅增加 GRU 或 gate 不能声称新原理。
- [SHINE v2，§3–4](https://arxiv.org/html/2602.23960v2)已有多尺度时序表征及按目标/时间的局部—上下文融合；输入为 30 秒并联合重建包络和 Mel。它的任务、划分及评分不支持直接拿 r 排名。2026-11-02实际查看的[作者仓库](https://github.com/xuxiran/SHINE)仅展示 README，说明论文代码将于发表后发布；本次未取得可运行论文实现。
- [Lin 与 Li，§2–3](https://sites.duke.edu/dkusmiip/files/2023/12/NCMMSC23_cr.pdf)已有 S4、U-Net、被试调制组合。状态空间、长序列或条件化本身均不足以构成新颖性。
- [SDN-Net 作者预印本摘要](https://arxiv.org/abs/2501.08693)提出被试信息解耦与多尺度时间/通道注意。本次仅核摘要，不把其性能声明视为本项目已复现证据。

## DECAF 公开代码的新增核验

[论文作者组织仓库](https://github.com/JHU-LCAP/DECAF)实际指向 [carankt/DECAF](https://github.com/carankt/DECAF)。`git ls-remote`实际确定 main 为 `f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332`。仅保存该 commit 的七个源/许可文件于 `decaf_source_review/`，未 import 或执行，未下载 README 所列数据或权重。来源和内容 SHA 见该目录 `SOURCE_MANIFEST.json`。

核对 [trainer.py](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/src/train/trainer.py) 的具体分支：递归训练两段 teacher-forcing ratio 均为 1.0，验证调用 ground-truth-context 路径；测试另有无 teacher-forcing 路径。不能将这些验证成绩解释为纯 EEG 自由递归验证，也不能未经配置追踪就认定论文主结果使用某一分支。

核对 [two_branch_model_recursive.py](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/src/models/two_branch_model_recursive.py)：普通 `forward` 在推理时从 `eeg_context` 经 EEG branch 生成过去包络，再送 forecaster；该入口没有接收前一调用的最终融合输出。当前输出还有逐窗 min/max 缩放。公开 trainer 的另一个入口调用 `forward_recursive`，而本次所读类未定义该方法。具体注册配置及论文产物路径仍未完全确定，因此这里只记录实现差异，**不据此声称论文结果错误或整个仓库不能运行**。

核对 [eeg_dataloader.py](https://github.com/carankt/DECAF/blob/f4c5dbeafa93e97abb32bf189d6d1b5cb6df3332/src/data/eeg_dataloader.py)：EEG 当前/过去片段分别标准化，目标片段有 min/max 变换。此口径也不等于本项目固定原 TRAIN 标准化与全记录 Hann 输出。官方比较若后续实施，必须明确原生条件与统一条件适配，不能直接粘贴入口自动测试或下载外部 checkpoint。

## 对下一实验的约束

1. 当前可检验缺口是：强 FT+Ccap 固定锚点之外，完整时序历史能否通过非线性条件映射提供额外预测信息。现有组件名称和未测组合本身不是新方法贡献。
2. 必须有相同当前输入、匹配活跃容量/预算的 current 控制与预测包络历史控制，推理和选点均不输入真实过去包络；内容干预保持当前 EEG 与锚点不变。
3. 历史应在 Hann 融合涉及的全部当前输入之前。按现有窗坐标，候选严格过去索引从 `i-3` 起每次减 2；不能把单窗不重叠误写为物理输出层面的完全非重复信息。
4. 只做原 TRAIN/VAL 开发；VAL 已反复使用且人与部分 family 依赖，不能提供独立确认。旧 test、DTU、ds004408 不重开、不用于选架构或预算。
5. 下一轮必须冻结一个有界、可证伪问题并实际训练；阳性还需后续机制、竞争方法和未污染泛化证据，阴性不靠更名或无限扩网格挽救。

GitHub connector 本次连接失败，公开 API 也遇速率限制；以上代码来自固定 commit 的公开 raw URL，仓库状态由网页及只读远程 ref 查询交叉确定。没有推测未取得的实现或访问受限资源。
