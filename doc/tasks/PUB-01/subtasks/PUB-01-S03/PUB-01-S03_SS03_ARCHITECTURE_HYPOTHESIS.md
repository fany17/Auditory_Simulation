# SUBTASK — PUB-01-S03_SS03 — EEG Direct Auditory Decoder Architecture Hypothesis

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS03` |
| Parent Node | `PUB-01-S03` |
| Parent Task | `PUB-01` |
| Title | EEG Direct Auditory Decoder Architecture Hypothesis |
| Status | `BACKLOG`（需研究者审阅 SS02 后明确架构） |
| Blocking for Parent | YES |
| Patient Data | `FORBIDDEN` |
| Upstream | `PUB-01-S03_SS02` 独立验收，强基线与失效机制已明确 |
| Inputs | SS02 同任务 baseline、错误分析、冻结评测协议、声音特征定义 |
| Deliverables | 一项具体的可证伪架构假设、模型代码/训练配置、同预算对照、初步 direct-output 结果 |
| Acceptance | 核心算子对解决何种误差有明确预测，能逐模块消融；若无收益保留阴性 |
| Validation | 结构/shape/时间方向测试、budget-matched comparison、train-only model selection、独立审核 |
| Go/Hold/Fail | 没有可验证缺口/数据划分不清→HOLD；不能因成绩不如预期无限换网络 |
| Files Allowed to Change | 仅公共训练/模型/测试/报告路径与本节点任务文档；原始数据、S02/SS01 frozen 产物只读 |
| Last Updated | 2026-10-13 |

## Objective

**一个 EEG decoder，输入 EEG，直接输出声音包络 / Mel / 经批准的冻结语音表征。** 模型推理阶段不得要求候选声音或答案文本；后续可以用输出做后验检索。

## 进入 ACTIVE 前必须由研究者确定

1. 一个强 baseline 中真实发现的局限（如 EEG 空间观测、快速/慢速信息、对齐延迟或信号可靠性）；不能先假定一定是多尺度问题。
2. 只提出一个主要算子或结构假设；指出为什么普通 TCN/Conv/Transformer/SSM/通道适配不足。SaShiMi / TimeMixer++ 仅候选与反例，不自动加入训练。
3. 确定同一 direct output、训练数据、test 划分、模型参数/FLOPs、训练总步数、至少多个固定种子和对照。
4. 消融：去掉该算子、参数匹配替代、破坏其预期作用的输入/时间操纵；训练和评估不触碰 test 选择。
5. 新方法需同时支持直接连续特征评价与后验候选检索；若只擅长检索，必须如实降级论文目标。

## Completion

`NOT_STARTED`。本文件仅占位并冻结研究需求；**没有决定新架构、没有训练新模型，也没有创新性保证**。SS02 验收之后另行由研究者指定具体方案。
