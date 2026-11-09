# D13独立验收

2026-11-02；仅固定误差诊断，PUB-01-S03_SS07仍ACTIVE。

## Technical：PASS_WITH_LIMITATIONS

独立执行者协调会话在2203以CPU2线程运行独立review脚本，source SHA `7e89f64bbfc01c85e2c5c7bcaa4c146303e573f1c7274925eeb709c1c8076a26`，协议SHA `47404c336f76365c48fb847ac59007067420ea6701a82f806c82418c7b02221c`。实际 [REVIEW.json](coordinator_review/REVIEW.json) 原样保留。

- 三seed各426条TRAIN记录：独立Ccap重建、torch.fold物理融合、scalar loss autograd；7668个record梯度moments最大差1.65e−23，3834个record cos差5.55e−16；18个seed moments/9个cos通过。
- 全61704块r/MSE恒等最大差3.33e−16/4.93e−16，11781逐人聚合和198phase聚合差0；没有按分数删除对象。
- 复用此前独立真实非零VAL回放核对九last的r/MSE，差≤1.36e−8/3.03e−10；未新增重复NN。没有对本轮TRAIN非零网络作第二次独立前向回放，不能声称覆盖这一项。
- 主任务实际2,508,111 residual NN windows、0新encoder/head/fit/update；独立review0NN/fit/update。原始模型/数据只留2203。
- 主评分无undefined/低方差，辅助TRAIN tail与末single未定义值保留null；无epsilon、ridge或skipna补救。四张真实PNG已逐张目视检查，报告各有完整中文结果解释；没有将输出梯度当作参数梯度。

## Acceptance：PASS_WITH_LIMITATION

已回答本轮限定的“最后轮MSE/r为何出现不同方向、输出目标是否反向”问题：VAL MSE降低主要为偏置项，TRAIN物理MSE升高、r下降（均恶化）；三对输出梯度总体/record/person均正。该结果否定简单预设解释，但不能证明优化机制因果、历史普遍无效或更换loss有效。全量统计、区域、成本、来源、公式与未定义规则可追溯。

## Parent consistency：PASS

保持原公开TRAIN/VAL、固定split/目标/九last；未改变PUB-01范围或D12/D9结果，旧test/DTU/ds004408冻结；无患者/新数据/新权重/新训练、无PR合并。科研goal未完成。后续顺序对照须独立冻结方案/源和计账，不将其潜在结果写成本轮发现。

## 全文与图解释独立复核

协调会话实际阅读REPORT/REVIEW并查看四PNG后，主科学解释通过。封存前按其具体反馈更正：TRAIN物理MSE是升高、r下降；21,375辅助总行中21,240投影已定义，135不足点未定义（COMPLETED字段记录已定义投影数）；删除未做检验的“显著”措辞。只读顺序假说证据原样纳入，不当作因果结论。
