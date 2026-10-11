# PUB-01-S03_SS01 独立 review pass

日期：北京时间2026-10-11。审阅对象：拟合前任务书/配置、实际运行代码、server输出、逐记录/被试CSV、rerun和两张静态图。由执行agent另行进行独立review pass，并编写独立公式复核脚本；不是外部人工signoff或父任务Integration Review。

**Decision: PASS_WITH_LIMITATION。** Technical PASS，Acceptance PASS_WITH_LIMITATION，Parent consistency PASS。完整S03/PUB-01仍ACTIVE。

| 检查 | 实际证据与结论 |
|---|---|
| 预定scope不变 | S02 83名READY中按ID排序取sub-001–010；93条预定record均使用，无替换、跳过或被试失败；40条subject-condition指标 |
| 原split及family无泄漏 | 与服务器S02 manifest逐条一致；每被试train/val/test family不交叉，未重新分配 |
| lag/时间支持 | 明确envelope[t]←EEG[t+lag]，null加832；四条件每记录同t、同finite mask；n−864边界与CSV一致，test每条件27条/1,448,987点 |
| 只训练集归一化 | code Stats只在split=train更新，参数train centering/scaling；val/test不更新；未做PCA或train+val重拟合 |
| 只validation选alpha | 每subject/model/condition六候选，所选项等于validation record-macro r最大项；test仅打分 |
| 指标独立核验 | 首条test记录四条件，用独立设计矩阵、np.corrcoef及SSE/SST公式直接算；r最大误差2.78×10⁻¹⁷，R²最大误差2.22×10⁻¹⁶ |
| 汇总算术 | record→subject macro、subject→summary、样本数逐项一致；保留pooled副指标与train-mean基线 |
| 重现 | 首被试完整rerun的alpha/样本/r/R²与首次相同，最大差0 |
| 图与表对应 | 逐被试图保留sub-010负R²；示例是预定首test记录首20s，幅度弱，没有挑好看样例 |
| 科学解释 | 报告限定within-subject envelope reconstruction；单null不作p-value，不称语音语义、新被试或STN泛化 |
| 边界及parent | 只在2203读公共derivatives、拟合和读参数；本地只汇总/图；S03未全完成、不进入S04、不生成患者artifact |

技术结果见[technical_self_test.json](technical_self_test.json)、[technical_review.json](technical_review.json)；首次与rerun逐被试表均在本交付中。

限制包括预定10名非随机小样本、部分跨被试共用声音内容、发布方physical_sync/revision仍UNKNOWN、粗时滞网格、13秒单一描述性null，以及低解释方差。TRF虽高于null、9名record-macro R²正，但不能据此称可靠重建。零时滞多数负R²合法保留。深度EEG decoder和held-out-subject尚未比较。

没有为结果符号改scope/split/lag/alpha；没有测试集选层或升级claim。SS01实际交付完成，S03其他验收条件保持未完成。

2026-10-11补充方法来源复核：实际打开Crosse等2016原始mTRF论文和作者stimulus_reconstruction.m，确认backward方向及decoder权重不直接具有神经生理解释。引用已补入REPORT/WEEKLY_REPORT/SOURCES，实际Python配置、代码、结果未改；仅验证引用与陈述对应，不重跑已通过的实验。
