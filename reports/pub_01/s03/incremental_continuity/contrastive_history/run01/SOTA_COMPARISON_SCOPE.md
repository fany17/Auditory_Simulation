# SOTA性能目标与尚缺的强方法比较

2026-11-06，响应用户『有性能提升就行，核心目标是SOTA』。这是协调已核实的对照范围，不是新训练结果或全领域穷尽综述。当前任务是 EEG-only 连续语音包络重建；检索 Top-1、AAD 分类、MEG/语音文本指标不能替代本任务性能。

旧强 Transformer 的完整训练、FT 与 Ccap 是现有可核查锚点，尚不等于本任务所有公开强方法。D26在做三个完整方法的性能比较，历史机制仅为次级解释。此前反复开发的VAL与已经看过的test/外部结果都必须明确标注；不同数据划分的论文r不能直接与本项目开发r排名。

已有整臂开发候选继续保留。历史机制未成立不使实测性能收益失效；无历史的模型也可入选。下表直接引用冻结报告，不新增评分、拟合或选点，不表示已穷尽所有旧实验，也不改变D26三臂主比较。D27汇总时将它们列为历史开发参考，与同轮训练结果分开说明预算和选点差异。

| 已核实参考 | 三seed选中开发r | 相对同seed原锚点的平均Δr | 解释及限制 |
|---|---:|---:|---|
| 原FT+Ccap锚点 | 0.182339940 | — | 原共享模型，所有参考沿用其冻结指标。 |
| [D22 B：当前预测＋历史预测后处理](../../output_query/run01/REPORT.md) | 0.184580904 | +0.002240965 | 三seed均提升；三组均选epoch0，末轮均低于锚点，活动MSE未改善。不能只报道峰值而省略回落。 |
| [D18 P：预测历史对照臂](../../preprojection_memory/run01/report01/THREE_SEED_MEANS.csv) | 0.184379162 | +0.002039222 | 完整P臂保留；20轮FT、旧固定Ccap/前缀和VAL选点不同于D26的固定5轮及新Ccap，不称同预算消融。 |
| [D23 S：无历史的当前预测非线性校准](../../memoryless_output/run01/REPORT.md) | 0.183899636 | +0.001559696 | 三seed均提升、全部60个已训练轮次高于锚点；选中r低于D22 B，活动MSE仍未改善。无历史不构成排除理由。 |

这些是同一反复开发VAL上的描述性候选，不是独立测试证据。各行保留整臂的全部三个seed，不跨方法挑seed拼接；历史机制、相关提升、幅值误差和独立泛化分别作结论。下一步强对照也不能仅战胜较低锚点便略去已存在的更高开发参考，更不能凭此宣称SOTA。

| 方法 | 本次实际核实的一手依据 | 下一项可比性工作 |
|---|---|---|
| Sea-Wave | 已实际核读正式PDF、表7/8及12份固定作者源码，见[完整审计与来源指纹](../../metadata_review/code_review/seawave_protocol_20261106/REPORT.md)。论文Small周期5与公开JSON周期4、FT损失/hop/入口存在差异；原开发test作VAL有明确split说明，不据文件名断言泄漏。 | 后续实际清单证明VAL49人均在TRAIN，故[D27固定为覆盖已见人适配的3个initial＋147个personal FT](../../seawave_seen_reference/run01/COORDINATOR_DECISION.md)。initial群体结果只作未globalFT参考，不冒称完整S2。尚未开始Sea-Wave正式训练；不将personal与共享参数结果混称同适配条件。 |
| DECAF | [原文](https://arxiv.org/html/2602.19395v1)直接研究包络；本地已有[固定公开实现审计](../../metadata_review/code_review/decaf_protocol_20261106/REPORT.md)，包括HappyQuokka、forecaster、动态融合和训练入口。 | 完整方法是候选强对照；现有自定义P-prior不能替代它。代码的teacher验证、普通预测历史、3/3.5秒及500ms对齐差异须落实到同协议，不能直接跑自动test入口或以oracle评分充当EEG-only。 |
| SHINE | [v2原文](https://arxiv.org/html/2602.23960v2)是30秒EEG、包络+10Mel联合目标，SparrKULee分配263/207/192；不同于本项目426/49和3秒局部权限。2026-11-06通过GitHub接口实读[作者仓库](https://github.com/xuxiran/SHINE)，根目录仅README，正文仍称代码待发布。 | 无现成作者代码可直接运行。若后续依论文实现，标明实现者及差异；不将3秒单目标删改版称完整SHINE。广义长窗重建SOTA主张仍需处理该对照缺口。 |
| DPT-Net | [ICASSP2026官方会议记录](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=17226)及出版社提交的[Crossref元数据](https://api.crossref.org/works/10.1109%2FICASSP55912.2026.11464366)相互确认题名、作者、DOI `10.1109/ICASSP55912.2026.11464366` 和页码6796–6800；[实际元数据回执](../../metadata_review/code_review/dptnet_access_20261106/CROSSREF_METADATA.json)已保存。 | 目前未核全文和作者实现。2026-11-06正常IEEE文章入口返回HTTP202但未返回可读正文，Crossref无摘要，仅列出用于similarity-checking的staging链接，未据此推断开放全文。不能编造成绩、结构或排除相关性；后续需要可核验原文或作者实现，不重复用题名代替协议。 |

原文数字只用于定位强方法，不能构造跨协议虚假排行榜。尚未完成Sea-Wave/完整DECAF/SHINE的本项目同协议实测；也尚无充分证据称D26或现有锚点达到SOTA。

执行次序：PUB先实施已下达D26，协调并行审Sea-Wave与已有DECAF实现的最小固定比较，具体结构/训练/评分冻结后再调度，不能为了对照数量扩成庞大超参网格。强对照不以D26历史阳性为前提，也不因CC获胜而弃用CC。该后续对照使用既有合法TRAIN/VAL，不自动重开旧test、外部集或新数据。SOTA最终主张的基准范围、测试用途和所有适配必须单独明确。

本次源码证据：SHINE README Git blob `49203188da940913dcc05f711a454933d0ea1974`；Sea-Wave commit `910138cdac01c58c007459711008ae4991757ff2`，12份文件逐个核对Git blob一致，见审计目录PROVENANCE。源码审计完成不等于训练协议已冻结或实验已完成。
