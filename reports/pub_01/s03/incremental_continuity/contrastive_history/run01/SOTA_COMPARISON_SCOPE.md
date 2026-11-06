# SOTA性能目标与尚缺的强方法比较

2026-11-06，响应用户『有性能提升就行，核心目标是SOTA』。这是协调已核实的对照范围，不是新训练结果或全领域穷尽综述。当前任务是 EEG-only 连续语音包络重建；检索 Top-1、AAD 分类、MEG/语音文本指标不能替代本任务性能。

旧强 Transformer 的完整训练、FT 与 Ccap 是现有可核查锚点，尚不等于本任务所有公开强方法。D26在做三个完整方法的性能比较，历史机制仅为次级解释。此前反复开发的VAL与已经看过的test/外部结果都必须明确标注；不同数据划分的论文r不能直接与本项目开发r排名。

| 方法 | 本次实际核实的一手依据 | 下一项可比性工作 |
|---|---|---|
| Sea-Wave | [2024论文](https://doi.org/10.1109/OJSP.2024.3378594)直接研究ICASSP包络任务；[作者仓库](https://github.com/LiuyinYang1101/Sea-Wave)实际有models/train/finetune/utils及LICENSE。论文区分subject-independent与subject-specific微调。 | 优先审作者实际模型、配置、训练/评分路径，挑固定subject-independent强配置补同协议对照；不可把个体微调成绩当纯跨人结果。当前只核了公开目录与论文，尚未运行/逐源码审计。 |
| DECAF | [原文](https://arxiv.org/html/2602.19395v1)直接研究包络；本地已有[固定公开实现审计](../../metadata_review/code_review/decaf_protocol_20261106/REPORT.md)，包括HappyQuokka、forecaster、动态融合和训练入口。 | 完整方法是候选强对照；现有自定义P-prior不能替代它。代码的teacher验证、普通预测历史、3/3.5秒及500ms对齐差异须落实到同协议，不能直接跑自动test入口或以oracle评分充当EEG-only。 |
| SHINE | [v2原文](https://arxiv.org/html/2602.23960v2)是30秒EEG、包络+10Mel联合目标，SparrKULee分配263/207/192；不同于本项目426/49和3秒局部权限。2026-11-06通过GitHub接口实读[作者仓库](https://github.com/xuxiran/SHINE)，根目录仅README，正文仍称代码待发布。 | 无现成作者代码可直接运行。若后续依论文实现，标明实现者及差异；不将3秒单目标删改版称完整SHINE。广义长窗重建SOTA主张仍需处理该对照缺口。 |
| DPT-Net | [ICASSP2026官方会议记录](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=17226)确认论文与作者。 | 目前未核全文和作者实现，不能编造其成绩或排除其相关性；继续补一手来源，不把搜索标题当实验协议。 |

原文数字只用于定位强方法，不能构造跨协议虚假排行榜。尚未完成Sea-Wave/完整DECAF/SHINE的本项目同协议实测；也尚无充分证据称D26或现有锚点达到SOTA。

执行次序：PUB先实施已下达D26，协调并行审Sea-Wave与已有DECAF实现的最小固定比较，具体结构/训练/评分冻结后再调度，不能为了对照数量扩成庞大超参网格。强对照不以D26历史阳性为前提，也不因CC获胜而弃用CC。该后续对照使用既有合法TRAIN/VAL，不自动重开旧test、外部集或新数据。SOTA最终主张的基准范围、测试用途和所有适配必须单独明确。

本次目录证据：SHINE README Git blob `49203188da940913dcc05f711a454933d0ea1974`；Sea-Wave models tree `62d06cd335a2de1682dccecdc440955723bb2492`、train tree `1d10c54ab4994d855be29526878391e1b7efa9be`。这里只记录实际读取版本，尚未将这些目录当成执行冻结源。
