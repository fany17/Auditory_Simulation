# SS07 round01实际结果审阅

本轮预设实验和证据交付完成，研究成功准则未达到。持续goal不标complete，下一阶段尚未拟合。

| 层级 | 决定 | 实际依据与限制 |
|---|---|---|
| Technical | PASS | 54screen、37唯一full、五完整P/FT；47 checkpoint角色/14协议/106成本条目冻结；五seed×2最终权重VAL桥差0；掩码预测前锁定；全部五seed统一评分与聚合完成，无失败seed丢弃 |
| 实际独立复核 | PASS | raw v1的360项代表性重放最大差1.82e−12；metric v2共583129检查、165373内部块、43200外部trial配对；绑定同一manifest/READY/协议；V2综合门通过 |
| 本轮实验交付 | PASS_WITH_LIMITATION | 五真实结果图及四训练/VAL图全部实际查看，逐图中文解释、原始轻量统计、成本、阴性与初始化边界齐备；协调另行实际核对训练及最终五图 |
| 超越原P及历史增益 | NOT_ESTABLISHED | 内部M4−P −.000621、DTU +.000149，M7区间均跨0；matched current也跨0；内部静态容量/对齐/FT优于M4的预设区间为负 |
| 机制/投稿就绪 | NO_GO（当前证据） | content-blind只有未校正探索性95%区间为正，其他诊断无稳定区间优势；初始化/联合优化/3s–60s关系未分离；新意及新确认性证据缺口保留 |
| Parent consistency | PASS | 仅公开EEG/批准DTU，患者/STN FORBIDDEN；计算/权重/预测留2203；本地为轻量结果，旧SS02/SS06/原资料不改，无新下载/远端Git/PR合并/下游启动 |

内部比较是POST-HOC INTERNAL EXPLORATORY；DTU是固定输入适配、无目标域模型拟合，刺激/房间独立性未证。M7主比较、M5三seed探索、六history诊断和两个域不具联合FWER保证，近似区间亦非有限样本覆盖保证。区间跨0不证明等价；外部M4−FT为正不能替代M4−P。

当前round的head与残差从相同随机head起点联合训练，未测试“在训练完成P head上接入历史”。FT是共同更新encoder/head的强控制，其源域改善没有形成跨域一致优势；现有结果不足以将阴性单独归因于过拟合或历史信息不存在。

raw复核仅预定3条Sparr/6条DTU、五seed八main，复用原NN forward；history神经干预未重新raw重放，全部保存预测指标另行复算。R²/MSE的缩放数值门在测试前由合法VAL确定，本轮没有分数引导的阈值、模型或掩码改变。

主要入口：[完整中文报告](REPORT.md)、[数值来源](REPORT_NUMERICS.json)、[实际V2综合复核](INDEPENDENT_REVIEW_COMPLETED_V2.json)、[checkpoint清单](FINAL_CHECKPOINT_MANIFEST.json)、[图表来源](final_presentation01/PRESENTATION_MANIFEST.json)。后续只准备源域train/val机制方案，等待审阅；当前不运行新拟合，也不把本轮收束等同整个goal完成。
