# ds004408：整体gain适配后的固定读出外部迁移

完整结果与五图中文解释见[结果报告](report01/REPORT.md)；实际数据内容、来源/单位异常及固定适配见[预处理报告](PREPROCESSING_REPORT.md)。

全380记录、19人、988块及三seed均完成，独立全量数值复核通过。Ccap−P_FT* Δr=+0.00804082，family95近似CI[0.00511387,0.01097443]、Holm2 p=0.0000534058，三seed全正、18/19人改善。F−W Δr=+0.00015187，CI[−0.00020287,0.00048958]、p=0.345627，seed2正1负；F单窗退化与融合MSE较W更差均保留。

支持当前窗口静态读出的固定条件迁移增益，不能写连续历史机制成立。全record未来EEG参与380次无标签gain统计、物理单位未验证、模板空间映射及零偏移有来源假设、同一书籍叙述与预训练接触未知等限制详见正文。无新学习模型拟合/权重/旧test或DTU新推理/患者数据。

独立证据：[输入/掩码](COORDINATOR_FINAL_INPUT_AUDIT.json)、[4560记录/11856块与统计复算](COORDINATOR_INDEPENDENT_NUMERICAL_REVIEW.json)、[成本含审核](report01/COST_LEDGER_WITH_REVIEW.json)、[完成后资源](FINAL_RESOURCE_SNAPSHOT.json)。当前回合不等于SS07科学goal完成；不自动追加模型/窗口/数据或合并PR。
