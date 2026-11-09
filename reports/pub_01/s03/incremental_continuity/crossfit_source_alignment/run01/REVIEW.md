# D25 独立验收

- **Technical PASS_WITH_LIMITATIONS**。实际工程核对 12 份固定权重的 seed、epoch、role 与身份，在原 VAL 上验证 Hann 支持、共享 mask 和按记录平均评分的规则。正式门包含 105 个源码及 JSON、12 份固定权重、18 个 FT/READOUT 文件和 54 份 VAL 输入属性。[独立路由](COORDINATOR_ROUTING_REVIEW.json) · [正式源审核](COORDINATOR_EXECUTION_REVIEW.json) · [工程](ENGINEERING_COMPLETED.json) · [冻结门](FORMAL_GATE.json)
- **Acceptance PASS_WITH_LIMITATION**。独立重算全部 20,970 个块、1,335 个逐记录路由和 36 个分层/总体汇总，并校验旧同 epoch 参照和生产实际计数。六个固定对照的相对损失均缓解，同时 B 均低于同折锚点、S 与原全 TRAIN 锚点。该结果用于来源一致性诊断，连续性收益门未通过。[独立数值审核](COORDINATOR_RESULT_AUDIT.json)
- **Parent consistency PASS**。分析使用原公开 VAL49 和冻结权重。正式与工程均无新增拟合或统计拟合，已训练 epoch 0 和 19 全部保留。此输出历史后处理支线封存，SS07 总体仍为 ACTIVE。[完成回执](COMPLETED.json)
