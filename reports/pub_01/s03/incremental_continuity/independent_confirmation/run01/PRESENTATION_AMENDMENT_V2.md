# 结果呈现补充：共同纵轴和可追溯成本

在实际新数据 NN 为 0 时，根据协调独立预审新增 `pub_01_s03_ss07_openneuro_report_v2.py` 与 `PRESENTATION_PROTOCOL_V2.json`；原 v1 源码和协议保留。仅改变最终呈现，不改科学预处理、角色、参数、掩码、统计、运行队列或实际预测。

- 事先选定首条 canonical record、seed2026、前60秒。四面板使用相同纵轴，由目标及四个预测的全部有限值共同最小/最大值加3%留白确定；不裁极值、不重标幅度。实际范围写入 `EXAMPLE_AXES.json`。
- 成本账分列 header/预取首轮接续及v2、音频包络、EEG预处理v1接续及v2、三seed实际推理和评分。时间戳差仅表示启动到中断经过时间；v2中的 `total_elapsed_since_v1_seconds` 为包围区间，不与分阶段再相加。
- 单列本轮三seed继承的完整 pretraining/head/FT 数字、D8读出准备和27次岭拟合的成本清单、D9六次物理目标求解及worker成本；保留各manifest路径与SHA。此账不冒充全部SS07历史搜索成本。并行任务的wall time不相加为总历时。
- 未记录CPU/RSS/GPU等字段显式为null/缺失；XLSR上游预训练及原TRAIN统计准备成本未测量，不写成零。每record gain是新增无标签输入统计，与新学习模型拟合次数区分。
- 首次v2候选在复查时补上外部worker实际使用的 `peak_GPU_allocated_bytes` 字段映射及窗口计数；修改前源/协议以 `*_pre_review01.py` / `PRESENTATION_PROTOCOL_V2_PRE_REVIEW01.json` 保留。补正仍在真实新数据NN为0时冻结，不影响实际运行来源。

协调会话单独提供 `pub_01_s03_ss07_openneuro_independent_review.py`；合成自检完成不代表实际结果review已完成，执行者不调用或修改该源。三seed实际结果完成后由协调独立执行，最终图表及中文结论仍须整体审阅。

资源限制说明：现有 queue 的8小时上限作用于轮询守卫；同步的freeze/statistics子进程没有独立timeout。此限制如实记录，不重启已在等待的队列。
