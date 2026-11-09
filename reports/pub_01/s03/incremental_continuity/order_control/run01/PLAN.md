# D14：唯一干预为逐epoch全TRAIN窗口随机重排

2026-11-02，归属PUB-01-S03_SS07现有节点；D13封存701be5a后，协调在既有持续目标授权内审定此具体优化顺序对照，先完成源/协议及工程核对后由协调审阅实际冻结结果。不是新架构或方法创新实验。患者FORBIDDEN；旧D12/D13/D9、test/DTU/ds004408全冻结。

## 问题与唯一改变

D13固定last19显示TRAIN物理MSE升高/r下降，旧在线轨迹loss却下降；point-weighted whole-record仍恶化而最后record改善，见 `error_decomposition/run01/coordinator_review/ORDER_HYPOTHESIS_EVIDENCE.json`。顺序/尾部优化偏置是待检假说，尚未证实因果。

保持D12三seed1024/2026/31415、H_full/C_match/P_prior、全部模型参数/原component RNG初始化、原FT+Ccap/归一化/目标、Adam lr.001/weight_decay0、20epoch、原global64/尾38、原252454窗含3834fallback/248620有效、同梯度finite检查与后端设置。唯一改变为每epoch完整global窗口索引的均匀随机排列；不是record shuffle、不是更换loss/LR/容量、不是依据评分挑顺序。

排列来自独立NumPy `default_rng(SeedSequence([seed,20261102,epoch]))`；同seed/epoch三臂精确共用同排列，其60条实际digest在结果前冻结，独立RNG不消耗torch初始化。每epoch验证双射、不漏窗不重复，记录first/last16索引与fallback计数，无排列数组落盘。当前/过去输入始终根据原索引提取，history=(i−5,i−3)、older=(i−9,i−7)，同record，严格过去；batch邻接绝不视作历史。原i<9仍保留在loss分母、增量为零。不是时间逆序训练或跨record隐藏状态续接；原网络逐窗重置的两token GRU不改。

## 固定评价与matched旧参考

每fit 3945更新/epoch、78900更新，总710100。VAL每epoch0…19使用同冻结752块/49人、同strict >及incumbent−1/最早tie规则；selected VAL重放一次，H selected older一次，不新增模型/超参选择。所有评分、record/person/区域保留；常数/非有限不按成绩排除。

固定参数离线TRAIN在epoch−1/0/4/9/19记录同252454×192点未加权windowMSE及原6104完整60s块/70人r/MSE；−1直接锚点0NN，四个非零阶段各一次前向。TRAIN目标原record/cache float32再float64，VAL原连续float64，继承D13定义。在线每batch不同参数loss单独标注trajectory，不能混称离线loss。所有数据/权重/科学计算只在2203。

新增必要matched参考：原D12九个last19各仅一次TRAIN回放，计算同固定参数windowMSE和物理r/MSE并核D13一致，不重放旧全部epoch、不作旧VAL/test/外部评分或改写历史。新模型正式初始化重新构造，不继承旧last权重。

## 精确预算与文件

| 残差网络前向用途 | 窗口数 |
|---|---:|
| 新九fit：20epoch TRAIN更新 | 44,751,600 |
| 每epoch VAL | 5,410,620 |
| 四次固定非零epoch离线TRAIN | 8,950,320 |
| selected VAL | 270,531 |
| H selected older VAL | 90,177 |
| 新模型合计 | 59,473,248 |
| 旧九last19仅TRAIN参考 | 2,237,580 |
| 正式总计 | **61,710,828** |

每C/P为6,846,739，H为6,876,798。工程前向独立另列上限16,384窗，0工程optimizer更新/fit；新encoder/head前向0，无新学习统计/归一化/数据/权重。九新fit/710100更新如实计账。

server新ROOT `/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_order_control_20261102`，本地轻量报告 `reports/pub_01/s03/incremental_continuity/order_control/run01`；源码 `pub_01_s03_ss07_order_control_{core,prepare,train,queue}.py`。不触碰旧source/outputs、不建server Git、不回传信号/权重/数组。

最多两张实际compute PID空且free≥4GiB的GPU；每worker RSS8GiB/GPU allocated2GiB，正式总queue墙钟6h，新root128MiB，两盘各20GiB。无新历史/梯度/预测磁盘array；只last/best checkpoint在server保存，结构化离线block/record/person指标及轻量日志，预计root约80–110MiB（估计，实测guard）。预算/失败停止后不自动放宽或retry，核UID/PID/start_ticks仅停止自有worker；失败和原source保留。

## 工程、冻结与验收

先核D13实际protocol/source/input身份，重核九last缓存SHA/stat；冻结新source/PLAN、60排列digest与budget。工程以seed1024真实TRAIN及冻结非零D12 checkpoint：contiguous first64/tail38 gather tensor与旧batch精确一致；同一实际混record batch重排前后输入、非零输出、标量目标和全参数导数等价；全record metadata的四lags严格过去/不越界；60排列双射及torch RNG不变、所有seed/arm初始化重建一致。工程权重不保存/不拟合，实际NN窗另计。正式模型均干净重建。

实际协议/工程/源经协调独立review通过，写 `COORDINATOR_EXECUTION_REVIEW.json` 后才启动九fit。完成后Technical核输出/counts/全epoch permutation、离线同定义matched对照和独立数值；Acceptance明确PASS_WITH_LIMITATION/REWORK等，Parent consistency保持PUB-01。即使优化恢复也不自动证明历史超越强锚点、独立泛化或创新；阴性合法，不能据此改lr/loss再试。当前只是具体方案/源准备，未宣称运行完成。
