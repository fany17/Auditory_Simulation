# SS07 初始15个full：阴性结果的机制边界

本核查只读取2203已有train/val metadata与冻结源码，无新模型、拟合、预算、预测评分或test/DTU成绩。当前事实是：原三seed的15个候选在VAL60s/person上均低于同seed原P；这尚不能区分历史信息贡献、初始化/联合优化限制与窗口目标/连续评分差异。

本轮检验的是**每seed共用同一个本轮新预训练encoder，冻结其表征，从相同随机head起点联合训练head与残差**，不是“在训练完成的P head上接入历史”。五family没有各训一个encoder。`make`先重置seed，`NativeResidual`先构建与P相同的257参数head，再构建残差；残差最终投影置零使初始函数等于该随机head，未加载已训练P head。M3另含预设辅助目标。训练条件中只有FT显式加载已训练P encoder/head并继续优化；EMA是已训练P预测的后处理，没有head初始化或拟合。

因此，“相同强encoder/读出类与随机起点下联合训练残差”与“从已训练强head继续增量学习”是两个问题。当前阴性证据约束前者，不能写成后者已被否定。

[15行配对metadata表](INITIAL15_PAIRED_VALIDATION_METADATA.csv)并列全部候选与同seed原P的best3s VAL、选中checkpoint的VAL60s/person、两项差值、0-based选中epoch、实际epoch/update、停止原因与共同预算。候选均pat10停止（15–71轮），P三seed均250轮预算上限、选中249；两者使用相同max250/pat10规则，实际优化量不同。表保留原日志，不新增逐checkpoint评分或重选。

| 已确认事实 | 可证伪假说（未确立） | 当前无法区分的机制/证据缺口 |
|---|---|---|
| 同seed P与候选共享同一冻结encoder和随机head初始化；候选head/残差共同更新，未从训完P head出发。 | 若随机head/残差联合优化起点是主要限制，最终表现应对起点有稳定依赖，而不只取决于历史内容。 | 无已训练P head起点的历史残差条件；FT同时更新encoder且没有历史模块，不能隔离head起点效应。不能证明换起点会改善。 |
| 15个候选均有VAL60s负差；3s仅M4/31415高于P，其余14个更低。容量/当前输入/输出prior/静态对齐控制与真实历史干预仍待完整评估。 | 若真历史对本轮模块无有效贡献，固定当前EEG的既定配对诊断不应显示稳定的真历史内容/时间优势。 | 未获贡献不等于信息不存在；有限模块、表示、时间支持及优化失败也可造成阴性。尚不能把现象归因于“历史没用”。 |
| M4/31415的3s为.1522685305，P为.1519646136（差+.0003039169）；相应60s为.1656389903与.1663503571（差−.0007113668）。 | 窗口Pearson与Hann连续融合/60s→人聚合的不同约束和权重，可能导致选择目标与终点评分不一致。 | 这例直接反驳“3s更高就保证60s更高”，但不证明目标差异造成全部阴性。没有所有epoch的60s评分，不能指定另一最优checkpoint，也未区分融合、窗口幅度/偏置或人/块权重的作用。 |
| 部分在线train曲线继续升高而VAL持平/下降；全部候选按预设pat10停止。在线train均值随轮内权重变化，且样本池不同。 | 局部过拟合或验证泛化不良与部分轨迹相容。 | 在线train与固定VAL的差不是同checkpoint泛化误差；不能仅由该图证明过拟合、训练不够、信息不足或全局收敛。P上限停止也不是最优性证明。 |

源码依据：[P head重置随机seed并训练](https://github.com/fany17/Auditory_Simulation/blob/00cb514103b571a63347408bd27836566ac9a875/scripts/pub_01_s03_ss07_baseline.py#L202)、[残差head先初始化及最终投影归零](https://github.com/fany17/Auditory_Simulation/blob/00cb514103b571a63347408bd27836566ac9a875/scripts/pub_01_s03_ss07_incremental.py#L105)、[full make及新Adam](https://github.com/fany17/Auditory_Simulation/blob/00cb514103b571a63347408bd27836566ac9a875/scripts/pub_01_s03_ss07_train_full_v2.py#L74)、[FT加载原P head](https://github.com/fany17/Auditory_Simulation/blob/00cb514103b571a63347408bd27836566ac9a875/scripts/pub_01_s03_ss07_finetune.py#L75)。既有INCREMENTAL_ENGINEERING_CHECK_v2仅在合成输入验证相同seed初始化恒等，不是机制证据。

CSV来源为`full/<condition>_seed<seed>/{completed,history,resolved_config}.json`、`baseline/head_seed<seed>/{completed,history,cache_manifest}.json`与冻结EMA_SELECTION的tau0三seed既有值；全部来源encoder一致、最早3s选点/预算/停止原因已核对。FULL_PRIMARY_SELECTION仍为d43da5b521af0cabfed6a2a6fc4ce968ae39f20824411d07c305fb1fa08b69d2。仅补解释边界，不设计或启动下一轮，不改变既定评估，SS07/goal仍ACTIVE。
