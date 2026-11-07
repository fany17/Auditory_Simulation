# D4完成审阅

本轮54/54预定岭解实际完成，三seed、六臂、固定三λ，无科学运行失败/重跑、无NN新训练或test/DTU推理。不是仅合成检查或排队。正式源码0182fb055b7d684026ad3a21d13d715e70e7cd952831b42c436bcd6a4f54a410保持不变。

- **Technical：PASS_WITH_LIMITATIONS。** 协调独立在2203读取原代码/协议/全部54解及系数，核对17原来源与48补充target size/mtime指纹；全部系数有限且形状/参数数正确，54格共同49person/752block/30500window及block/subject/sample集合完全一致。独立从block→person→macro及record加权3秒重聚合最大差1.11e−16，全部共享λ/P incumbent规则正确，原P60s差0、正仿射不变性≤2.78e−16，实际三张PNG已由双方查看。
- **Acceptance：PASS_WITH_LIMITATION（本轮有界证据交付）。** Ccap开发VAL均r=.176897、对P三个seed均正；CF对C/Ccap三个seed均负，CF48相对summary也未提供增益。结论是CURRENT_READOUT_VALIDATION_GAIN; NO_ADDITIONAL_HISTORY_BENEFIT_ESTABLISHED，不是新方法成立、外部确认或整个goal通过。
- **Parent consistency：PASS。** 仍在SS07公开EEG连续增量/强current控制范围，无患者/STN、额外载荷/权重、测试调参、其他节点、远端Git或PR合并；数据与权重/系数/预测仅2203，原历史证据不改。

原样独立[代码review](COORDINATOR_CODE_REVIEW.json)和[结果review](COORDINATOR_RESULT_REVIEW.json)保留；后者SHA aa67f4233f7f1264784ba219c05ad651334471a245feb9ff576dc5d63b5dd87f。独立检查复用保存的指标/系数和原神经模块，没有另做NN raw重放；也不是外部实验室复现。现有验证已覆盖实际修改及风险，不扩大复核。

主要限制：反复使用source VAL选参和报告；相关刺激/被试重复；加性线性残差及固定3–0/6–3秒历史，不覆盖历史×当前非线性交互；参数数不是独立信息或有效自由度。强current读出改变容量/函数/目标，不能与warm joint跨轮直接当等预算胜出。

target载荷指纹/线程及GPU/CPU观察在运行中补充，非run前冻结。补充timing初始误写保留并已澄清；首观察实际在第二seed、18解之后。GPU只读观测500MiB，内置peakRSS约6.31GiB；独立连续全过程GPU上限证明、纯solve/score分项CPU时间未测，不伪造、不为成本字段重跑。

完整中文[REPORT](REPORT.md)覆盖三图、全λ与共享选择、逐seed/逐人/逐块、完整成本及解释边界。SS07、S03、PUB和科学goal均ACTIVE；D4交付完成不自动扩大本矩阵。下一步仅提出单独冻结的最小映射交互机制设计，先由协调审定，保持本轮强current Ccap作为锚点。
