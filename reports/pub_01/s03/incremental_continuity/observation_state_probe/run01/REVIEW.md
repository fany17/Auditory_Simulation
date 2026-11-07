# D11 复核与三级验收

Technical validation：**PASS_WITH_LIMITATIONS**。15个固定岭解、6个无拟合干预、24评分集合实际完成；三seed原Ccap full桥差0，32直接区间与rolling descriptor最大差1.0814e−13，floor/clip/负variance修正0；生产正规方程max7.03e−17。

独立review由已获用户授权的协调chat 01a134e4-2036-7e02-81e9-1289a07606f2完成，实际证据为[方案/源码审阅](COORDINATOR_PLAN_REVIEW.json)、事前冻结[数值scope](COORDINATOR_NUMERICAL_REVIEW_SCOPE.json)、[实际audit](COORDINATOR_NUMERICAL_AUDIT.json)及其[脚本](independent_review/COORDINATOR_NUMERICAL_AUDIT.py)。一次审核、无失败/重跑：26来源hash、551输入size/mtime、475支持、15正规方程、60直接原EEG状态、48缓存预测及24full/warm/cold聚合均通过。独立正规方程max7.03e−17、截距差0、状态cache差max1.135e−7，缓存r/MSE/r3s与聚合差0，cold fallback保持；非载荷全内容hash，不复现原神经数学。audit wall3.276s、CPU20.704s、RSS221003776B，无fit/NN/NPY。

Acceptance validation（本次有界诊断交付）：**PASS_WITH_LIMITATION**。完整逐seed/逐人/逐块/来源描述、全部六primary、secondary Past−Add、既有F full逐seed及SHA、三图与中文逐图解释均齐备，阴性与指标分歧不省略。协调端已实际读完REPORT和三PNG，同意结果并要求精确修正PCA白化措辞，现已落实；没有改变源码/系数/分数。最终署名JSON为 COORDINATOR_RESULT_REVIEW.json。

科学候选判读：**NOT_SUPPORTED / 本轮六条件未满足**。Past−Ccap/Prefix/Person/Older/Neutral均三个seed负；仅Past−Current full全正，warm2/3正。Past−Add、−原F full亦全负。Past MSE改善仍保留，不能称所有指标恶化；静态参考较Ccap略益不能称动态历史机制。此状态不等于关闭SS07或持续goal。

Parent consistency：**PASS**。仍为PUB-01-S03_SS07内原TRAIN/VAL公开EEG观测状态probe，无新parent问题、数据scope、旧test/DTU/患者/STN或私人材料。没有下载或训练新backbone，不改原split、目标、seed、λ、时长、PCA维度或已完成结论；不自动启动新实验或患者handoff。

主run预算实核：[COMPLETED](COMPLETED.json) wall3023.156s<3600、RSS2.61GiB<8GiB；全NPY26270880224B<25GiB，新state45273920B，数值101036740B<128MiB，图表后root约142.46MB<256MiB，free约25.60GB>20GiB。原NN前向/更新0，继承训练/搜索成本非零。唯一PID656537正常结束，源commit5c50719及protocol a238ed63…b5f11冻结不变。

限制：重复known-person/source VAL、原derivative revision/物理同步未知、离线预处理非实时因果；当前3秒与过去30秒估计样本数不同，57024参数相同不等于effective df相同。协方差含信号与其他成分，非已测噪声；没有EA/AEA/ASR复现、h加性/交互独立拆分、独立确认或创新充分性。PCA有V/sqrt(eigenvalue)缩放；64通道EEG没有协方差逆变换/空间白化。

归档：本轮三图HTML run_id=20261101_pub01_s03_ss07_observation_state_probe；实际生成和副本/说明/链接检查以归档脚本输出为准。PR6维持Draft，科学goal ACTIVE，不以技术PASS宣称目标完成。
