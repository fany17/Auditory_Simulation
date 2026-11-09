# SS07 现有公开EEG暴露核查（metadata-only，2026-11-01）

目前没有得到足以宣称论文级独立确认集的现有资产。原SS02的seen/heldout及SS06 audit已有开发暴露；重新切分、改变采样率或换模型不能消除暴露。下述核查没有新增神经网络预测或读取旧准确率用于选择。

## 范围与反证

既有SparrKULee清理索引666条；S02原split为659 READY、7 UNASSIGNED_SPLIT。SS02范围619条（426train/49val/69seen/75heldout），范围外47条不能全部当新holdout：其中40条已属旧S02 val/test；7条仅表示没有进入该S02训练划分。

7条的主体为sub-017四条及sub-077三条，详见EXPOSURE_RECORDS.csv。2026-11-01在2203复读pub_01/s02_acoustic_baseline_20260924/split_manifest.csv：均UNASSIGNED、selected=False，原因subject_has_2_source_families_lt_3。feature_compare及resource分支的split_manifest.csv则为UNASSIGNED但selected=True，因此不能只凭一个选择字段断言从未拟合/选择。该两分支所有实际normalization/PCA/SSL fit成员的全链证明尚缺；这项未知保留。

sub-077有明确相反证据：旧HDPS 100Hz M中podcast_34为train（166候选）、audiobook_14_1为val（155）、audiobook_14_2为test（182），均selected=True。证据为本仓库reports/pub_01/s03/strong_baselines/evidence/continuation01/author_split_input_coverage.csv及available_20261015/hdps_m/candidate_trace.csv。不能因其未进入64Hz Transformer就称独立，也不需要读取其准确率来判定暴露。

sub-017四条在上述两个HDPS表无对应行，S02原声学sample_manifest及旧线性/扩展CNN所检查的split也无其拟合成员，但这仍是已检查范围内的缺席证据，不是所有历史程序从未使用的证明。全部已进入S01 QC/时序/描述性探索；不称原始数据从未查看。当前SS07的426条train-only归一化及新pretrain明确不含该subject，不能据此反推其他历史统计也不含它。

## 质量与可用范围

admission_inventory.csv中sub-017 run01/02/04为INCLUDE/PASS_WITH_LIMITATION；run05为HOLD/LARGE_TAIL_HOLD，common-prefix48747点。run05与发布方另行排除的run03不同，不能混淆。后续用户允许探索/共同区间分析不等于其正式补充评估已自动准入；需依据实际尾差和后续记录作单独决定，不阻塞主SS07。

即使sub-017最终证实未参与拟合/选择，也仅一个subject、两个source families，部分刺激已在既有研究出现；最多作为锁定方法后的独立补充个案，不能冒充外部多被试确认。任何补充载荷评估先冻结清单、排除理由、主/敏感性范围、五seed与统计；不改原train/norm，不混入主seen/heldout加权均值。若发现历史无监督fit包含它，明确标记transductive，不能改叫无暴露。

## 当前决定

sub-077三条判定EXPOSED；sub-017四条判定INDEPENDENCE_UNPROVEN，其中run05另有QUALITY_REVIEW。未产生这7条新模型分数，未将其添加训练或主测试。继续全量train/val；真正外部确认需单独批准最小新增公开数据范围，方案见EXTERNAL_CONFIRMATION_PROPOSAL.md。
