# SUBTASK — PUB-01-S03_SS01 — 初步 EEG→声音包络解码

| Field | Value |
|---|---|
| Subtask ID | PUB-01-S03_SS01 |
| Parent Node | PUB-01-S03 |
| Parent Task | PUB-01 |
| Title | Initial public EEG envelope decoding |
| Status | COMPLETED |
| Blocking for Parent | YES |
| Patient Data | FORBIDDEN |
| Upstream Subtasks | S02 已有 acoustic/feature comparison PASS_WITH_LIMITATION，保留 HuBERT HOLD |
| Inputs | 2203 上 SparrKULee RDR V3.1 公共发布方 EEG/envelope、S02 frozen split/sample manifest |
| Deliverables | decoder code/config、逐被试/记录 test/null CSV、静态图、报告、中文周报与模型角色表 |
| Acceptance | 冻结10名范围及两种ridge模型的test/null或明确失败均报告；技术复核与独立review pass；阴性合法 |
| Validation | synthetic unit/metric/index tests + split/support检查 + first-subject rerun + separate review pass |
| Go/Hold/Fail | PASS_WITH_LIMITATION允许初步交付；失败/阴性不自动扩范围；不完成S03整体 |
| Files Allowed to Change | scripts/pub_01_s03*、configs/pub_01_s03*、reports/pub_01/s03/**、doc/tasks/PUB-01/**、doc/CURRENT_TASK.md、doc/TASKS.md；远端新增pub_01/s03_envelope_decode_20261011/** |
| Last Updated | 2026-10-11 |

## 1. Objective / Scope
按统筹转达用户本周继续PUB并交实际解码结果的要求，在已有S03公开EEG benchmark内增加方向明确的包络重建基线。只运行本子节点；S03其余neural/retrieval/held-out-subject比较保持未完成。不是语音、语义重建或患者泛化。

## 2. Inputs / Pre-execution gate
主目录旧S02记录陈旧，较新a063工作区只读报告与2203真实文件已核验：83 subjects/659 records、acoustic/log-mel/frozen facebook/wav2vec2-base encoding比较完成。S02 upstream按其已交公开分支阶段通过，不把HuBERT HOLD解释成完成。SparrKULee公开数据已有非商业研究使用许可门（reports/pub_01/s01/AUDIT_REPORT.md）；不请求restricted对象。数据读取、拟合与评估只在2203。现有用户未提交修改保留；不提交/推送。

ID/filename/registry匹配，parent与权限继承、已有资产、范围、split、可判定交付及validation均通过。配置在拟合前写入 configs/pub_01_s03_envelope_decode_20261011.json；不覆盖旧报告或数据。

## 3. Frozen execution
按S02 READY被试ID字典序取前10名，失败不递补；全部所属READY记录。精确复用S02被试内跨source-family train/val/test，不重新分配。Backward ridge/TRF用EEG[t+0,4,...,32]（64Hz，0–500ms）重建envelope[t]，576列；zero-lag用64列。train-only设计列/目标标准化，无PCA。Ridge目标为mean-MSE+alpha L2；alpha=[1e-4,1e-3,1e-2,0.1,1,10]，只用validation record-macro Pearson r选，不合并train+val重拟合。

null独立拟合EEG[t+832+lag]→envelope[t]，不循环不跨记录。四条件统一t∈[0,n−864)，共享target及所有matched/null TRF输入有限掩码。记录每条n与排除数。主要指标为被试test逐record宏平均r/R²、matched−null配对Δr；同时保留pooled指标和train-mean基线。示例预定为首被试首test记录前20秒，不按test成绩筛选。

## 4. Controls / limitations
未知physical synchronization/historical preprocessing revision沿用S02限制，不能通过相关结果声称硬件同步。使用未来最多500ms EEG，属于离线重建，不承诺实时零延迟。13秒错配是描述性null，不是置换分布或显著性检验。每个被试独立训练，不称新被试泛化；不因为阴性改变范围、lag、split或加模型。

## 5. Deliverables / Acceptance / Validation
必要文件：run_summary.json、metrics_per_subject.csv、metrics_per_record.csv、alpha_selection.csv、split/sample/skip manifests、performance/example静态图、REPORT.md、REVIEW.md、WEEKLY_REPORT.md。server_only模型参数不回传。首次被试完整rerun对比表；合成测试核验lag方向/非循环边界、指标与解析ridge；独立review pass重新读取冻结配置、代码、CSV和图核验结论及parent consistency。

## 6. Completion Record
2026-10-11实际完成，Acceptance **PASS_WITH_LIMITATION**；Technical PASS，Parent consistency PASS。

- Actual input：SparrKULee RDR V3.1公共publisher EEG/envelope；精确沿用S02 frozen split/sample，预定sub-001–010，93/93条record可用。0读取跳过、0被试失败；test每条件27条/1,448,987点。
- Code/config：scripts/pub_01_s03_envelope_decode.py、scripts/pub_01_s03_review_decode.py、configs/pub_01_s03_envelope_decode_20261011.json；完整实际命令在reports/pub_01/s03/envelope_decode_20261011/COMMANDS.md。main / HEAD 354eca9为接入Git基线；本轮实现未提交，不将该HEAD冒称代码版本。实际脚本和配置副本留2203运行目录。
- Actual outputs：2203 `pub_01/s03_envelope_decode_20261011/run01`与`reproduce_sub001`；Windows `reports/pub_01/s03/envelope_decode_20261011/`内CSV/JSON/静态图/REPORT/REVIEW/WEEKLY_REPORT及上游轻量报告副本，无信号/参数回传。
- Results：backward TRF matched record-/subject-macro r=0.18191283，R²=0.01780734；13s null r=−0.00328993，R²=−0.02042896。zero-lag matched r=0.07368206、R²=−0.00984629；null r=−0.00373046、R²=−0.01634919。10/10 TRF Δr>0，9/10 matched R²>0；sub-010负R²保留，不改范围。
- Validation：六项合成检查PASS；首被试完整rerun所有审查指标最大差0；独立首test记录直接公式r/R²最大差2.22e−16；冻结范围/精确split/共享支持/alpha只由val选择/表格算术复核PASS；两张实际图已目视审阅。独立review pass由同一agent另行进行，非外部人工签字。
- Changed paths：上述新增代码/config/报告/本任务书；必要增量更新SUBTASKS、S03/S02任务书、父TASK、CURRENT_TASK、TASKS，保留既有用户修改和历史provenance；较新a063只读。
- Failures/deviations：无失败、无scope/lag/alpha/split的结果后修订；没有可用性强制替换。深度temporal CNN未训练（原授权可选）；HuBERT仍HOLD。
- Limits：前10名被试内跨source-family、发布方坐标非hardware-sync证据、future EEG最多500ms离线、单描述性null非显著性检验、低解释方差；不称语音语义重建、新被试或患者泛化。
- Parent effect / stop：本SS01完成；S03完整neural/encoding/retrieval/held-out-subject等仍未完成，S03/PUB-01继续ACTIVE。未进入S04、未生成ART-AUDREP-v1、不提交/推送。已交本周新增周报段落，9月旧结果另列。
- 2026-10-11同节点小补充：根据统筹提供的原始方法来源，实际读取Crosse等2016 mTRF论文及作者stimulus_reconstruction.m；补充方法引用、backward方向及decoder权重解释边界至REPORT/WEEKLY_REPORT/SOURCES/REVIEW。实验代码、冻结配置及数值不变，不新增节点或重跑实验。
