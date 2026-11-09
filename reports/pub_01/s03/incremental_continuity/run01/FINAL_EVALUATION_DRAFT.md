# SS07 最终内部评估与机制统计规则说明（未执行test）

2026-11-01。协调已科学审阅并按下述澄清冻结独立不可变EVALUATION_RULES.json；历史文件名保留。规则冻结不等于checkpoint/target-mask/inference child已具备，所有训练结束后还须锁定后者。这里不授权当前读取test载荷或成绩。原test已暴露，所有结果均POST-HOC / INTERNAL EXPLORATORY；区间和多seed不能恢复独立性，不以本轮结果证明投稿级外部泛化。

## 锁定顺序与模型

全部54 screen、15 family full、FULL_PRIMARY_SELECTION、主模型/匹配controls五seed、原P五seed、强FT五seed与EMA_SELECTION完成并核验后，锁定每个checkpoint及全部source/config、固定有效块和供体清单、统计随机种子/contrast清单，再进行一次预定主评估。FT采用原P同seed best encoder/head，允许epoch−1 incumbent；额外预算及validation pass独立列示。EMA只使用三个预设seed选择全局tau，同tau用于五seed。最后两个seed不重新选择主模型/宽度/tau。

保存5family×3seed完整结果，不把screen controls当新候选。主模型及其same-budget current/static/output-prior共五seed，M3另current-teacher、M4另static alignment、M5另prior-only。原P、P+EMA、P+FT分别作强锚点，不能因其中某个分数更高而事后改主比较对象或隐藏失败。

## 坐标、mask与endpoint

保持native零lag、3s窗口/1.5s hop、原EEG train-only统计、完整record Hann融合、原60s非重叠段、covered>1e−8及不少于3800有效点；target非有限/常数及几何不足的块在任何模型预测前固定。预测非有限/常数判该condition失败，不能按模型删块。记录首尾与无历史头两窗照常评分。信号/latent/融合预测全留2203，本地仅静态汇总。

逐seed逐块计算原始Pearson r；先按全部有效块平均到person，再seen/heldout等人权重；总体为2/3 seen +1/3 heldout，保持原定义，不以Fisher z替换主endpoint。保存seed×person×record×story×source-family×块的轻量r和覆盖计数，不能把60s块或seed当独立被试。source-family由冻结S02 source-recording映射join，不能默认等于story；缺失映射先报错。

## 预设contrast与区间（请协调审阅）

主contrast为固定primary−原P，另primary−current、−static、−output-prior、−P+EMA、−P+FT，共6项；按family确定的teacher/static-alignment/prior-only最多再1项。M在最终primary child确定时由上述规则自动生成，不看test决定。每项均报告五seed配对差、seen/heldout及原weighted差，不能只给最大seed。使用20,000次配对交叉seed/person bootstrap（随机种子20261101）：每次抽一组seed索引、seen一组person索引、heldout一组person索引，同一person抽样跨所有seed/模型/contrast共享；person不是嵌套在seed内，不能各seed独立抽人后当五倍样本。主要推断endpoint只有原weighted r，固定M仅作用于此endpoint的contrast族。报告95%区间及“Bonferroni调整的近似bootstrap区间”；不保证有限样本覆盖，不作确认性p值。seen/heldout为预设支持性描述/95%探索区间，不能声称M同时覆盖三个endpoint；若另声称三endpoint联合覆盖需3M，本轮不作该声称。

共享刺激不能靠person bootstrap解决。另作明确标注的crossed sensitivity：person及source-family各用独立正Exponential(1)权重，family权重跨seen/heldout共享，同一个block的所有seed/模型共享权重；先在person内按其block×family权重归一，再用person权重汇总，保持2/3+1/3。正权重避免少量共享故事下bootstrap使大量person无有效刺激的问题。该区间为交叉权重敏感性，非已证明校准的频率学置信区间；同时逐family、leave-one-family-out展示。样本刺激数少仍是泛化限制，不以seed数补足。

全部五family×3seed对P的比较单列探索性表，预设M=5近似调整区间，不与已选主模型五seed的主contrast混充独立复现，不暗示两比较族联合FWER。历史干预在可用的预定paired subset内分别比较true−intervention，标注全部eligible人数/记录/块，不能将供体不可用的不同覆盖直接与全主成绩相减。外部数据已批准仅DTU两文件，另冻其dataset-specific endpoint/split，不混入本表。

## 历史干预与支持检验

固定主模型每个当前z/encoder/head和评分坐标，逐窗重建true、reset、训练均值内容盲、同人错source-family、同story错person、固定打乱顺序、older-outside-fused-support。逐窗操作，不能只改60s第一状态。供体从已授权公开619-record池按metadata筛选，同人错内容须不同source-family，同story错人须subject不同；优先满足供体window数≥当前record，按多余window最少、record词典序定唯一供体，截取同local时间长度。没有供体则预先标记该干预不适用，不回退挑好分供体，不改主endpoint覆盖。跨phase/跨person供体仅离线分布干预，不是可部署预测条件。

内容盲每窗输入固定train-only observation均值，保留历史更新次数及record-local时间；不等同每步reset，不能将其性能解释为生理机制。打乱使用固定随机种子20261102、record词典序确定permutation，同record保持相同观测多重集；可能含较晚观测，明确只作离线次序分布诊断，不能称严格因果可用模型。

metadata-only冻结SOURCE_FAMILY_JOIN.json已逐一join619条native记录到S02真正共享音频family（不是每人独有EEG source_recording）；DONOR_METADATA_PLAN.json在模型输出前为144条test记录固定唯一donor、几何块起点与缺失原因，同人错family119条可用、同story错人144条可用。true与干预还须共享后来按预设target/coverage规则冻结的最终eligible keys，未适用记录不回退或重选。STATISTICS_ENGINEERING_CHECK.json合成验证交叉抽样手算误差2.78e−17、constant/zero保持和nonfinite拒绝；不是科研结果。

older-outside-fused-support用ctx_before(i−1)，最后观测源i−3，原始支持结束不晚于当前预测实际Hann融合baseline输入并集起点。代码按每窗实际nonzero Hann及首尾/coverage检验；该条件同时改变recency与奇偶流相位，不能把差值叫纯delay效应或唯一删除重叠信息的因果效应。wrong-history变化也是分布外干预，若收益下降只能支持对历史分布敏感，需要结合训练的current/static/output-prior等控制判断。

只有稳定战胜原P及强对照、与容量/平滑/额外成本区分、内容/时间及支持诊断一致，才支持进一步方法研究；本内部数据仍不单独满足ICML/NeurIPS级创新/外部证据要求。阴性和未定结论照实保留，不根据test加宽度、换tau、改split或补seed。
