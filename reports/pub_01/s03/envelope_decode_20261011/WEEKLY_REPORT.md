# 听觉公共模型周报段落（2026-10-05至2026-10-11）

**本周新增实际工作：** 在server2203完成公开SparrKULee的初步EEG→声音包络解码，冻结被试ID排序前10名、93条记录，复用已有被试内跨source-family训练/验证/测试划分。实现backward ridge/TRF（64通道、0–500ms EEG时滞）及零时滞ridge，均使用train-only标准化和validation-only选正则化；四条件采用相同测试区间，并独立拟合非循环13秒时间错配对照。Backward TRF的test逐记录/被试宏平均r=0.1819、R²=0.0178，错配r=−0.0033；10名相关性均高于各自错配，9名R²为正，sub-010为负。零时滞ridge为r=0.0737、R²=−0.0098。已交逐被试/记录CSV、静态图、命令及配置；合成测试、首被试重跑与独立指标复核通过，验收PASS_WITH_LIMITATION。

**解释与模型可用性：** Backward TRF已验证可作为公开EEG离线声音包络解码基线，预测关系初步可见，但幅度重建弱，平均R²仅约0.018；不能表述为可靠语音/语义恢复、新被试泛化或STN结果。零时滞ridge可作简单对照，其负R²保留。轻量temporal CNN EEG decoder仍待比较，本周未训练；没有为了凑模型数量扩范围。

**9月旧结果，本周仅核对而非新增：** 9月25日S02已有83名/659条公开记录的声音特征→EEG encoding比较（envelope/onset、log-mel、冻结facebook/wav2vec2-base）；wav2vec2按validation选层后宏平均test mean-channel r=0.02502、R²=−0.00153。它是声音编码器和encoding comparator，不是EEG decoder，且与本周单目标包络解码的范围/指标不同，不能直接比较r大小作为“提升”。HuBERT identity/cache分支继续HOLD。本周校正主目录陈旧的S02状态并登记S03_SS01；完整S03/PUB-01保持ACTIVE，未进入颅内迁移S04或患者侧任务。

证据：[完整报告](REPORT.md)、[技术与验收复核](REVIEW.md)、[实际图](performance.png)、[逐被试指标](metrics_per_subject.csv)。

方法依据：[Crosse等，2016 mTRF](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00604/full)，backward方向为神经响应→刺激特征。Decoder权重本周仅作预测参数，不解释为神经机制；实际运行使用冻结的Python ridge实现。
