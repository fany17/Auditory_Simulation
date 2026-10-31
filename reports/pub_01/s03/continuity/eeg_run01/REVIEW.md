# SS06 后置审阅：完成有限制

2026-10-23。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS。本文件记录**同一agent在执行后另作的separate post-execution pass**。协调会话的独立只读source/split/cache/runtime审阅另列，二者均不是外部第三方科学签署。

## 执行前及冻结审阅

协调会话实际确认249/63/41record、13/5/5family、路径/story/family隔离、归一化fit IDs恰为249train，以及当时audit未开。其提出的完整record覆盖、M5模式隔离、M4zero-lag缓存、正式SSL seed/预算/source/共享checkpoint、实际访问记录与eligible分离、实质all-training gate及CUDA输出/梯度/分块核验均在formal freeze前关闭。也逐式核对了M5手写一阶BPTT、observation辅助loss和初态梯度，并查看7项最终适配测试日志。该审阅不是独立重跑训练。

原型38测试/71子测试与最初3适配测试实际通过；最终7适配测试另行通过。原日志保留TorchScript弃用和非连续GRU权重warning，不称无警告。最终实测train/val坐标/标准化、实际60s CPU-GPU/分块/梯度核验PASS。observer方法在开audit前另用validation36相关检验，max差9.10e−8，错误seed/预算/pilot权重被拒，AR递推/因果性通过。

协调会话在结果产生后另实际读取review.json、冻结协议、全部8个primary contrasts/history_deltas、bootstrap源码及开封门证据，并在2203从per_record_all_seeds独立按seed/person重聚合全部SUMMARY，最大差5.55e−17；39run/3seed/每run3262步一致。此为独立逐record汇总复核，**不是由协调者重跑全资产模型**。下方原始64Hz资产+原型CPU复算由执行者另作，两个范围不能混称外部完整科学签核。

## 后置实际指标审阅

1. 39run每次3262步，实际监督record/block顺序一致，全部249记录/3262块覆盖；3正式SSL各3511个完整段覆盖全部新train完整时间，eligible、actual_records、span_sequence分别记录。所有source/split/norm和checkpoint身份核验；M3frozen/finetune同seed同正式初始化，pilot/异seed不可混入。
2. 每1000步及最终完整validation最大r最早位置选点，自主prior按MSE选择。全部训练跑满，测试前核查实际checkpoint内容/finite/seed/variant/selected step。全训练完成、输出控制val选择、实质review通过之后才原子开audit；时间顺序核验PASS。
3. 每selected模型一次预声明assessment，全部538完整块/41record/36person/5family保持共同坐标，历史条件另有每record首评分块41段。243行真实结果，逐seed/逐record/逐人证据完整；没有按audit追加模型、lag、预算或重新挑seed。
4. 对所有分段重算record均值、再person均值、再macro，Pearson与MSE最大汇总误差0。自主prior的近常数r为null，不冒作0；重置瞬态的有效r另列。无效r计数不被隐藏，未据此删掉其他模型的负结果。
5. 从既有原始64Hz输入数组（非新cache、非从BDF重做清洗）重新读取并用新train统计标准化，采用原型CPU路径独立核验39run×2record×3状态条件。222个有效r max差1.5513e−7；234个MSE max差6.5220e−8；12个近常数r未定义核验一致。不是仅复读summary，也未声称重跑全部538块的所有模型。
6. 三seed在record内平均，subject bootstrap和subject/source-family交叉bootstrap各10000次，八primary Bonferroni单项99.375%区间。M1multi−long、M3finetune−frozen、M4adaptive−uniformfixed三项为正；其余五项跨0。没有挑subject-only较窄区间宣称M2/M5稳健胜出。
7. 五主框架真实−错配state的secondary交叉95%均跨0。state-only current EEG和local encoder特征逐点相同；全前缀边界效应分列。首块一次reset与全记录周期reset端点/频率不同，不混合解释。M4/M5部分次要reset现象未作多重性确证。

## 验收、局限与交付完整性

完成限定真实实验不要求阳性。M2最高点估计不等于超过GRU；M3微调超过冻结不等于SSL超过从头；M4只对uniform固定控制，不证明真实动态时延；M5相对EEG-only/输出AR没有稳健额外收益。参数/硬件/额外SSL成本不同，均已披露。只有5family、历史训练暴露、人员重叠、单pass、固定低层目标及physical_sync UNKNOWN，使本轮最多为INTERNAL/EXPLORATORY证据，不是外部独立复现、SOTA、语义或生理机制证明。

E盘回传耗尽空间后，仅本轮未提交partial evidence被移动至C盘备份，未永久删除任何资料。服务器完整结果/weights未改；Git精简JSON只去重复访问序列，保留逐run真实指标、来源身份、完整源路径/摘要及覆盖核验；完整轻量JSON仍在2203和C盘delivery01。package_review在服务器分别核对完整包和Git精简包，指标必须完全等于实际run evidence；CSV、图、source/config与无信号/权重载荷检查见机器结果。

SS06 COMPLETED/PASS_WITH_LIMITATION，不完成S03或PUB，不产生ART-AUDREP-v1。PR#6维持Draft，不自行合并；主目录CNN/TRF/stash与旧SS02/03/05 frozen证据不改。当前STOP，SS04/S04/父S05/GPT不自动启动。
