# PUB-01 协调核验记录

**2026-09-20 最终状态：S06已COMPLETED，获准公开范围数据下载/初步清洗与文档整理已独立验收PASS_WITH_LIMITATION。当前结论与证据入口见 [COORDINATOR_REVIEW.md](COORDINATOR_REVIEW.md)。以下为按时间保留的历史过程，不再代表当前执行状态。**

2026-09-19 持续 goal：在 server2203 完成 SparrKULee 与 ds004703 的合规下载、初步清洗/QC及实际产物复核。goal 保持 ACTIVE，不能以已派发、已建进程或只写报告代替完成。

## 当前结论（服务器时间 2026-09-19 15:50 UTC）

- **整体 ACTIVE，未验收。** 以下旧段落按核验顺序保留，历史 PID、计数和问题不代表当前状态；实际进度以2203实时文件和进程为准。
- ds004703：377个官方对象大小对账通过，11 EDF / 270 WAV全样本QC完成；319个可用片段经虚拟loader全量读取，33处独立数值比较最大差0。初步准备技术结论 PASS_WITH_LIMITATION；尚未证明物理同步，也未解锁训练。
- SparrKULee：下载PID107053持续运行；15:48 UTC本次进程已处理785项 / 26,120,393,604 bytes，0下载失败。目标4,142公开对象 / 135,967,030,049 bytes；196受限对象不在已授权公开下载范围。
- 739个NPY（666 EEG、73 envelope）已通过全样本格式检查。独立复核发现save metadata的668条EEG映射只有666个不同derivative；重复来自两组同名.bdf与.bdf.gz来源别名，执行者正在按derivative路径去重并保留别名。v2的668计数不得当作独立录音数量或最终验收证据。
- 格式QC最新1,562个不同对象：1,561可读、1个pickle截断HOLD；解析器兼容问题已采用已知函数引用的不可执行占位符修复，并非执行发布方序列化代码。PID113424继续检查新下载对象。格式可读不等于已完成清洗。
- 下一验收点：666个唯一EEG的可执行清洗索引、缺陷文件真实原因、全公开范围下载/格式对账、原始通道/单位/时间轴来源和限制。

15:51 UTC续核：SparrKULee v3索引已修复重复计数，独立按冻结官方清单比较，666行=666唯一路径=全部公开EEG derivative，0缺失/额外；全部有限值、平直通道及数值比较检查通过，26条large-tail保留。该虚拟derivative初步技术验证可接受，仍非全库下载/完整准备验收。剩余pickle截断定位为podcast_35-1.data_dict（116,391,936 bytes），执行者正做有界原因核验。下载进度33,351,894,060 bytes，0失败。原始BDF通道/单位及官方bad标记映射仍待实物审阅。

15:55 UTC续核：podcast_35-1.data_dict独立重新获取后与原件逐字节相同，两次均pickle截断，证据在2203的podcast_35_1_independent_refetch_20260919/validation.json。辅助save metadata仅列72个data_dict，遗漏的恰是该坏对象；刺激审计v2已修复为官方73项全集，72项与发布envelope逐数值一致、1项HOLD、0缺失。666条可用EEG索引没有引用podcast_35-1。原件及重取副本均保留，坏对象未假装修复。下载PID107053、格式检查PID113424仍存活，已处理40,911,556,102 bytes；空间余量约663GiB。

2026-09-20（本地）审计表复核：S01/current_audit_20260920_v1的group/pair/timing三表各1,104行，dataset+sample_key集合完全一致，subject无空值、split全为UNASSIGNED。ds004703为319 PASS_WITH_LIMITATION / 119 EXCLUDED，SparrKULee为666 PASS_WITH_LIMITATION；完整benchmark准入仍HOLD。另对官方718个BDF.gz清单复核：无重复basename，666个derivative各有唯一来源，49项原始source受限、617项公开，不把受限头信息当作已读取；三条作者明确触发器不足的bad recording未被这666项引用。公开原始文件下载/全格式QC仍继续。

独立BDF数字/物理单位验证：对实际sub-009 restingState run-02原gzip前2个record，创建仅改record数的服务器诊断摘录，原件不改；64 EEG通道×2048样本的独立int24与header仿射校准和MNE读取比较，最大绝对差3.0923e-11微伏（容差1e-8），正负数字值均覆盖，PASS。代码scripts/pub_01_verify_bdf_calibration.py，轻量证据review/bdf_calibration_20260920_v1/validation.json；数据摘录仅留2203。这是解码/校准的独立抽样检验，不替代660份公开原始BDF的全读取或物理同步验证。

协调者补完原始刺激全量审计：确认204份NPZ格式PASS且无同类审计进程后，使用执行者最新raw_stimulus_audit代码在2203生成review_raw_stimulus_complete_20260920_v1，204/204 available、0HOLD、72份原音频与data_dict中波形逐样本完全相等。实际fs全部48000Hz，204唯一路径与官方清单相同，无缺失/额外。轻量证据位于review/raw_stimulus_complete_20260920_v1，载荷仅留2203。初版角色是文件名启发式：75项标AUDIO中3项无同名data_dict，实际为audiobook_3_noise、audiobook_6_1_swn、triggers；不能宣称75项均为已确认语音，已要求基于既有metadata更正角色及依据，保留v1，不重跑72份波形。其余60项noise前缀与69项t_前缀也须写明角色依据。全量数值扫描无需重复；物理神经同步仍UNKNOWN。

角色v2复核通过：204条既有数值证据/采样率/比较字段与v1无差异，逐项role_basis齐全；72项PUBLISHED_STIMULUS_AUDIO不证明语音内容/实际播放，70项TRIGGER_BY_FILENAME仅名称依据，62项AUXILIARY_ROLE_UNVERIFIED。旧表保留。

ds004703余96对象已实际补检并独立复核：ds_remaining_formats_20260920_v1的96唯一路径恰好覆盖官方剩余清单，无缺失/额外；18 NIfTI全部样本解码，shape乘积=读取样本数，nonfinite=0；88格式PASS、7文本读取PASS、1历史pyc仅按字节读取并排除执行/数据使用。与既有281 EDF/WAV全样本QC合计覆盖377对象，原始文件未修改。仅做已defaced影像格式读取，未生成影像重建/可视化或身份推断；影像不进入benchmark。

## 已完成的管理工作

- 9 份 doc 历史文件已移入 doc/archive/pre_pub_20260919，逐项保留原始字节数并核验新文件存在；旧路径映射见 ARCHIVE_MAP.csv。当前 README/索引/归档入口及子任务路径检查通过。
- S06 数据下载和初步清洗任务书已落盘，用户授权记录写入 parent Amendment Record。
- 原执行任务曾回到旧 M6A 报告复核；已纠正并改名为 Auditory_Simulation — PUB-01 数据下载与初步清洗。首个真实 PUB 进度文件已核对。

## 真实数据状态与审核

- 服务器现有 ds004703 约14 GiB，卷余量约702 GiB；不据此宣称下载齐全。
- SparrKULee 官方V3.1清单：4,142 public文件共约126.63 GiB；196 restricted文件约6.05 GiB。官方来源和使用条件经查阅，公开部分非商业科研下载已放行；完整访问权限已询问用户，公开分支继续。
- ds004703 S01表：11录音、270音频文件、438连续事件组；header+首秒抽样的事件越界与非有限值均为0，不是全录音QC完成。
- 初版配对438/438 HOLD：示例因 all_stimuli 和 excerpts/Block 下同一刺激的目录副本触发双候选；已要求依据目录角色、已有映射和服务器直接音频比较核验，保留失败表，不能直接认定真实配对全部不可用。

## 未完成与下一步

公开数据下载已核验实际运行：server2203 PID 105526，状态文件 pub_01/s06/sparrkulee_v3.1/download_status.json；本次抽查完成68/4142文件、6131 bytes（首先下载小型元数据）、0失败。该状态远未完成全库，不能以文件数量忽略载荷字节数。完整下载计数/字节数对账、全部纳入文件可读性、确定性初步清洗和独立抽查未完成。必须继续检查2203的下载进程/日志、实际文件和QC产物；受限项单列，不能虚报全库完成。原始数据只读，数据与派生物只留2203。

## 官方来源

- https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND
- https://homes.esat.kuleuven.be/~spchdata/corpora/auditory_eeg_data/
- https://github.com/exporl/auditory-eeg-dataset
- https://openneuro.org/datasets/ds004703

## 持续核验更新（2026-09-19）

- 独立脚本 pub_01_verify_ds_inventory.py 已在2203完成官方S3实时清单对账：377/377文件大小相符，0缺失/不符，14,173,350,514 bytes；其中11 EDF、270 WAV、27 anatomy、69 metadata/other。证据：review/ds_inventory_20260919_v1。该证据不证明不可变版本或内容完整性。
- S06 ds004703全样本QC实际表已核对：11录音、270音频，非有限值0，事件越界0，48个平直通道；1346个可用通道跨录音计数；438事件组中319个可用、78个控制片段排除、41个Catalan来源/时长冲突排除。现阶段仅无损排除索引和标准化metadata，不是滤波/重参考后的信号文件。
- 已要求提供实际索引消费loader或清洗文件，并验证全部索引可读性与抽样数值；阈值须冻结到配置，零可用结果须有显式schema。这些条件尚待实物验收。
- 下载执行者为优先获取可QC载荷调整了下载优先级并续传，独立ps确认新PID107053活着；旧PID已终止，旧日志保留。重启后的completed计数是本次运行已处理项，不等于累计目录中的文件数，复核最终以冻结清单逐项对账。

## ds004703 初步清洗技术复核

独立查阅2203实际v2执行日志及loader代码后：11录音、319可用片段全部索引被实际分块读出；33次抽样数值比较最大绝对差0。结合377文件官方大小对账，ds004703初步数据准备技术结论为 PASS_WITH_LIMITATION：产物是可执行的无损通道/片段选择（虚拟derivative），不声称已经滤波、重参考或ICA；原始信号保持不变。Catalan来源冲突、physical-sync UNKNOWN、S01完整准入与split审阅仍需明确保留。

证据服务器目录：pub_01/s06/ds004703_qc_20260919_v2；运行日志 ds004703_virtual_validation_20260919_v2.log。下一重点是SparrKULee完整公开下载与实际初步QC/清洗。goal仍ACTIVE。

独立本地表间核验：ds004703 v2的clean_channel_index与channel排除项无交集，319个clean segment key均有PASS全片段读取记录，samples_read与半开采样边界一致。SparrKULee初版NPY QC中204个EEG均因PAIR_LENGTH_MISMATCH为HOLD，须查作者同步/截段规则，不能直接截短并当作配对正确。

## SparrKULee 配对判据复核

作者 Brain Pipe 的 default_drift_correction 保留末trigger后约2秒并处理整秒边界；SplitEpochs本身只拆分数据字典。因此原版QC把EEG与envelope长度严格相等作为必要条件会误判，需按实际trigger与版本核验有效重叠时域。133点差值约2.08秒仅是可能符合尾段的推断，2005点约31.3秒仍待调查；不授权盲目截断或宣称current master就是V3.1生成版本。

已实际读取官方来源并发送给执行者：
- https://raw.githubusercontent.com/exporl/brain_pipe/master/brain_pipe/preprocessing/brain/trigger.py
- https://raw.githubusercontent.com/exporl/brain_pipe/master/brain_pipe/preprocessing/brain/epochs.py
- https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/preprocessing_code/sparrKULee.yaml

进一步官方依据：technical_validation/util/split_and_normalize.py 将EEG转为time-first，并使用min(EEG长度,envelope长度)从零索引取公共区间；technical_validation/README.md明确下载的derivatives可直接用于作者验证。因而有来源的公共重叠时域loader可用于初步数据准备，必须保留原始长度、裁去尾部计数及异常flag，并明确不等于独立物理同步验证。已授权执行者采用该消费约定，不照搬作者相邻时段80/10/10划分或归一化，不解锁模型训练。
来源：https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/technical_validation/util/split_and_normalize.py
