# 恢复同论文复现所需的最小资源

2026-10-15用户已允许缺少部分记录，以下完整论文资源**不再作为现有合法子集训练的前置门**。HDPS76人原100Hz/M真实test已完成（Top1 12.03%、Top10 42.10%），Transformer正在public可用条件执行。此清单只保留完整原paper/official test等价性的恢复需求，不得再次据此整体停工。

用户目标是同论文条件的数值对照。当前没有同条件测试结果，不能判断“一致”或“不一致”。不启动缩减cohort、64Hz适配或新模型来替代。

## HDPS：作者split中的5人原输入缺口

固定作者commit `42a5e1ec9649f4a673bfc018d377bbf38e289ed4` 的split有81人647 record-phase对。既有公开对象已下载齐全；以下缺口属于发布方restricted范围，未请求或读取。

| Subject | 必需但不可用的train | val | test | 备注 |
|---|---:|---:|---:|---|
| sub-019 | 8 | 1 | 1 | 原listening raw均restricted |
| sub-020 | 6 | 1 | 1 | 原listening raw均restricted |
| sub-021 | 8 | 1 | 1 | 另1条restricted raw无作者story精确匹配，不计必需 |
| sub-022 | 7 | 1 | 1 | 另1条public train raw存在；另1条restricted raw无精确匹配 |
| sub-026 | 8 | 1 | 1 | 原listening raw均restricted |
| 合计 | **37** | **5** | **5** | **47条作者split精确对应记录** |

49条restricted listening BDF包括这47条，以及2条`NO_EXACT_STORY_MATCH`。不把后2条强行映射或列为恢复必需。sub-022公开train `audiobook_5_1` 的BDF `file_id=135984`不计raw缺口；其APR `file_id=39013`仍restricted，初次v6处理因此失败。随后核对GPL代码：APR只添加SNR，而选定grouper只消费stim_file/trigger_file，EEG数值步骤不消费SNR；公开events还明确给SNR100及原文件坐标。v7显式记录容器适配，使用公开events，不请求APR，不改变trigger/crop数值步骤；该1条public raw现已成功生成64×90900的100Hz产物。**APR不再单列为该记录的恢复资源要求**，也没有用此修复掩盖47条原raw缺失。

最小恢复资源优先是：**这5人的合法、同源100Hz处理包**，带subject/story/phase、原始来源和处理版本；或者47条必要raw及无法由公开信息等价恢复的处理metadata的合法访问。不能由现有64Hz衍生信号上采样恢复。细目见[逐对象映射](evidence/continuation01/restricted_raw_mapping.csv)。来源是已有[官方inventory](../../s01/sparrkulee_official_inventory.csv)，例如sub026run01 `file_id=38723, restricted=True`。各公共events的story在2203读取映射。

## Transformer-CLIP-EEG：对应冻结模型与官方评价载荷

作者commit `8be963abe5be5c0568e54df3bceb2600861f895b` README给出的两个模型入口经2026-10-13实际打开：

- [Google Drive作者checkpoint](https://drive.google.com/file/d/1EO1BTlU2wb4S5x6BOeCfKEUFiKqhL7XO/view)：页面显示请求的文件不存在。
- [KU Leuven作者checkpoint目录](https://homes.esat.kuleuven.be/~lbollens/sparrkulee/clip-transformer-eeg)：HTTP404；trailing slash与父目录/zip有界检查也未找到有效迁移。

公开仓库issues/releases及两版README没有提供替代checkpoint。已终止循环补查，保存事实；失效不被推断为永久不可获得。检查来源见[当前证据记录](evidence/continuation01/external_resource_checks.json)。

最小恢复资源是：**论文对应、获准使用的冻结EEG模型**（报告ensemble须相应15模型及组合定义），以及官方seen70/heldout14测试sample IDs、signal/candidate映射和目标包的合法位置。普通长录音、71人公开from-scratch模型或单模型不能替代论文180人训练/15模型ensemble。无需请求非公开完整训练集。

## 执行端继续处理的条件

- HDPS对应GroNLP Dutch-base9/GPT2-small9权重的条款尚未明确。模型卡和关联代码repo是分开的权利对象，不从关联Apache自动推定权重许可。恢复资源也可包含已合法生成、对应论文版本的三特征冻结包及来源/许可。
- 原论文全文文字已经读取；TableI/II在PDF第8页的图像数值仍未核验，M对应论文数字留空。取得可渲染公开副本后核读，不猜值、不用MWG20.0%代替M。
- 原论文与公开code的lr/FFL/log压缩/采样率文件头等差异已经预先记录在`configs/pub_01_s03_hdps_paper_m_20261013.json`，不会根据test追分选配置。
- 全部数据、权重和计算继续仅在server2203；资源位置可以是2203现有目录或获准的下载URL，本地只接收轻量来源/路径/汇总。

无同条件模型训练/测试，无新性能数字，无ART-AUDREP-v1。上述是恢复同论文测试的输入要求，不是新的研究任务。
