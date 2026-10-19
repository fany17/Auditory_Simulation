# SS05后置审阅：有效阴性，完成有限制

2026-10-19。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS。本审阅为同一agent在执行后对照任务书、冻结源码/协议、真实checkpoint和输出的separate review pass，**不是第三方独立审查或外部科学签署**。

## 核查及实际证据

协调会话另外核对了SUMMARY、primary_contrasts、review、FORMAL_PROTOCOL、CODE_AUDIT与DATA_CONTRACT的轻量文件一致性，未代表独立重跑训练或第三方科学签署。其指出的上游计数文案已按实际manifest修正：排除193条为49旧validation+144旧test；冻结split的legacy字段名不准确，成员/字节身份及全部数值未变，见upstream_phase_audit.json。

1. 原代码/合成已在2203重跑，15项冻结版本tests PASS。发现原local带两past token并在新wrapper修正；forecast残差未受显式预测监督、实例化不等于实际活动容量等问题已在CODE_AUDIT说明。旧报告保留原文，不以新结果覆盖其来源。
2. 新split仅426条旧SS03train，family实际映射23/7/7组，record353/26/47。新normalizer的fit IDs严格等于新353train，与holdout不交；旧SS02统计未用来训练新模型，原文件不改。已打开的旧SS03 test payload/分数/checkpoint未参与SS05。
3. 五组四lag的有限train/val pilot后，只用validation选lag8。source/config/环境、六primary contrasts、2000步/200步val/三seed/七干预在新holdout信号前冻结。所有15run都跑满相同预算，checkpoint等于完整val最大r的最早位置，均1600步；训练全完才构建新holdoutcache。
4. 每个selected模型holdout01只运行一次，47record/21person/7family/517段在所有模型条件保持一致，0无效段/无选择性排除。模型输入仅EEG；历史干预复用当前编码/目标，仅换过去。同record连续性、有限上下文reset、无跨record状态与local零历史影响均通过。
5. 对全部分段、逐record、逐person和macro独立重算，最大汇总误差0。从原始EEG及新norm独立CPU重建90个60s/条件相关，maxr差8.1273e-6，低于预设1e-4技术容差；未声称所有段全部重跑CPU。冻结来源及时间顺序检查通过，见review.json。
6. 三seed先在record内平均，person bootstrap与subject/source-family交叉cluster bootstrap各10000次。六primary contrast校正后区间均跨0，阴性仍是合法验收结果；未根据结果缩短窗口、换lag、挑seed、加模型或增加预算。
7. H1未建立稳定真实历史收益；H2没有关联记忆相对state/GRU的稳健增益；H3没有full历史残差额外贡献的证据，且不能把其称已识别预测误差机制；H4有标签toy能力可用但EEG相似性代理不能证明语义内容特异检索。未把“未拒绝零差异”说成严格等价或普遍无效。

## 验收边界

本轮是prospective INTERNAL source-family holdout，因为数据曾用于SS03训练，不是外部独立复制；只7family、21人，人与train重叠。60s有限历史/每query reset、3s中心化目标函数、低层包络、固定2000步、不同active参数和墙钟、共享刺激及physical_sync UNKNOWN限制均披露。没有与旧SS02/SS03跨协议排名、没有新SOTA或患者/生理时标推断。

RESULTS含105个真实模型/seed/条件行，per_record/per_subject表、六primary及25secondary history delta均完整；常数control零方差Pearson为null，非伪造0。机器审阅见review.json，最后报告/来源/无payload交付检查见package_review.json。

SS05 COMPLETED/PASS_WITH_LIMITATION不等于S03或PUB完成。PR#3按用户新授权已合并main@8d21d0a；本PR#4保留Draft不自行合并。主目录CNN/TRF/stash和SS02/SS03不改；所有信号、normalizer、cache、token/weight/逐窗数组仅2203。当前停止，不启动SS04/S04/父S05/GPT或ART-AUDREP-v1。
