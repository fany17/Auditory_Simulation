# PUB-01-S01 当前审计入口

更新：2026-09-20。S01状态 REVIEW；S06数据准备已通过最终独立验收并COMPLETED，当前无执行中的primary子任务。此处不把阶段许可门或数据准备技术通过写成S01完整benchmark准入。最终决定见[协调者验收](../COORDINATOR_REVIEW.md)。

## 当前机器可读产物

当前快照为 `current_audit_20260920_v2/`（v1阶段快照保留，不再当作当前下载状态）：

- `dataset_manifest.csv`：两数据集版本、来源、范围、服务器路径及UNKNOWN/HOLD。
- `license_use_boundary.csv`：使用/衍生/再分发边界与官方链接。
- `audio_neural_pairing_audit.csv`：ds438事件块与EEG666独立derivative的1104条证据。
- `timing_event_inventory.csv`：采样率依据、时间映射、未证实physical sync。
- `group_split_keys.csv`：subject/session/recording/story/block键，全部UNASSIGNED。
- `exclusion_inventory.csv`：ds非passage/Catalan排除、EEG大尾段标记、损坏孤立对象。ds详细通道排除另见S06 v2 channel_qc。

根目录原 `audio_neural_pairing_audit.csv` 是首次尝试：汇总音频副本被当成独立候选造成438/438 HOLD。它保留失败provenance，**不再是当前配对结论**。新快照依据实验脚本角色规则及直接波形比较纠正；原表未覆盖。

## 已核验与仍未解决

SparrKULee官方RDR V3.1冻结4,338对象：4,142公开、196受限，公开135,967,030,049 bytes。非商业科研公开下载门经协调者审核通过；限制访问仍HOLD。CC-BY-NC-4.0要求非商业使用及共享时保留署名/来源/许可/修改说明，不能默认衍生artifact自由商用。[官方来源](https://doi.org/10.48804/K3VSND)，[许可正文](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)。

666个独立EEG与各自envelope已按作者消费约定完成共同区间全读取与数值核验；不是668个独立录音。两组source别名去重，所有source alias保留。26个较大尾差继续REVIEW。作者当前代码给出64Hz及微伏处理依据；实际data_dict中的previous_steps与envelope逐值相符。**stimulus_sr=64只对应重采样envelope，不用于原stimulus_data时长**，原trigger_sr另列。生成时确切软件revision仍UNKNOWN。[作者消费脚本](https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/technical_validation/util/split_and_normalize.py)。

刺激审计以官方73个data_dict为全集：72个可读且与对应envelope一致，1个`podcast_35-1.data_dict`源对象截断。它不在作者save metadata中、无同名envelope、666 EEG中无同名story引用；正常`podcast_35`是不同对象，不受此孤立对象的直接引用影响。独立重新下载与原件逐字节相同，均截断，保留HOLD及两份文件；不无限重试。

666个EEG的raw来源全部能在官方清单定位：617公开、49受限。617公开来源header已全部核验，缺失/未知为0；49受限未请求，不误标下载失败，其原始channel/header无法独立核验的限制保留。另1条公开raw的stimulation.tsv受限，因此616条story匹配、50条UNKNOWN（49+1）。全660公开BDF已解码/校准有效；实际公开来源头1024Hz与dataset sidecar8192Hz差异明确保留，derivative64Hz与raw音频48000Hz不混用。

ds004703现存v1.1.0：协调者377对象大小对账齐全；本轮11EDF/270WAV全样本读取，事件边界0异常，319英语passage进入虚拟derivative。78控制块与41Catalan块排除。6个Block中Catalan标记音频与提示音波形相同，不能由另一份文件静默替换。模板一致不等于真实声学同步。CC0元数据与README非商业/禁止重识别要求并存，保守遵守README；无限制再分发仍HOLD。[官方数据](https://openneuro.org/datasets/ds004703/versions/1.1.0)，[实际README](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004703/master/README)。

## Leakage / split准入

ds004703的README“6 blocks × 7 passages”与实际打包Block目录计数存在差异，应以逐文件/事件清单展示实际范围，不将README概数强行当成已验收protocol。此差异不靠修改数据消除，保留为发布元数据限制。

本轮只冻结分组键，不分配train/validation/test。禁止相邻片段随机跨split、重复story泄漏，以及test参与归一化/PCA/特征/层选择。后续需要按父任务明确subject/session/story held-out评估口径并验证分组可行性，不能复用作者80/10/10相邻时段split。S02未获准启动。

## 验证与决定

Technical：现有数据的真实读取、索引消费、数值比较已做；轻量快照可重现于 `scripts/pub_01_audit_handoff.py`。S06全公开下载已完成：4142文件/135967030049 bytes，0 missing/partial，4141格式可读+1明确截断源对象排除。ds余96对象格式/静态检查补齐（18 NIfTI全体素finite，旧pyc排除执行），与281信号QC覆盖377。最终交付见[交付入口](../s06/README.md)及[清洗清单](../s06/final_handoff_20260920_v1/cleaning_manifest.json)；S06已由协调者最终验收为COMPLETED / PASS_WITH_LIMITATION。

Acceptance：SparrKULee公开许可/身份阶段门PASS；ds与EEG虚拟初步准备技术PASS_WITH_LIMITATION已由协调者独立复核。完整S01 benchmark admission仍HOLD，不能把这些阶段结果升级为完整科学接受。

Parent consistency：范围只含两套批准公开数据，无患者/STN、无模型训练、无哈希审计；所有载荷留2203，不改变父任务acceptance。
