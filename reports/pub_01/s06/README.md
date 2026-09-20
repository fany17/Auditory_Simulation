# PUB-01-S06 当前产物入口

2026-09-20 当前 **REVIEW / READY_FOR_REVIEW**。获准公开范围下载、完整格式盘点和初步虚拟清洗已实际完成，等待协调者最终独立验收；不是S01 benchmark准入，不启动S02。

正式根级交付：`download_inventory.csv`、`download_status.json`、`recording_qc.csv`、`exclusion_inventory.csv`、`cleaning_manifest.json`、`validation.json`，均来自服务器 `final_handoff_20260920_v1/`。4,715对象=4,338 SparrKULee（含196未请求受限项）+377 ds004703；recording表的11 EDF、666已发布EEG虚拟视图和660 raw BDF是不同证据层级，不能相加当作独立样本量。

| 范围 | 当前轻量证据 | 状态 |
|---|---|---|
| ds004703 | `ds004703_qc_v2/` | 初步虚拟准备技术PASS_WITH_LIMITATION，11录音/319片段/33次数值比对 |
| ds004703余格式 | `ds_remaining_formats_20260920_v1/` | 96/96盘点，18 NIfTI全样本有限；1旧pyc排除执行，合并信号覆盖377对象 |
| SparrKULee共同区间 | `sparrkulee_virtual_v4/` | 666独立EEG、全部索引实际验证；26大尾段flag |
| SparrKULee刺激provenance | `sparrkulee_stimulus_audit_v2/` | 官方73对象：72PASS、1截断HOLD |
| 截断复取证据 | `podcast_35_1_refetch/` | 两次独立获取相同字节且均不可解码，不覆盖原件 |
| 原始刺激 | `raw_stimulus_roles_20260920_v2/` | 204 NPZ全检查，72精确匹配published waveform；其余角色限制明确 |
| Raw来源/channel | `sparrkulee_channel_provenance_20260920_v3/` | 617公开header已核验、49受限，0缺失；stimulation额外1受限 |
| 最终下载/全格式QC | `reconcile_20260920_final_v2/` | 4142公开文件/135967030049 bytes，0缺失；4141格式通过+1截断源对象HOLD；660 BDF全解码 |

过程、失败与边界见 `PROGRESS.md`、`EXECUTION_AND_REVIEW.md`。原根级recording/exclusion两表已移至`historical_root_ds_qc_20260920/`备份，可恢复，未删除。其余根级ds专属早期表和`ds004703_cleaning_manifest.json`是历史；当前ds细节以`ds004703_qc_v2/`为准。`sparrkulee_qc_v1`不等长HOLD是早期保守规则，已按作者消费契约修正并保留历史。

虚拟derivative不是经过本轮滤波的信号；loader依赖原始服务器文件，单位/时轴来源与未知生成版本明确记录。本地只传回表和报告，不含信号数组。

26项服务器pytest通过（36条NumPy弃用警告），最终结构化验证PASS。独立审阅见`../review/sparr_completion_20260920_v1/independent_summary.json`及ds/stimulus审阅目录；实际runtime见`../review/runtime_20260920.json`。physical sync UNKNOWN、split UNASSIGNED、26个大尾差flag、49受限来源与ds Catalan来源问题均未消除；S01 admission仍HOLD。无Git提交/push。
