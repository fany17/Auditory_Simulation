# SUBTASK — PUB-01-S06 — Server Dataset Download and Initial Cleaning

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S06` |
| Parent Task | `PUB-01` |
| Status | `COMPLETED` |
| Blocking for Parent | `YES` |
| Patient Data | `FORBIDDEN` |
| Upstream | S01 license/access and identity gate per dataset; full S01 acceptance before benchmark use |
| Inputs | Official SparrKULee / ds004703 sources; verified existing server assets |
| Deliverables | Server raw inventory, download status, cleaned derivatives, per-recording QC, reproducible config/code, review report |
| Acceptance | Downloaded scope reconciled to official inventory; actual files readable; initial-cleaning/QC outputs verified; raw preserved; missing/excluded/blocked items explicit |
| Validation | Inventory reconciliation + all selected files readability/QC + independent sample rerun + parent consistency |
| Go/Hold/Fail | PASS only for actual completed scope; access/license/disk/network gaps remain HOLD, never treated as downloaded |
| Files Allowed to Change | public configs/scripts/src/tests/schemas; reports/pub_01/s06/**; doc/tasks/PUB-01/**; necessary TASKS/CURRENT_TASK status; server project only |
| Last Updated | `2026-09-20` |

## 1. Objective and authorization

用户要求持续推进到至少完成 PUB-01 数据集下载及初步清洗。范围为 SparrKULee 与 ds004703 的官方公开可获取、获准使用的数据；先盘点全量官方对象、字节数与磁盘余量，再下载和复用既有资产，不能只下一个样例就宣称全库完成。

## 2. Execution location and safety of records

全部下载、续传、解压、数据读取和处理在 `server2203:/home/fanyu/auditory_simulation_m6a` 下执行，使用新的 PUB 产物目录，原始数据和历史结果保持只读。Windows 只保留轻量代码/配置/清单/报告，不下载载荷或回传音频、神经信号、权重和特征张量。服务器不可用不得回退本地。

## 3. Download procedure

1. 官方来源核验版本、许可、访问方法和范围；未知项不猜测，不接受本助手不能代表用户接受的限制性合同。
2. 建立逐文件 expected inventory，记录 source URL/version、相对路径、预期字节数（若来源提供），以及已有/缺失/部分下载状态。
3. 核对可用空间、现有下载进程和既有数据，禁止重复启动相同对象下载；使用可续传、限重试、持续日志的下载方式。
4. 下载并抽样检查容器；对全部纳入文件执行格式/可读性检查和计数/字节数对账。沿用不主动 checksum/hash 审计规则，不把大小相符宣称密码学一致。
5. 断网/临时限速等可恢复失败诊断后续传；长下载以 PID、日志更新时间、完成数量/字节数可核验，不在只有后台进程时宣称完成。

## 4. Initial cleaning and QC

先依据实际数据格式冻结最小、确定性的清洗配置。至少记录每个 recording 的采样率、通道名/类型/单位、样本数/时长、非有限值、平直/缺失通道、事件边界、音频时长与时轴匹配、官方 bad 标记。无效记录/区间给出原因并从可用清单排除；原始文件不修改。

在服务器独立 derivatives 目录生成标准化元数据、可用 recording/segment 索引、必要的无损格式统一和清洗产物及变更日志。数据已由发布方预处理时保留其 provenance，避免二次盲目滤波/重参考。任何滤波、重采样、重参考、去伪迹均需在配置中给出参数与方法依据并验证时间轴；本阶段不要求自选 ICA 或跨被试拟合，不把 QC 索引冒充已做信号处理。

缺失值/坏道不静默填补；禁止按最终模型效果选择清洗参数；不跨 train/test 拟合统计量。不训练模型、不使用患者数据。

## 5. Deliverables

- `reports/pub_01/s06/download_inventory.csv`、`download_status.json`：official expected / existing / downloaded / verified / missing / failed，计数及字节数可对账。
- `recording_qc.csv`、`exclusion_inventory.csv`、`cleaning_manifest.json`：实际 QC、清洗动作/未做动作、服务器产物路径与限制。
- 服务器 download log/status、raw 与 derivatives；本地仅轻量 summary/manifest。
- 中文执行及独立复核报告，附可复现命令、config、实际范围与未解决问题。

## 6. Acceptance / review

分别判定每个数据集：下载范围是否齐全、已纳入文件是否可读、QC/清洗是否真实执行、原始数据是否保留、排除与未下载项是否可追溯。明确区分 full download、partial download、initial QC、cleaned derivatives。只完成下载或只完成小样例不能把本 goal 标为完成。

## 7. Completion Record

2026-09-19：协调者核验 S01 SparrKULee V3.1 公开许可/身份阶段门并授权先行。Primary 切换为 S06；4,142 public 文件 / 135,967,030,049 bytes，196 restricted 不请求。实际服务器独立下载 PID 105526，状态/日志见 `reports/pub_01/s06/PROGRESS.md`。下载与QC尚未完成，非验收；S01 pairing/timing/split 仍待完整审阅，不执行S02。

2026-09-20最终执行交付：下载PID107053及格式PID113424正常结束，公开4142对象/135967030049 bytes对账完整，0 missing/partial/下载失败；4141格式可读+1已独立复取确认的孤立截断源对象排除，196 restricted未请求。660 BDF全解码、校准有效；617公开source header核验、49受限保持限制。204 NPZ全检查、72原音频精确匹配；666独立published EEG虚拟共同区间全读取/数值检查，26大尾段flag保留。ds004703现存377对象大小及格式范围盘点完整（281全信号、余96格式/静态检查，1旧pyc排除执行）；319虚拟片段全读取、33次数值比对差0。

正式交付`reports/pub_01/s06/{download_inventory.csv,download_status.json,recording_qc.csv,exclusion_inventory.csv,cleaning_manifest.json,validation.json}`，服务器对应`pub_01/s06/final_handoff_20260920_v1/`。26项pytest、py_compile、最终结构化验证PASS；独立分支核验见review目录，最终整体验收交协调者。技术PASS_WITH_LIMITATION候选，S01 benchmark准入HOLD、physical sync UNKNOWN、split UNASSIGNED；不执行S02。旧快照/失败保留，无永久删除，无Git提交/push。当前REVIEW，非自行DONE。

2026-09-20协调者最终决定：**COMPLETED / PASS_WITH_LIMITATION（获准公开范围）**。独立逐文件实物对账、全格式结果集合、617公开通道头、666唯一EEG、319可消费片段、377 ds对象、204刺激对象及最终4715行交付表均已核验。原始载荷保留，196受限对象和1坏源对象的限制已完整列明；文档归档/入口与2203规则完成。证据见 `reports/pub_01/COORDINATOR_REVIEW.md`、`review/final_artifact_acceptance_20260920.json`。本决定完成数据准备goal，不解除S01模型准入HOLD，不启动训练。
