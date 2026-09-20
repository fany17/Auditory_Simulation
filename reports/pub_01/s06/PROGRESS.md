# PUB-01-S06 实际进度

2026-09-19：协调者审核 S01 SparrKULee 公开许可/身份阶段门后，primary 从 S01 切换为 S06。S01 完整 pairing/timing/split 仍 REVIEW，不解锁模型。

## 最终交接（2026-09-20）

当前REVIEW。公开4142/4142文件、135967030049 bytes，下载16:47:36Z完成；格式16:48:06Z完成（4141可读+1源截断HOLD），660 BDF全解码，两个进程已退出。最终617公开header通过、49受限、0missing。ds377范围已完成；204刺激审计与666虚拟EEG、319 ds虚拟片段证据齐备。

正式根级五类交付及validation已从`final_handoff_20260920_v1/`同步；4715对象、1337不同证据层级recording行、1529排除/flag项，最终validator PASS。S01快照更新`current_audit_20260920_v2/`；S06 README/执行报告顶部及subtask/registry已更新REVIEW。根项目README/CURRENT等全局入口由协调者维护。26 tests通过，36弃用警告；实际runtime见review记录。

汇总首次重复dataset字段TypeError及继发无产物FileNotFound已记录在报告，修复加测试后成功，无源数据覆盖。原根级历史ds recording/exclusion移到`historical_root_ds_qc_20260920/`可恢复备份，没有删除。协调者已独立确认全公开文件大小/QC集合、0partial/受限payload、660校准、前64无整段平直等。

待协调者最终验收，未标DONE；S01 benchmark仍HOLD，未启动S02，没有commit/push。以下为原始阶段日志，非当前状态。

## 已启动的真实下载（历史）

- 主机：server2203 / nercn。
- PID：105526（启动后实际 status 读取；需结合日志时间核验存活）。
- runtime：`/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python`。
- 脚本：`/home/fanyu/auditory_simulation_m6a/pub_01_s06_download.py`。
- 清单：`pub_01/s01/audit_20260919_v1/sparrkulee_official_inventory.csv`。
- 输出根：`/home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1`。
- 公开范围 4,142 文件，135,967,030,049 bytes；196 restricted 明确不请求。
- 2026-09-19T15:24:36Z 状态：34 文件 / 2,714 bytes，0 failed，RUNNING。这是 metadata 小文件起步，**不是全库下载完成**。
- 独立 nohup 进程，3 workers，逐文件 `.partial` 续传、三次有限重试、文件锁防重复启动、仅大小检查，不声称密码学完整性。
- 状态：`download_status.json`；逐文件流水：`download_inventory_<UTC>.csv`；日志：`launch_20260919_retry1.log`。

## 失败保留与下一步

第一次启动用了不存在的环境绝对路径，未下载；`launch_20260919.log` 保留。修正为实际查询路径后成功启动。

下载进行中继续实际可读性/QC、元数据时轴与刺激角色核验。所有信号和衍生物留2203。完整下载、全范围QC、独立复跑尚未完成，不能验收或结束 goal。

## 已完成实质QC / 持续下载（15:45Z更新）

- 当前下载PID **107053**；前105526经核验为本任务进程后停止，用相同raw/partial恢复并优先core provenance/derivatives。先前log、ledger全部保留。
- 15:45:28Z恢复进程已完成754文件 / 18,890,816,116 bytes，0 failed；不是全公开范围完成。
- ds004703 v2：11 EDF与270 WAV全样本QC；319候选片段，1,346个录音-通道项；消费loader全索引读取与33次直接原数值比较通过（最大差0）。独立协调审阅判初步准备技术PASS_WITH_LIMITATION；虚拟derivative非滤波信号，S01最终准入仍限制。
- ds v2 manifest/validation/索引已同步 `ds004703_qc_v2/`，v1和首次HOLD表保留。7项服务器unittest通过；schema/计数/资格检查通过。
- SparrKULee首批277 NPY全样本QC原先因不等长HOLD；已找到并读取作者正式消费脚本，其明确从0取共同长度。新增可实际消费的common-overlap虚拟loader，保留尾部计数、大尾段flag、不改原数据、不继承相邻split或归一化。首快照验证624个EEG，44个暂缺、26个大尾段flag；metadata共668项，正在与官方文件清单逐项区分真正缺失和未发布引用。
- 官方代码证据：`technical_validation/util/split_and_normalize.py` L48–86；`brain_pipe/preprocessing/brain/trigger.py`的对齐实现保留尾段，SplitEpochs只拆字典。当前master不等于当时生成revision；physical sync仍UNKNOWN。
- 全格式增量QC独立进程已实际检查超过1,500文件（NPY/JSON/TSV/text），其余随下载到达核验。BDF.gz将流式解码所有int24，记录通道/物理单位与校准；NPZ.gz顺序展开留服务器，低于50GiB余量停止；不把NPY检查冒充全范围。
- data_dict已核对作者brain_pipe 0.0.4官方包保存器：pickle。限制型读取首两文件发现序列化`__main__.temp_stimulus_load_fn`引用，失败保留；改为不可执行sentinel后重试，绝不运行其序列化函数。其他未许可全局仍拒绝。

详见 `EXECUTION_AND_REVIEW.md`。目前继续S06，完整下载、全格式QC、SparrKULee最终虚拟derivative验收仍未完成。

## 15:52Z 实际修复与阴性证据

- SparrKULee下载35,218,191,230 bytes，后台继续。666个独立EEG已完成虚拟共同区间全读取/数值验证；26个大尾段flag。668 metadata source映射的两条重复是同recording压缩/非压缩别名，已按derivative路径去重，保留alias；不能再说668独立EEG。
- data_dict格式来源已核对官方brain_pipe 0.0.4发行代码，实际文件头为binary pickle。受限unpickler仅新增标准库OrderedDict和不可执行作者函数占位符，禁止任意global/REDUCE执行；14项测试通过，包括任意global拒绝、占位符不能执行、BDF校准头和共同区间边界。
- `podcast_35-1.data_dict` 官方大小116,391,936 bytes但不可解码。独立目录重新获取后与原件直接字节流比较完全相同，二者均报`pickle data was truncated`；确认不是本地续传少字节的解释。报告远端`podcast_35_1_independent_refetch_20260919/validation.json`，此项保持HOLD，原件和第二份均保留，不宣称全库零错误。
- 新增逐刺激data_dict→envelope直接数值/采样率/provenance审计，进行中。全格式QC与下载继续，不启动训练。

## 2026-09-20 00:03 本地时间检查点

首个全清单对账快照`reconcile_v1/`：4,142公开对象中2,810已大小匹配（40,913,260,023 bytes），1,332未完成；2,788对象已格式检查，1项HOLD；196受限不请求。实际PID107053仍活跃，状态PARTIAL_ACTIVE，不是完成。

S01当前6表已写`../s01/current_audit_20260920_v1/`并验证：2数据集、2许可记录、1104唯一分组/配对/时轴项，未分配split。入口`../s01/AUDIT_REPORT.md`；旧根级438/HOLD表明确历史。

SparrKULee刺激最终本轮快照v2：官方73对象=72PASS+1孤立截断HOLD，0 missing。666 EEG当前虚拟v4绑定冻结config；49 raw来源受限、617公开，均官方可定位。raw通道头随下载后核验，受限项不被误标下载pending。

当前服务器环境pytest可用：14 tests passed（36条NumPy旧内部命名deprecation warning，无失败），2份cleaning manifest schema通过。此环境结果不沿用历史M6A环境的pytest缺失判断。

## 00:10 实際原始载荷核验

下载45,318,940,447 bytes / 3,462文件；全格式已检3,454，含103 NPZ.gz、73 BDF.gz，仍仅1源对象截断HOLD。

原始NPZ采样率从真实`fs`零维字段读取：首快照96/204可用，0HOLD，22个原音频与data_dict stimulus_data逐值完全相等。来源不是被重采样步骤改写的stimulus_sr字段。channel provenance v2已核验32公开raw header，49 restricted保持未访问，585公开等待下载；每条要求64通道名字/单位/rates完备且rate正、校准有效。实际头采样率与dataset级JSON的8192字段不一致时单独标记，使用实际头，不静默混用。

协调者另行独立完成BDF缩放抽查（`../review/bdf_calibration_20260920_v1/validation.json`）：64通道×2048样本、正负digital均覆盖，独立int24仿射物理校准与MNE最大差约3.09e-11 uV，阈值1e-8，通过。不重复该计算，也不把它冒充全库/同步验证。

新增channel header测试3项通过（至少64通道、正采样率等）；现17项相关测试已分别运行通过。持续等待原始公开载荷完成，并逐批做真实格式/QC，不以这一检查点结束goal。

## 对象级权限补充

已直接查询冻结官方清单并与协调者独立结果一致：`sub-022/ses-shortstories01/...run-09_eeg.bdf.gz`公开，但该录音的103-byte stimulation.tsv及_eeg.apr受限。因此stimulation story额外1条UNKNOWN（总50：49受限raw加此1条）不是下载缺失，不补请求。通道来源表新增独立stimulation_access字段，与raw source_access区分。

raw NPZ来源审计已改为只读取格式日志中明确PASS完成的expanded项，避免将后台正在写入的NPZ临时文件误判源损坏。全量结束后的最终快照会重新对齐完成范围。

## 00:29 全量NPZ证据与角色修订

协调者完成 `../review/raw_stimulus_complete_20260920_v1/` 全量审计：204/204 NPZ，0 HOLD，实际 fs 均48000，72原始波形与publisher stimulus_data逐样本精确相同。本worker不重复波形读取。

基于该完成表生成 `raw_stimulus_roles_20260920_v2/`，不覆盖旧证据。72项标 PUBLISHED_STIMULUS_AUDIO（不证明语音内容/实际播放）；70项标 TRIGGER_BY_FILENAME（含triggers.npz.gz，功能未独立验证）；62项标 AUXILIARY_ROLE_UNVERIFIED（含audiobook_3_noise、audiobook_6_1_swn及noise前缀项）。逐项role_basis保留依据，不能宣称75份已确认原语音。对应脚本支持metadata-only修订。

16:24:18Z 下载3,753/4,142公开文件、77,416,246,959 bytes、0下载失败；格式3,748项，BDF266/660，唯一已知源对象截断HOLD。两个后台进程仍存活；完整公开范围尚未完成。

20项服务器pytest全套通过，36条已有NumPy deprecation warnings，无失败。协调者要求补齐ds004703余96对象格式检查，已启动`pub_01_ds_remaining_formats.py`（PID123435）：18 NIfTI分块解码/有限值，文档/表/JSON静态解析，py只AST不执行，pyc仅读字节并排除使用；不重读281信号、不做成像重建。输出将为server `pub_01/s06/ds_remaining_formats_20260920_v1`，尚未完成时不宣称96已通过。

最终汇总门禁已按独立审阅修正FAILED_FORMAT缺省字段，新增真实截断记录形状与错误路径拒绝测试。门禁5项通过；累计25项相关测试分别通过。尚未执行最终汇总（明确拒绝不齐范围）。16:31:38Z下载95,549,424,451 bytes，0失败；ds剩余格式进程实际正在读取sub-SD015 defaced NIfTI，非停滞。

## 00:34 ds004703余96项完成

`ds_remaining_formats_20260920_v1/` 已回传轻量清单/summary：18 NIfTI全体素分块解码且nonfinite=0；10 CSV、35 TSV、23 JSON、1 ipynb（只JSON）、1 DOCX、2 py（只文本/AST）、1 pyc（只字节/排除执行）、5文本。88 PASS_FORMAT，7 PASS_TEXT_ONLY，1 EXCLUDED_NOT_EXECUTED，无HOLD。与已有281 EDF/WAV全样本QC并集覆盖377；这不把旧代码当可运行，也不将解剖加入benchmark。未执行成像重建、可视化或身份推断。原文件未改。

SparrKULee 16:33:28Z已100,310,599,408 bytes /3,917文件，格式3,915项（BDF433/660），失败仍仅既知孤立截断；继续主下载/QC。

协调者随后独立核验并认可ds余96项与官方待检集合完全相同、18 NIfTI样本数等于shape乘积，角色v2仅改角色依据而数值字段全保留。本worker不会重复这些已完成分支。16:38:59Z下载114,105,054,091 bytes，BDF527/660已解码。全部25项pytest一次性运行通过；最终目录、报告顶部、S01快照及S06 registry将在实际全量完成后刷新为REVIEW，根README/CURRENT等全局入口仍由协调者统一维护；无提交/push。

16:43:38Z下载125,951,935,532 bytes /4085文件，0下载失败；格式4078项、BDF596/660，仅既知坏源1项。最终汇总/验证/审计/通道来源脚本在2203 py_compile通过。实际环境以协调者`../review/runtime_20260920.json`为准：Python3.11.15、numpy2.4.6、mne1.11.0、nibabel5.4.2、pytest9.0.2等，未安装/更改运行环境；保留协调者新增nibabel依赖。
