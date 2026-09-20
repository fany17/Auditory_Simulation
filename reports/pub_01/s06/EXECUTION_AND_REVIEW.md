# PUB-01-S06 公开范围完成报告与独立复核

最终状态更新（2026-09-20）：协调者已完成独立验收，**COMPLETED / PASS_WITH_LIMITATION**。见[最终验收报告](../COORDINATOR_REVIEW.md)。下文REVIEW / READY_FOR_REVIEW保留为执行方提交时的状态，后续以最终验收决定为准。

2026-09-20。状态 **REVIEW / READY_FOR_REVIEW**，获准公开范围数据准备已实际完成，交协调者最终验收；不自行写DONE。S01 benchmark准入仍HOLD。所有信号操作在server2203，Windows仅轻量表与代码，无训练、无患者/STN、无hash审计、无Git提交/push。

## 最终结果（以下明确标为历史的段落只保留过程）

- SparrKULee V3.1：4,142/4,142公开文件，135,967,030,049 bytes，0 missing/partial/下载失败；196受限不请求。逐项格式4,142覆盖=4,141可读+1孤立截断源对象HOLD。660 BDF全样本int24解码且校准有效；独立审阅确认前64 EEG通道无整段平直。617公开来源header核验、49受限来源保留，0未知/待下载。实际公开来源头1024Hz，与dataset sidecar8192Hz冲突显式保留，使用实际头；发布derivative64Hz，不混用。
- 204 NPZ实际fs均48000；72原始波形与published stimulus_data精确一致。角色修订为72 published stimulus audio、70 filename trigger、62 auxiliary unverified，不声称75确认语音或证明播放。73 data_dict中72可读且对应envelope一致，1 `podcast_35-1`孤立截断已独立复取确认，无EEG/envelope引用；不掩盖、不反复重试。
- 666独立EEG（668映射含2别名）完成common-prefix虚拟视图全部样本核验与3位置直接数值比较；26大尾差flag保留。ds004703复用377对象/14,173,350,514 bytes并完成281信号+96余格式盘点；319虚拟片段全读取、33次原EDF比较差0。没有新增盲目滤波、重采样、归一化或ICA。
- 最终交付`final_handoff_20260920_v1/`及根级五类正式文件；总4715对象条目、1529排除/审阅/受限条目。recording表1337行是11EDF+666虚拟EEG+660raw格式证据，存在来源重叠，不能当1337独立录音。`validation.json`计数、路径唯一性、数值证据与边界检查PASS。

下载于16:47:36Z完成；格式于16:48:06Z终止；两进程均已退出。最终只读对账`reconcile_20260920_final_v2/`，通道表`sparrkulee_channel_provenance_20260920_v3/`，S01当前6表已刷新为`current_audit_20260920_v2/`，不再保留RUNNING为当前状态。

独立证据：协调者`../review/sparr_completion_20260920_v1/independent_summary.json`核验所有公开路径、大小、0受限payload、4142 QC唯一全集、660 BDF校准与前64通道、TSV列宽；另独立核验ds96对象与NIfTI shape乘积、204刺激表和角色修订、ds虚拟loader、666独立EEG及BDF物理缩放。26项pytest全套通过、服务器py_compile通过；36条NumPy内部命名弃用警告保留。实际环境见`../review/runtime_20260920.json`（Python3.11.15、numpy2.4.6、mne1.11.0、nibabel5.4.2）；未安装依赖或改运行环境。

最终汇总首次因官方表已含dataset字段而出现TypeError，后续validator因无产物FileNotFound；发生在输出目录创建前，未覆盖任何结果。已修复字段合并并新增重复dataset列/错误dataset拒绝测试，重跑汇总和验证通过。早期失败与快照继续保留。原根级两份历史ds表移到`historical_root_ds_qc_20260920/`备份，无永久删除。

技术结论：获准公开范围下载/格式盘点与初步虚拟清洗 **PASS_WITH_LIMITATION（待最终独立验收）**。科学边界：物理同步UNKNOWN、split UNASSIGNED、历史publisher生成版本未知、26大尾差、49受限raw及额外1受限stimulation、ds不可变版本身份/许可冲突和Catalan播放来源未解决；不升级为模型可训练/科学有效性接受，不执行S02。

### 当前收尾复现入口

在2203使用上述实际Python环境，脚本位于`/home/fanyu/auditory_simulation_m6a/`。下列`<new>`必须是该项目下不存在的新目录，保留全部旧快照；不需要重跑既有波形检查来复现轻量汇总。

1. `pub_01_s06_reconcile.py --output <new-reconcile>`：依据冻结官方清单、现有载荷大小与format日志。
2. `pub_01_sparrkulee_channel_provenance.py --raw <S06/sparrkulee_v3.1/raw> --inventory <S01/audit_20260919_v1/sparrkulee_official_inventory.csv> --virtual-index <S06/sparrkulee_virtual_20260919_v4/cleaning_index.csv> --output <new-channels>`：来源header，不重复全波形。
3. `pub_01_s06_final_handoff.py --reconcile <new-reconcile> --channels <new-channels> --output <new-handoff>`；随后`pub_01_validate_final.py <new-handoff>`：拒绝不完整公开范围、错误HOLD路径/重复字段与非唯一对象。
4. `pub_01_audit_handoff.py --output <new-S01-snapshot>`：不改变split，生成当前六表。
5. 角色metadata修订：`pub_01_sparrkulee_raw_stimulus_audit.py --metadata-only-source <S06/review_raw_stimulus_complete_20260920_v1/raw_stimulus_inventory.csv> --output <new-role-snapshot>`。

本次最终本地6份根级产物与服务器final快照副本逐文本相等；1104个S01分组项全UNASSIGNED，当前dataset无RUNNING；617公开header均1024Hz/uV，616 story匹配加1受限stimulation UNKNOWN。所有检查只读轻量文件，没有主动生成/读取hash。

### 最终协调者独立验收回执

协调者已报告最终产物验收通过，机器证据见`../review/final_artifact_acceptance_20260920.json`：4715唯一对象与官方全集匹配、大小和访问计数准确、617来源头各64维/49受限/0pending、1104分组/配对/时间键一致。本worker保持执行交付REVIEW；全局入口、COMPLETED及goal关闭由协调者统一处理。没有继续实验、训练、提交或推送。

## ds004703：真实初步QC与虚拟清洗

输入是服务器现存 `data/ds004703/v1.1.0`。协调者的独立清单核验位于 `reports/pub_01/review/ds_inventory_20260919_v1/`：377个官方当前S3对象与现存文件大小一致，总14,173,350,514 bytes。这不是本轮重新下载，不代表不可变版本或内容完全一致；无需重复下载大小已齐全范围。本轮实际新计算如下。

- 11个EDF全部神经样本按20秒块读取；270个WAV全部样本读取。
- 377项大小对账另有完整格式范围盘点：281个神经/音频文件全样本QC；余96项已补齐格式/静态读取，包括18个defaced NIfTI全体素分块检查、表格与文档解析，以及1个旧pyc明确排除执行。27项anatomy类别含影像及sidecar，不纳入benchmark/重建；没有解剖重识别。格式可读性仍不等于全部语义或科学有效性。
- 神经样本未发现非有限值；事件越录音边界为0。平直通道逐记录存在并记录，未静默填补。
- 按官方bad标记、README的C前缀闲置规则、非神经通道、未知单位/元数据、非有限/精确平直规则生成独立通道排除。
- 438个事件连续块逐一与模板的word/POS/phone、duration和offset核对。319个英语passage进入候选清洗索引；78个control与41个Catalan块排除。
- 根据发布包实验脚本第178–183行选Block目录、第88–98行选root控制音频。直接解码波形比较汇总副本：59对完全相等、7对不相等。6个Block中的Catalan标记音频均与pleasePressSpace波形相等，不能以另一个catalan_v2文件“修好”播放来源。该冲突保留。

canonical新结果：`server2203:/home/fanyu/auditory_simulation_m6a/pub_01/s06/ds004703_qc_20260919_v2`。本地副本：`ds004703_qc_v2/`。v1及S01最早438/HOLD表保留为历史尝试，不覆盖。

### 可消费的虚拟derivative

`scripts/pub_01_virtual_derivative.py` 提供 `VirtualDerivative(folder).read(recording,start,stop)`；利用已冻结通道/片段索引从原EDF读原采样值，通道名与原索引、采样率、sample_offset=0、time_scale=1写入映射。片段窗口用floor(onset*rate)到ceil((onset+audio_duration)*rate)的半开区间；这保留整段并最多扩一采样边界，不改变时轴。

已实际通过loader读取全部319片段的每个索引样本；11录音各3个位置另开原EDF直接比较，33次最大绝对差0。验证见 `virtual_validation.json`、`virtual_segment_validation.csv`、`virtual_numeric_validation.csv`。这是真实可消费的索引型虚拟derivative，依赖原EDF可用，不是独立信号副本；未滤波、重参考、重采样、ICA或插值。

### 冻结阈值依据与边界

实际执行配置存于结果 `executed_config.json`，源为 `configs/pub_01_s06_initial_qc.json`。模板duration容差1ns和offset spread 1µs只用于发现文本时间模板不一致，远小于采样间隔，不证明生理/声学同步。audio-template尾差[-0.05,5]秒沿用既有公共metadata adapter的粗截断筛查阈值，非从本轮结果拟合，也非新科学时窗；较长未标注尾段/越界被排除，不为增加PASS放宽。physical audio sync仍UNKNOWN。

空可用索引按显式schema输出；未知通道类型/单位和音频非有限值明确排除。零可用数据是有效失败结果，不靠索引空表崩溃隐去。

## 历史过程：SparrKULee首批下载与QC（非当前状态）

公开范围4,142文件 / 135,967,030,049 bytes；196 restricted不请求。原PID105526在核实是本任务进程后仅对其发TERM，保留已有文件/partial/log；因小文件排序阻塞早期信号QC，改为core provenance→derivatives→其它metadata→其余public，恢复PID107053。仍3 workers，无重复并行下载任务。

2026-09-19T15:35:35Z，恢复进程已完成315文件 / 5,155,121,390 bytes，失败0。此计数为本次恢复进程已访问完成的对象，另有先前已落盘文件尚待重访；不能当全磁盘总数或全库完成。

首批快照 `sparrkulee_qc_v1/` 扫描277个完整NPY（204 EEG、73刺激envelope），全样本检查格式/非有限/平直；204 EEG因与对应envelope样本长度不一致HOLD。64Hz仅有作者当前pipeline依据，NPY自身未携带完整channel/unit/timebase，尚不能认为已证明本次历史文件的时轴。无擅自裁剪、补零、重做滤波。官方预处理参考：<https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/preprocessing_code/sparrKULee.yaml>。raw BDF/NPZ和data_dict格式QC尚未完成；没有危险地unpickle未知对象。

## 复现命令（2203）

使用 `/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python`：

1. `pub_01_s06_download.py --inventory <S01官方CSV> --output <S06/sparrkulee_v3.1>`；自动复用完整大小文件/续传partial，文件锁防同路径双启动。
2. `pub_01_s06_qc_ds004703.py --config <pub_01_s06_initial_qc.json> --output <新的QC目录>`。
3. `pub_01_virtual_derivative.py <QC目录>`；读取全虚拟片段并直接比对原数据。
4. `pub_01_s06_qc_sparrkulee.py --raw <S06/sparrkulee_v3.1/raw> --output <新的快照目录>`。

## 历史阶段独立复核结论（当前结果见顶部）

- Technical：ds全样本QC与虚拟loader实际验证通过；下载器5项无网络测试通过。完整SparrKULee范围未完成，不能写整体PASS。
- Acceptance：ds初步QC/虚拟derivative可供协调者审阅；S01物理同步、完整版本准入和leak-safe split仍待解决。SparrKULee下载ACTIVE、配对HOLD。完整S06未验收。
- Parent consistency：不改变PUB范围、acceptance或科学endpoint；原始数据保留，全部载荷留2203，未解锁模型。
- 尚需：继续下载/对账所有公开文件，完成SparrKULee全范围格式与实际时轴/单位核验，给出最终分支验收；受限权限等待用户，不绕过。

失败记录：S01首版语法错误已修复，服务器py_compile通过；S06初次环境路径错误未启动；官方GitHub YAML服务器读取45秒超时，本地官方网页可读，未伪称服务器来源文件已取得。这些尝试均保留，不影响已核验事实。

## 后续纠正与当前快照（2026-09-20）

上文首批NPY不等长HOLD仅是v1历史判断，不是当前结论。已直接核验作者 `technical_validation/util/split_and_normalize.py` L48–86，作者从0取共同长度。因此按此正式消费约定提供CommonOverlap loader，**不继承相邻split或归一化**。v4含冻结config：666独立EEG全共同区间读取和3位置原数值对照通过；26条超过192样本（3秒）的尾段差仅作为审阅flag，不通过改变阈值消除。668 source条目中2条是同录音.bdf/.bdf.gz别名，已去重并保留alias。

官方73个data_dict已覆盖：72可读并与对应envelope直接数值一致，1孤立对象podcast_35-1截断。受限unpickler新增仅标准OrderedDict和不可执行作者loader占位符，不运行任意序列化函数；首版不兼容错误和源对象截断错误分开。源对象独立复取字节相同且同样截断，未修复/覆盖原件。该孤立名字无对应envelope/EEG引用，正常podcast_35独立使用其正常对象。

14项服务器unittest通过，包括任意global拒绝、占位符REDUCE调用拒绝、empty schema、断点续传不被服务器支持时保护partial、BDF头校准、共同区间样本边界。666路径与冻结官方清单相符亦获协调者独立复核。当前入口见README.md，S01新增六类机器可读当前快照及AUDIT_REPORT.md，不再把原438/HOLD表当最新状态。

## 2026-09-20 余格式与刺激角色独立审阅

ds004703余96对象已由`pub_01_ds_remaining_formats.py`实际完成只读检查，产物`ds_remaining_formats_20260920_v1/`。18个已defaced NIfTI通过header及分块全体素解码/有限值检查，未进行成像重建、可视化或身份推断。其余表格、JSON、DOCX、文本均静态解码；2份py通过AST但未执行，1份pyc读原始字节后明确排除执行与数据使用。合计88格式PASS、7文本PASS、1字节码排除，无HOLD；与既有281 EDF/WAV全样本QC合并覆盖377对象。

协调者独立核验96唯一待检官方路径无缺失/额外、18 NIfTI样本数等于shape乘积且nonfinite=0，认可该范围盘点；未重跑整个信号分支。377大小相符仍不等于immutable snapshot身份或物理同步证据。

协调者完成204 NPZ全量审计，72份与published stimulus_data逐样本一致；本worker基于完成metadata修订角色，保留原v1。`raw_stimulus_roles_20260920_v2/`中72项为published stimulus audio、70项仅filename trigger、62项auxiliary role unverified。协调者已独立确认v2的全部204路径与原数值/率/比较字段未改，仅角色与依据修订；不声称75份确认原语音，也不以filename证明播放内容。

相关测试累计25项已分别通过，最终汇总新增完整范围、已知失败精确路径、缺少format字段的真实失败行形状测试。完整SparrKULee下载/格式仍在进行，整体完成以最终对账为准。
