# SS07 外部迁移：ds004408 v1.0.8 共享无标签整体增益校准与一次冻结评估

状态：数据范围已获真人授权；协调设计审阅已通过，获取/预处理可继续；以下精确澄清并入预测前freeze，尚未运行任何新数据模型预测。首条真实EEG按header解析的幅度异常，物理增益仍UNVERIFIED；协调已决定唯一全record无标签整体gain校准，见AMENDMENT_GAIN_CALIBRATION.md。这不是单位修复或纯零校准确认。
归属PUB-01-S03_SS07，不新增S/SS编号，不改D11及更早结论。授权原文见 metadata_review/OPENNEURO_SCOPE_AUTHORIZATION_20261102.json；仅server2203。
服务端ROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102；本地只源码、来源/许可/路径/结构化汇总和静态图报告。

## 已核事实与获取

固定Git commit b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be，DOI10.18112/openneuro.ds004408.v1.0.8，CC0，19人×20记录及20音频。20段来自同一《The Old Man and the Sea》叙述，不能称20个独立故事/说话人。
全部760个实际vhdr/vmrk已按annex SHA256和长度核验；380 EEG均128通道A1..D32、512Hz、MULTIPLEXED/IEEE_FLOAT_32、各通道resolution=.1µV，全部marker为空。119唯一channel TSV已在2203逐blob核验并关联380记录，285记录有declared bad，0–34个，不按坏道数排除被试。
首尝试raw GitHub指针URL实际超时，确认UID/argv/start_ticks后终止仅自有metadata进程；日志和switch记录保留。attempt02用同commit官方codeload元数据tar并逐Git blob SHA1核验，随后相同S3文本header校验完整完成，不换版本，不当科学阴性。

原EEG载荷19435625984B，最大125218304B；20WAV共640222064B。不能整套raw落盘。
保留20音频及首条EEG51,380,224B作独立reader单位canary，其余EEG按record内存获取、完整SHA256核验后处理，不保存第二raw副本；不永久删除任何文件。HTTPS Range分块也必须验证每块Content-Range/长度、固定对象版本及最终完整SHA，失败仅重试传输，不改变数据/模型。
下载完成即做有限描述性数据概览和单位/长度/坏道诊断，不等待模型评分，不把数据可读性当独立确认。

## 一个共享适配，预先选择而非按成绩筛选

1. 数值单位：binary float×header resolution是header声明约定，当前全部float×.1；独立MNE reader以V返回并与float×1e−7逐值一致，但真实物理增益未核实。TSV单位V不证实发布转换链，输出不称实测µV。原TRAIN mean/std只读保持；最终数值输入采用下述唯一共享无标签整体gain，不搜索倍率/不按成绩选scale。
2. 空间：明确采用已有MNE1.6.1的标准BioSemi128位置假设，并映射到既有Sparr64合同顺序/标准BioSemi64位置。个体坐标未提供；README BioSemi与JSON BrainProducts差异保留，pybv0.7.5 header只证明转换格式，不证明采集厂商/实测电极。
   不能按同名C1/C2或前64截取。排除TSV明确bad的source道，仅用其余good电极做一次固定spherical-spline映射到64目标位置（MNE alpha=1e−5），随后64道pointwise CAR。每record算子仅依赖固定模板和既有bad标签，不学习目标统计；无其他候选空间算子或成绩驱动排除。报告good道数、row-sum/finite/条件数等无成绩诊断，不把模板当实测几何。
3. 时间滤波：每record原512Hz全长，order1 0.5Hz Butterworth high-pass、sosfiltfilt odd padding，采用brain_pipe0.0.4的MATLAB-emulation padlen规则；空间映射/CAR后scipy.resample_poly到64Hz。公开源管线的MWF/artifact interpolation没有在新数据重拟合，作为域差异明确披露；不是声称重现全部Sparr旧derivative。
4. 音频：librosa原采样率float32 mono（立体声按其固定算术平均，先报告通道结构），使用已安装brain_pipe0.0.4 GammatoneEnvelope默认28个50–5000Hz ERB频带，abs^0.6后**求和**，再同包ResamplePoly到64Hz。不把求和换均值、不做新目标标准化/增益/lag拟合。软件源码/版本/默认参数与现有官方公开管线一并pin；旧published envelope具体revision仍未知，MSE注明音频幅度/处理域差异。
5. 起点：依据README已起点对齐的声明固定0样本偏移；不做新EEG–目标相关峰搜索、time-stretch或目标驱动裁剪。EEG可比audio长，处理后只按二者实际64Hz共同长度裁到min。空marker没有硬件同步证据，零偏移是明确的来源假设，不能宣称物理同步已证实。
6. 无标签输入校准：HP/空间/CAR/64Hz全程FP64，按音频实际长度裁到共同L后，用全部[0,L)样本（包括Hann未覆盖边界）计算 A_record=sqrt(mean_c(mean_t((x_ct−mean_t(x_ct))²)))，ddof=0；A_source=sqrt(mean(original_TRAIN_std²))。单一正gain=A_source/A_record乘全record所有道，不对输出demean、不逐道缩放/新增norm/clamp/网格；只用EEG和音频长度，不用目标包络值/预测/成绩。原good道在filter前全部时间常量，或A_record非finite/<=0，机械失败。校准后才保存64×L C-contiguous float32，沿用原raw_windows的float32→原TRAIN mean/std→float32、双stack/global64跨record顺序和FP32/eval/no_grad上下文。全record EEG估计gain使用未来EEG，明确offline/transductive、无标签整体增益校准后的迁移；不是原TRAIN固定增益纯零校准确认，也不是在线因果。
完整data/units/channel/time/audio长度和mechanical mask在首次预测前冻结，数据接触本身不构成确认资格。

## 模型、比较与无历史新增

三原seed1024/2026/31415；四角色只读固定权重：
- P_FT*：D8原完整FT selected epoch4，原TRAIN positive affine标定用于描述性MSE，Pearson与原raw P等价；
- Ccap：D8已定λ=.01、原TRAIN latent/P/square变换和系数；
- W/F：D9已冻结物理目标系数，原相同TRAIN变换/λ=.01/192；
- 每record i<4保持原P_FT* fallback，原Hann192/stride96一致，无新bias规则或预先去病态处理。

不加入D11阴性PastCov，不依据新成绩添加角色/seed/λ，不训练新模型、不下载新权重。
NN每seed只一次global64全数据同次捕获Z/P，三份cache保存供独立重算；原Decoder路径首尾和非64 record起点工程桥<=1e−6。新数据没有既有NN成绩gold，此桥只证同一冻结实现，不证模型泛化/神经数学复现。

## 评分、统计与F的已知风险

主端点：原共同非零Hann覆盖、完整60s块（3840样本，原网格start0/3840/...；至少3800 covered），record-local。每record每eligible块分别计算Pearson；每person全部这些块等权平均，再在person内平均三seed。不把不同segment拼接后计算一个Pearson。equal-clip secondary先在每record内等权平均块r，再在人内等权平均record并平均seed。
新参与者的独立n为实际纳入人数，最多19，绝不是19×3、380或时间块数。20段共享同一叙述；所有推断条件于这个故事/说话人及冻结适配，不能宣称跨故事独立泛化。

两个固定primary：Ccap−P_FT*、F−W。先每person内平均三seed差值d；sign-flip统计量固定abs(mean(d_person))，n<=19枚举全部2^n符号（含观察排列），p=count(Tperm>=Tobs−1e−15)/2^n，两项p采用Holm校正。“exact”仅指完整枚举，仍假定人独立且零假设下符号对称/可交换，不是无条件distribution-free或因果检验。报告每seed方向和person值；另固定20,000次person-paired bootstrap，seed20261102，percentile分位数[.0125,.9875]，即两primary Bonferroni family95近似区间，Holm2 p另外报告。明确bootstrap近似覆盖及共享刺激限制。禁止重采样window充当独立人/故事。
其余F−Ccap/W−Ccap、各模型绝对r、equal-clip后person平均均为预定secondary描述，不据其选择新primary。

尾段：不完整60s尾不进入主端点；覆盖>=192样本时另报该尾的Pearson，所有尾报MSE/长度/覆盖。whole-record与equal-clip聚合作描述。音频实际时长必须实测，不硬编码论文“约180秒”。
mask在预测前锁定：原np.hanning(192)及stride96构造coverage，coverage>0为covered；完整块covered数>=3800且该covered目标全finite、max>min方eligible。非完整尾段covered数>=192、目标finite且max>min才计算尾r；所有有限目标的covered点均计尾MSE。常量判断为精确max==min，不按观察到的方差/模型成绩增加阈值。失败对象如实记录，其余可进行。模型预测恒定/不有限不得悄悄删掉该模型的低分record：保留失败并报告受影响primary不可评估。所有角色使用相同record/block mask，实际record/block清单在预测前冻结。

独立性证据仅比较既有Sparr/DTU与本集的study/language/book及可用audio identity manifests，不重跑旧NN。本集collection/audio与既有来源的差异不等于能够排除XLSR大规模预训练的音频接触；后者无法逐条核实，明确保留。不得声称20独立故事。实际brain_pipe安装源码/参数、旧TRAIN单位来源/norm、三seed模型和四角色读出fingerprints在最终freeze中pin；MNE单位reader只证header解析一致，不证物理同步或旧Sparr包络revision一致。

F已知窗口MSE约352、近相消bias不能省略。主计算采用D9的FP64 readout/Hann fusion；四角色均报r3s、window MSE、uniform overlap误差、weightedSingle/fused MSE、disagreement与恒等式、max|prediction|、first4/末端行为。
另固定W/F各一个全FP32 readout+FP32 Hann accumulation诊断（相同FP32 Z/P，无额外NN），对主FP64报告波形最大差、fused r/MSE差。不得按诊断切换主precision、移除F或改boundary，不把旧“coef-only FP32”当全路径验证。

## 硬预算、代码与验收

metadata/headers实际核验STORAGE_FEASIBILITY：
- processed EEG未裁音频的保守上界1,214,814,720B；
- NN全长窗口上界48,867，三seed Z1536/P192 float32 cache含header上界1,013,306,880B；
- 保留20音频640,222,064B、首EEG canary51,380,224B；
- 加128MiB metadata/code/report余量，新ROOT保守3,053,941,616B<3GiB；
- 全SS07 NPY保守28,499,925,984B<28GiB；每次写入前检查实际累计与free>=20GiB。NN cache不因扩展名绕cap。
不得复制整套raw EEG，不覆盖旧数组，不删历史/他人作业。

CPU BLAS主计算最多8线程（避免D11小矩阵过多线程和每record重扫开销）；header网络并发8、body最多4；RAM每worker<=8GiB，推理每GPU<=4GiB，两L40可并行两个seed。资源巡检按固定每50record/每100batch以及写大文件前，不把每record全目录扫描当模型成本。wall分获取/处理/三seed推理/评分/独立审核单列；初始总预算各stage4h上限，失败保留、按实际transport和预算透明纠正，不无限自动重试。
缓存总量在完全冻结data mask后复核更紧的准确值；只有存储可行才开始批量processed/NN缓存写入，超预算先压缩存储策略，不按科学成绩改模型。

允许新改范围：SS07 openneuro_*获取/adapter/prepare/evaluate/statistics/report代码、本run轻量config/报告与既有任务入口/Completion Record/唯一SUBTASKS登记；不改D11及更早frozen证据、parent科学问题或患者边界。
工程门：760header/119TSV/全部载荷SHA，MNE单位reader、模板/坏道算子常数/shape/finite、原TRAIN norm单位、固定gammatone/resample代码、真实时间长度共同mask、原Decoder同次Z/P桥、FP64读出与独立缓存重建。所有门不使用新成绩选adapter。
协调审阅本PLAN→完整source/config/model/mask/statistics freeze→一次正式四角色×三seed评估→独立原输入/数值及统计重算→中文逐图解释与三级验收/归档。任何正结果都不自动建立连续性机制/方法新意/ICML或NeurIPS充分性，scientific goal仍未预判完成。
