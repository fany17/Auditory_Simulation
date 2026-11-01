# SS07 warm round02 实现与工程记录

2026-11-01协调端独立审阅8701673方案后，明确批准独立warm round02六格三seed、固定5epoch、epoch−1候选、原3s选点与所选VAL60s；总预算预先最多18fits。当前只实现与2203工程核验，正式train/val尚未开始。持续goal ACTIVE，不是新确认测试或投稿验收。

新server ROOT：`/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_warm_20261101`；本地`reports/pub_01/s03/incremental_continuity/run02/`。round01源码、信号/缓存、预测、权重与报告只读；无旧test/DTU推理、新下载、患者/STN、远端Git或PR合并。

## 实现要点与预执行检查

- `pub_01_s03_ss07_warm.py`仅接受train/val，保留原seed预训练encoder及训练完成head。新readout在隔离CPU/全部CUDA RNG的构造域内建立并加载训练完成head，M4末投影零；frozen保持encoder eval及requires_grad=false。
- joint的raw输入沿用原RawWindows/归一化，全局64先完整encoder前向，之后按record拆M4/state，再拼回global64只调用一次原head，使零残差保持原完整head batch；每global64一次原Pearson和Adam更新，更新后detach不reset，record/epoch/phase重置。past仍为非重叠parity历史，current同结构当前输入。
- 全六格三seed固定5epoch，epoch−1在更新前实际全VAL并保存候选，严格最早3s改善选点；incumbent/selected均报告原Hann60s/person，不以该统计重选checkpoint。输入、优化器、LR、窗口重复和上游成本分别记录，不把更新数一致当FLOPs一致。
- `warm_check.py`以合成数据检查全部可训练tensor实际非零梯度可达、within-batch历史梯度、跨batch detach、数值state保留、reset与跨record/未来路径；actual epoch−1覆盖六格三seed全部30500 VAL窗，并独立重建legacy构造/CPU-CUDA RNG与跨record global64。
- `warm_prepare.py`只盘点或在协调源码/实际工程证据独立核查后冻结。`warm_queue.py`两GPU最多18fit，无自动重试/扩预算；失败保留，其他已启动合法cell不被擅自杀死。源码/上游/工程证据均绑定精确hash。新权重best及临时双份有界，不建新latent cache。

## 工程attempt01（已终止，保留）

PID405605正常作为失败退出；`engineering01.log`和`engineering_attempt01_input/`保存原始日志与全套源码，未开始科学fit。合成参数梯度部分执行后，历史依赖探针在eval模式请求cuDNN GRU backward，触发PyTorch明确错误。修正仅为探针使用train模式；直接喂synthetic z，不调用随机encoder，不改科学网络/容差/预算。正式训练本来就在train模式。

## 工程attempt02（已终止，保留）

PID406944在合成分块一致性检查失败后退出，`engineering02.log`/`engineering_a02/`和`engineering_attempt02_input/`保留。2203独立诊断同一synthetic z：FP32 CUDA两种recurrent长度分块最大输出差.0300827，最大输出206.223，scaled差.000437799；同值CPU float64差2.84e−14/scaled2.88e−16。原TF32 cuDNN下不同GRU长度内核不能主张逐bit/绝对1e−6分块恒等。诊断只有合成z，无新科学fit或实际VAL调门。

## 工程attempt03（已完成，原版本证据保留）

PID409806已正常结束，实际18格×30500 VAL起点及合成检查通过，用时78.66秒；`engineering03.log`/`engineering_a03/`与`engineering_attempt03_input/`保留原完整源码和证据。分块代数、detach/reset的参考比较使用2203 CPU float64、绝对1e−10；GPU within-global64历史梯度与无未来/重叠路径仍直接验证。真实全VAL波形容差1e−6、loss1e−7，以及原encoder精度均未放宽；正式训练固定global64，不承诺任意GRU分块GPU输出一致。

实现另将M4的transform按record计算后拼回global64，再一次调用原head。原H16/core/shift/.1公式与参数不变，同一完整64上直接对照冻结NativeResidual；避免零残差时因拆head batch改变cuDNN卷积内核。旧FT历史runtime RNG/backend快照未记录，当前重建等价不证明历史逐bit路径；执行前固定18格共同实现，不是发现旧函数/seed不匹配，也不读新成绩触发。

## 工程attempt04（最终评分修正后实际通过）

协调独立代码核查指出，np.std单独判常量可能接受0.1重复导致的均值舍入伪方差。新warm实际correlation加入精确max==min拒绝并断言r有限；macro全常量窗亦先拒绝。补实际评分函数的非二进制常量pred/target及macro拒绝、正常非恒定macro与原corrcoef不变检查。round01代码不改。a03结束后才修改源码，其综合标记/复用决定移入engineering_a03原位保留，不将新hash冒充已验证的旧内存版本。

完整a04已正常完成，79.32秒，18格×30500实际VAL。各行current/past与P的波形和loss最大差均0；9例构造CPU/全部CUDA RNG/完整global64 dropout重建及训练零投影波形差0；对旧FT initial VAL loss最大差1.48767e−8。新评分常量pred/target与实际macro拒绝通过，正常非恒定macro与原corrcoef差3.12e−17。实际旧FT checkpoint keys只有epoch/model/seed/validation_loss，复用技术边界如上。

独立engineering04.log/engineering_a04/及最终ENGINEERING_COMPLETED仅由a04运行生成。ENGINEERING_COMPLETED SHA=5318eb3c4dd98241d8c5f73cb5a18edc72cf0e52c24f992d7bb365440f0265bf；复用决定SHA=a82f08d1346178cc30b4a15ba863ebb9998833b407d2e44d13e3a9663a7064c6。最终源码7fb74d6，SOURCE_REVIEW_BUNDLE绑定全部来源；轻量JSON已回传，不含synthetic/真实信号数组。此时无正式fit或测试评分。已交协调端做最终独立核验，通过后冻结并进入正式train/val，goal不在工程PASS处完成。

## 独立核查、冻结与实际启动

协调端直接在2203写入[INDEPENDENT_CODE_REVIEW](INDEPENDENT_CODE_REVIEW.json)，decision=PASS_FOR_TRAIN_VAL，SHA0d208252e5ad26328237721aeeea2197b096088f3c1beff1a806ee59dac87587；24项current source、a04和三份proof/复用决定实际核验，round01训练/eval/review来源仍匹配。此文件原样回传，执行者未改写独立归因。

已调用已审prepare --freeze，[WARM_PROTOCOL](WARM_PROTOCOL.json) SHA=a9e1d220170a9c6a166c71c8113ed6ede0b6570a5aab3152e5a3ee5a685481a8。18fits/355050更新/22720860监督窗上限在任何正式fit前固定，无test操作。两GPU队列416910启动，首批frozen_p/1024 PID416911与frozen_p/2026 PID416912；实际/proc UID1012、argv与start_ticks见[START_VERIFIED](START_VERIFIED.json)。启动不是完成，后续全18格结果与科学接受仍待。

## 全矩阵汇总排队

独立`report_input/`放置warm_report及report_queue，与冻结训练`input/`分开；不改变其24 source/7 proof。REPORT_PROTOCOL SHA217a3e41758b600add934659c7126d1e08f605182d1d483868f36d61c7a060bb，固定只在18格全部完成后运行一次，全格/全seed/全部配对差与完整成本，不训练、重选模型或读取测试。静态语法检查通过，实际报表运行/图形检查尚待。

report queue PID423181的实际argv已核实，WAIT_ALL18_TRAIN_VAL，按/proc argv/start_ticks验证上游416910活跃，不把状态文件当进程证据。训练/汇总均无自动重试；异常保留证据。实际检查时已完成7/18，frozen_p/current前三seed全完成、past1024完成，其余按固定队列继续，未按途中成绩改变设计。汇总后还需实际图解、独立复核和科学判断，自动报表不是验收。

## 最终选中checkpoint只读复核准备

新增隔离report_input/warm_verify，source SHA87d230eaf13839243138aa998bb53f73c48ef3119dd1cc4434dc1e4698b5cc66，只在18格训练及完整报表结束后运行；训练source/protocol/报告source不变。固定检查全部18格，逐tensor比较原P/warm初始与所选权重，冻结encoder必须精确不变，epoch−1选择时全部权重必须不变；其他参数的实际变化完整报告，不因某参数未变而补跑。

复核从既有未归一化EEG derivative重建原VAL的global64输入及M4状态，复用冻结neural模块，不调用主Data.batch/latent_forward/validate/macro；独立调用者和float64公式重算30500窗与49人/752块，每格都保留差异。frozen格另对同权重的既有latent cache波形；最终3s/60s均值容差1e−7、块/人1e−6、raw/cache波形1e−6，同一数组两套Pearson数学保持1e−10。上述为数值复现检查，不能当科学增益阈值。不是原BDF/音频预处理重做或独立neural数学，也不是测试、拟合或新选择规则。

在2203已做独立公式合成检查：非二进制常量pred/target拒绝通过，正常公式与corrcoef差2.43e−17；最终证据见FINAL_REVIEW_MATH_CHECK_FINAL。此前初稿检查单独保留。全18格真实权重/重放尚未运行，不预报PASS。脚本等待实际report handle和完整标记，无自动重试，异常保留；随后协调端仍需实际查看结果与图表作科学判断。

frozen/joint行同时改变encoder train/eval随机行为，交互不能作为纯encoder梯度因果效应。warm vs旧随机起点还改变上游head训练与预算，不是严格初始化因果消融。旧内部测试和DTU已暴露；后续确认资格另审，不以本轮工程通过或VAL改善完成科学goal。
