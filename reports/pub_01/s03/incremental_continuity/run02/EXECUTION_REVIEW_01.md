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

## 工程attempt03（运行中）

PID409806，`engineering03.log`/`engineering_a03/`独立保留。分块代数、detach/reset的参考比较使用2203 CPU float64、绝对1e−10；GPU within-global64历史梯度与无未来/重叠路径仍直接验证。真实三seed×六格epoch−1的全VAL波形容差1e−6、loss1e−7，以及原encoder精度均未放宽；正式训练始终固定global64，不承诺任意GRU分块下GPU输出一致。

实现另将M4的transform按record计算后拼回global64，再一次调用原head。原H16/core/shift/.1公式与参数不变，同一完整64上直接对照冻结NativeResidual；避免零残差时因拆head batch改变cuDNN卷积内核。实际证据待完成后交协调端独立核查，不预报PASS。旧FT复用或共同实现18fit的决定仍须在正式fit前依据技术证据固定，不读新成绩触发。

frozen/joint行同时改变encoder train/eval随机行为，交互不能作为纯encoder梯度因果效应。warm vs旧随机起点还改变上游head训练与预算，不是严格初始化因果消融。旧内部测试和DTU已暴露；后续确认资格另审，不以本轮工程通过或VAL改善完成科学goal。
