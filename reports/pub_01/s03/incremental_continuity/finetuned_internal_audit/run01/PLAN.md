# D10：冻结强FT读出与物理目标的旧内部泛化审计

状态：具体矩阵/桥/预算待协调审定，现仅只读JSON/源/NPY header盘点；未读取新test数值、编码或评分。仍属PUB-01-S03_SS07，不新建任务节点。D4-D9全部冻结。

## 1. 范围、暴露与固定角色

一次审计既定P_FT*/P-only/C/Ccap/W/F×seed1024/2026/31415×seen/heldout，共**36角色–split评分**。这是第一次将D8/D9对应FT的冻结读出参数用于这些旧测试；原P_FT本身已经在round01评估。所有记录历史已暴露，统一标 **POST-HOC INTERNAL EXPLORATORY**，即使本轮预测前冻结，也不称新确认、无偏泛化或独立验证。

不拟合任何参数/统计、不换λ/seed/角色、不按test选择方法；DTU不再前向、不下载、不启动新数据或患者/其他节点。保留D9单窗严重表示退化与固定Hann最终输出的小增益两项事实，不以本次分数回改D8/D9。

P_FT*为原epoch4完整FT encoder/head输出经**D8 TRAIN**固定正a,b；P-only用D8共享λ=.01、C用.0001、Ccap用.01，W/F用D9已完成系数（λ_scalar=.01/192）。FT latent/P/平方统计、常量mask、block scaling、所有bias均继承D8/D9，不在测试重标定。W/F的0/191固定端点和大bias原样保留。

六角色均按原192点/96stride/Hann/64Hz输出；五个learned角色每record首4窗回退P_FT*，其余只当前窗输入。P_FT*本身无新边界修改。原FT旧测试没有learned首4回退，故不得把不同读出的边界策略和预算差异称纯loss或等预算消融。

## 2. 固定mask、汇总和问题

| 原split | records | people | source-family labels | windows | 有效60秒块 |
|---|---:|---:|---:|---:|---:|
| test_seen | 69 | 69 | 9 | 36573 | 872 |
| test_heldout | 75 | 13 | 8 | 48119 | 1172 |

heldout仅相对原训练fold的13人，非整个研究未暴露；来源/刺激/同人依赖不变。原prediction-independent mask共2044块不改，块→person宏r，再原2/3 seen+1/3 heldout计权，不按记录/窗数加权。

三个预定主要**内部诊断**配对：F−W（D9目标差）、Ccap−P_FT*、Ccap−P-only（D8再读出差）。每项同时给seen/heldout/原2:1及三seed点差；secondary F−Ccap、W−Ccap、C−P-only。不做p值/CI、显著性/等效或新SOTA声明，不由新排名选λ或再次训练。

并列报告3秒r/undefined、uniform窗口MSE、fused/加权单窗target MSE、Hann disagreement、uniform96 overlap；用原D7 measure/aggregate的常量/undefined规则，不按窗口表示退化删窗或删有效块。F原大单窗值及物理融合结果都呈现；不同split均可有合法阴性，不能追阳性扩矩阵。

历史同三seed原P_FT已保存原2:1均r=.168944726368672，2044块keys相同，来源为D7 COORDINATOR_HISTORICAL_FT_REFERENCE；它不是新角色/新推理，也不把旧五seed均值当匹配三seed参考。D10原P_FT*应通过正仿射Pearson不变恢复此参考，先做实现桥，不将其当未见确认。

## 3. 原FT测试实际路径与资产

原路径是`SparrRecords.finetuned_predictions → eval_common.predict_raw_phase`：每phase独立连续global64，**可跨记录混batch**。seen=572批、尾29窗；heldout=752批、尾55窗。它不是native latent的record256，也不是按record切开的aligned64 head路径。

原raw_windows从同record EEG按TRAIN mean/std做`np.stack(...).astype(float32)`；helper再`np.stack(pending)`、torch.from_numpy→device，保留实际input stride、FP32、no_grad（非inference_mode）、eval、requires_grad=True、TF32 matmul=False/cudnn=True及原默认benchmark/deterministic签名。每batch按原Decoder `encoder(x).transpose(1,2) → head`，**P与Z同次获得**，不从落盘重排Z再算head。

原144×3条P_FT只有未缩放float64 fused波形，未保存raw单窗P或FT test latent。已核canonical final_latent属于native encoder，不能充当FT；D8只存FT TRAIN/VAL。实际旧raw review已对每seed3条Sparr P_FT记录核过原单位波形（非全144记录），本轮将对每seed全144记录做桥。

新代码复用原`raw_windows`和`predict_raw_phase`，但**不调用SparrRecords构造器、encode_test或旧eval main**，避免创建/覆盖历史final_latent及启动DTU。可使用只含all_rows/mean/std的只读context调用原unbound raw_windows。加载原FT Decoder/epoch4 checkpoint，保持同源码/参数标志；用无hook的capture wrapper执行同encoder→head操作并记录phase顺序Z/P，避免hook改变Transformer路径。只新增D10独立目录，不改原checkpoint/cache/frozen evidence，不初始化远端Git。

## 4. 真实测试前的必要核验及正式顺序

实现/源/协议在真实新数据前冻结。仅验证本次改变路径：小合成多record/双phase global64 scatter/尾批和首4回退；不重跑旧ridge/PCG/metric公式。

1. 只用既有FT VAL缓存，六角色×三seed×原首/尾record=36项cached读出映射桥，对D8/D9保存block r及当前canonical预测（预测≤1e−8、block r≤1e−10）。这不编码test或拟合统计。
2. 每seed只对原VAL首64/末36 global batch，用新capture在同FT模型上前向，与D8已存gold Z和未缩放P核对，Z/P最大绝对差均≤1e−6。共300额外encoder/head窗，不保存新VAL数组、不用a缩误差；同时记录actual strides、grad/eval/precision。它核对新capture的latent/P与既有FT编码路径。
3. 上述通过后，按原两phase global64正式一次编码，保存各phase连续float32 Z/P。每phase首/尾batch同输入另调原Decoder.forward与capture P做原单位≤1e−6桥：每seed64+29+64+55=212、三seed636额外窗。主编码中的P/Z不重算、不重复整phase。
4. 每seed全部144记录的raw P用原Hann融合，对旧保存**未缩放**P_FT波形逐点最大绝对差≤1e−6；另核P_FT*对旧原FT每块r差≤1e−10及全部2044块/原2:1。这是fused波形桥，不能称旧单窗raw gold或独立神经数学复现。完整seed桥通过后才应用五个既定learned读出，不根据成绩决定是否继续。
5. 对固定六角色完成原集合评分，所有first4/shape/finite/mask/constant规则保持。输入header和权重/transform/coef身份在前后核对；失败保留，不放宽门限或静默换batch/layout，其他无关可独立工作不被阻断。

正式main encoder/head254076窗/3972批，工程额外300+636=936窗，合计255012窗；无NN优化或新ridge解。旧raw wave全432记录桥与新增capture路径核对只服务实现验证，不称科学确认。

## 5. 精确空间与运行预算

server ROOT：`/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_internal_audit_20261101`。metadata实际盘点见INPUT_INVENTORY，未读test数组值或算新分数。

| 项 | 精确预算 |
|---|---:|
| 当前全SS07 NPY | 24469431456 B |
| 三seed×双phase Z/P载荷 | 1756173312 B |
| 12个128B NPY header | 1536 B |
| 新NPY总量 | 1756174848 B |
| 预计全NPY | 26225606304 B |
| 25GiB上限 | 26843545600 B |
| NPY剩余headroom | 617939296 B |
| D10整个new root上限 | 2GiB = 2147483648 B |
| metadata盘点free | 27662708736 B |
| 扣满2GiB后free | 25515225088 B |
| 必保free reserve | 20GiB = 21474836480 B |

只保存两phase Z/P，不复制旧fused、EEG或learned fused全数组。无额外full feature/cache；评分逐record临时运算。每份新cache写前验证预计总量，严格同时守25GiB总NPY/2GiB root/20GiB余量；不删除历史数据腾空间。图文/audit属于root上限，NPY尚有约589MiB余量，不批准其他数组。

单GPU、三seed顺序，encoder torch2线程、CPU读出BLAS≤8线程；GPU allocated≤2GiB、RSS≤8GiB、全run wall≤60min。硬界不是未经实测的时间预测，代码记录主前向/额外工程窗、全部36评分/record/window/block数、head/encoding/读出/score/写入成本及继承训练成本。Windows/Git仅轻量JSON/CSV/源/静态图和中文HTML，不回传Z/P/原信号/权重/coef。

## 6. 独立复核与交付

技术核验包括实际FT输入签名、300窗Z/P gold、636窗原Decoder桥、432record原raw fused桥、baseline旧block r不变、共同mask及36完整分数。协调独立重聚合全36项、核source/cache/参数身份，并按原manifest两phase各首/尾record×3seed×6角色=72项cached重建，覆盖F窗口/融合严重差异，不加新神经前向或新拟合。具体review scope在正式分数前保存。

三张主静态图（seen/heldout/2:1六角色绝对值；六组固定配对；fused与window/disagreement联合度量，必要时log展示大值）及完整中文解读、全表/成本/失败、REVIEW三级判定、轻量HTML、正常commit/push。数据范围/暴露/有限预算/历史阴性及F表示限制全部保留；若呈现大尺度差需补图，仅已存结果展示并保留原图，不重开科学矩阵。

Technical/Acceptance/Parent consistency分别判定；阴性亦是合法交付。D10一次内部审计完成不代表SS07/S03/PUB或持续goal完成，不宣称新确认、通用artifact或额外历史贡献。PR6仍Draft，不合并；不自动启动下一模型/数据/患者任务。
