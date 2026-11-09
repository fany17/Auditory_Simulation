# D8：完整微调强encoder上的最小TRAIN/VAL读出迁移

状态：只读盘点已完成，具体方案待协调审定；尚无新encoder前向、cache、拟合或评分。仍属PUB-01-S03_SS07，不新建任务节点。

## 1. 科学问题与依据

检验D6在原冻结encoder上出现的监督再读出/latent映射增益，是否也存在于原完整P_FT encoder/head。依据为D6的TRAIN/VAL归因与保留强baseline的研究要求，不以D7旧测试排名选择角色或超参数。D7/D4-D6结果冻结；本轮仅原TRAIN/VAL，不读取旧test/DTU，不下载、不加新backbone/历史模型、不进行NN优化。

旧P_FT继承完整预训练/head，另有每seed5epoch/19725更新/1262270监督窗、3372617可训练参数的联合微调；三个seed的既有best均epoch4。checkpoint、原normalization和历史选点只读。不能用nativeP latent或系数代替FT。

## 2. 实际资产、空间与一次前向预算

2203根：`/home/fanyu/auditory_simulation_m6a/pub_01/s03_finetuned_readout_transfer_20261101`；本地只代码、配置、汇总、静态图文。实际盘点见`INPUT_INVENTORY.json`，仅JSON/文件metadata/NPY header，没有读取数值或模型。

三seed1024/2026/31415的FT目录只有原checkpoint和历史，无TRAIN/VAL特征/预测。每seed需252454 TRAIN窗（426记录/70人、3945个global64批次）及30500 VAL窗（49记录/49人、477批），保存float32 Z[N,8,192]和同次原head输出P[N,192]，不保存EEG副本或新fused全数组。

| 预算 | 精确值 |
|---|---:|
| 新Z载荷 | 5215408128 B |
| 新P载荷 | 651926016 B |
| 12个NPY header | 1536 B |
| 新NPY合计 | 5867335680 B |
| 已有7个SS07根NPY | 18602095776 B |
| 预计8根总NPY | 24469431456 B |
| 25GiB总NPY上限余量 | 2374114144 B |
| 预计未压缩moments / coefficients | 270812184 / 71705088 B |
| D8整个新root上限 | 7GiB = 7516192768 B |
| 盘点时free / 扣整个上限后 | 33947099136 / 26430906368 B |
| 最低free reserve | 20GiB = 21474836480 B |

单GPU、三seed顺序，每seed每phase一次原路径前向，共848862主encoder/head窗、13266个global64批；仅原路径工程对照再加每seed首/尾VAL批，最多384额外encoder/head窗，实际尾批36窗。每个case保留输出，不重复整phase。8 CPU线程、RSS≤8GiB、GPU allocated≤8GiB、wall≤120min、NN更新0；全过程监测空间/RSS/时长，失败保留不自动重试、不删除任何历史cache。7GiB上限包含图表/metadata等所有D8输出，25GiB另独立约束8根全部NPY。

## 3. FT路径桥与新TRAIN统计

直接复用冻结`pub_01_s03_ss07_finetune.RawWindows`、Decoder和原objective，不调用其main、旧evaluate或任一测试读取。保留原TRAIN EEG输入normalization（该已训练模型的固定输入变换）；重拟合的是FT latent/P/平方分支的TRAIN统计与全局a,b。

cache严格复现原完整phase连续global64，可跨记录；不改成native record256。RawWindows原`np.stack(...).astype(float32)`、tensor布局、eval/no_grad、FP32、TF32 matmul=False/cudnn=True、原requires_grad标志不变。拆开的encoder→transpose→head与Decoder.forward操作一致；P当次保存，不从改变layout的持久化Z重新生成基准P。记录Z view stride及落盘顺序。

每seed先VAL一次前向，首/尾批与原Decoder.forward波形最大绝对差门1e−6（原未缩放单位），同global64/float32 objective重聚合与该FT已保存best validation negativePearson差≤1e−7。首尾额外前向不写cache、不参与选点。任一桥失败则不进入该seed TRAIN编码/拟合，保留原失败并诊断，不缩a或放宽门。原FT没有保存VAL波形，标量历史loss桥和首尾同路径实现桥不能冒称全VAL历史波形桥；D7的FINAL_WEIGHT_VAL_BRIDGE不作FT证据。

桥通过后TRAIN一次编码；按D4相同物理点等权Hann融合，用对应FT TRAIN监督拟合单个正a,b（a≤0失败、不截断），定义P_FT*=aP_FT+b。对VAL原r验证正仿射不变≤1e−10；不按VAL拟合a,b。随后只TRAIN内i≥4的250750窗拟合逐坐标Z/P均值、标准差、常量mask及标准化Z平方的TRAIN均值/标准差。每块按sqrt(dim)缩放，全部使用float64读出；不复用native normalization/Gram/coef/PCA，也不生成任何历史特征。

## 4. 固定最小矩阵与选点

零残差P_FT*为每臂共同候选。三个learned角色Ponly/C/Ccap × 三seed × 三λ(.0001,.01,1)=**27套闭式岭解**，不加S/U/新λ/新seed/新网络。输出维192、共同自由截距、不惩罚截距，目标为原TRAIN窗口y−P_FT*；损失/正则/坐标缩放沿用已审D4/D6。

- Ponly：192维当前P_FT*，37056参数；不使用Z。
- C：当前Z1536 + P192，331968参数。
- Ccap：C + 当前标准化Z平方1536，626880参数。

三者TRAIN同i≥4拟合；VAL每记录首4窗共同P_FT*回退，其余仅当前窗输入，无历史/未来额外数据。完整30500窗/49人/752块与原prediction-independent VAL mask不变；Hann192/stride96、64Hz、原60秒块→person宏r用于选点。3秒r（undefined则null+覆盖）、window/fused/加权单窗MSE、overlap及Hann disagreement均次要、不选点。复用实际已审D7 metric/constant规则与D4岭/stat/calibration模块，不重复所有旧公式实验；只核对FT来源/路径/新TRAIN统计与此矩阵变化。

每角色按三seed mean VAL60s选共同λ；同分取较大λ，严格优于三seed零残差均值才采用，否则三个seed共同回退P_FT*。保存全部27解/指标/正规方程误差/λ候选/失败，不依据某seed分数单独择λ。两项主要诊断：Ccap−P_FT*、Ccap−Ponly；C−Ponly、Ccap−C为次要。报告每seed/人/块，不新增p值/CI或显著性。选定差在三seed均正且均值为正时只称本重复VAL上的方向一致；否则该控制下不支持稳定增益，不能追结果扩网格。与既有D6摘要只作描述，encoder+head+优化差异不能当单因素因果效应。

## 5. 验证与交付

协议/源/输入身份在任何新前向前冻结；既有权重不修改。仅新增FT cache桥、FT TRAIN统计/标准化与Ponly/C/Ccap系数约束/有限性/正规方程检查。协调独立审代码、来源与共同mask，实际重聚合全部27格；复用缓存输入独立重建选定3seed×3learned的VAL预定首/末记录，不扩大神经前向。完整中文三图（逐角色/seed r、全λ及主要配对、MSE/一致性）、全表/成本、REVIEW、轻量HTML及正常commit/push；PR6保持Draft，不合并。技术/Acceptance/Parent三级判定，本轮交付不等同持续goal完成。

## 6. 后续可检验假说（此轮只提出，不执行）

对非退化且epsilon不激活的窗口，逐窗Pearson目标不识别每窗正尺度/偏置。假说：原监督可能留下跨窗物理单位未识别的自由度；用真实目标约束的连续物理时间监督，能在强current控制之上改善融合目标误差及相关表现。小a或低overlap本身不是该机制证据。

若另行审定，固定同架构/初始化/更新/输入权限，对比原逐窗PCC、逐窗监督MSE、固定Hann物理时间融合监督MSE；共同target单位与预先固定的loss尺度/预算，报告r、加权单窗MSE、fused MSE、disagreement和退化检查。Hann恒等式显示disagreement也可产生融合分散收益，故不直接最小化无target的一致性、不宣称越平滑越好、不学习测试融合权重。该设计既不自动具备方法新意，也不在D8中增开模型/搜索；在强控制、历史干预与后续可信确认前不作机制结论。
