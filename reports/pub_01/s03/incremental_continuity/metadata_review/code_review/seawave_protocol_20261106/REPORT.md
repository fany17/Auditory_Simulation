# Sea-Wave 一手论文与公开实现核验

2026-11-06，协调端只读审计。**可进入同协议强对照设计；尚未在本项目训练，不能据此声称达到或超过 SOTA。** 用户以性能为主，是否利用历史不影响方法作为性能候选。

## 已核实的来源

- [正式论文与 DOI](https://doi.org/10.1109/OJSP.2024.3378594)，IEEE OJSP 5 (2024), 686–699；[作者机构提供的正式 PDF](https://lirias.kuleuven.be/retrieve/8a19355d-6c50-4d10-91ff-c48f1b1c66ff)。实际下载、抽取全文，并渲染核对期刊页690、693的训练说明、Table 7/8及Fig.7。
- [作者源码固定版本](https://github.com/LiuyinYang1101/Sea-Wave/tree/910138cdac01c58c007459711008ae4991757ff2)，MIT；12个实际读取文件完整保存在 `upstream/`，逐文件 Git blob 与远端一致。版本、字节量和SHA256见[PROVENANCE.json](PROVENANCE.json)。未导入执行作者脚本，未下载数据或权重。
- PDF留在本地只读资料缓存，未提交完整论文。该代码版本早于正式论文，不能默认二者所有配置相同。

## 正式论文限定

Table 7 的 Small 为32通道、20层、膨胀周期5、kernel3、感受野249、约150k参数；仓库JSON却是周期4。Table 8报告Small的S1/S2 r为.2258/.1158。Methods IV.D将S1配个体微调、S2配群体模型，摘要措辞不同，故不能将.2258直接当作未经个体适配的群体成绩。论文还描述群体微调，不能省略后将弱化版本称完整方法。

作者将原开发集train+val合为训练、原开发test用作验证；最终challenge test另设。因此代码中`test_-_*`用于选点**不单独构成测试泄漏证据**。本项目只能按明确426 TRAIN/49 VAL清单映射，绝不套文件名前缀去重开旧test。

## 代码实际行为

| 路径 | 核实事实及实现含义 |
|---|---|
| `models/EEGWaveRegressor.py` | 非因果、对称padding膨胀卷积；GELU输入投影、门控tanh×sigmoid、dropout、共享残差/skip、每周期输入skip、输出零初始化。无时间维LN/BN。构造支持周期5，但不能把默认周期4称论文Small。保留作者weight_norm与初始化次序；移植需验证，不能悄悄更改。 |
| `train/distributed_train.py`及JSON | 5s窗、.5s hop、每GPU B32、dropout .2、lr1e-4。配置n_iters200，而循环含上界，fresh为epoch0至200共201轮。启动脚本配置名大小写与实际文件不同，另有机器绝对路径，不能直接启动。 |
| `train/train.py` | AdamW wd1e-5；真正反传MSE，Pearson只记录/选点；初始阶段没有early stop。单GPU未设shuffle，多GPU采样不同。验证按batch均值Pearson选点，非本项目逐人60秒宏平均。 |
| `utils/CustomDatasetPytorch.py` | train切窗；`toTrain=False`返回整条文件，验证并非调用参数写的3s滑窗。EEG与包络共同坐标，未证明与本项目归一化/lag完全一致。 |
| `utils/cyclic_scheduler.py` | 自定义cosine restart10、周期乘1.2；除lr外还逐步缩放weight_decay，含batch/epoch_size/restart_period因子。不可替换成普通CosineAnnealingWarmRestarts后声称完全相同。 |
| `finetune/fine_tune_subject.py` | 公开的是逐人FT；AdamW默认wd .01；loss为.6 MSE+.4负Pearson；B8、lr1e-5、最多epoch0..20，首次不改善即停，best初始为0。与论文所写.8/.2不同。硬编码加载checkpoint62，失败只打印后继续，不能沿用此静默回退。 |
| FT launcher/config | 8s、hop .117经整数转换为7个64Hz采样点、dropout .65；论文写.1s。JSON仍为周期4，预训练路径名却指128通道16层模型。不能以路径名证明实际加载成功；必须核权重形状/来源并fail closed。 |

FT默认每2轮保存一次，可能未保留奇数轮的最佳模型。后续本项目应明确保存实际VAL选中权重，这属于执行修复，并保留与作者入口的差异。公开树中未找到完整独立的群体FT入口；不能声称只跑该逐人脚本就复现了所有论文阶段。

## 后续固定比较的实施边界

建议以**论文Small周期5的群体方法**为首个Sea-Wave对照，不做宽泛结构搜索。复用公开模型数学形式；初始训练和群体FT均须实施。初始阶段按公开优化器/调度器及上限，FT按论文群体描述落实，.8/.2损失中的相关项采用最小化负r并明确是执行解释；窗口整数化、FT上限/早停、选点指标应在第一次训练前冻结，不能事后依性能在代码/论文版本间挑选。此处尚未冻结该实验预算，也未开始训练。

使用既有合法TRAIN和新拟合统计，三seed，统一49人/752个60秒VAL块评分，输出每人、每块及源家族差异。不得把逐人FT结果混成纯群体结果；不因它更强而省略、不因它更弱而调测试集。若以后加入逐人适配，应作为明确不同的方法条件。

保留Sea-Wave原生整记录推理及有限双侧感受野，必要时用足量halo分块并验证等价；不要截成3s来削弱它。原Transformer的3s离线上下文与Sea-Wave不同，属于**相同数据/指标的离线方法比较，非严格同时间权限/同算力比较**。须分别披露唯一数据、重复样本、更新量、训练阶段、参数与成本。论文原r只定位方法，不能与本项目VAL直接排名。

D26继续原已授权矩阵；Sea-Wave补对照不以历史贡献阳性为前提。这里的来源差异是可解决的复现适配问题，不是作者造假或本项目必须停工的证据。

