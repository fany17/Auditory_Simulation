# SS07 下一阶段train/val方案草案

**APPROVED_FOR_IMPLEMENTATION / ENGINEERING_NOT_YET_RUN。** 2026-11-01协调端独立审阅8701673后批准隔离warm round02实现、2203工程核验和后续正式train/val；用户持续goal授权覆盖该范围，不重复询问。当前尚无新fit，不开放测试。沿用SS07，不新增编号；round01源码/预测/权重/报告完全冻结，使用独立ROOT与run02结果目录，持续goal ACTIVE。

批准补充：总预算预先最多18fits。若执行前工程证据不能证明旧FT三格满足下述条件，固定在共同实现重跑三格P_joint，保留旧FT为历史参考；不按新分数或seed挑选是否重跑，不再为已授权重跑另设许可门。具体源码/diff、RNG复用决定与工程证据交协调端独立代码核查后冻结并进入正式train/val。frozen/joint两行还同时改变encoder train/eval随机行为，交互不能解释为单纯encoder梯度效应。没有新确认测试授权。

## 问题与依据

只问：在**训练完成的原P**上，既有M4-H16是否能通过有限源域继续训练获益，且真历史能否超过同结构current控制；encoder联合更新是否改变这种关系？沿用原M4结构，不加新架构/宽度/层/LR搜索。

立项依据仅为测试前已留存的源train/val事实：[14/15 best3s低于P及3s–60s不一致](FAILURE_MECHANISM_CHECK.md)；原候选与P同随机head起点而非训练完成P；[五FT源VAL改善](FINETUNE_SNAPSHOT.json)记录时间1793484216早于[READY](EVALUATION_READY.json)的1793500382。不是用本轮DTU/旧test挑超参或新分支。M4是原VAL冻结的结构，不按测试重选。

这个矩阵检验原轮未覆盖的warm incremental问题。旧随机起点full与本方案上游head训练/预算/停止不同，**不能当严格的初始化因果对照**，也不能据warm改善声称随机起点已被证明是失败原因。

## 最小配对矩阵

固定前三seed1024/2026/31415，各自加载原轮同seed预训练encoder与训练完成head。三列都从同一P函数开始；新增M4最终投影零初始化、其余既有core正常初始化，head继续训练。

| encoder模式 | P继续训练（无增量） | M4-H16 current | M4-H16 past |
|---|---|---|---|
| 冻结、eval模式 | 仅原head继续训练 | head+原M4，仅current输入 | head+原M4，非重叠历史 |
| 联合更新、train模式 | 原P_FT，满足下述条件才复用 | encoder+head+原M4 current | encoder+head+原M4 past |

每行的past−current与past−P是预定配对问题；两行差的交互只描述，不作“初始化因素已隔离”的机制解释。current控制排查新增容量/当前映射，但不代替所有可能静态/prior控制；本阶段不新增其网格。原P（不继续训练）作为零更新incumbent单独保留，不能用继续训练P代替原P标签。

## 共同输入、更新与选择

- 只原合法train426/val49及原train-only归一化。每轮252454个3s窗，global64、固定record顺序、不shuffle、最后不足64保留，跨record只拆head/state计算，不增加optimizer步；每轮3945次更新。joint encoder必须先以完整global64 raw窗前向，再按record拆增量计算，不能因边界拆encoder batch或改变dropout/原FT批次语义。
- 两种M4均保留原H16、tau、4/8/16 latent shift、残差系数.1、原conditioning定义及非重叠i−2/i−4 parity/reset/TBPTT边界。所有臂当前3s离线权限相同，保持record/phase/epoch状态重置，每global64更新后detach；不反传跨更新历史。joint历史来自此前实际前向及旧更新时刻的状态，不用冻结latent cache冒充当前encoder；验证以选中checkpoint重新从record起点建立状态。
- 冻结encoder臂可复用同seed既有合法train/val latent cache，验证其来源；joint各步从raw重算。共同目标仍原3s negative Pearson eps1e−6；新Adam状态、head/增量LR=.001，joint encoder LR=1e−5，无weight decay/clip、原精度。无新预训练、teacher/SSL目标或辅助loss。
- **全部固定5epoch**（每fit19725更新、1262270监督窗口呈现），不因曲线加轮或提前挑有利预算。更新前保存epoch−1 incumbent并纳入checkpoint候选；0…4为训练epoch，每轮按原完整3s VAL、严格改善和最早tie选择，epoch−1可最终胜出。不得看60s或测试后改变checkpoint选择。
- 全部臂报告incumbent及所选checkpoint的VAL60s/person（原Hann/有效块/人汇总规则）、3s VAL、逐seed配对差和训练曲线。无需补造旧FT未保存的中间权重，也不据缺失中间60s更改选点。3s选点不保证60s最好。
- 初始P与两个零投影M4的VAL输出须在预先工程容差内相同；原FT已声明loss绝对1e−4仅用于B64 raw/B512 cache incumbent桥，不能泛化为科学增益阈值。新实现的角色/zero投影/record边界/梯度/state检查先在2203按工程协议审阅，当前不执行探针或新科学运行。

## 复用原FT的必要条件

联合P格拟复用原前三seed FT，需在执行前用源码/metadata证明：同源encoder/head/fold/norm、相同raw global64及训练模式、精度、样本/更新顺序、新Adam组/LR、incumbent候选及最早3s选点一致。原pat3实际从未触发，三seed均跑满5轮且best4；固定5轮可匹配其实际轨迹，而非伪称协议文字完全相同。

必须额外证明**配对的encoder dropout随机流**可与旧FT一致：额外M4构造消耗RNG不能平移encoder随机流，计划采用隔离模块初始化RNG并恢复旧FT构造完成后的CPU/CUDA状态；不能只凭seed相同认定匹配。M4运行若引入额外随机操作需隔离相应生成器。还需核对原环境的随机/确定性设置，不声称仅seed即可逐bit等价。

若这些条件不能在读metadata/工程核验后证明，按上述已批准总预算在任何正式fit前冻结重跑三seed联合P格，原FT仅作历史参考；不能看新分数再补跑。无论15或18新fits，决定与具体工程证据均交协调端核查后冻结，不在工程原型处收工。

## 有界成本、决策与停止

| 情形 | 新fits | 新epoch | 新更新 | 新监督窗口重复 |
|---|---:|---:|---:|---:|
| 原FT三格可复用 | 15 | 75 | 295875 | 18934050 |
| 经复核决定联合P也需匹配重跑 | 18 | 90 | 355050 | 22720860 |

上述预算已在2203做metadata算术核对，不是新增训练。上游五预训练/原head成本在总账保留且不重计共享部分；旧FT成本与本轮新增成本分列，额外incumbent/验证遍历单列。epoch/update匹配不意味着FLOPs、墙钟或延迟匹配。继续原资源边界：新array/cache≤25GiB、余量≥20GiB，best/latest有界，不删历史、不动他人进程；joint不建新全量latent cache。执行前读盘点不满足则只处理受影响作业，不回退Windows。

所有六格/三seed、失败及incumbent胜出的情况均报告，不丢负seed。只作源域开发决策：若某encoder模式下past在全部三seed的选中VAL60s都高于同模式current与继续训练P，且超过已核验数值误差，可记为“值得后续审查”，不是推断性显著、研究成功或论文验收。若不能一致改善或选中incumbent，同样按预算停止并回到审阅；不自动加epoch/seed/宽度/backbone、换选择指标或另开“再试一个”。技术失败保留，不自动重试。

本阶段只train/val。旧Sparr测试和已暴露DTU不重新打新确认标签，不打开其推理/评分作调参；任何后续确认性数据资格、来源独立性与访问许可需重新明确，当前不下载、不拟合目标域，不申请患者/STN访问。完成本草案审阅不等于授权新的数据范围、确认测试或PR合并。
