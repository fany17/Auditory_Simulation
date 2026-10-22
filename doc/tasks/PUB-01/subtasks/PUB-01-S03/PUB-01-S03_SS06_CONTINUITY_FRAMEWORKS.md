# SUBTASK — PUB-01-S03_SS06 — 五种连续信息解码框架

| Field | Value |
|---|---|
| Subtask ID | `PUB-01-S03_SS06` |
| Parent Node / Task | `PUB-01-S03` / `PUB-01` |
| Status | `ACTIVE` — `REAL_EEG_EXECUTION`；prototype01 保留 PASS_WITH_LIMITATION |
| Blocking for Parent | NO；此包不替代 S03 最终 benchmark |
| Patient Data | `FORBIDDEN` |
| Upstream | SS02/SS03/SS05 冻结报告；2026-10-23 用户明确授权 |
| Inputs | 现有方法证据、随机合成张量；本轮不读取 EEG/声音载荷 |
| Deliverables | 五框架结构/目标/对照、统一接口、可运行原型、合成工程测试及独立代码审阅 |
| Acceptance | 五模型前向/反向、分块连续性、因果性及核心对照通过；真实训练为 NOT_RUN |
| Validation | Technical → Engineering acceptance → Parent consistency；不冒称科学验收 |
| Go/Hold/Fail | 工程通过后停止；运行失败修复；服务器不可用记 HOLD_RUNTIME，不转本地计算 |
| Files Allowed to Change | 本任务书/接口契约；`src/m6a_public/continuity/`；`tests/test_ss06_*.py`；SS06 script/config/report；当前入口/登记的必要更新 |
| Last Updated | 2026-10-23 |

## 当前授权更新（2026-10-23）

用户在原型交付后明确“启动吧”，现授权既定五框架进入真实公开EEG训练与比较。执行以 [SS06_REAL_EEG_EXECUTION.md](SS06_REAL_EEG_EXECUTION.md) 为最新补充，先实际train/val pilot，再冻结预算/对照/划分并推进合格评估。以下PROTOTYPE_ONLY与NOT_RUN描述为prototype01历史，不再阻止本轮已授权实验；旧SS02/03/05结果仍冻结。

## 1. prototype01 历史授权与目标

用户原话：“先把这五种框架细化成任务书，写清网络结构、训练目标和对照。然后快速搭建基本框架，实际训练和执行暂不开始，确保能跑通先。”

本轮只执行任务书、模型实现与合成输入工程检查。允许一次损失求导检查；**不执行 optimizer.step、不做任何合成训练、不读取真实 EEG、不运行 pilot/预训练/正式训练或测试集评估**。后文训练目标和实验对照是待执行设计，不是已冻结正式方案。完成后停止，不自动晋级。

全部数值检查在 `server2203:/home/fanyu/auditory_simulation_m6a/pub_01/` 新目录；Windows 独立 worktree 只保存代码与轻量报告。基线 main 为 `60c340494bbdb218f62114bab9e4c1ee203dec67`；原 CNN/TRF、stash、既有 worktree 与历史结果保持原位。

## 2. 科学动机及不能推断的内容

SS05 在既定五模型、数据和预算下未检出历史/关联残差额外收益，不能推断所有连续性无用。其每个 3 秒窗口经 mean/absmax 压为 token，历史每次从有限上下文左端重建，训练损失按 3 秒而评估按 60 秒。这些是可检验的设计限制，**不是已经证明的阴性原因**。

本包将连续性拆为：密集时序轨迹、跨块状态、可预测表征、缓慢对齐、状态预测与观测校正。先分别验证假说，避免一次把五种全部叠加而无法归因。框架 1 为公共基础；2/3 为优先研究候选；4/5 为条件性探索。没有任何一种已被证明优于现有强基线。

## 3. 共用输入、网络与边界

- 工程输入 `x[B,T,64]`，示意采样率 64 Hz；逐样本输出 `ŷ[B,T,1]`。hidden=32。短测试 T=256；长前向 T=3840（60 秒合成输入）。这些是工程配置，不是正式 EEG 采样/窗口选择结论。
- 共用 LocalEncoder：左补零 causal Conv1d(64→32,k=9) → GELU → causal depthwise Conv1d(32,k=9) → GELU。保留 `[B,T,32]`，不做全窗池化。单独该编码器感受野 17 样本。
- 五模型统一 `forward(x, state=None) -> DecoderOutput(prediction, state, aux)`。状态显式返回，不写 module 内的隐式历史。`detach_state` 仅截断计算图，保留数值；`state=None` 才清零。
- 原型 EEG lookahead=0。任何未来实验的 EEG→音频反应时延都另由 train/val 冻结，不能把 EEG 因果前向等同零声学延迟；不得给某模型额外未来 EEG。
- 换 subject/session/record/split 时清状态；缺口不能跨接。`StreamRunner` 检查记录身份和 sample offset；任一 batch lane 更换身份会清整批状态。原型不支持混合批次逐 lane 独立 reset，未来 loader 须按边界组织批次或补齐该功能。
- 当前 EEG 固定，仅前缀为真实/清零/错配三种。错配历史不能跨 split 或偷用后续目标；正式历史干预需保留同一当前区间、长度、模型权重和预处理。`compare_histories` 是包含 encoder 缓存的全前缀干预；`compare_state_histories` 额外固定真实 encoder 缓存，只干预其后的状态，避免把前 16 样本局部边界效应误归因给长记忆。

## 4. 框架 1：连续密集多尺度解码（M1）

**问题**：保留连续局部轨迹与长时包络变化，能否优于摘要 token 或短感受野？

结构：LocalEncoder → 四个并行 causal Conv1d(32→32,k5) 分支，dilation=1/8/64/256，分支 GELU 后均值融合，再线性混合 → 与当前局部特征融合 → 逐样本线性包络头。最大有效感受野为 17+4×256=1041 样本，约 16.27 秒；有限卷积上下文是明确限制，不能称无限记忆。分块缓存保留恰好所需历史。

拟议训练目标：60 秒连续预测拼接后一次计算 `L_env = 1-r_60s + λ_MSE·MSE`；原型 λ=.1 仅用于检验梯度，最终权重在 train/val 冻结。不要先算每个 3 秒 r 再取平均冒充 r_60s。MSE 用未来新 train-only 标准化目标。

主要对照：`multi` vs `short` vs `long`；另保留相同主干、相同长 span 的单尺度参数/算力匹配设计。当前三开关仅为功能对照，**不保证有效参数匹配**。后续须记录实际参与梯度参数、MAC/内存、wall time；不能用注册参数相同掩盖失活分支。

最小验收：任意分块含单样本块与整段前向一致；改变未来 EEG 不改过去输出；三分支模式能构造、前向、求导。失败风险：长分支只学平滑、边界缓存错误、较长 span 带来额外历史量。

## 5. 框架 2：持久快慢状态（M2）

**问题**：跨计算块持续积累状态能否提供超过固定卷积历史的 EEG 证据？

结构：LocalEncoder 的 f_t → fast GRUCell(f_t,h_fast,t-1)；slow state 用有界可学习更新率 α 更新，`h_s,t=(1-α)·h_s,t-1+α·tanh(W_s f_t+b_s)`；concat(f_t,h_fast,t,h_s,t) → 包络头。每个样本推进一次，跨连续 chunk 传递三个状态（encoder/fast/slow）。α 初值 .01，64 Hz 时约 1.55 秒指数时常，可学习且有界；当前只验证接口，不能把它描述成已学会 60 秒记忆，也不赋予生理时常意义。

拟议目标：同 M1 连续 L_env；只增加预先声明的稳定性约束才可进入正式比较。未来训练可 TBPTT，但 detach 点不清零，记录实际反传跨度、热身和前缀长度。

主要对照：同一权重持久状态 vs 每 3 秒 reset vs 记录边界 reset；fast-only/slow-only 留作后续明确配置，当前不假称全部已实现。真实/错配历史固定当前 EEG；普通 GRU 相同状态维度及单独预算匹配；local+因果输出 EMA 排除纯包络平滑收益。

最小验收：整段与分块输出一致；detach 前后输出不变且历史梯度被切断；跨记录自动清零；错配前缀不改变当前输入。风险：累积漂移、身份/伪迹记忆、长样本训练预算更高。

## 6. 框架 3：连续性预测表征（M3）

**问题**：先学习 EEG 中可预测的时间结构，是否改善下游声音重建？

结构：LocalEncoder → causal GRU context c_t → 包络头；另设 horizon={4,16,64} 样本的预测头 p_h(c_t)。EMA teacher LocalEncoder 生成未来目标 z_teacher,t+h，冻结梯度。前向推理只接受 EEG 与过去 state，不能输入未来 latent 或声音。

拟议预训练：平均各 horizon 的归一化 latent 预测距离 + online 特征的方差防塌缩正则。EMA 更新只能在将来实际优化之后由显式方法调用，当前 smoke 不做数据依赖优化；独立 EMA 单元测试允许固定常数扰动和显式 EMA 操作以验证边界。这里采用无负样本目标，并非已经实现 CPC InfoNCE；CPC 仅提供预测表征的研究动机。

拟议下游：`no_pretrain` 从头训练；`frozen` 冻结 encoder/context、只训包络头；`finetune` 联合微调。三者应同一 downstream 数据/预算，并分别披露额外预训练算力。预训练只能使用未来新 train 的 EEG；没有声音标签也不能偷用验证/测试 EEG。

当前 `predictive_loss` 仅支持 fresh state 开始的整段输入和尾部 padding，拒绝内部缺口及带历史 state 的辅助损失；长流式教师缓存/跨 chunk horizon 目标尚未实现。流式包络推理接口本身仍需通过 chunk 等价性测试。

最小验收：teacher 与 target_source 没有梯度；未来目标不会影响当前前向；encoder/context/predictor 有有效梯度；模式冻结有效；不足 horizon 的序列与非法 mask 明确拒绝。风险：塌缩、预测眼动/漂移而非听觉信息、额外训练量混淆收益。

## 7. 框架 4：连续自适应时间对齐（M4）

**问题**：EEG-目标有效时延是否缓慢变化，固定 lag 是否丢失信息？这是待验证假说。

结构：LocalEncoder → 非负滞后特征 bank f_(t-l), l={0,4,8,16} 样本 → EEG 驱动门控 logits 的慢 EMA → softmax 权重 → 加权特征 → 包络头。滞后范围及最大可见 EEG 在 fixed/adaptive 两组完全相同。slow_decay=.98 的工程值可配置；不把该约 .77 秒指数时常作为已拟合生理参数。

拟议目标：同 L_env，加 lag 权重时间变化惩罚 `λ_lag·mean((w_t-w_t-1)^2)`；未来 λ 仅 train/val 选择。当前基本包提供权重轨迹用于该损失，正式系数未冻结。

主要对照：固定均匀权重、train/val 冻结的固定 lag/权重 vs adaptive；同 bank/读出/输出延迟。禁止 test 包络互相关寻峰、逐记录最优重对齐或目标驱动门控。

最小验收：lag 无未来索引、权重非负且和为 1、固定模式权重不随 EEG 变、adaptive 前缀连续、分块一致。风险：收益源自更宽 RF/输出平滑，lag 权重不可自动解释为神经时延。

## 8. 框架 5：声学潜状态预测与 EEG 校正（M5）

**问题**：声音的可预测结构与当前 EEG 新证据能否显式分开？

结构：`z_prior,t = F(z_post,t-1)`；预测 EEG 特征 `f_hat,t=H(z_prior,t)`；实际 EEG 特征 `f_t=LocalEncoder(x)_t`；innovation `e_t=f_t-f_hat,t`；有界逐维 gain 校正 prior，`z_post,t=tanh(z_prior,t+gain_t⊙e_t)`；包络头从 z_post,t 输出。这里 innovation 有明确观测定义，仍不能直接称大脑预测误差信号。

拟议目标：连续 L_env + stopgrad(f_t) 的观测预测误差 + 可选状态稳定约束；预训练/辅助权重仅为将来设计。当前提供 prior/innovation/gain 以构造辅助目标，不把尚未执行的目标写为结果。

主要对照：`full`；`prior_only` 完全跳过 EEG（包含前缀），检查先验自身预测；`eeg_only` 仅局部 causal encoder 与读出；local+EMA 输出平滑；预测包络历史先验。禁止真实历史声音 teacher forcing 混入主推理条件。真音频驱动 oracle 如未来需要只能另列上限，当前未实现。

最小验收：prior_only 对任意 EEG 值替换不变；eeg_only 不传递 latent 递归状态；full 创新由实际观测产生且有梯度；整段/分块一致。风险：latent 不可识别、prior 塌缩、gain 把所有信息复制成普通 RNN、输出自相关制造虚假改善。

## 9. 将来真实实验的共同对照与停止点（本轮不执行）

| 因素 | 共同规则 / 对照 |
|---|---|
| 数据与划分 | 旧 SS03/SS05 已查看测试集只能保留历史标签；不得重新称新独立 test。新方案须先确认可用的未见分组或明确仅探索。当前未创建新 split。 |
| 归一化 | 在未来新 train 内重拟合；teacher/自监督也不读保留组；不覆盖旧统计。 |
| 当前 EEG 与历史 | 对比时当前 EEG 完全相同；只变前缀或状态。声学候选/正确音频不能作为输入。 |
| 目标/尺度 | 同目标、同因果/声学 offset、同评分段；60 秒段→record→subject 聚合设计待冻结，不能直接与不同历史分数排名。 |
| 输出自相关 | local+因果 EMA（仅 train/val 选系数）、预测包络历史 prior、错配/清零历史。 |
| 算力与参数 | 记录实际梯度参与参数、处理样本数、预训练量、wall time/峰值内存；分别报告固定样本预算和可行算力匹配，不假称所有架构天然相同。 |
| 推断 | 将来至少 3 seed、预设主要比较与多重性校正、逐人配对差及区间；阴性保留。当前无 seed 性能比较。 |

优先顺序：基础 M1 → 单独 M2 与 M3 → 有时延证据再 M4 → M5 机制性探索。当前同时搭建骨架用于后续审阅，不触发上述实际实验。

## 10. 本轮验收、失败与交付

1. 结构/损失/主要对照/失败假说全部写清；代码与文档相符。
2. 五类及指定控制模式可实例化、输出有限 `[B,T,1]`；主路径和辅助路径的应有梯度有限且非空，冻结路径无梯度。
3. 因果后缀扰动、任意分块、reset/detach、边界及错配历史控制通过；invalid API 输入有明确报错。
4. 五个默认模型在 64 通道、60 秒合成输入做前向；短合成输入求导。性能数字只报告工程耗时/参数，不报告科学准确率。
5. 第二 Agent 独立读实际代码/测试/日志审阅；报告明确工程可运行 ≠ 科学收益。所有真实训练/预训练/数据加载/正式 test 为 `NOT_RUN`。
6. 只提交自有代码、配置、轻量日志和报告。SS02/03/05 frozen 文件无改动；不生成模型权重与训练 checkpoint，不扩大数据范围。

`REWORK`：等价性、因果性、冻结或对照失败，先修正再复测相应风险。`HOLD_RUNTIME`：2203 无法执行。`PASS_WITH_LIMITATION`：工程通过，但真实 EEG loader、预算匹配、split/lag、有效性均未验证；完成后不自动训练。

## 11. 方法依据（背景，不是本项目有效性证据）

- [CPC, van den Oord et al.](https://arxiv.org/abs/1807.03748)：用未来表征预测组织自监督任务；本包使用的 EMA/无负样本形式不是论文复现。
- [Mamba, Gu and Dao](https://arxiv.org/abs/2312.00752)：输入依赖的连续序列状态建模背景；M2 是 GRU/EMA 最小原型，不称 Mamba 实现。
- [mTRF Toolbox, Crosse et al.](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00604/full)：连续刺激与神经响应滞后建模背景；M4 自适应假说尚未验证。
- [KalmanNet, Revach et al.](https://arxiv.org/abs/2107.10043)：动态状态预测与学习观测校正背景；M5 非已校准 Kalman 滤波器，也不输出可信不确定度。
- 本项目直接证据：SS03 `reports/pub_01/s03/architecture/REPORT.md`；SS05 `reports/pub_01/s03/predictive_memory/codex_run01/REPORT.md`。结论不被本设计修改。

## 12. Completion Record

2026-10-23：五种原型、统一接口/目标/核心对照及任务书完成。server2203 Python3.11.15/PyTorch2.11.0 CPU实测38测试+71子测试通过；相关DE警告断言detach修正后10测试+26子测试无警告通过。五默认64ch/32hidden模型的60秒合成前向/257样本分块max差<1e-7，短段梯度有限；主smoke参数不变，optimizer steps=0。AB/DE跨作者源码审阅及AB对实际日志/JSON/修正代码的独立闭环，发现全部关闭；root汇总核验。Technical PASS / Engineering acceptance PASS_WITH_LIMITATION / Parent consistency PASS。详情reports/pub_01/s03/continuity/prototype01/REPORT.md、REVIEW_CDE.md、SMOKE_FINAL.json；代码/config/命令同报告。

base main60c3404；分支codex/pub-01-s03-ss06-continuity。原CNN/TRF与stash、旧SS02/03/05证据未改。真实EEG加载/预训练/pilot/参数优化/正式test全部NOT_RUN；未来split/lag/预算未冻结，无准确率改进结论。SS06仅原型包COMPLETED后STOP；S03/PUB仍ACTIVE，不启动SS04/S04/父S05。
