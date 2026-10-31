# 当前执行：SS07 全量原强基线+连续残差（2026-11-01）

用户最新明确授权尽快全量实验、设goal、使用2203全部可用算力，研究目标超过原baseline并积累ICML/NeurIPS级证据。SS07 ACTIVE；旧STOP/单pass/NO_GO不覆盖本授权，旧结果保持只读。先执行PUB-01-S03_SS07_FULL_SCALE_INCREMENTAL_CONTINUITY.md及本轮INTAKE/BASELINE_PROTOCOL，原完整Transformer/XLSR19两阶段、fold完整train/val、100epoch/pat15预训练与250epoch/pat10head至少3seed；仅在强模型上增加五类有界残差，精确矩阵协调审阅。开发只train/val，旧test均内部暴露，未冻全计划不评估。无阳性/论文接收保证。

唯一计算2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101；本地reports/pub_01/s03/incremental_continuity/run01。新cache≤25GB、余量≥20GB、best/latest有界，不删历史/不动他人进程。沿用SS06 worktree/branch/PR6 Draft，可commit/push、不合并，不动主目录CNN/TRF/stash，无患者/远端Git/擅自新数据。持续goal不以smoke、技术PASS或单seed分数完成。

2026-11-01执行接续：五native完整两阶段已完成，screen/full/FT按冻结矩阵继续；full v2 controller14537。最终调用链source4e33d69、自动队列88448依全部固定训练完成→最终checkpoint→实际最终权重VAL桥→mask→一次内外评估→聚合接续；真实READY尚未建立。独立read-only复核source2f699c2经协调审阅、37项数学/77482项合成caller/实际VAL原始输入检查后冻结INDEPENDENT_REVIEW_PROTOCOL（5499cb54029da9497259e769aa4b9a6760783e821cf367155d2865f6ce16b795），队列113154实际live/WAIT_COMPLETE_FIXED_EVALUATION，核对上游88448。其只在完整final_summary+ready后按预先metadata选定3条Sparr/6条DTU×5seed全main重放raw，并复算全部保存预测/统计；精确常数两侧拒绝，任一技术失败保留不重试。不得重启或改活跃冻结源、不提前读分数。实际计数看服务器status，不以文档快照代替进程核验。DTU仅原两文件白名单；不扩大下载。复核排队/工程PASS不代表实际复核已完成，图文解释与科学验收仍待。

---
以下是旧交付/STOP历史，非当前路由。
# 当前交付 — SS06 run02 COMPLETED / PASS_WITH_LIMITATION（2026-10-26）

用户授权的匹配比较已完成：C/O各3×3262、P各639pre/2035head三seed、统一9模型seed/538块、原始derivative CPU/独立汇总复算与3图齐备。C/O/P r=.123014/.151762/.141466；C相对五框架/GRU六校正区间全跨0，不能认定普遍不如强因果控制或等价。O/P未来信息、P额外学习/不同loss与单轮限制明确，POST-HOC INTERNAL EXPLORATORY/5family。报告continuity/eeg_run02_matched_baseline/REPORT.md、REVIEW.md。Technical PASS / Acceptance PASS_WITH_LIMITATION / Parent consistency PASS；协调独立SUMMARY误差2.78e-17。当前STOP，无自动下一READY，同#6 Draft不合并。run01冻结与原资料/CNN-TRF/stash不动；无下载/患者/远端Git，下游不启动。

## run01历史交付（已被新启动授权覆盖STOP）

# 历史交付 — SS06 run01 已完成有限制（2026-10-23）

用户“启动吧”授权的SS06真实实验已完成：39×3262监督、3×3511 SSL，新train统计/完整连续record/一次内部审计/后置复核齐备。M1multi .122913、M2 .128164、M3finetune .123747、M4adaptive .113400、M5full .114178；GRU .125221。八primary三项为正，其余跨0；不把M2点估计当稳健胜出，未建立SSL相对GRU或M5额外收益、内容特异历史收益。只5family且旧训练暴露，INTERNAL/EXPLORATORY，非外部确认。

报告reports/pub_01/s03/continuity/eeg_run01/REPORT.md及REVIEW.md；243真实结果/四图/逐seed人表及机器review可读。原型CPU222r max差1.55e-7、协调SUMMARY独立重算max差5.55e-17，完整与精简包检查PASS。SS06 COMPLETED/PASS_WITH_LIMITATION；当前STOP，无自动下一READY。分支codex/pub-01-s03-ss06-continuity / PR#6 Draft、不自行合并；S03/PUB ACTIVE，SS04/S04/父S05 BACKLOG。E盘回传空间不足已用本轮C盘备份及精简JSON收束，原资料/旧worktree不删；数据/权重继续只在2203。

## prototype01 历史状态（以下非当前训练禁令）
# 当前任务 — PUB-01-S03_SS06（2026-10-23）

用户批准五种连续信息框架的任务书和基本代码，仅做合成工程检查。五模型已在2203完成前向/梯度/状态测试，独立代码/运行证据审阅已完成，原型包COMPLETED/PASS_WITH_LIMITATION；真实EEG、预训练、pilot、正式训练/测试全部NOT_RUN。本原型包已完成，当前STOP，不自动训练。

- 任务书：doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md。
- 证据：reports/pub_01/s03/continuity/prototype01/REPORT.md；状态以PUB-01/SUBTASKS.md为准。
- 独立worktree分支codex/pub-01-s03-ss06-continuity；main起点60c3404。PR#3/#4均已按既有授权合并，旧模型/结果冻结；主目录CNN/TRF与stash不动。
- S03/PUB保持ACTIVE，SS04/S04/父S05 BACKLOG；未生成ART-AUDREP-v1。

## 历史工作板快照（以下非当前执行指令）
> **Issue#5 / SS05已完成有限制并停止**：15run真实训练/一次新holdout测试、7种同当前EEG历史条件及后置review已完成。五组r约.091–.092，六预设比较区间均跨0，未检出复杂记忆额外收益。报告predictive_memory/codex_run01/REPORT.md。PR#3已按新授权合并8d21d0a，PR#4保持Draft，不合并/不扩实验。

# Auditory_Simulation Current Workboard

更新时间：2026-10-19（Codex 分工纠偏）。父任务状态由 `doc/TASKS.md` 定义，子任务状态由 `doc/tasks/PUB-01/SUBTASKS.md` 定义。**以下当前区是唯一执行路由，其下为历史记录。**

## 当前交付与停止

- **任务：`PUB-01-S03_SS05`，COMPLETED / PASS_WITH_LIMITATION。** 已交真实内部holdout阴性证据及同agent separate pass；协调只核对轻量文件，不称第三方科学签署。当前STOP，无自动下一READY节点，不启动SS04/S04/父S05。
- **执行归属**：Codex 写代码、运行 tests/合成验证/2203公开EEG、复核与提交；协调助手维护总任务并核对指定轻量文件。旧协调助手合成初稿已另由Codex实际复核，accuracy复现而A-assoc MSE有差异，详见CODE_AUDIT；不称第三方科学签署。
- **工作位置**：独立managed worktree pub-01-s03-ss05，分支work/pub-01-s03-ss05-predictive-memory/PR#4 Draft、base main。PR#3已由协调者按新授权合并8d21d0a；当前Codex不更动主目录CNN/TRF/stash或合并#4。
- **证据入口**：codex_run01/REPORT.md、REVIEW.md、FORMAL_PROTOCOL.json、105行RESULTS及15run轻量证据；新holdout来源限旧train，重拟合统计防泄漏，没有用旧SS03 test调参。原始EEG独立CPU90相关复核max差8.13e-6，全部汇总误差0。
- **不可外推**：本轮没有支持关联/历史残差的额外收益，不等于证明所有历史记忆无用；不能声称超过GRU/旧SS02/SS03、预测误差机制成立或第三方科学签署。PR#4继续Draft，后续由研究者另行决定。
- **其他状态**：SS01/SS02/SS03已完成有限制并冻结；SS04 BACKLOG；S04公开颅内支线BACKLOG；S05版本化artifact整合BACKLOG；公共项目与患者/STN项目单向版本化衔接，禁止患者数据进入本项目。

## 历史执行/状态快照（仅供追溯）

### 历史：SS05启动前入口（已被上方交付状态取代）

**2026-10-19新授权 SS05：ACTIVE / SYNTHETIC_PROBE_COMPLETED；正式公开EEG TRAIN/TEST NOT_RUN。** 已写PI-MRAM（预测误差＋多速率状态＋关联delta记忆）及同窗口GRU控制、10项本地CPU单元测试与两套合成实验。延迟两键干扰160步：full/assoc_no_pred 100%，GRU62.5%；GRU600步也100%，不能声称不可替代的算法能力；预测误差独立贡献尚不成立。代码和结果仅在PR#4独立工作分支。下一步按 `PUB-01-S03_SS05_PREDICTIVE_MEMORY.md` 与 `reports/pub_01/s03/predictive_memory/NEXT_PUBLIC_EEG_PROTOCOL.md`，先做公开source-family和新内部测试划分可行性审计，SS03旧测试集不可再用于SS05选模型。**不自动启动SS04/S04/S05，不产生artifact。**


2026-10-19交付：`PUB-01-S03_SS03` **COMPLETED / PASS_WITH_LIMITATION**。真实输入对齐/CPU-GPU smoke/100步pilot→测试前预算冻结→4×3正式训练及每模型一次test→同agent separate review已完成。multiscale seen/heldout r=.10305/.11380；long .06411/.06979；middle .09297/.10413。当前四原型/固定数据预算下的H1获支持，不是生理机制或超过SS02。报告`reports/pub_01/s03/architecture/REPORT.md`。

当前STOP：PR#3保留Draft，工作分支work/pub-01-s03-ss03-multiscale/独立managed worktree；主目录未提交CNN/TRF和stash未动。S03/PUB仍ACTIVE，SS04/S04/S05保持BACKLOG，未产生ART-AUDREP-v1。后续须研究者指定，无自动下一READY节点。下方10-15为SS02历史交付。



2026-10-15交付：用户授权“缺几个就缺几个”后，两套指定强基线及Transformer直接包络readout均完成真实训练/验证/一次测试与独立review。SS02 **COMPLETED / PASS_WITH_LIMITATION**；当前执行停止，没有自动下一READY节点。报告`reports/pub_01/s03/strong_baselines/REPRO_REPORT.md`。

- Parent：`PUB-01 — Public Auditory–Neural Representation`，`ACTIVE`。
- 当前优先支线：**A / SparrKULee / EEG 听觉直接解码**，归属 `PUB-01-S03`，`ACTIVE`。
- **本轮完成节点**：`PUB-01-S03_SS02 — Strong EEG Auditory Decoding Baseline Reproduction`，**COMPLETED / PASS_WITH_LIMITATION**。
- 执行入口：`doc/CODEX_START_HERE.md` → `doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS02_STRONG_BASELINE_REPRODUCTION.md`。
- 任务：按冻结的可执行公开条件复现HDPS与Transformer-CLIP-EEG，明确原论文条件差异；当前HDPS仅M，Transformer仅单模型及其原方法包络readout，不启动新架构/GPT重排序。
- 实际结果：HDPS 76人、100Hz/M、作者split及valid索引，Top1/5/10=12.03/30.45/42.10%，shuffle Top1=0.53%。Transformer作者fold4、70人训练、69seen/13heldout，64Hz/XLSR19；2AFC=83.53/80.08%，加权82.38%；冻结encoder原Conv32直接包络r=.1688/.1403，加权.1593，错配近0。三运行均review PASS_WITH_LIMITATION；官方完整cohort/hidden test/ensemble未复现。
- `S03_SS03` 架构与 `S03_SS04` 统一验证均 `BACKLOG`，需要前一环节审阅后再决定。

## 两条数据支线

| 数据与职责 | 当前状态 | 阻塞关系 |
|---|---|---|
| A / SparrKULee EEG decoding / `S03` | ACTIVE；SS02 COMPLETED / PASS_WITH_LIMITATION | 两套公开可用条件基线已交；SS03已授权原型启动，实验待运行 |
| B / ds004703 public intracranial / `S04` | BACKLOG；SS01/SS02 尚未执行 | 自身 sEEG baseline 与 A 线可独立；**仅 EEG→sEEG 迁移需 A 线 frozen 模型** |
| `S05` artifact integration | BACKLOG | 两条支线的结果与失败/许可验收后评估 |

参见 `doc/tasks/PUB-01/RESEARCH_BRANCHES.md` 与 `BENCHMARK_REGISTRY.md`。

## 已经完成且不重跑的证据

- `S01`：公开 SparrKULee / ds004703 数据可用范围下载、审计、初步虚拟清洗、图形化探索已阶段验收；历史 ds004703 319 候选片段**不等于正式模型输入合法性**。
- `S02`：83 名参与者、659 条记录 acoustic/log-mel/frozen wav2vec2 → EEG encoding，完成有限制；HuBERT HOLD。
- `S03_SS01`：10 人、93 条记录 64 Hz EEG→声音包络 backward TRF，test r≈0.1819、R²≈0.0178、错配 r≈−0.0033；2026-10-11 PASS_WITH_LIMITATION，证据 `reports/pub_01/s03/envelope_decode_20261011/REPORT.md`。

## 硬边界 / Stop

- 所有数据、权重和实际计算 **server2203**；本地 Windows Git 只存轻量代码、配置和汇总；先只读盘点已有工作，不覆盖用户未提交内容。
- 不读取任何患者/STN 数据与 embedding；公共模型只通过冻结、版本化 `ART-AUDREP-v1` artifact 向患者侧 `INT-01` 单向交付。当前尚未冻结。
- 当前两套方法的可用条件复现不等于完整论文数值一致性；完成SS02三级审阅后停止，不自启SS03/S04或GPT新模型。
