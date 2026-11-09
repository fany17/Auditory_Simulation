用户最新已明确“启动吧”，这是对 SS06 五种连续信息框架进入真实公开 EEG 训练与比较的授权，覆盖 prototype01 的仅原型/不训练/完成后停止条款。请现在作为原 PUB 执行会话继续，原模型设置 gpt-6.1-sol/high，直到完成实际结果与审阅；不要只交准备报告或再次征求启动许可。

工作位置：E:\BCI_LFP_EEG\Project\0_EEG_Explanation_Auditory\_worktrees\Auditory_Simulation_ss06，分支 codex/pub-01-s03-ss06-continuity，PR https://github.com/fany17/Auditory_Simulation/pull/6。这是协调者已经建好的独立 worktree；授权你接管该工作树，协调者在派发后不再编辑此分支文件。先只读 status，确认当前分支和最新远程一致，再继续。不要切换/重置主目录，不动原 CNN/TRF、stash 和旧 worktree。PR#3/#4已合并 main60c3404；SS02/SS03/SS05历史证据冻结不动，不回去续跑。

必读 AGENTS.md、doc/CODEX_START_HERE.md、doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_REAL_EEG_EXECUTION.md、同目录 PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md，以及 reports/pub_01/s03/continuity/prototype01/REPORT.md/REVIEW_CDE.md。这些文件记录本次人工启动决定与完整执行契约。五模型原型和38项测试/71子测试已有2203工程证据，但真实 loader/训练尚未做；不要冒称已开训。

仅 server2203:/home/fanyu/auditory_simulation_m6a 内处理现有合法公开 EEG、数据、缓存、权重和训练；在 pub_01/s03_continuity_20261023/eeg_run01 新目录运行，不覆盖 prototype01 或旧结果，不在服务器初始化Git，不新增下载/数据范围/架构或患者/STN。已有Python入口 /home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python。SSH加 BatchMode=yes/ClearAllForwardings=yes/ConnectTimeout=15。

立即开始：核对实际 source_family/story/record/subject 与旧split谱系，补真实连续EEG loader并重拟合新train-only EEG和目标统计；先M1/local真实train-val pilot，再推进M2/M3/M4/M5和既定对照。首轮每候选100步、明确工程需要最多200步，单seed；基于实际吞吐、内存和train/val曲线冻结有限正式预算、共同lag/目标、至少3seed及对照/统计，然后从头训练并按合格条件一次评估。你可自行解决可逆工程问题和合理实现选择，不要因为需要loader、预算匹配或数据不齐收工。

数据约束：SS05全部426条来自旧SS03train；旧SS03/SS05 test已查看，不能调参或改名成新test。优先从SS05原train23个source families按元数据/固定seed一次冻结train/val/audit_holdout，在新拟合/预留信号读取前完成。诚实标记本轮前瞻内部审计、曾用于既往模型训练，不声称独立外部确认。新train-only归一化和M3自监督不能包含预留组。若无足够可信预留组，继续完成全部可行train/val探索，标EXPLORATORY/HOLD_NEW_TEST；没有独立test不能整体停工。

连续性关键：batch=1连续record或边界同步批次；60秒同一连续输出拼接后一次Pearson/MSE和optimizer step，不能退回3秒平均损失；TBPTT detach保留状态，record/subject/session/split/gap边界才reset。M3当前辅助loss只支持fresh-state完整连续段，正确处理teacher/EMA；frozen/finetune共享同一预训练初始化，no_pretrain可与同结构GRU合并计为一个基线。历史干预使用固定encoder缓存的state-only对照，确保当前EEG/局部特征相同；全前缀干预另列。local+EMA、预测包络历史prior、M4fixed/adaptive、M5prior/EEG-only/full等按任务书落实或明确缺口，不能用简陋prior-only假称已排除全部输出自相关。报告实际参数/样本/时间成本，阴性照实，不按test追分或加模型。

允许修改SS06模型/新loader/scripts/tests/configs、eeg_run01轻量报告及必要任务登记；禁止覆盖旧frozen证据。代码和轻量报告commit/push同PR#6分支，不自行合并、不启动SS04/S04/父S05。用户已授权你与协调会话相互汇报问题和结果，协调会话ID 01a134e4-2036-7e02-81e9-1289a07606f2（local）。启动后请回报真实阶段、服务器目录、入口/进程或调度标识；随后内部问题由双方解决，用户只接收真实结果、重大失败或必须行动的最小问题。最终提交EXECUTION_LOG、SPLIT_PROVENANCE、FORMAL_PROTOCOL、RESULTS、REPORT、REVIEW和逐seed/逐人证据，完成后停止。
