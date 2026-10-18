# PUB-01-S03_SS02 协议核对矩阵

2026-10-15最终实测：HDPS **AUTHOR_PROTOCOL_AVAILABLE_SUBSET**，原100Hz/M/76完整被试/599对，test12776query，100–235候选，Top1 12.03%/Top5 30.45%/Top10 42.10%。Transformer **DERIVED_PUBLIC_AVAILABLE_PROTOCOL**，原架构/public fold4/原XLSR19，70训练、69seen/13heldout；longrecord±4s2AFC=83.53/80.08%，加权82.38%；冻结EEG encoder原Conv32包络readout按60s段r=.1688/.1403，加权.1593。三项运行review均PASS_WITH_LIMITATION，SS02 COMPLETED。非完整paper cohort、MWG、official packet或ensemble。旧OFFICIAL_PROTOCOL的NOT_RUN仍指完整原条件，不覆盖新增真实子集结果；以RESULTS逐行实际protocol为准。

10-13历史追加状态：原raw100Hz/Mel70story、独立论文功能模型batch256工程smoke已执行；602条产物与作者647对中600对就绪，必要47条raw受限；当时尚无训练/测试。Transformer EEGcheckpoint失效，官方评价载荷未齐，XLSR原声音权重已在2203取得。10-15改为从头训练功能复现，不依赖失效EEGcheckpoint；最新事实和差异见`REPRO_REPORT.md`。

以下保留首次preflight矩阵的历史快照，不能把“未取得索引/模型smokeNOT_RUN/100Hz无产物”当当前全部工程状态。固定split与候选定义不改，未用内部分数补齐。

核对日期：2026-10-13。结论：**官方协议已定位，A+B 的训练/模型 smoke/性能评估均 NOT_RUN，验收 HOLD。** 本矩阵冻结核对对象与比较边界，不表示正式训练准入已通过。现有内部 64 Hz split 未被替换为作者 split，也未启动内部替代训练。

## 固定源码版本与许可

| 项目 | 固定 commit | 实际入口 | 代码许可核验 |
|---|---|---|---|
| [HDPS](https://github.com/bobwangPKU/HDPS) | `42a5e1ec9649f4a673bfc018d377bbf38e289ed4` | `utils.py` → `train_clip_SparrKULee.py` / `train_reg_SparrKULee.py` | 整个源码 tree 无 LICENSE/COPYING/NOTICE；README 给运行说明，未定义使用/修改/分发许可证；本项目许可门 HOLD |
| [Transformer-CLIP-EEG](https://github.com/exporl/transformer-clip-eeg) | `8be963abe5be5c0568e54df3bceb2600861f895b` | `preprocess_audio/create_wav2vec.py` → `train_clip_final.py` | 整个 tree 无许可文件；权重链接不等于权重许可；许可门 HOLD |
| [Challenge 2023](https://github.com/exporl/auditory-eeg-challenge-2023-code) | `8de4eb84c37f7a6455ace94eff8709e515eae09a` | `task1_match_mismatch/experiments/dilated_convolutional_model.py`、`task2_regression/experiments/linear_baseline.py`、`test_scripts/score_*.py` | LICENSE 原文为 GPL-3.0-or-later；GitHub API 的 NOASSERTION 不代表无许可证 |
| [Challenge 2024](https://github.com/exporl/auditory-eeg-challenge-2024-code) | `bd6b018ffd11c0f6a9bfdf4ac40bdf6ce846d365` | `preprocessing_code/`、`task1_match_mismatch/experiments/`、`task2_regression/experiments/` | GPL-3.0-or-later |

固定 tree、37 个指定源码/文本文件及许可证只作静态阅读，未 import/执行作者代码。服务器副本保存在 `input_run02/source/` 和 `run02/source_inspection_only/`；本地静态镜像被 `.gitignore` 排除，未将未明确许可的作者代码纳入本项目 Git。逐文件 URL、读取与 AST 结果见 `evidence/run02/source_manifest.json`。

## 官方协议与实际运行分列

| 方法 / 原任务 | 数据与采样/窗口 | 官方 split / 候选与负例 | 表示、输入输出、先验 | 评分口径 | 本轮实际状态 |
|---|---|---|---|---|---|
| HDPS / 多候选 contrastive 检索 | SparrKULee；源码 `model_config.py` 明确 100 Hz、3 s、64 EEG 通道 | 作者 `derivatives/SparrKULee/split.json`：81 人、647 record-phase 对（465 train / 91 val / 91 test）；`valid_model` 对每名被试取得该 phase 的全部有效片段为候选，**不是固定 100 个**；实际候选数依有效片段索引和特征长度，尚未生成 | EEG encoder 输入 EEG + subject ID；声音端可用 Mel80 / Dutch wav2vec2-base layer9 / Dutch GPT2 layer9、20词上下文；融合三者包含语言先验。候选声音用于相似度检索，不能视为可听语音重建 | test Top1、Top10 先逐被试再平均；Top5 仅见 training batch rank 日志，正式 test Top5 需明确新增评价；不能移用训练 Top5 | OFFICIAL_PROTOCOL；模型 smoke/训练/评估 NOT_RUN；HOLD。647 对均找到现有 **64 Hz** 文件，只是文件覆盖，不证明 100 Hz 准入 |
| HDPS / regression | 同上；与 contrastive 分开训练 | 作者 split，同样须核验有效片段和目标特征 | `train_reg_SparrKULee.py`；连续声音特征目标；与三特征融合检索结果分开 | 需完整原文/脚本评价核对后方能固定正式 direct endpoint | OFFICIAL_PROTOCOL；NOT_RUN / HOLD；未用 SS01 TRF 充当本方法结果 |
| Transformer-CLIP-EEG / representation + 2023 2AFC、envelope regression | 64 Hz；训练 3 s、不重叠；Dutch XLSR-large layer19、1024维；批大小128；论文总194人，训练71+109=180人 | 官方测试：70名 seen subject 的新故事 +14名 held-out subject；`fold_split.json` 当前代码 train_subs=91，test_subs=14，9个故事fold和`test_set_2023_1`；代码 manifest 与完整论文训练规模不同，不能静默等同。2AFC 测试须官方 sample ID、候选映射和测试载荷 | EEG latent encoder + speech encoder；contrastive 训练负例来自同一录音近邻片段，跨被试 shuffle；无 GPT2；固定预训练声音表示仍是 prior。envelope 回归是固定 embedding 的后续线性映射，不等于直接输出原音频 | 逐被试 mean；2023 官方汇总 `2/3*seen_macro + 1/3*heldout_macro`；单模型与 ensemble 分开。论文有16个模型统计和15模型logit ensemble，不能以单次训练冒充 ensemble | OFFICIAL_PROTOCOL；NOT_RUN / HOLD；缺官方 TEST_task1/2 载荷与候选映射；现有71/91 train subjects、14/14 test subjects仅文件存在 |
| Challenge2023 / match-mismatch | 85人版本；预处理 EEG 64 Hz，3 s（192 samples） | 官方测试84人，sub-001无test；2候选；同故事近邻 mismatch，相对匹配起点±4 s；70 seen新故事 /14 unseen被试分列 | EEG+2段声音表示→二分类；baseline可用envelope/Mel；无预设LM | 每被试accuracy，再按2/3与1/3汇总；缺sample ID计错。官方repo现有84份match标签文件，**不能笼统声称标签隐藏** | 协议审查完成；OFFICIAL_PROTOCOL；模型/官方测试 NOT_RUN / HOLD，当前缺信号测试包和candidate mappings；不得追加受密码限制载荷下载 |
| Challenge2023 / regression | 64 Hz；30 s测试（1920 samples） | 与match任务测试区间分开；同上两个泛化组 | EEG→envelope；目标明确固定 | 按segment相关，再subject，再两组加权；对应官方score脚本 | 协议审查完成；NOT_RUN / HOLD；标签文件存在不等于官方测试信号可得 |
| Challenge2024 / match-mismatch | 105人版本；测试20新被试，64 Hz，**5 s**（320 samples） | 5候选=1 matched+4同故事随机window imposters；官方mapping确定候选位置，不能换成旧2AFC | EEG+5声音段→5类；音频表示可自行处理；baseline无LM | 逐被试accuracy后宏平均；缺sample ID计错；description有多test-case表述，最终test_set页明确subject-macro，须保持对应评估版本 | 协议审查完成；OFFICIAL_PROTOCOL；NOT_RUN / HOLD；未发现官方20人测试载荷/候选映射；本次不扩数据下载 |
| Challenge2024 / regression | 64 Hz、30 s；10频带Mel、0–5000 Hz | 新被试和held-out stimuli；官方任务2载荷/版本独立 | EEG→10-band Mel | segment各band Pearson，再subject-macro；description写1920×10、test_set提交页写10×1920，执行时必须按测试loader明确转置；不能猜测schema | 协议审查完成；NOT_RUN / HOLD；保留2023-12-13 raw/MWF对齐修订，preprocessed版本不变 |
| 既有 S02 / SS01 内部参考 | 发布方64 Hz；S02 83人659 READY录音；SS01 10人93条 | per-subject、跨source-family；非官方challenge、非HDPS split | S02声音→EEG；SS01 EEG→envelope TRF；无LM | 历史报告的r/R²；不写入本轮强基线成绩 | INTERNAL_PROTOCOL；历史参考，仅用于盘点。**没有新内部训练结果** |

## 拆分、防泄漏与缺失字段

- HDPS 原 manifest 的全局 train 与 val/test **精确 story label**无交集，但81名被试的val/test均共享至少一个 source-family（如 audiobook_6_1 / audiobook_6_2）。这是作者part级定义的边界，不能写为严格未见source-family；本轮没有修改它。需要 source-family 测试时应单列内部协议，不重命名官方协议。
- HDPS 每个3 s EEG segment按其时间轴作normalization；不能把这种segment-local输入标准化称为train-only统计。其`valid_seg_dict_6.pkl`是仓库公开索引，**本次未下载/反序列化**，不是“作者未公开”；`feat_len_dict.pkl`需按原流程生成。没有这两者和实际100 Hz目标，精确候选数不可估计。
- 当前训练seed：HDPS源码固定1024；Transformer本轮未确定可重跑seed/ensemble模型对应关系，标UNVERIFIED；challenge实际seed均NOT_RUN。没有可据以计算跨seed不确定性的数据。
- train/validation-only选择、test一次评分、candidate identity和duration/normalization按各协议冻结，不能因失败改测试难度。此轮**没有形成可用于训练的internal共同协议或最终candidate manifest**。
- EEG-shuffle / wrong-EEG、audio-only、prior-only、候选池敏感性与误差表均NOT_ESTIMABLE：没有训练模型或预测；不能从fixture、随机初始化或历史TRF补齐。

## 一手来源

- HDPS原论文：[Wang等，IEEE TNSRE，DOI 10.1109/TNSRE.2026.3688564](https://doi.org/10.1109/TNSRE.2026.3688564)；[PubMed原摘要](https://pubmed.ncbi.nlm.nih.gov/42048207/)。publisher全文访问遇robot challenge，未核验其全文细节；上表HDPS协议以固定原代码可读行为为依据。
- Transformer论文：[Bollens等，Scientific Reports 2025](https://www.nature.com/articles/s41598-025-13646-4)。论文开放许可不被推断为独立代码仓库/权重的许可证。
- [2023 task1定义](https://exporl.github.io/auditory-eeg-challenge-2023/task1/description/)、[2023 task1测试包](https://exporl.github.io/auditory-eeg-challenge-2023/task1/test_set/)、[2023 task2测试包](https://exporl.github.io/auditory-eeg-challenge-2023/task2/test_set/)。
- [2024 task1定义](https://exporl.github.io/auditory-eeg-challenge-2024/task1/description/)、[2024 task1测试包](https://exporl.github.io/auditory-eeg-challenge-2024/task1/test_set/)、[2024 task2定义](https://exporl.github.io/auditory-eeg-challenge-2024/task2/description/)、[2024 task2测试包与修订](https://exporl.github.io/auditory-eeg-challenge-2024/task2/test_set/)。
