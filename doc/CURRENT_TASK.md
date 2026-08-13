# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 状态 | `ACTIVE_EXECUTION` |
| 执行授权 | `AUTHORIZED_WITH_GATES` |
| 当前方向 | `M6A-PUBLIC：公开声音—神经表征对齐` |
| 唯一任务 | `M6A-PUBLIC-001` |
| 正式任务书 | `doc/tasks/M6A-PUBLIC-001.md` |
| 机器配置 | `configs/m6a_public_001.json` |
| 计算位置 | `server2203:/home/fanyu/auditory_simulation_m6a` |
| Git 边界 | 本地唯一 Git 工作区；远端计算目录不作为第二 Git 工作区 |

## 当前决定

人工已授权建立并持续完善 `M6A-PUBLIC`。首轮采用以下可逆默认值：

1. 数据集冻结为 OpenNeuro `ds004703 v1.1.0`；
2. CC0 元数据与 README 限制共同生效，执行更严格的非商业、禁止再识别、禁止原始数据外传边界；
3. 首个声音模型为冻结的 `facebook/wav2vec2-base`，用于纯 speech SSL 逐层表征，不使用 ASR 微调模型作为首个主模型；
4. 首轮主终点为 `Audio -> Brain` layer-wise ridge encoding；retrieval/contrastive、CCA、RSA、CKA 留待后续节点；
5. 防泄漏优先，先冻结 stimulus/block 分组、recording 内时间邻域隔离、null、指标和 artifact schema；
6. 所有大数据、权重、特征和训练只位于 2203，本地只保留轻量可审计资产；
7. 不进行任何 SHA、MD5、checksum、文件哈希或 Git 对象哈希操作。

跨项目接口的 revised DRAFT 已通过 STN consumer 二审与协调独立复跑：`REVISED_DRAFT_REVIEW=ACCEPT`，原 16 项、non-finite 与 layer-order 反例均 fail closed。当前准确状态是 `REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION` / `READY_WAITING_M6A_CANDIDATE`。Auditory 内部运行 manifest 与 M6A→M6B exchange manifest 保持分离；当前没有真实 manifest、method package 或 canary，`CONSUMER_CROSS_TEST=NOT_RUN`，contract 仍非 accepted/frozen，也未发布正式 v1。

## 当前节点

当前节点为 `FINAL_EMBARGO_CANDIDATE_AWAITING_COORDINATOR_REVIEW`：

- 协调者已于 2026-08-13 接受 G2 candidate；准确状态为 `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE`，不是整条 M6A PASS/FROZEN；
- 唯一主模型 `facebook/wav2vec2-base` 已在 2203 项目专用缓存通过同一 `hf-mirror.com` endpoint 落盘并冻结为 `SEMANTICALLY_VALIDATED_REMOTE_ONLY`；`model.download_allowed=false`。使用 mutable `main`，第三方镜像与 no-hash 政策不能提供密码学完整性或不可变 provenance；
- synthetic sentinel/canary 与真实 319 行 audio identity gate 已并列验证输入隔离：48 个唯一 `audio_file` 均只属于一个 stimulus/block/split，跨 split 音频文件为 0，采样率全为 44100、单声道、来源均为 `BUNDLED_BLOCK_AUDIO`。跨 split 输入 overlap 候选为 0.0 s，final embargo 候选为 2.0 s，319 行 split guard 在该候选下通过；
- 当前输出为 `FINAL_EMBARGO_CANDIDATE_AWAITING_COORDINATOR_REVIEW`；协调复核前 `baseline_final=false`、`neural_extraction_allowed=false`；
- 禁止读取真实神经波形生成 target、正式提取 acoustic/wav2vec2 特征、运行 baseline/G3 smoke 或建立 M6A→M6B exchange candidate。

## 硬停止条件

- 下载接口要求账户本人点击同意、签署新条款或提供个人凭据；
- 出现商业用途、再识别或原始数据外传需求；
- 实际版本、许可、音频—事件关系或时间轴无法审计；
- split 不能阻断 stimulus、block 或 recording 内时间邻域泄漏；
- 需要读取 STN 患者数据；
- 预计新增存储超过 500 GB、连续 GPU 运行超过 24 小时或需要重大不可逆操作；
- 任务需要微调声音基座、引入第二数据集或改变项目边界。

## 当前证据状态

- 正式科学结果：`NOT_YET_AVAILABLE`；
- `ds004703` 许可：`ACCEPTED_WITH_STRICTER_README_BOUNDARY`；
- 数据下载/读取：保留首次 374/377 与 Range 切换历史；2026-08-13 仅补齐 SD011/SD019 的 2 个缺失区间，复用已有 350 chunks，0 失败，并原子装配 SD011/SD019/SD022。full audit 为 377/377 paths、14,173,350,514 bytes，missing/unexpected/byte mismatch/partial 均为 0；G2 candidate 已由协调者接受用于进入 audio-context gate，但这不是整条 M6A PASS/FROZEN；
- 专用环境：`G1 PASS`；2203 独立环境、自检和远端 10 项测试均通过，证据见 `reports/remote_selfcheck_2203.json`；
- 轻量 manifest/split：`SPLIT_REWORK_COMPLETED`；438 segments 中 319 个 English passage segment 进入确定性 primary split，train/validation/test 为 223/48/48（69.9%/15.0%/15.0%）；旧 208/32/79 分配已被审核否决并保留失败历史；
- 原 `stimulus+recording` 方案：单一 connected component，不可三分；现采用 stimulus+block 分组。依据 2 s preliminary minimum、0.5 s maximum lag、1.091796875 s neural filter edge、0.0 s audio cross-split overlap 与 0.0006349206349206349 s audio resampling edge，final embargo 候选为 2.0 s；候选 guard 已重跑通过，但协调接受前仍不是 baseline-final；
- Catalan：显式标为 `ca`，但 block 音频与 cue 波形冲突，暂不进入 baseline，不能声称跨语言泛化；
- 神经 metadata：11 份 sidecar/channels/events 可读，采样率为 512/1024 Hz，工频为 60 Hz；README 的 C-prefix 排除规则已机器化。标准 `electrodes.tsv`/`coordsystem.json` 均缺失，仅有 9 份非标准 contact RAS CSV 且无脑区标签；
- reference metadata：11 份 sidecar 的 `iEEGReference` 由审计器逐 recording 记录并应一致为 `scalp electrode, not included with data`；首轮主分析预声明 `AS_RECORDED_SCALP_REFERENCE`，禁止猜测缺失 reference 或按 contact 名构造 bipolar pair；
- anatomy gate：`ANATOMY_MAPPING_NOT_READY`；它不单独阻塞后续电极级 smoke，但 region summary 为 `NOT_ESTIMABLE`，在坐标系/atlas/映射可审计前禁止从 contact 名称猜脑区；
- neural target method：协调二审已 `ACCEPT`，状态为 `METHOD_FROZEN_AWAITING_EXECUTION_GATES`；主目标为排除 110-130 Hz 的六个等宽 10 Hz high-gamma 子带，有限 FIR edge 最大值 1.091796875 s。旧 70-150 Hz 方案仅为 sensitivity；`neural_extraction_allowed=false`；
- 声音模型门禁：`facebook/wav2vec2-base@main` 只有 `pytorch_model.bin` 权重；通过 16 kHz mono、单 passage、无 batch padding、`local_files_only=true`、`trust_remote_code=false`、`weights_only=true` 且禁止降级、tensor-only 与关键 shape 检查；projected + 12 Transformer layers、49 帧、20 ms 步长和首帧中心 0.01246875 s 的 synthetic canary 通过。Transformer 可在单 passage 内全局注意，这不等于局部 receptive field；
- 剩余执行门禁：final embargo 候选协调验收与 baseline-final 状态接受；方法冻结、G2 接受和本候选均不等于 G3 启动、真实 target、M6A exchange candidate、整条 M6A accepted/frozen 或科学结果；
- layer-wise alignment：尚未运行，不能声称存在模型层—脑区功能对应。
- 当前 split 只支持 within-subject unseen-stimulus/block generalization；不支持 subject-held-out、speaker-held-out 或 cross-language；
- M6A→M6B exchange contract review：`REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION`；consumer 为 `READY_WAITING_M6A_CANDIDATE`。真实 exchange candidate 尚无、consumer cross-test 未运行，contract 未 accepted/frozen；candidate 准备仍受 final embargo/split guard 和真实 method/runtime/canary 门禁约束。
- G2 promotion gate：历史机器入口为 `G2_CANDIDATE_AWAITING_COORDINATOR_REVIEW`，所有 required checks 为 true；`g2_pass_claimed=false`、`candidate_contains_raw_data=false`。首次因额外要求 C-prefix 名称唯一而 fail closed 的报告已保留；实际协调要求的 eligible 名单唯一性与 1346/727 聚合均通过。协调者独立验收后当前 G2 状态为 `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE`。
- 最新联合验证：2203 专用环境为 `108 passed, 169 subtests`，Ruff 通过，mypy 对 34 个源文件通过；audio-context 单一机器 gate 的 22 项 checks、neural target semantic gate、main config gate 与 formal-src direct-convolution scan 均通过。首次 audio-context gate 因运行时环境名缺失而 fail closed 的报告已保留；模型下载工具在冻结配置下实测拒绝且未改变缓存。
