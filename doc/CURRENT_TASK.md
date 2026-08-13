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

当前节点同时受两个候选门禁约束：`G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW` 与 `G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW`：

- 协调者已于 2026-08-13 接受 final-embargo candidate；`final_embargo_seconds=2.0`、`final_embargo_status=FINAL_EMBARGO_COORDINATOR_ACCEPTED`、`split_status=BASELINE_FINAL_COORDINATOR_ACCEPTED`、`baseline_final=true`。这只接受当前 split/final embargo，不是整条 M6A、artifact 或科学结论 PASS/FROZEN；
- 唯一主模型 `facebook/wav2vec2-base` 缓存保持 `SEMANTICALLY_VALIDATED_REMOTE_ONLY`，`model.download_allowed=false`；mutable `main`、第三方镜像与 no-hash 政策不能提供密码学完整性或不可变 provenance；
- G3 单 recording 候选已由协调者接受，机器状态为 `G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY`。它只证明预声明一个 recording/一个 train passage 的有界读取、时间轴、shape 与数组可读性，不是科学结果。其既有 wav2vec2 raw-input representation 仅保留工程 shape/time 证据，状态为 `MUST_NOT_REUSE_FOR_G4_SCIENTIFIC_BASELINE`，不重算 G3；全数据 `neural_extraction_allowed=false`；
- 已接受的 G4 原协议历史报告保持 `G4_PROTOCOL_COORDINATOR_ACCEPTED`。当前实质性修订只新增 passage-wise wav2vec2 preprocessing：固定 16 kHz、逐 passage 独立 zero-mean/unit-variance、无 padding、`return_attention_mask=false` 时不传 attention mask，并用本地 `Wav2Vec2FeatureExtractor` 严格等价核对。修订在协调复核前不得静默继承 accepted 状态；
- G4 范围继续只覆盖同一受试者、同一 `sub-SD012_ses-02_task-PassiveListen` recording 的 24/8/8 train/validation/test passages，排除 `ses-01`，并冻结 exact `t+lag`、所有 lag 共用的 train-only target transform、固定 20 维 log-mel PCA rank 门禁、ridge SVD 复用、test-only stimulus derangement 与 max-statistic smoke 流程；
- preflight 候选只使用 synthetic longest-passage waveform 和只读空间盘点。`g4_execution_authorized=false`；本轮不读取新 EDF/audio、不提取真实特征、不运行 ridge/null/指标。任何真实 G4 执行仍需协调接受协议修订与 preflight 后另行授权。

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
- 原 `stimulus+recording` 方案：单一 connected component，不可三分；现采用 stimulus+block 分组。依据 2 s preliminary minimum、0.5 s maximum lag、1.091796875 s neural filter edge、0.0 s audio cross-split overlap 与 0.0006349206349206349 s audio resampling edge，final embargo 为 2.0 s；协调已接受 baseline-final split，但这不授权 baseline 运行；
- Catalan：显式标为 `ca`，但 block 音频与 cue 波形冲突，暂不进入 baseline，不能声称跨语言泛化；
- 神经 metadata：11 份 sidecar/channels/events 可读，采样率为 512/1024 Hz，工频为 60 Hz；README 的 C-prefix 排除规则已机器化。标准 `electrodes.tsv`/`coordsystem.json` 均缺失，仅有 9 份非标准 contact RAS CSV 且无脑区标签；
- reference metadata：11 份 sidecar 的 `iEEGReference` 由审计器逐 recording 记录并应一致为 `scalp electrode, not included with data`；首轮主分析预声明 `AS_RECORDED_SCALP_REFERENCE`，禁止猜测缺失 reference 或按 contact 名构造 bipolar pair；
- anatomy gate：`ANATOMY_MAPPING_NOT_READY`；它不单独阻塞后续电极级 smoke，但 region summary 为 `NOT_ESTIMABLE`，在坐标系/atlas/映射可审计前禁止从 contact 名称猜脑区；
- neural target method：协调二审已 `ACCEPT`，状态为 `METHOD_FROZEN_AWAITING_EXECUTION_GATES`；主目标为排除 110-130 Hz 的六个等宽 10 Hz high-gamma 子带，有限 FIR edge 最大值 1.091796875 s。旧 70-150 Hz 方案仅为 sensitivity；全数据 `neural_extraction_allowed=false`，仅 G3 配置中的单 recording/单 passage 读取获授权；
- 声音模型门禁：`facebook/wav2vec2-base@main` 只有 `pytorch_model.bin` 权重；通过 16 kHz mono、单 passage、无 batch padding、`local_files_only=true`、`trust_remote_code=false`、`weights_only=true` 且禁止降级、tensor-only 与关键 shape 检查；projected + 12 Transformer layers、49 帧、20 ms 步长和首帧中心 0.01246875 s 的 synthetic canary 通过。Transformer 可在单 passage 内全局注意，这不等于局部 receptive field；
- G3：协调者已 `ACCEPT`，状态为 `G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY`；真实波形仅分段读取 36 channels、18,850 samples（125.427734375–162.244140625 s），未 preload 整段。输出 1,732 个 recording-origin 50 Hz frames，其中 common valid 1,622；未拟合正式 train-only transform、未运行 baseline 或科学指标；
- wav2vec2 preprocessing 修订：2026-08-13 在 2203 无代理探测清华 TUNA 目标路径返回 HTTP 404；同一轮有限回退 `https://hf-mirror.com` 返回 HTTP 200。远端 `preprocessor_config.json` 为 159 bytes，机器核对 `feature_size=1`、`sampling_rate=16000`、`padding_value=0.0`、`do_normalize=true`、`return_attention_mask=false`、`padding_side=right`。缓存保持 remote-only、下载关闭；第三方镜像、mutable `main` 与 no-hash 边界不提供密码学完整性或不可变 provenance；
- G4 synthetic preflight：实际 free bytes 为 978,024,435,712；data/cache/outputs/log/code 选定合计 21,523,777,809 bytes，项目根合计 21,525,428,164 bytes；data+cache+预计新增 20 GB 为 34,553,621,929 bytes。最长 passage 77.08981859410432 s 对应 `ceil(duration*16000)=1,233,438` 个 synthetic samples；冻结预处理与本地 feature extractor 最大绝对差为 0，不传 attention mask。一次 passage-global forward 产生 13 个 `[1,3854,768]` finite tensors，wall 0.07231187901925296 s，CUDA peak allocated/reserved 为 1,660,685,824/1,962,934,272 bytes；保守估算单次 150 s、40 次合计 1.6666666666666665 h。该候选只证明资源与运行 shape 可行，不授权 G4；
- layer-wise alignment：尚未运行，不能声称存在模型层—脑区功能对应。
- 当前 split 只支持 within-subject unseen-stimulus/block generalization；不支持 subject-held-out、speaker-held-out 或 cross-language；
- M6A→M6B exchange contract review：`REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION`；consumer 为 `READY_WAITING_M6A_CANDIDATE`。真实 exchange candidate 尚无、consumer cross-test 未运行，contract 未 accepted/frozen；candidate 准备仍受 final embargo/split guard 和真实 method/runtime/canary 门禁约束。
- G2 promotion gate：历史机器入口为 `G2_CANDIDATE_AWAITING_COORDINATOR_REVIEW`，所有 required checks 为 true；`g2_pass_claimed=false`、`candidate_contains_raw_data=false`。首次因额外要求 C-prefix 名称唯一而 fail closed 的报告已保留；实际协调要求的 eligible 名单唯一性与 1346/727 聚合均通过。协调者独立验收后当前 G2 状态为 `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE`。
- 最新联合验证：2203 专用环境为 `147 passed, 349 subtests`，Ruff 对完整 `src scripts tests` 通过，mypy 对完整 46 个文件通过，主 config gate 为 `PASS`。已接受原 G4 协议的历史机器报告仍为 `reports/g4_protocol_candidate_20260813_v3.json`；当前修订候选 `reports/g4_protocol_amendment_candidate_20260813_v2.json` 为 9/9 required checks true，当前 preflight 候选 `reports/g4_resource_runtime_preflight_candidate_20260813_v3.json` 为 15/15 true。两者均不得在协调复核前写为 accepted。
- 候选治理：协议修订仅 v2、preflight 仅 v3 可标记 `current_candidate=true`；无版本协议修订报告与 preflight v2 均为 `SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE`，原始 preflight 与镜像无版本报告继续保持 `FAIL`。原协议 accepted v3 报告的 config 路径现已发生 amendment，其机器接受范围明确排除 `WAV2VEC2_PREPROCESSING_INPUT_CONTRACT`；不得据此宣称 amendment accepted。首次 hf-mirror 403 只保留 provenance，不构成当前 preflight PASS 的必要条件。
