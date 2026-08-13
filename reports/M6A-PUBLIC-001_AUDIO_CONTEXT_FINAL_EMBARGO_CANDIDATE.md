# M6A-PUBLIC-001 Audio context 与 final embargo 接受记录

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-13 |
| 状态 | `FINAL_EMBARGO_COORDINATOR_ACCEPTED` |
| Coordinator review | `ACCEPT`（2026-08-13） |
| G2 前置状态 | `G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE` |
| 单一机器证据 | `reports/audio_context_final_embargo_candidate_20260813.json` |
| 失败证据 | `reports/audio_context_final_embargo_candidate_failed_runtime_env_v1.json` |
| 完整性策略 | `NON_HASH_AUDIT` |

## 完成内容与运行证据

- 清华 TUNA 的旧 Hugging Face 模型路径在 2203 实测 404；固定选择单一 `https://hf-mirror.com` endpoint。小型 `config.json` 为 1842 bytes，JSON 可读，`model_type=wav2vec2`。镜像探测见 `reports/wav2vec2_model_mirror_probe_20260813.json`。
- 唯一模型为 `facebook/wav2vec2-base`、revision label `main`。第三方镜像、mutable `main` 与 no-hash 政策不能提供密码学完整性或不可变 provenance。
- 模型只位于 `/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813`。四个文件共 380,271,415 bytes：`README.md` 1,997；`config.json` 1,842；`preprocessor_config.json` 159；`pytorch_model.bin` 380,267,417。
- 缓存完成语义验证后已冻结为 `SEMANTICALLY_VALIDATED_REMOTE_ONLY`；主配置 `model.download_allowed=false`，neural method `execution.model_download_allowed=false`。受控下载脚本在当前配置下必须非零退出；实测拒绝前后四个文件的名称、字节数和时间戳未变化。
- `pytorch_model.bin` 仅以 PyTorch `weights_only=True` 读取；运行时若不支持即失败，不降级。218 个 state-dict 值均为 tensor，卷积首层、feature projection、Transformer 第 1/12 层 q-projection 的关键 shape 均符合冻结配置。
- 模型加载固定 `local_files_only=True`、`trust_remote_code=False`，未执行仓库自定义代码。base encoder 无缺失或 shape mismatch；仅排除预声明的 7 个 pretraining quantizer/projection head 参数。
- 推理输入为单个独立 passage 的 44.1 kHz mono 波形，按半开样本区间裁剪，使用 8,821-tap 对称 Kaiser 多相 FIR 独立重采样到 16 kHz；原输入支持半径 28 samples，边缘为 0.0006349206349206349 s。禁止隐式 downmix、相邻 passage 读取和 primary inference batch padding。
- synthetic forbidden-path sentinel 只读取 `train/passage.wav`，validation/test sentinel 未读取；与其并列的真实 319 行 audio identity gate 机器核对了所有 `audio_file` 非空、48 个唯一音频文件、每个文件只映射一个 stimulus/block/split、跨 split 音频文件为 0 且名单为空、采样率全为 44100、声道全为 1、来源状态全为 `BUNDLED_BLOCK_AUDIO`。两项证据共同支持跨 split 输入 overlap 候选 0.0 s。Transformer 允许在当前 passage 内全局注意，未把它表述为局部 receptive field。
- synthetic model canary 返回 projected + 12 Transformer layers。1 s 裁剪后为 16,000 samples；卷积核/步长给出 400-sample convolutional receptive field、320-sample stride，期望/实测均为 49 帧，首帧中心 0.01246875 s、帧步长 0.02 s。
- `final_embargo=max(2.0, 0.5, 1.091796875, 0.0, 0.0006349206349206349)=2.0 s`。真实轻量 319 行 split 在 2.0 s 下通过并获协调接受：train/validation/test=223/48/48，固定 block assignment，English=319、Catalan=0，`baseline_final=true`。
- 单一机器 gate 的 22 项 required checks 全部为 true；`execution_remains_blocked` 同时要求 `scientific_result_claimed=false`。首次以环境内 Python 绝对路径启动导致运行时环境名缺失，gate fail closed，随后用专用 Conda 正式入口重跑通过，失败报告未覆盖。
- 最终 2203 复跑为 `108 passed, 169 subtests`；Ruff PASS；mypy 对 34 个源文件 PASS；主配置、冻结 neural method、formal-src direct-convolution 禁用扫描均 PASS。反例覆盖空 audio_file、跨 split、stimulus/block 漂移、采样率/声道/来源状态漂移、候选证据内单条 assignment 漂移、缓存下载重开及科学结果越级声明。旧运行入口失败历史仍保留。

## 科研可声称与不可声称

可声称：单一模型的非密码学镜像缓存、配置/权重安全读取、单 passage 输入隔离、有限支持重采样、synthetic hidden-state/frame-time 行为、0.0 s 跨 split 输入 overlap 及 2.0 s final embargo/baseline-final split 已有机器证据并获协调接受。

不可声称：第三方镜像或 `main` 得到密码学固定；Transformer 具有局部 receptive field；ridge/null/指标、G4、M6A→M6B exchange candidate、整条 M6A accepted/frozen 或任何科学结果已经完成。

## 失败、阻塞与停止点

- TUNA 旧镜像 404 与 `model.safetensors` 不存在均作为路径选择证据保留；实际仓库使用 `pytorch_model.bin`，并由安全权重门禁约束。
- 首次机器 gate 仅因 `CONDA_DEFAULT_ENV` 未记录而失败；修正启动方式后通过，未降低门禁。
- `baseline_final=true` 只表示 split/final embargo 已接受；全数据 `neural_extraction_allowed=false`、`exchange_candidate_exists=false`、`scientific_result_claimed=false`，模型下载权限保持关闭。
- 后续仅可按独立 scoped G3 配置执行单 recording/单 passage 工程 smoke，不得把本接受记录升级为整条 M6A 或科学结果。

## Git 与下一节点

本节点已由协调任务提交推送并独立接受。后续 Git 与 G3 状态见 `doc/CURRENT_TASK.md`；本文件不作为 G3 或整条 M6A 接受声明。
