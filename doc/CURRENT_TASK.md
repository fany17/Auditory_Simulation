# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
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
5. 防泄漏优先，先冻结 stimulus/recording 分组、时间邻域隔离、null、指标和 artifact schema；
6. 所有大数据、权重、特征和训练只位于 2203，本地只保留轻量可审计资产；
7. 不进行任何 SHA、MD5、checksum、文件哈希或 Git 对象哈希操作。

跨项目接口已完成 producer 侧 16 项返工：Auditory 内部运行 manifest 与 M6A→M6B exchange manifest 保持分离。exchange contract 当前为 `REVISED_DRAFT_AWAITING_CONSUMER_REVIEW`，尚无真实 manifest、method package 或 canary；须经 STN consumer 复核，且后续真实 candidate cross-test 与协调接受后才能另发正式 v1。当前不存在冻结 M6A artifact。

## 当前节点

当前节点为 `G0-G2 启动门禁`：

- 建立正式任务书、配置、schema 与测试；
- 建立 2203 专用 Conda 环境并完成连接/环境自检；
- 记录 `ds004703` 官方版本、许可、README 限制和轻量文件清单；
- 获取公开数据并完成最小可读性、BIDS schema、显式 language/stimulus manifest 和 split 泄漏测试；
- 在上述门禁通过且资源正常时，进入第一批冻结音频表征提取与 ridge smoke baseline。

## 硬停止条件

- 下载接口要求账户本人点击同意、签署新条款或提供个人凭据；
- 出现商业用途、再识别或原始数据外传需求；
- 实际版本、许可、音频—事件关系或时间轴无法审计；
- split 不能阻断 stimulus、recording 或时间邻域泄漏；
- 需要读取 STN 患者数据；
- 预计新增存储超过 500 GB、连续 GPU 运行超过 24 小时或需要重大不可逆操作；
- 任务需要微调声音基座、引入第二数据集或改变项目边界。

## 当前证据状态

- 正式科学结果：`NOT_YET_AVAILABLE`；
- `ds004703` 许可：`ACCEPTED_WITH_STRICTER_README_BOUNDARY`；
- 数据下载/读取：`AUTHORIZED_ON_2203`，尚待实际运行证据；
- 专用环境：`G1 PASS`；2203 独立环境、自检和远端 10 项测试均通过，证据见 `reports/remote_selfcheck_2203.json`；
- layer-wise alignment：尚未运行，不能声称存在模型层—脑区功能对应。
- M6A→M6B exchange contract：`REVISED_DRAFT_AWAITING_CONSUMER_REVIEW`；producer 侧 33 项远端测试、Ruff 与 mypy 均通过，consumer cross-test 尚未运行，未冻结或接受。
