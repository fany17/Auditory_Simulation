# M6A-PUBLIC-001 G0-G1 阶段报告

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 阶段 | `G0 task gate + G1 remote environment` |
| 状态 | `PASS` |
| 下一门禁 | `G2 ds004703 metadata/readability/split audit` |

## 完成内容

- 建立正式任务书、机器配置、artifact schema、no-hash 配置门禁和 split 泄漏检查器；
- 本地 10 项单元测试通过；
- 在 2203 新建独立 Conda 环境 `auditory_m6a_public_001`，未复用或修改其他环境；
- 远端同一代码快照 10 项测试通过，配置门禁为 `PASS`；
- 保存轻量环境版本清单和 `reports/remote_selfcheck_2203.json`；
- 大数据、模型权重和大体积输出均未进入本地仓库。
- 将原 schema 降级并改名为 Auditory 内部运行 manifest；另建 M6A→M6B exchange contract 草案，二者字段和发布状态明确分离。

## 运行证据

- 主机：`nercn`
- Python：3.11.15
- PyTorch：2.11.0+cu128
- Transformers：5.14.1
- CUDA runtime：12.8
- GPU：2 × NVIDIA L40
- 自检时可用磁盘：1,046,709,411,840 bytes
- 必需包缺失数：0
- 本地测试：10 passed
- 远端测试：10 passed
- 配置门禁：PASS

完整包列表见 `environment/resolved_packages_2203.txt`；该列表只记录版本，不提供密码学完整性保证。

## 科研可声称

- M6A 首轮任务边界、数据/模型默认值、split/null/指标和 artifact schema 已正式冻结；
- 2203 专用环境可连接、可导入必需软件并可使用两张 L40；
- 当前防泄漏检查器能识别 stimulus、recording、speaker 和时间邻域跨 split 泄漏。
- 当前只建立了 `DRAFT_PROPOSED_BY_M6A` 的跨项目接口草案，不存在冻结 M6A artifact。

## 科研不可声称

- 尚未证明 `ds004703` 文件完整、时间轴可审计或可形成有效 split；
- 尚未下载或读取正式神经数据和模型权重；
- 尚无任何 acoustic、wav2vec2、ridge、null 或 held-out 指标；
- 不能声称模型层与脑区对应、存在神经机制或适用于 STN。
- 不能声称当前 exchange contract 已被 STN 接受，也不能把内部 run manifest 当作跨项目合同。

## 失败、修正与阻塞

1. 默认 Conda channels 要求交互接受 Anaconda 条款；未代替用户接受，改为 `conda-forge` + `--override-channels` 后创建成功。
2. 第一版 NumPy 固定值仅支持 Python 3.12；安装在解析阶段停止，改为 Python 3.11 可用的固定版本后成功。
3. pip 构建一个小型第三方 wheel 时自动打印了内部摘要行；项目没有请求、读取、比较或据此验证完整性。后续 bootstrap 已改为 quiet 安装，避免重复暴露该类输出。
4. 当前无外部阻塞；下一门禁是 G2 数据审计。
5. 跨项目接口为 `REDESIGN_REQUIRED`；不阻塞 G0-G2，但冻结发布必须等待协调审核和 STN consumer cross-test。

## Git 提交与推送状态

- G0 启动任务书、配置、代码和测试已提交并推送到 `main`；
- 本报告、G1 环境修正和自检证据随本阶段结束提交并推送；
- 不记录或核验提交对象标识。

## 下一节点

在 2203 获取 `ds004703 v1.1.0`。先核验 DOI、CC0 声明、README 非商业/禁止再识别限制、文件名/字节数/时间戳/数量和抽样可读性，再构建真实 recording/stimulus manifest 并运行 split guard。只有 G2 通过才进入模型权重与单 recording 特征 smoke test。
