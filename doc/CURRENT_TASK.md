# Auditory_Simulation 当前任务状态

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-11 |
| 状态 | `PLANNING_ONLY` |
| 执行授权 | `NOT_AUTHORIZED` |
| 当前方向 | `M6A-PUBLIC：公开声音—神经表征对齐` |
| 权威边界 | `doc/PROJECT_BOUNDARY.md` |

## 当前决定

项目目标已从“先建立一套可训练听觉基座，再进入神经对齐”调整为：

> 优先利用冻结的公开预训练声音模型和公开神经数据建立声音—神经表征对齐基线；是否微调或重新训练声音基座由后续证据决定，不再作为进入对齐工作的强制前置条件。

当前只是完成项目边界与路线归属调整，没有批准下载数据、安装环境、运行模型或创建正式结果。

## 下一项候选任务

候选编号：`M6A-PUBLIC-001`

候选名称：`ds004703 数据冻结与最小 layer-wise alignment 任务设计`

本任务在获得人工批准前只允许规划。建议范围：

1. 冻结 OpenNeuro `ds004703` 的 dataset version、license、文件和大小清单；
2. 核验实际音频、event、subject、electrode、brain-region 和 timing 字段；
3. 定义 subject/story/clip/time-neighborhood 防泄漏 split；
4. 冻结 wav2vec 2.0 模型 revision、逐层输出、采样率和预处理；
5. 定义 acoustic baseline、layer-wise ridge encoding 和 neural->audio retrieval/contrastive baseline；
6. 定义 observed、matched control、null、held-out 指标和停止条件；
7. 估算下载、存储、GPU、运行时间和缓存位置；
8. 形成正式任务书草案，等待人工批准。

## 当前明确不做

- 不下载 `ds004703` 或其他数据集；
- 不下载或运行模型；
- 不新建 Python/Conda 环境；
- 不启动大规模模型比较；
- 不同时加入 emotion2vec、BEATs、CLAP、MERT；
- 不读取或迁移 STN 患者数据；
- 不把异构模型参数预测称为表征对齐；
- 不把公开脑区—模型层相关性写成生理机制同一；
- 不初始化 Git、提交或推送。

## 进入执行前的人工门禁

需要人工确认：

1. 是否批准 `ds004703` 作为首个正式数据集；
2. 数据下载与缓存位置；
3. 首个声音模型和精确 revision；
4. 是否只做 frozen feature，还是允许训练 neural encoder；
5. 允许的 GPU、磁盘和运行时间；
6. 首轮主终点是 Audio->Brain encoding、Brain->Audio retrieval，还是二者的预注册组合；
7. 正式任务编号是否继续使用 `M6A-PUBLIC-001`。
