# SS06 五种连续信息框架：任务书与工程原型

日期：2026-10-23（Asia/Shanghai）。Task：`PUB-01-S03_SS06`。范围：`PROTOTYPE_ONLY`。

**五种框架已实现并在 server2203 通过合成工程检查。真实数据加载、预训练、pilot、参数优化、正式训练及测试全部 NOT_RUN。当前只能说明代码与控制接口能运行，不能判断 EEG 准确率或是否改进。**

## 交付

- [完整任务书](../../../../../doc/tasks/PUB-01/subtasks/PUB-01-S03/PUB-01-S03_SS06_CONTINUITY_FRAMEWORKS.md)：每种框架的科学问题、网络结构、拟议训练目标、主要对照、风险和停止点。
- [接口契约](../../../../../doc/tasks/PUB-01/subtasks/PUB-01-S03/SS06_IMPLEMENTATION_CONTRACT.md)：显式状态、统一 `[B,T,64]→[B,T,1]`、因果性与工作分工。
- `src/m6a_public/continuity/`：五类模型、共享编码器、模型工厂、连续目标、输出平滑和历史干预。
- `configs/pub_01_s03_ss06_prototype_20261023.json` 与 `scripts/pub_01_s03_ss06_smoke.py`：只允许零优化、无数据清单、无 checkpoint 的合成 smoke。

## 实际工程证据

执行位置：SSH alias `server2203`（运行时 hostname `nercn`），目录 `/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/prototype01`。Python 3.11.15、PyTorch 2.11.0+cu128；本轮只用 CPU、单计算线程，无 GPU 训练。

| 框架 | 核心机制 | 注册参数 | 下游梯度参与参数 | 60秒整段/分块最大绝对差 |
|---|---|---:|---:|---:|
| M1 连续密集多尺度 | 不池化轨迹；四 dilation causal Conv 分支 | 40,481 | 40,481 | 9.69e-8 |
| M2 持久快慢状态 | 逐样本 GRU + 可学习慢更新 | 29,345 | 29,345 | 2.98e-8 |
| M3 预测连续表征 | 因果 context + 独立未来 latent 目标 | 47,105 | 25,153 | 5.96e-8 |
| M4 连续自适应对齐 | 同一 lag bank 的慢 EEG 门控 | 18,950 | 18,950 | 7.45e-8 |
| M5 潜状态预测与 EEG 校正 | prior → EEG observation innovation → gain correction | 23,009 | 23,009 | 8.94e-8 |

来源：[SMOKE_FINAL.json](SMOKE_FINAL.json)。hidden=32，输入通道64；短反向为2×256样本，长前向为1×3840样本（64Hz示意60秒），分块长257样本。输出、梯度均有限；主 smoke 前后参数与 buffer 完全不变，optimizer steps=0。M3 注册参数包含未用于下游的 teacher/辅助头；“梯度参与”指在计算图上取得 grad 的参数标量数，不保证每个 grad 非零。M1 短输入仅35361个标量的 grad 非零，长 dilation 的部分历史 tap 尚无支持；长前向/分块覆盖完整感受野。这些计数不等于公平训练预算，耗时也不构成性能排名。

完整合成单元测试：**38 passed，71 subtests passed**，见 [pytest_initial.txt](pytest_initial.txt) 与 JUnit [pytest_initial.xml](pytest_initial.xml)。覆盖：

- 因果后缀扰动、单样本及任意分块、卷积长 dilation 缓存、状态不被原地改写；
- subject/session/record/split 边界 reset，记录内 gap 拒绝，detach 保留数值且切断历史梯度；
- 五模型主路径及 M3 辅助目标有限梯度，teacher/未来目标 stopgrad、冻结模式、padding/horizon 拒绝；
- M1 分支、M4 固定/自适应、M5 prior-only/EEG-only/full 与 prior-only 完全不依赖 EEG；
- 真/清零/错配历史固定当前 EEG；状态专用干预还固定局部编码缓存，三条件当前局部特征逐点相同；
- 真正整段 Pearson 目标与逐短窗平均的区别、mask/常数目标、因果输出平滑。

初测有一条测试断言将 requires_grad tensor 转 scalar 的 PyTorch warning；修正为显式 detach 后，相关10测试/26子测试复测通过且无 warning，见 [pytest_de_final.txt](pytest_de_final.txt)、[pytest_de_final.xml](pytest_de_final.xml)。该修正只影响测试断言，不修改模型。随后补充长序列分块等价检查，五模型全部通过。没有科学实验失败、隐藏运行或准确率结果。

M3 的 EMA 单元测试使用常数扰动和显式 EMA 验证 decay=0/1；M2 的系数边界检查也使用确定性常数。这是临时测试实例的接口验证，不是拟合、优化或训练。主 smoke 本身不改变任何参数。

## 独立审阅与修正

M4/M5 实现者审阅其他人的 M1/M2/common；M1/M2 实现者审阅其他人的 M3/M4/M5 及 root 集成代码。不是外部第三方科学签署。详细文件：[REVIEW_AB.md](REVIEW_AB.md)、[REVIEW_CDE.md](REVIEW_CDE.md)。root 检查全部模块、配置、实际运行日志及范围。

已处理：

1. `nonzero_gradient_scalars` 按实际非零标量计数，保留注册/可训练/图参与参数的区别。
2. 任务书准确区分零优化与确定性参数扰动单元检查；M1 为完整通道卷积而非 depthwise 分支，M2 α 为更新率，M5 为逐维 gain 校正。
3. 全前缀干预标记可能包含最初16样本局部缓存效应；新增 `compare_state_histories` 固定真实 local encoder 缓存，仅替换后续状态，并通过实际测试。
4. 长60秒分块检查补足 M1 最大感受野 beyond 1041 samples 的整体模型覆盖。

## 实现范围与仍需工作的部分

| 当前已实现 | 未来实验前仍需实现/冻结（未启动） |
|---|---|
| 五基本模型、显式状态、核心模式开关 | 真实连续 EEG loader、按记录/分组连续性与新 train-only normalization |
| 连续 `1-r + λ MSE` 损失及 M3 多 horizon 预测/方差目标 | M4 权重平滑正则/M5 观测预测辅助目标的正式权重与训练集成 |
| 真/错配/清零全前缀与固定encoder缓存状态干预；因果EMA函数 | 预注册历史匹配、评分 warmup、预测包络历史先验、强GRU/参数和算力匹配 |
| M3 no_pretrain/frozen/finetune 接口及 teacher 隔离 | 实际预训练、预训练权重交接、teacher流式状态/跨chunk辅助目标/TBPTT |
| CPU随机输入前向/反向与流式等价检查 | 真实split/声学lag/预算/种子/正式统计比较、GPU/混合精度与长训练稳定性 |

M1 只有约16.27秒有限卷积历史；M2 α初值.01约1.55秒指数时常，并未学习长时信息；M3 当前辅助损失仅支持从新边界开始的整段输入；M5 prior-only 只是与时间位置有关的自主动力学对照，不是已训练的语音先验。`StreamRunner` 任一 batch lane 换身份会重置整批，offset也按整批对齐。上述均已明示，不能当成生产级数据管线。

旧 SS03/SS05 测试已查看，不能在后续模型选择中再次冒充独立测试；当前没有创建新 split、读取任何数据或改变旧结果。下一步须由研究者明确何时开始真实实验；当前到此停止。

## 复核命令（仅在2203，新目录，无训练）

```bash
cd /home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/prototype01
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKPY=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python
"$TASKPY" -m pytest -q tests
"$TASKPY" scripts/pub_01_s03_ss06_smoke.py \
  --config configs/pub_01_s03_ss06_prototype_20261023.json \
  --output evidence/smoke_recheck.json
```

实际脚本运行使用 subprocess 并保存 stdout/JUnit；第一次SSH环境探测因 PowerShell引号出现SyntaxError，改为stdin脚本后成功，没有运行模型/改动资料。app worktree工具因当前会话cwd是repo父目录返回Not a git repository，改用明确repo上的 `git worktree add` 创建独立工作树；未切换或修改主目录。

## 交付判定

Technical：PASS。Engineering acceptance：PASS_WITH_LIMITATION（限定原型可运行）。Parent consistency：PASS。Scientific efficacy：UNVERIFIED。真实实验：NOT_RUN。SS06原型包完成不等于S03/PUB完成。

Git：`codex/pub-01-s03-ss06-continuity`，base main `60c340494bbdb218f62114bab9e4c1ee203dec67`；最终源码与报告同提交版本，提交号以本文件所在提交为准。只新增SS06自有代码/配置/测试/轻量证据及当前导航；SS02/03/05 frozen证据未修改，无新下载、患者资料、远端Git或模型权重。
