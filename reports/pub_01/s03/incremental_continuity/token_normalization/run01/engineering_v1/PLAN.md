# D16：单一逐token无仿射归一化对照

2026-11-02；PUB-01-S03_SS07，D15已封存1bde328。具体授权/假说/禁止项见 [协调决定](COORDINATOR_DECISION.md)。原完整FT+Ccap、H/C/P、三seed、float64 centered negative cosine、20epoch、Adam .001/wd0、60排列、global64/尾38、全部fallback/目标/支持/VAL选点保持。**唯一网络改变**为token末端GELU后、两步GRU前：

```python
tokens = F.layer_norm(tokens, (32,), weight=None, bias=None, eps=1e-5)
```

每B×2×32的32个特征独立，用population correction0方差及sqrt(var+eps)；不跨batch/时刻/record/split、不存running统计、不学习仿射参数。初始化state_dict与D15所有9组合逐tensor相同，H/C/P参数仍165792/165792/164784，正式干净初始化、不继承旧late权重。新增算术与runtime明确记录，0新参数不等于FLOPs相同；它改变表示的尺度/平移信息及函数约束，不包装成纯优化器干预或新方法创新。其定义与[PyTorch2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LayerNorm.html)核对，原论文[Ba et al.](https://arxiv.org/abs/1607.06450)只作既有方法来源，不声称复现其整个归一化RNN。

## 精确预算与观察

正式九fit各78900更新，总710100；残差NN正式59473248窗不变：TRAIN更新44751600、每轮VAL5410620、四非零离线TRAIN8950320、selected VAL270531、H selected older90177。新encoder/head0，不重新跑D15参照。结果后协调独立复算另至多631239窗，需独立实现该新归一化，不能直接沿用D15未归一化forward。

每epoch原首shuffle batch已有forward/backward只增detached标量：token前后RMS/max、各token均值/方差摘要、GRU两步状态|h|≥1−1e−6比例和max、原五模块梯度nonzero/L2。没有另一次forward/step、没有图或数组留存。epoch0零输出头上游梯度0为预期；首批输入随固定shuffle变，不能声称固定输入轨迹/全数据状态，也不参与选点/earlystop/调参。

同D15五个离线TRAIN检查点−1/0/4/9/19，同一次预测记录负cosine/MSE/物理r/MSE/幅度/overlap/常量与eps计数，−1直接锚点0NN。VAL原752块/49人、TRAIN6104块/70人和原目标精度不改；selected包含−1/最早tie；同一H selected older一次。先分别回答通路退化与任务收益，H必须与同归一化C/P及强Ccap比较。梯度更活跃但无任务收益应明确阴性，已有归一化算子或开发收益不自动满足科研创新/独立确认目标。

## 必要工程，≤4096NN/0fit/optimizer

不重跑未改loader、原严格过去、完整排列与目标QC；继承D15实际冻结身份/支持，核源/输入stat和SHA并绑定D15已验收结果。新算子CPU/GPU与独立NumPy correction0公式一致、轴独立、常量输入和反传finite、CPU/CUDA RNG不变；九初始化参数/state_dict精确一致。真实metadata固定TRAIN64窗中48有效：fresh零头三臂144NN；固定非零output探针三臂144NN验证输入/GRU梯度；诊断开/关输出等价再48NN。旧D15 late H/C各native/同权归一化一次192NN，只用于有限的分布改变/梯度状态诊断，不当作已训练归一化候选，不据loss选变体。预计工程528NN，实际计账，上限4096不扩。所有probe权重丢弃于内存，不进入正式初始化，0optimizer/学习fit；原常量/近零目标语义继承D15不删窗。

## 路径、资源与验收

server唯一新ROOT `/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_token_normalization_20261102`；本地 `reports/pub_01/s03/incremental_continuity/token_normalization/run01`。入口四源 `pub_01_s03_ss07_token_norm_{core,prepare,train,queue}.py`，所有数据/模型/科学计算仅2203，源/指标/静态图回本地，权重/原信号/特征张量不回传。

完全沿用已实测5+4九进程调度/身份保护，torch2/BLAS4，每launch freeGPU≥4GiB、MemAvailable≥16GiB且该卡compute PID仅本轮已核UID/start_ticks/argv的worker；有陌生PID等可用slot，不终止他人、不改MPS。6h总queue、每worker RSS8GiB/GPU allocated2GiB、两盘各20GiB、root160MiB且观察值+28MiB在途保留仍低于门；消失临时路径容错、其他OSError传播。新字段仅少量标量，D15实测root约116.6MB，按D15存储保守界沿用并记录本轮实际大小，无运输tar/新数组。失败仅有界停止本轮已核身份worker，不自动放宽预算/重试。

源/预算/预设观察先冻结，实际工程通过后交协调一次独立审阅，绑定最终protocol与source才正式九fit；无需用户重复许可。Technical核输出/计数/数值与独立复算，Acceptance分别判断通路条件改善和历史任务收益，Parent consistency保持PUB范围/原split。D12/D13/D14/D15/D9、旧test/DTU/ds004408冻结，无患者/新数据/新权重/损失或LR/归一化位置网格、无服务器Git或PR6合并。
