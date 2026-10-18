# SS03实际执行入口（server2203）

Windows独立managed worktree：`C:/Users/fanyu/.codex/worktrees/pub-01-s03-ss03/Auditory_Simulation`，分支`work/pub-01-s03-ss03-multiscale`，PR#3 Draft。原主目录main及用户CNN/TRF未提交修改保持原样；未动stash。服务器不初始化Git。

```bash
TASKROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019
TASKPY=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python
cd "$TASKROOT/input"
# write-once缓存/证据；同名目录存在即拒绝，不覆盖SS02。
"$TASKPY" pub_01_s03_ss03_bootstrap.py --config pub_01_s03_ss03_multiscale_20261019.json
"$TASKPY" pub_01_s03_ss03_freeze.py --config pub_01_s03_ss03_multiscale_formal_20261019.json
"$TASKPY" pub_01_s03_ss03_constant_control.py
"$TASKPY" pub_01_s03_ss03_run.py --config pub_01_s03_ss03_multiscale_formal_20261019.json
# 所有真实结果完成后才运行的独立审阅，不再拟合模型。
"$TASKPY" pub_01_s03_ss03_review.py
```

实际bootstrap和formal runner使用nohup、独立log和started/status JSON持久运行。四组分别3seed；两GPU worker独立顺序执行各自任务，所有train完成后统一核查，再开启test worker。分别保存`train_<variant>_seed<seed>.log`、`test_...log`；各run为`formal01/<variant>_seed<seed>/`，含history、best_validation.pt、train_completed.json与test01。所有权重/arrays/per-window预测留服务器；本地仅显式白名单结构化统计、代码/config和PNG。

bootstrap来源起点PR head `9ffef2656d06e648073d5a7fa59e78ba03d5a648`；新增执行源码的精确SHA256在formal_protocol，不能把起点commit误当包含新增源码。冻结后所有train/test入口核对这些源码/config身份。PR保持Draft，不合并main，不启动SS04。
