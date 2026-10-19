# SS05 Codex实际执行日志

2026-10-19，Issue#5/PR#4。独立worktree C:/Users/fanyu/.codex/worktrees/pub-01-s03-ss05/Auditory_Simulation，分支work/pub-01-s03-ss05-predictive-memory，起点5724a4ac690b67fa3def4f60e74e827c67a82b63；主目录main/CNN/TRF/stash未动。

用户最新授权已允许审查合并PR#3，远程main merge为8d21d0a0bc3b22ec9fe2d27c7bd5522beb379c6f；PR#4改base main且保持Draft。旧“#3保持Draft”只作历史，不阻断本次，也不改SS03科学结果。不自行合并#4。

## 已实际完成

1. 只读审查完整任务书/NEXT协议/Issue/原型及报告。2203 hostname nercn，2×L40；env01 PyTorch2.11.0+cu128、NumPy1.26.4，精确环境在FORMAL_PROTOCOL。无新下载、无server Git。
2. intake01.log/started.json记录PID543180。旧train426条按真实source-family一次冻结353/26/47记录。原10tests与cue160/key160/GRU600重跑；额外有标签事件/延长gap probe、旧新数值对照已完成，不把toy充作EEG。
3. 只在新train重拟合归一化；trainval_data01.log实际缓存379记录4735段，未读新holdout。新wrapper修正local带前两窗问题，新增5tests和真实CPU/GPU smoke通过。逐标量梯度资源审计区分实例化、tensor reachability和实际活动经验下界。
4. 五组×四lag×100步、seed1024、batch2的真实train/val pilot完成，仅validation选共同lag8。正式从头，不复用pilot权重。
5. 任何新holdout payload前write-once冻结FORMAL_PROTOCOL/source/config身份/split/norm/环境/六primary contrasts。五组×三seed、2000步、每200步完整val、AdamW.001/wd.01/clip1、batch2、无early-stop，由两GPU worker执行。全部15训练完先核对预算/val selection，再构建holdoutcache并每selected模型一次评分全部7条件。

## 实际入口（2203）

```bash
TASKROOT=/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_memory_20261019/codex_run01
TASKPY=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python
cd "$TASKROOT/input"
"$TASKPY" pub_01_s03_ss05_intake.py
"$TASKPY" pub_01_s03_ss05_data.py --stage trainval
"$TASKPY" -m unittest discover -s tests -p test_pub_01_s03_ss05_finite_history.py -v
"$TASKPY" pub_01_s03_ss05_pilot.py
"$TASKPY" pub_01_s03_ss05_synthetic_review.py
"$TASKPY" pub_01_s03_ss05_resource_audit.py
"$TASKPY" pub_01_s03_ss05_freeze.py
"$TASKPY" pub_01_s03_ss05_run.py
```

各阶段nohup/独立log与started/status JSON持久运行，cache/协议/run目录默认exist即拒绝；不覆盖SS02/SS03。SHA256仅本任务明确要求的轻量source/config/契约身份，不做全量raw checksum审计。evidence/intake保存实测轻量复核资料。

目前无工程失败重启。A-assoc合成MSE差异、未显式训练forecast误差等局限已披露。当前继续正式结果/review，不能因准备完成停止；完成SS05后停，不启动SS04/S04/父S05。
