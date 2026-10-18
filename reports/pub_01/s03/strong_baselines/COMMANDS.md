# SS02 本次实际入口与重跑约束

最新追加执行见本文件末节：2026-10-15已实际训练/测试HDPS，并执行Transformer训练；下方run01/run02及10-13准备为历史命令。所有原输出目录保留，不直接重跑到已有目录。

下面是实际执行的命令结构。`input_run01/run01`网络失败、`input_run02/run02`成功完成preflight；它们都没有执行作者模型。使用新run目录重跑，原目录exist即拒绝。Source源码为固定commit文本阅读；不复制Git工作区到2203。

## Windows：只读接入与轻量源码

```powershell
git show main:doc/CODEX_START_HERE.md
git status --short
git branch --show-current
git log -1 --format='%H %s'
ssh -o ClearAllForwardings=yes -o BatchMode=yes server2203 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'
```

run01输入为初始config（run_id/output=run01，无source_staging），现已保存在`evidence/run01/config_snapshot.json`及服务器`input_run01/`。首次执行：

```bash
HTTPS_PROXY=http://127.0.0.1:62347 HTTP_PROXY=http://127.0.0.1:62347 \
 /home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python \
 /home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run01/pub_01_s03_strong_preflight.py \
 --config /home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run01/pub_01_s03_strong_baselines_20261013.json
```

run01代理关闭连接；随后直接HTTPS探针超时。保留它而另建run02。Windows按config指定repo/commit/files，分别调用官方GitHub tree API和raw.githubusercontent.com；将tree JSON和37个源代码/文本写入`reports/pub_01/s03/strong_baselines/source_inspection_only`。可重跑等价入口：

```powershell
# 用一个尚不存在的destination，避免覆盖静态镜像。
./scripts/pub_01_s03_stage_source.ps1 -Destination reports/pub_01/s03/strong_baselines/source_inspection_only_new_run
```

本次复制的路径：

```powershell
scp -o ClearAllForwardings=yes scripts/pub_01_s03_strong_preflight.py server2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run02/scripts/
scp -o ClearAllForwardings=yes tests/test_pub_01_s03_strong_preflight.py server2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run02/tests/
scp -o ClearAllForwardings=yes configs/pub_01_s03_strong_baselines_20261013.json server2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run02/
scp -r -o ClearAllForwardings=yes reports/pub_01/s03/strong_baselines/source_inspection_only server2203:/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run02/source
```

## 2203：技术tests与真实数据preflight

```bash
cd /home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/input_run02
PY=/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python
$PY -m unittest discover -s tests -p test_pub_01_s03_strong_preflight.py -v > technical_tests.log 2>&1
$PY scripts/pub_01_s03_strong_preflight.py --config pub_01_s03_strong_baselines_20261013.json > preflight.log 2>&1
```

实际tests之后才执行preflight；4项tests通过。测试fixture留在2203 `/tmp/strong_preflight_fixture_*`，没有永久删除。输出只含路径、shape、finite flag、软件版本和协议覆盖 metadata，绝不写出signal values。

## 单独review pass

把4个mandatory报告、RESULTS及轻量evidence复制到服务器`review01/`，然后执行：

```bash
BASE=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013
/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python \
 "$BASE/input_run02/scripts/pub_01_s03_review_strong.py" \
 --report "$BASE/review01" --server-base "$BASE"
```

该检查只验证HOLD包的实际证据、空指标和边界，不能证明模型性能。`review_checks.json`、`SERVER_MANIFEST.csv`为本次实际运行产物。恢复未来模型执行需要先满足REPRO_REPORT中的条件；上述审计代码没有隐藏的train开关。

## 追加实际执行（2203）

```bash
TASKBASE=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/continuation01
TASKPY=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python

# 首条1024Hz BDF→100Hz，版本v6增加只读坐标记录。
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 "$TASKPY" "$TASKBASE/pub_01_s03_hdps_100hz_v6.py" --base "$TASKBASE" --mode smoke --run smoke06

# 作者split81人原raw；8个独立进程，全部已结束，不是agent委派。
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_raw_batch.py" --base "$TASKBASE" --workers 8

# 先smoke，再70story原Mel参数。各目录独立，不能重复原命令覆盖。
OPENBLAS_NUM_THREADS=4 "$TASKPY" "$TASKBASE/pub_01_s03_hdps_mel_v2.py" --config "$TASKBASE/pub_01_s03_hdps_paper_m_20261013.json" --smoke
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_mel_v2.py" --config "$TASKBASE/pub_01_s03_hdps_paper_m_20261013.json"

# 真实输入工程forward/backward，非科研训练或性能评价。
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_model_smoke.py" --config "$TASKBASE/pub_01_s03_hdps_paper_m_20261013.json"
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_engineering_review_v2.py" --base "$TASKBASE"

# 缺APR的public sub022run09：已证明数值流程不消费snr，显式容器适配。
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 "$TASKPY" "$TASKBASE/pub_01_s03_hdps_100hz_v7.py" --base "$TASKBASE" --mode full --run author02_sub-022 --subject sub-022
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_restricted_mapping.py" --base "$TASKBASE"
"$TASKPY" "$TASKBASE/pub_01_s03_hdps_continuation_summary_v2.py" --base "$TASKBASE"
```

smoke01–04失败（NumPy2/brian2、动态class pickle、runner pair顺序、save helper注册）与05/06成功均在2203保留版本/日志。默认torchaudio的CUDA13不兼容失败保留，cu128 wheel装至独立`runtime_torchaudio_cu128`target；未卸载或改原环境。新增代码AST和JSON核验在2203完成。

已执行`scripts/pub_01_s03_download_xlsr.sh`，1.2GB原XLSR Dutch权重仅2203；这是明确Apache2的声音模型。未下载GroNLP未明条款权重、EEGcheckpoint或受限dataset。

追加验证已实际执行：`$TASKPY "$TASKBASE/test_pub_01_s03_paper_guard.py"`（1项PASS）及`$TASKPY -m compileall -q "$TASKBASE"/pub_01_s03_*.py`（PASS）；仅JSON guard fixture保留2203，不读取signal、不创建科研train目录。

`pub_01_s03_hdps_prepare_paper.py`、`pub_01_s03_hdps_train_paper.py`已保存，**NOT_RUN**；后者拒绝有inputfailures的cohort。内部64Hz配置/prepare入口也NOT_RUN，不是本轮交付路线。恢复原条件需先满足RESOURCE_RECOVERY，使用新run及新冻结配置，避免覆盖现有证据。没有后台作业或自动下一节点。

## 2026-10-15授权可用子集：实际执行入口

前节NOT_RUN/等待是10-13状态。10-15按输入可用性冻结cohort后执行，未改作者每人split/有效索引；严格完整模式仍保留。下列为已用入口，不能复制到同名已有输出覆盖；重现须改新运行根并先固定配置，不得依测试改参。

```bash
TASKPY=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python
HDPS=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/hdps_m_available01
TRANS=/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01

"$TASKPY" "$HDPS/input/pub_01_s03_hdps_prepare_paper.py" --config "$HDPS/input/pub_01_s03_hdps_m_available_20261015.json"
"$TASKPY" "$HDPS/input/pub_01_s03_launch_train.py" --config "$HDPS/input/pub_01_s03_hdps_m_available_20261015.json" --script "$HDPS/input/pub_01_s03_hdps_train_paper_v2.py"
"$TASKPY" "$HDPS/input/pub_01_s03_review_hdps_result.py" --config "$HDPS/input/pub_01_s03_hdps_m_available_20261015.json"
"$TASKPY" "$HDPS/input/pub_01_s03_hdps_candidate_trace.py" --config "$HDPS/input/pub_01_s03_hdps_m_available_20261015.json"
"$TASKPY" "$HDPS/input/pub_01_s03_hdps_descriptive_errors.py"

# 既有READY 64Hz EEG + 作者fold4；XLSR Dutch19原声音模型，所有特征留2203。
"$TASKPY" "$TRANS/input/pub_01_s03_transformer_inputs.py" --config "$TRANS/input/pub_01_s03_transformer_available_20261015.json" --stage freeze
# features/prepare成功版本为attempt03；日志与失败版本保留。
"$TASKPY" "$TRANS/input/pub_01_s03_transformer_job.py" --config "$TRANS/input/pub_01_s03_transformer_available_20261015.json" --start features --attempt 03
# 首次train受DataParallel/LSTM InferenceMode工程错误中断，未评分test；修复no_grad后相同seed从头训练。
"$TASKPY" "$TRANS/input/pub_01_s03_transformer_job.py" --config "$TRANS/input/pub_01_s03_transformer_available_20261015.json" --start train --attempt 04 --train-script pub_01_s03_transformer_train_v2.py --run-name training_seed1024_attempt04
# 顺序完成当前训练后review→原Conv32包络readout→独立CPU review；不改backbone或测试规则。
"$TASKPY" "$TRANS/input/pub_01_s03_transformer_finalize.py"
```

最终器记录`finalize_status.json`，分阶段日志为`contrastive_review.log`、`envelope_readout.log`、`envelope_review.log`；配置与训练源码保存于各run根input。Windows仅scp代码/config到2203，回传白名单仅轻量JSON/逐被试CSV/静态PNG；不回传npy/pt/音频/信号。
