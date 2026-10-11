# 本轮实际命令与运行版本

以下命令均在server2203上执行，Python固定为已存在环境；不是Windows本地训练。输入代码与config由项目轻量文件复制，不含数据载荷。原输出已存在，脚本拒绝覆盖；复现时使用新的输出目录。

```bash
ROOT=/home/fanyu/auditory_simulation_m6a
PY=/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python
IN=$ROOT/pub_01/s03_decode_inputs_20261011

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$PY" "$IN/pub_01_s03_envelope_decode.py" --self-test

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$PY" "$IN/pub_01_s03_envelope_decode.py" \
  --config "$IN/pub_01_s03_envelope_decode_20261011.json" \
  --out "$ROOT/pub_01/s03_envelope_decode_20261011/run01" \
  > "$IN/run01.log" 2>&1

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$PY" "$IN/pub_01_s03_envelope_decode.py" \
  --config "$IN/pub_01_s03_envelope_decode_20261011.json" --subject sub-001 \
  --out "$ROOT/pub_01/s03_envelope_decode_20261011/reproduce_sub001" \
  > "$IN/reproduce_sub001.log" 2>&1

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$PY" "$IN/pub_01_s03_review_decode.py" \
  --run "$ROOT/pub_01/s03_envelope_decode_20261011/run01" \
  --rerun "$ROOT/pub_01/s03_envelope_decode_20261011/reproduce_sub001"
```

完成时间：主运行UTC 2026-10-10 16:57:35（北京时间2026-10-11 00:57:35）；耗时58.114s。配置与代码拟合前冻结，所有93条record使用原始发布方共同区间；每条统一修剪13s+0.5s边界并应用共同有限掩码。无新增下载、PCA或模型layer选择。

运行时Python 3.11.15，NumPy 2.4.6，SciPy 1.17.0，Matplotlib 3.11.1。没有深度模型训练、模型权重下载或GPU设置。
