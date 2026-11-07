# D26 最终验收

正式完整矩阵包含 6 个对比预训练、9 个监督头、9 个固定五轮联合微调和 9 个固定 Ccap。三臂均提交原 49 人、752 块和相同选择规则的结果。保存结果的全矩阵核验覆盖 31,584 条块记录、2,058 条逐人记录和 1,176 条逐人配对。九个 HEAD 全部到 250 轮上限，选择 249，末轮 stale=0。原训练与结果范围符合 SS07 的公开 EEG 开发任务。[正式回执](FORMAL_COMPLETED.json) · [保存指标与成本独立复核](COORDINATOR_FINAL_REVIEW.json) · [汇总指纹](report01/REPORT_ARTIFACTS.json)

V02 独立原始路径复算完成九个模型、42 项比较，只覆盖固定首条 VAL 记录的首块。全部 r 和校准 affine/Ccap 符合原 1e-4 绝对容差，最大差分别为 6.4134×10⁻⁵ 和 6.4163×10⁻⁶。九个 raw MSE 全部超出原绝对容差，绝对差约 0.192249 至 3.513021，严格判定为失败。阈值与原模型保持原定义。[V02 终态与逐项失败](coordinator_final_replay_v02/COMPLETED.json)

固定 1024/HH 首块的 CPU 路径对照中，正式 40/64 批的前 40 个输出逐元素相同。独立 CPU 与正式 CPU 的 raw MSE 差约 6.96×10⁻⁷，正式 CPU 与原 CUDA 的 raw MSE 仍差 1.309422。此案例的 CPU 实现与 CPU 批形不足以解释跨端超差，具体 kernel 和 TF32 作用仍未确认。原严格失败继续计入。[CPU 对照](coordinator_cpu_bridge_v01/RESULT.json)

| 独立运行 | backbone 窗 | 最终投影和 head 窗 | 手算 MHA | 库 MHA | CPU 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V01 失败 | 102 | 40 | 31 | 31 | 4.973 |
| V02 九案 | 1,290 | 600 | 465 | 465 | 63.935 |
| 单案例 CPU 对照 | 378 | 144 | 31 | 117 | 17.555 |

三次独立运行均为 CPU，零拟合、零更新。V01/V02 的原生未使用投影分别为 102 与 1,290 窗，CPU 对照为 102 窗。三者分别保留退出、核对与成本回执。[V01 失败](coordinator_replay_v01_failure/FAILURE_1024_HH.json) · [V02 完成回执](coordinator_final_replay_v02/COMPLETED.json) · [CPU 对照完成回执](coordinator_cpu_bridge_v01/RESULT.json)

技术验收 **PASS_WITH_LIMITATIONS**，保留原 raw MSE 严格失败、跨端成因和有限首块支持。固定研究矩阵验收 **PASS_WITH_LIMITATION**，HH 为开发候选，性能提升尚不稳定。父任务一致性 **PASS**。SS07 保持 ACTIVE，D27 按既有协议接续。
