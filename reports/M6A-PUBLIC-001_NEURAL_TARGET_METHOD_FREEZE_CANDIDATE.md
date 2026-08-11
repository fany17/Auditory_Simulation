# M6A-PUBLIC-001 neural target method freeze candidate 阶段报告

| 字段 | 内容 |
|---|---|
| 状态 | `METHOD_FREEZE_CANDIDATE_AWAITING_COORDINATOR_REVIEW` |
| Neural extraction | `NOT_RUN / NOT_ALLOWED` |
| Model download | `NOT_RUN / NOT_ALLOWED` |
| Baseline | `NOT_RUN / NOT_ALLOWED` |
| G2 | `PENDING_3_EDF_DOWNLOADS` |

## 完成内容

- 预声明六个等宽子带主目标，并将 110-130 Hz harmonic guard 排除；
- 将原 70-150 Hz 方案降为不可替代主目标的 sensitivity；
- 冻结 `AS_RECORDED_SCALP_REFERENCE` 及禁止 reference 猜测/contact-name bipolar 的边界；
- 冻结 512/1024 Hz 有限 Kaiser FIR、square + finite low-pass power、overlap-add backend、50 Hz 时间网格、train-only 变换、lag 方向与 support mask；
- schema 使用 `additionalProperties=false` 和 `prefixItems/const` 固定六子带与两 sampling profiles；
- 添加 semantic/config gate 与 fail-closed synthetic tests；
- G2 metadata audit 增加逐 recording `iEEGReference` 机器记录和一致性判定。

## 运行证据

2203 项目专用环境 `auditory_m6a_public_001`：

```text
66 passed, 76 subtests passed in 1.36s
All checks passed!
Success: no issues found in 22 source files
neural target method gate: PASS
main config gate: PASS
formal src direct convolution scan: PASS
```

synthetic tests 覆盖 512/1024 Hz 的 FIR 对称性与实际 taps、overlap-add/direct 一致性、frequency response、impulse support、step、60/120 Hz 与 95 Hz passband sine、短输入、NaN/±Infinity、退化 transform、train-only 拟合和 support mask 边界。G2 metadata v4 另证实 11/11 `iEEGReference` 逐 recording 一致性 gate 为 PASS；该报告整体仍因 3 个 EDF 未落盘而 FAIL。

## 科研可声称 / 不可声称

可声称：method candidate 的结构、数值参数和 synthetic 验收阈值已预声明，且真实执行仍被 gate 阻止。

不可声称：candidate 已接受或 frozen；真实神经目标已提取；G2 已 PASS；final embargo 已完成；任何 Audio↔Brain baseline 或模型层—脑区结论已产生。

## 失败、阻塞与下一节点

- 旧 A/B 待选择 proposal 已被协调否决并保留历史；
- 首轮 mypy 因缺少 SciPy 类型声明及测试类属性注解失败；已在本项目专用环境增加 `scipy-stubs 1.17.1.5` 并修正注解，复跑通过；
- 一次 code snapshot 同步把轻量文件平铺到远端根并短暂覆盖 snapshot README；正式 README 已恢复，多余文件已移动到 `/home/fanyu/auditory_simulation_m6a/log/interrupted/sync_layout_20260811_2058/`，数据目录与下载进程未受影响；
- 当前等待协调二审；
- G2 仍等待 3 个 EDF 下载完成与 full audit；
- final embargo 仍等待声音模型 context 的未来测量和 split guard 重跑；
- 下一节点不是 G3，而是协调复核 method candidate，同时继续 G2 下载。
