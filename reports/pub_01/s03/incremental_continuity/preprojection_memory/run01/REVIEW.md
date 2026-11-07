# D18 复核与验收记录

2026-11-06。执行者目视 8 张实际 PNG，核对图轴、对照和相应保存数值。协调独立完成训练与评价证据的核验。报告全文和归档交付以另附协调审阅回执为准。［[作者图像核对](AUTHOR_FIGURE_INSPECTION.json) · [完整报告](REPORT.md)］

| 验收维度 | 结论 | 依据与范围 |
|---|---|---|
| 技术 | 通过，保留核验范围 | 正式 12 组均跑满原 20 轮。恢复协议、预算、选点、60 个固定 TRAIN 和 24 份 best/last 核验通过。 |
| 科学 | 预设历史收益门未满足 | H−C 两正一负，H−T 在 31415 为负，H−P 三组均负。near−older 三组均正，评价范围为重复开发 VAL。 |
| 父任务与数据边界 | 通过 | 本轮沿用公开项目 TRAIN/VAL 边界、冻结协议和既定 49 人主指标。整体科研目标尚未达成。 |

协调对 H 三组分别核算保存的块与逐人指标，对其余九组核对成绩、预算和选点。H/C 的三种子对照额外核对监督顺序、旧锚点及评分坐标。第三种子四臂矩阵复核了每臂的组件公式。检查点审阅覆盖每种子各 8 份、共 24 份真实 best/last，范围为序列化权重完整性及其元数据。保存指标复算的输入为现有标量，独立原始预测重放未开展。［[H/C 核验](COORDINATOR_H_C_THREE_SEED_AUDIT.json) · [第三种子矩阵](COORDINATOR_SEED_MATRIX_AUDIT_seed31415.json) · [检查点 1024](COORDINATOR_CHECKPOINT_AUDIT_seed1024.json) · [检查点 2026](COORDINATOR_CHECKPOINT_AUDIT_seed2026.json) · [检查点 31415](COORDINATOR_CHECKPOINT_AUDIT_seed31415.json)］

固定 TRAIN 审阅覆盖三个种子的 60 个实际检查点及其同轮 VAL 记录。评分时的旧锚点、回退前缀和边界标量保留。八个 selected TRAIN 缺口由固定检查点规则决定，CSV 明列缺失。末轮、选点、在线轨迹和固定模型成绩分别保留。［[固定 TRAIN 1024](COORDINATOR_OFFLINE_TRAIN_AUDIT_seed1024.json) · [固定 TRAIN 2026](COORDINATOR_OFFLINE_TRAIN_AUDIT_seed2026.json) · [缺口](report01/SELECTED_TRAIN_GAPS.csv) · [逐窗边界汇总](report01/BOUNDARY_SELECTED.csv)］

## 呈现后的四项协调审阅

原呈现快照保留其当时的文件和计时。下表补列快照后完成的四项实际审阅，秒数直接取自各自回执。四项新增神经前向、反向和优化器更新均为 0。CPU 权重审阅含导入、序列化读取及校验。单独披露顶层计时，避免同已存子计时或其他范围相叠加。［[原清单快照](report01/REVIEW_COST_INVENTORY.csv)］

| 补充审阅 | CPU 秒 | 壁钟秒 | 实测范围 | 来源 |
|---|---:|---:|---|---|
| P31415 指标、预算和选点 | 0.018285712 | 0.018351078 | 960 个标量比较，最大绝对误差 2.39×10⁻¹⁶ | [回执](COORDINATOR_COMPLETED_FIT_AUDIT_P_prediction_history_seed31415.json) |
| P31415 固定 TRAIN | 0.420173662 | 0.421435833 | 32,170 个标量比较，最大绝对误差 5.55×10⁻¹⁶ | [回执](COORDINATOR_OFFLINE_TRAIN_AUDIT_P_prediction_history_seed31415.json) |
| 31415 四臂矩阵 | 0.019116105 | 0.019192696 | 同协议、监督顺序、锚点及组件公式通过 | [回执](COORDINATOR_SEED_MATRIX_AUDIT_seed31415.json) |
| 31415 的 8 个 best/last | 1.500269239 | 1.609779596 | CPU 权重文件、头信息、参数、有限性和相等性核验通过 | [回执](COORDINATOR_CHECKPOINT_AUDIT_seed31415.json) |

## 保留项与交付边界

两项正式中断和全部工程尝试按真实成本保留。完整队列和呈现均已自然结束。报告保留未知的 H 终止点和中断 CPU/峰值，以及修正后的存储预测与原协议估计。保存报告的科学结论限于此候选四臂、固定预算和已见参与者的混合来源 VAL。历史替换的响应涉及内容与新近程度，无法单独识别纯时延。后续科研决策仍归父任务。［[中断及恢复](INTERRUPTED_FORMAL_COSTS.json) · [成本账本](report01/COST_LEDGER.json) · [划分解释](COORDINATOR_SPLIT_SCOPE.md)］
