# 仅新推理缓存的物理位置修订：两盘各保留20GiB

2026-11-02，所有真实新数据NN/模型分数/学习模型拟合仍0。初始完整存储预算在获取后因实际共享盘余量变化失去约0.28GB裕量，不能只检查当前free后开始写全部三seed缓存。

最终方案由COORDINATOR_CACHE_FILESYSTEM_DECISION.json明确批准：现有ROOT保持 `/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102`，仅尚未生成的三seed Z/P缓存直接写 `/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102/caches`。两者仍在server2203，不在本地存数据；没有现有资产移动、删除或symlink，没有第二Git仓库。

只读检查确认这是不同filesystem；/sata5共享根的权限未改。新建各级目录均UID1012自有、0700，逐级拒绝symlink和异主路径。CACHE_STORAGE.json记录实际device、路径、权限来源及创建前后空间。初始cache盘free179592044544B，三份FP32缓存完整942016512B后仍远高于20GiB；home当前free减去剩余全部processed EEG和128MiB报告余量后亦高于20GiB。最终输入完成、每次NN启动前再核完整未来投影，运行中继续检查。

预算必须跨两root合计：本run home+cache全部实际文件bytes<=3GiB；既有SS07 NPY+本run home NPY+cache NPY合计<=28GiB。两个filesystem各自reserve>=20GiB，不因另设物理目录或改扩展名逃cap。预测cache仍是原FP32 Z(N,8,192)/P(N,192)，无压缩/降精度/新模型/新适配/新评分规则；来源和CACHE_PATHS在每seed结果目录明确记录。

此前19GiB备选只生成了projection JSON，没有修改/执行任何19GiB源码、队列、NN或阈值。该提案作为未采用工程历史保留，RESOURCE_OPTIONS_RESOLUTION明确最终唯一方案。当前EEG预处理PID1395718及其冻结source不动，仅在核清旧waiting queue UID/argv/start_ticks/无children/无NN后接续新的evaluate_v3和queue_v2；原queue源、日志、launch保留。

最终唯一EVALUATION_PROTOCOL将pin本修订、CACHE_STORAGE、协调决定、实际数据/模型/mask/统计和新源码。正式NN仍等待协调绑定该protocol及新queue SHA的一次独立审阅。
