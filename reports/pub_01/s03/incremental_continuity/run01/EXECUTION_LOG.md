# SS07 实际执行记录

2026-11-01接管协调批准任务书、持续goal ACTIVE。工作树起点bddc8fa；唯一新增外来改动为SS07任务书，现已由本执行者接管。父amendment/唯一SS07登记/S03/current入口更新；原SS02/06、主目录CNN/TRF/stash只读。

实际2203 intake：426train/49val/69seen/75heldout；原426train统计不混SS06；原lag0。native loader每epoch820train/98val batch，完整128负例；原随机同story distinct-time语义，epoch0实际104960窗呈现、72852 unique、32108重复、53489 eligible未访，不假称exact-sweep。head252454train/30500val重叠窗，目标缓存独立于旧encoder，顺序/offset核验后复用；旧latent/cp不复用。

真实batch128 native与明确对称loss差0、3,800,713参数全gradient finite；原FP32/matmulTF32false/cuDNNtrue，真实step.306s，峰5.48GB。工程权重丢弃。BASELINE_PROTOCOL冻结原预训练100/pat15/AdamW.001wd.01/no clip/StepLR10*.1、head250/pat10/Adam.001/no wd/clip/shuffle，原两main/自动test禁用。只train/val载荷，XLSR亦phase过滤。

三seed已正式更新：driver1024 PID4114164(cuda0)、2026 PID4114165(cuda1)、31415 PID4114166(cuda0)，NN子4114167/68/69；日志baseline_seed*.log/status/history持续记录。按原法820updates/epoch/max82000，head3945/epoch/max986250。允许原validation早停，不硬编码旧24epochs。协调实际SSH读protocol/check/status/GPU确认正式启动且通过协议核对。

来源元数据修正：冻结native cfg模板仍seed1024，实际RNG/外层checkpoint seed是各seed；未改活动源/重跑，新增每seed RESOLVED_CONFIG sidecar和BASELINE_METADATA_CLARIFICATION明确reference vsactual，不改变科学结果。

协调在test前预先固定最终5seed=[1024,2026,31415,42,12345]、前三筛选；新增42/12345原full baseline已按独立SEED_EXTENSION_01冻结/driver启动，预算/源/旧进程不改。所有成本单列，不能择seed。5seed latent预计8.69GB，原prepared/XLSR不复制，cachecap25GiB且留20GiB，best/latest有界，历史不删；当前初始余58.6GB。

已保存独立SCIENTIFIC_DESIGN_REVIEW；增量精确矩阵/三seed筛选另审阅并行开发。尚无SS07 test评分或性能增益/投稿完成声明，goal不因baseline工程通过完成。
