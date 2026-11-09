# D15纯存储上限决定

2026-11-02，协调会话01a134e4-2036-7e02-81e9-1289a07606f2已直接核2203 STORAGE_ESTIMATE.json及尚无PROTOCOL/QUEUE_STARTED，明确批准仅D15新ROOT的存储上限由128MiB调整为160MiB=167772160B。

原128MiB估算记录保留不覆盖：实际anchor序列化样本1543726B；每float按32字符展开1767140B，45份79521300B；同18checkpoint12007098B，其他fit输出含余量20490314B、独立review含余量19211736B、源/图/log预留3MiB，保守合计134376176B，略高于原134217728B。预检0NN/fit/optimizer，wall12.236s、CPU43.338s、RSS5027124KiB。原estimate/source与此决定一起绑定新protocol；直接白名单SCP，不在server root另建运输tar。

此次调整不改变9fit/59473248正式残差窗/710100更新、epochs、模型、损失、LR、split、每worker内存/GPU或任何科学预算；不删除/丢失原证据，不重开已完成D14/D12/D13/旧test/DTU/ds004408。独立预声明review最大另631239残差窗、0fit/update，计入存储估计并单列算力成本，由协调结果后执行，不在prepare阶段运行。
