# D18 最终呈现准备（不是完整结果）

当前10/12 fits完成。必须等原queue2050834自然写出`QUEUE_COMPLETED.json`且12个`fits/<arm>_seed<seed>/COMPLETED.json`齐全；实际等待句柄为server2203 canonicalROOT下该文件。最后T/P seed31415未完成时不生成完整矩阵、图和最终报告。训练源/预算/选点不变，不追加NN、验证缓存、旧test/DTU/ds004408读取或训练。

后处理沿用D16 CSV/静态图框架，D18适配独立在`reporting/pub_01_s03_ss07_preprojection_present.py`；不import torch/runtime/model，只允许本ROOT JSON/JSONL/gzip JSON。`--readiness`已真实核验10/12返回WAIT且不创建report01；完整渲染尚未执行。一次等待入口只在自然完成后执行后处理一次，失败保留、无自动重试，绝不启动或更动训练进程。

预设8图：逐epoch VAL与选点；selected四臂及H相对全部对照/同query near-older；固定TRAIN与同epoch VAL r；固定TRAIN objective/windowMSE；幅度/物理MSE/overlap；逐人/逐来源描述；既有首batch Q/K/V梯度与delta活动；分组件/资源/参数成本。每张实际生成后必须目视，并完成中文Results级逐图解释，再归档完整中文报告。不用无误差条的3seed折线冒充显著性，也不另做事后显著性检验。

CSV保留逐seed、逐人、逐来源、逐块及边界标量、240epoch在线轨迹、固定TRAIN60行与相同epoch VAL、缺失selected TRAIN、已有首batch活动、组件阶段和资源。只用OFFLINE_TRAIN -1/0/4/9/19；selected若为1/2/5等无对应检查点，保持缺失，不插值/补跑。online_trajectory不是固定模型性能。

现有协调H/C三seed证据可先确定原全方向科学门已不满足：H-C为+.000593059/+.000567643/-.001172727、均值-.00000400839。仍等待原完整T/P矩阵，不把10fit当完整12fit结果。H近-远均小幅正不证明胜过当前信息或纯延迟因果效应；older替换内容/新近程度但固定slot位置码不变。

最终必须准确交代已见49参与者、混合来源、反复开发VAL；29条来源与TRAIN重叠、20条仅相对当前TRAIN来源未见，不改成主结果或独立test。不以文件标签无交集证明波形独立，不把seed/重复参与者当额外独立样本。

成本保留完整12fits与2个中断attempt、H18400–18500/C23000更新界、28次丢弃工程更新；未知CPU/精确H终点为null/范围。queue总wall已包含原尝试/停机/等GPU，不再相加；fitCPU包含setup/train/eval，不加其嵌套子计时。review receipt逐文件列出，不无依据求和嵌套时钟。组件窗不是唯一数据或FLOPs；H冻结E0两窗、C一次复用、T无memory、P多640参数/既有prior上游共享成本，不能声称公平同算力或零上游成本。

保留冻结历史643174935B估计，按真实checkpoint zip-name/header复核后719087919B保守投影单列，768MiB硬cap不变。不得改旧协议掩盖估计修正。协调已有选点/指标/检查点/配对审阅只引用、不重复执行；剩余第三seed完整检查点/最终独立审阅由协调完成。本轮即使收束也不等于总体goal实现，不自动开新结构/阶段或合并PR6。
