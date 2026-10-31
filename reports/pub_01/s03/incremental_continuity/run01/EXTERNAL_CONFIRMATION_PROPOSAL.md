# 外部确认最小方案（提案及后续DTU有界批准记录）

2026-11-01。原数据已有暴露，sub-017至多一人补充个案；外部公开EEG是论文泛化证据的实际缺口。已在2203请求官方JSON元数据、在本地查阅许可/说明网页；未下载数据zip、EEG、音频或权重。官方API摘要保存EXTERNAL_METADATA_ONLY.json。

| 方案 | 最小文件/许可 | 优点与限制 |
|---|---|---|
| A：DTU auditory attention | DATA_preproc.zip 1,838,947,136 bytes；必要时AUDIO.zip 385,086,237 bytes，合计约2.224 GB压缩；CC-BY-NC-4.0 | 官方18人、64通道Biosemi；约50s trial、双人竞争语音及重复单人条件，与主数据域不同，不能用原60s endpoint或跨trial拼接伪造连续历史 |
| B：Natural Speech | 仅Natural Speech.zip，官方页面3.46 GB；CC0-1.0；不取其余四个数据包 | 自然叙述更接近主问题；128→64的物理montage映射、参考方式、同步/音频与包内元数据仍待确认，禁止取前64通道凑形状 |

来源：[DTU官方发布页](https://zenodo.org/records/1199011)、[Dryad官方发布页](https://datadryad.org/dataset/doi:10.5061/dryad.070jc)。DTU许可来自其官方/api/records/1199011的metadata.license.id，Dryad许可来自官方API dataset metadata的CC0字段，页面未显式展示时不自行猜测。DTU预处理包官方说明已包含EEG和audio，因此AUDIO.zip只在预处理包不满足必要声学检查时才需要；优先限额1.839 GB。

建议先核准A为最低存储/通道适配成本的外部补充，B作为自然叙述确认的备选，而非两者自动并行下载。核准内容应明确仅server2203、文件白名单、压缩+解压峰值≤12 GiB、始终保留≥20 GiB空余；若仅可整包且超额，先报告，不突破磁盘预算。未确认解压尺寸，不承诺12 GiB一定足够；可先在服务器读取zip目录元数据后定具体提取范围。

任何外部成绩前须另冻数据适配协议：渠道物理对应与参考、64Hz重采样/包络生成、刺激/EEG事件对齐、有限可读性/尾差规则、记录级reset和不跨trial状态；baseline及增量使用完全相同适配。先用官方元数据定义subject/story分组和dev/test，外部test不用于主候选、适配超参或数据质量排除。若需要外部训练/适配，必须划定外部train/dev/test与train-only统计，给baseline同预算，并与严格零样本结果分开。

A应预设完整trial→subject的相关endpoint及3s窗口辅助指标，不能事后裁成好看的片段；不与主数据60s weighted r混合。B只有连续记录满足时才用60s；不同预处理/语言/刺激域不直接比较跨数据集r绝对值。冻结SS07主候选、全部controls及五seed之后一次执行确认；可读性/适配失败照实报告。新外部协议仍属SS07确认工作，不启动S04颅内支线。

本提案等待协调端在既有授权范围内决定具体数据范围；没有授权前继续现有train/val训练和机制实现，不下载新载荷。

## 后续决定（2026-11-01；取代上文待定状态）

协调端基于用户11/1全量公开EEG研究与投稿证据目标，独立核对官方API/页面/脚本后批准SS07扩展；这是协调研究范围决定，不是用户点名DTU。唯一载荷白名单为DATA_preproc.zip（1,838,947,136 bytes，官方MD5 1c1ca1390ad99ff41c33f36ca67f8418）及preproc_data.m（6252 bytes，官方MD5 7723ddc5334f5aa1f473143251fdcffd），非商业科研、署名/来源保留。AUDIO.zip、EEG.zip、Dryad及新权重均未批准，本轮不下载。

2203 range探测成功，只读取ZIP中央目录：18个S*_data_preproc.mat，解压总1,838,944,108 bytes，archive+extraction约3.43GiB，低于12GiB；磁盘检查同时为原5seed尚未生成latent cache留预算，始终保留≥20GiB。已启动仅白名单下载/官方MD5核对及验证路径留在独立external_dtu/extracted的提取，保留archive/partial/旧资料，无永久删除。schema/header检查排队，尚未作外部模型表现或调整Sparr候选。

官方脚本审阅发现预处理会移除无wavB的单人trial，不能由原实验说明推断预处理包包含单人trial；EXG1/2 mastoid可能仍参与平均参考，必须读实际label选scalp并报告参考，不能仅取前64列；wavA/B是filterbank/abs^.3后的64Hz envelope，不能当raw audio提XLSR。后续正式适配与独立trial→person评估另冻协议，Sparr全部选择完成后一次评估，零样本与外部训练/校准分开。
