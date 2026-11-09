# DTU 外部公开EEG：下载与实际内容核验

2026-11-01。2203已完成获批的DATA_preproc.zip与preproc_data.m下载、官方MD5核对、安全路径提取及全部18人的可读性/通道/事件检查。只下载这两个文件，zip及既有资料保留；本地仅保存本报告和轻量JSON，没有EEG、包络或MAT载荷。

实际内容为18人×每人60个trial，共1080个trial。每个trial均为3200点、64Hz，即50秒；EEG为3200×66，wavA/wavB各3200点。1080条的EEG及两包络全部有限、包络非恒定，长度一一相同，18个MAT均可读取。这里的PASS是可读性和坐标事实，尚无模型表现、参数拟合或正式外部验收。

| 项目 | 实际核验结果 |
|---|---|
| 物理通道 | 64个Biosemi scalp名称逐一匹配既有SparrKULee provenance顺序，另有EXG1/EXG2；各trial一致 |
| 选通道规则 | 按物理名称查找并重排64 scalp；检查后索引恰为0..63，不能把“取前64”当未经核验的普遍规则 |
| 参考 | 官方脚本和包内cfg均记录最终all-channel average，66列中含两mastoid；若采用scalp64 CAR，须另冻同适配规则 |
| 目标 | 官方脚本按attended speaker指定wavA、另一speaker指定wavB；两者是filterbank→abs^.3→平均频带→64Hz envelope，不是raw audio |
| 事件 | 每trial本地sample=1（MATLAB 1-based），value为1/2 attended gender code；不能用本地起点证明物理硬件同步 |
| 单人条件 | 官方处理删除无wavB的single-talker trial，本包不按原实验介绍泛称含单人数据 |
| 空间 | zip约1.839GB、提取约1.839GB，合计3.43GiB；保留≥20GiB及原五seed待生成cache预算，未回传载荷 |

还有两项实际缺失信息，先保留，不用模型成绩解决：MAT中未显式存储EEG物理单位；最终结构只含data，没有完整expinfo，cfg目前只看到首trial音频路径，尚不能保证每trial的story/room元数据可恢复。已有同长度和事件标签不等于全部刺激分组/物理时延已验证。未新增下载AUDIO.zip/EEG.zip来填补。

下一步是冻结可审阅的外部适配、刺激/被试分组和trial→person endpoint。trial之间必须reset，不能拼接成连续长记录或60秒段。Sparr主候选、controls、五seed、EMA全部锁定前不作外部模型评估；任何外部新训练/校准需独立方案，baseline同适配/预算，不能混充无目标域拟合的零样本结果。

许可为CC-BY-NC-4.0，仅当前非商业公开研究，保留发布方归属和来源。官方来源：[Zenodo1199011 v1](https://zenodo.org/records/1199011)。协调端基于用户11/1全量公开EEG研究目标批准此有界扩展，不表述为用户点名DTU。

证据：DTU_DOWNLOAD_EXTRACT_COMPLETED.json、DTU_SCHEMA_HEADER_AUDIT.json、DTU_READABILITY_CHANNEL_EVENT_AUDIT.json、ZIP_DIRECTORY_PROBE.json、INTAKE_AUTHORIZATION.json。实际payload和更细记录只在2203的pub_01/s03_incremental_continuity_20261101/external_dtu/。
