# SS03 直接EEG→声音包络输入契约

任务 `PUB-01-S03_SS03`，公开SparrKULee available-author-fold4；计算仅server2203。上游SS02只读，未修改其数据/模型/报告。新运行根 `/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019`。

| Phase | 记录 | 人 | 3s窗口（hop1.5s） |
|---|---:|---:|---:|
| train | 426 | 70 | 252454 |
| validation | 49 | 49 | 30500 |
| test_seen | 69 | 69 | 36573 |
| test_heldout | 75 | 13 | 48119 |

来源 `pub_01/s03_strong_baselines_20261015/transformer_available01` 的 `prepared/manifest_frozen.json`、`scope/manifest_frozen.json`、`prepared/train_only_normalization.npz`，以及 `envelope_readout_seed1024/manifest_frozen.json` 与各phase `_envelope.npy`。没有新数据/权重下载，也没有读取患者/STN资料。

逐条按readout manifest的record/phase/phase_offset构建原始64Hz EEG窗口：第i窗起点i×96，长度192。将原数组显式定向为time×64，逐通道套用SS02冻结train-only mean/std，float32输出 `[N,192,64]`；目标为原 `_envelope.npy` 的 `[phase_offset+i,0,:]`，训练时转置为 `[N,192,1]`。没有插入或调用Transformer编码器、XLSR、音频或候选ID。

619条记录全部检查：与原prepared record/phase/common support一一对应；窗口数、offset连续且一致；EEG和target finite；原envelope相同坐标切片cast float32后与已有目标**逐元素相同**，max绝对误差0。各phase总数与上游一致；不因结果排除被试或改变样本。新cache只写 `cache01/`，完成前不供训练加载。

train/val/seen-test精确story ID集合无交叉；heldout人未进入train。这里的新故事指作者story ID，不保证所有source-family独立；heldout人组也不保证全部音频故事未见。原本不READY/无官方文件的对象沿用SS02排除，不扩大cohort。

验证和测试均将3s预测按50%重叠Hann overlap-add恢复到原录音坐标；每60s完整段算Pearson，再段→人→组macro。仅保留覆盖≥3800/3840样本且target/prediction非恒定的完整60s段；任何排除均写原因，不静默补0；未满60s尾段不评价。SS03不修改时间轴或以峰值调整对齐。

`physical_sync=UNKNOWN`。当前结果只能说明现有发布方时间坐标下的预测关系，不能证明硬件同步或推断生理潜伏期。所有模型离线、非因果，输入相同3s窗口。

详细record/path/窗口清单和检查在 `evidence/bootstrap01/manifest.json`。依本任务明确要求，SHA256仅标识轻量source/config/冻结manifest及小型归一化参数；不做全量原始数据checksum审计。其原始source identity保留在manifest的 `metadata_source_sha256`，正式训练源码身份在 `evidence/bootstrap01/formal_protocol.json`。本地不含raw EEG、包络数组、特征张量或权重。
