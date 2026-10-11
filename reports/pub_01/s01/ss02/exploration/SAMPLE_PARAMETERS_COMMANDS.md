# SS02 exploration sample / parameter / command list

本目录只包含静态图、轻量索引摘要、Markdown/HTML 和绘图脚本；没有导出原始信号、音频数组或交互式原始载荷。

## Deterministic selection

- SparrKULee：在现有 cleaning_index.csv 中按 subject, source_recording 排序，取前三个最先出现且 raw source 为 PUBLIC、tail flag 为 WITHIN_3S_ADVISORY 的不同被试；另取排序后第一个 raw source 为 RESTRICTED 且有公开 EEG derivative 的样本。受限 raw 不读取。
- ds004703：在现有 clean_segment_index.csv 的 eligible=1 行中按 subject, recording, trial 排序，取前三个不同被试；读取原 EDF 的选定通道/小时间窗，不读取全库。
- 尾差：全部 26 个 LARGE_TAIL_REVIEW 行画长度关系；代表尾端图取 source_recording 字典序第一行。3 秒只作为 advisory marker。

## Parameters

- SparrKULee 使用发布方 64 Hz 预处理 EEG；PSD 上限为 32 Hz Nyquist。
- ds004703 使用原 EDF 记录采样率；PSD 上限为各样本的 fs/2。
- 多通道图读取最多 8 个通道；时域窗口最多 12 s。ds004703 事件相对窗口为 -0.5 至 12 s。
- ds004703 音频包络由公开 WAV 在服务器上即时计算（Hilbert amplitude + 20 Hz low-pass）；没有把该数组写入输出。
- SparrKULee 的 publisher envelope 直接从公开 derivative 读取；物理同步仍标记 UNKNOWN。时间零点是数组/metadata 参考，不是硬件同步证明。

## Server command

/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python /home/fanyu/auditory_simulation_m6a/pub_01/s01_ss02_exploration_20260924/plot_exploration.py --project-root /home/fanyu/auditory_simulation_m6a --output /home/fanyu/auditory_simulation_m6a/pub_01/s01_ss02_exploration_20260924

## Run notes

- 本批只复用既有 full index，并做确定性小样本读取；坏对象按同一排序规则跳过并记录。

### Audio notes

- sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz: raw stimulus audio read from /home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1/raw/stimuli/eeg/audiobook_5_1.npz.gz
- sub-019_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz: raw stimulus audio read from /home/fanyu/auditory_simulation_m6a/pub_01/s06/sparrkulee_v3.1/raw/stimuli/eeg/audiobook_6_1.npz.gz
- sub-SD010_ses-01_task-PassiveListen_trial-3: public WAV read; envelope=Hilbert+20-Hz low-pass
- sub-SD011_ses-01_task-PassiveListen_trial-3: public WAV read; envelope=Hilbert+20-Hz low-pass
- sub-SD012_ses-01_task-PassiveListen_trial-3: public WAV read; envelope=Hilbert+20-Hz low-pass
