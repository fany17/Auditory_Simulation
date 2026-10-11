# 来源、许可与路径

- 数据：SparrKULee RDR V3.1，官方来源 https://doi.org/10.48804/K3VSND 。历史许可审计 `reports/pub_01/s01/AUDIT_REPORT.md` 已确认公开范围非商业科研门；CC-BY-NC-4.0（https://creativecommons.org/licenses/by-nc/4.0/legalcode.en），保留来源、署名、许可和修改说明，不默认衍生产物可商用。本轮内部研究授权沿用既有用户阶段安排。
- 服务器根：`/home/fanyu/auditory_simulation_m6a`（server2203）。现有公共derivatives路径由 `pub_01/s06/sparrkulee_virtual_20260919_v4/cleaning_index.csv` 指向，不下载restricted对象。
- 本轮split/sample来源：`pub_01/s02_acoustic_baseline_20260924/{split_manifest,sample_manifest}.csv`。首次拟合前检查S02 83 subjects/659 READY records；本轮冻结10名93条，精确沿用分配。
- 声学历史输出：`pub_01/s02_acoustic_baseline_20260924`。特征比较历史输出：`pub_01/s02_feature_compare_20250925`。后者目录名的20250925是历史别名；报告UTC创建时间为2026-09-24T18:10（北京时间2026-09-25），本轮没有改名或搬动。
- 较新Windows工作区（只读复用）：`C:\Users\fanyu\.codex\worktrees\a063\Auditory_Simulation`。其reports/pub_01/s02下两份已读原报告轻量副本保存在 `upstream_evidence/`，仅供provenance，不将旧报告里的相对图链接当作本目录当前产物。
- 当前代码和记录写在项目主目录 `E:\BCI_LFP_EEG\Project\0_EEG_Explanation_Auditory\Auditory_Simulation`；主目录和a063既有修改保留。不初始化、切换、重置Git；本轮不提交/推送。
- 运行配置是本目录 `config_snapshot.json`，源为configs/pub_01_s03_envelope_decode_20261011.json；运行代码scripts/pub_01_s03_envelope_decode.py，复核代码scripts/pub_01_s03_review_decode.py。实际代码未提交，以运行目录所存脚本与配置为执行版本，不将HEAD当成未提交代码身份。
- 实际服务器新输出：`pub_01/s03_envelope_decode_20261011/run01`；首被试rerun：同父目录 `reproduce_sub001`。输入脚本/config和log：`pub_01/s03_decode_inputs_20261011`。
- 参数：只在server的run01/server_only/*.npz；未回传。Windows回传白名单为run01顶层CSV/JSON/PNG及首被试metrics/summary，不包括信号数组、模型参数或cache。
- 方法方向参考：[MNE官方stimulus reconstruction示例](https://mne.tools/stable/auto_examples/decoding/receptive_field_mtrf.html)，访问2026-10-11。本轮实际实现为自写ridge，不冒称MNE运行；没有借用示例结果。
- 原始方法论文：Crosse MJ, Di Liberto GM, Bednar A, Lalor EC (2016), *The Multivariate Temporal Response Function (mTRF) Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli*, Frontiers in Human Neuroscience 10:604，DOI 10.3389/fnhum.2016.00604。[全文](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00604/full)已于2026-10-11实际读取，核验backward为神经响应→刺激重建及权重解释边界。
- 作者实现来源：[stimulus_reconstruction.m](https://github.com/mickcrosse/mTRF-Toolbox/blob/master/examples/stimulus_reconstruction.m)，2026-10-11读取master网页。确认示例Dir=-1、训练内CV选lambda和test评估；未将其数据划分、归一化或lag数值照搬为本轮规则，不声明与该MATLAB实现数值等价，也未增加运行依赖。
