# TB001-DEMO001 单页展示

直接双击 `index.html` 即可打开。页面完全使用本目录内的音频和真实推理数据，不需要联网。

如果浏览器限制本地音频，可在 PowerShell 运行：

```powershell
& "C:\Users\fanyu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  -m http.server 8000 `
  --directory "E:\BCI_LFP_EEG\Project\0_EEG_Explanation_Auditory\Auditory_Simulation\deliverables\TB001-DEMO001_simple_showcase_20260807"
```

然后访问 `http://127.0.0.1:8000/`。看完按 `Ctrl+C` 停止服务。

默认展示为 2×语速、直接去掉 L5；原模型与干预后的最终转写字符编辑距离为 6。每个 Transformer 层都带有其信息流位置说明；“减弱一半”只保留该层新增量 `Δh=h_out-h_in` 的 50%，不是把权重或整层激活除以二。

点击中间的“卷积前端”可以切换标准约 50 Hz、约 100 Hz 和约 200 Hz 三种时间分辨率。2×语速下，约 100 Hz 的实际转写由标准前端的 `AS RITINGS THERE ARE TWO KINDS BRISH AND VON` 改为 `AS FOR ETCHINGS THEY ARE OF TWO KINDS GRITISH AND FOREIGN`；相对参考文本的字符错误率由 31.6% 降至 1.8%。这只是未重训练的单条语音实验，不代表更密的卷积输出总会改善识别。

右侧按“数据集真实标注 → 标准模型输出 → 当前干预输出”排列。两份模型输出中的红色词表示与真实标注不一致，并分别显示相对真实标注的字符错误率（CER）。

页面下方新增整体结果图；也可单独打开 `figures/frontend_speed_cer.svg` 或 `figures/frontend_speed_cer.png`。图的源数据保存在 `figures/frontend_speed_cer_source.csv`。虚线补偿策略由同一条语音选择，只用于展示解决思路，不是独立验证结果。
