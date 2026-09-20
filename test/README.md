# 历史材料与演示

`test/` 保留历史研究尝试、验证材料和演示包，不是 pytest 自动测试目录，也不决定当前 PUB 任务状态。

| 目录 | 用途 |
|---|---|
| `03_materials/` | 历史任务材料与演示交付参考 |
| `03_wav2vec2_independent/` | 早期独立 wav2vec2 工作材料 |
| `03_wav2vec2_independence_validation/` | 对应独立性验证材料 |
| `03_historical_remote_agent_attempts/` | 历史远端执行尝试记录 |
| `deliverables/` | 历史演示交付包 |

原文件名和路径保留，以维持来源追溯。当前任务入口是 `../doc/CURRENT_TASK.md`；正式自动测试位于 `../tests/`，由 `../pyproject.toml` 配置。

新 PUB 数据集下载、解压与处理全部在 `server2203`，不得在本目录增加数据载荷、模型权重或计算缓存。
