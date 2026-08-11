# 2203 专用环境

- 环境名：`auditory_m6a_public_001`
- 远端项目根：`/home/fanyu/auditory_simulation_m6a`
- 创建入口：`scripts/bootstrap_2203.sh`
- 自检入口：`scripts/remote_selfcheck.py`

环境必须新建，不得激活后修改 `auditory_demos` 或其他科研环境。环境解析后的实际包版本以 `python -m pip list --format=freeze` 的非哈希文本记录为准；该记录只能声称版本清单，不构成密码学完整性证明。

本地仓库不保存 Conda 包缓存、模型权重或数据。代码同步到 2203 时使用普通文件复制，不在远端初始化第二个 Git 工作区。
