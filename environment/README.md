# 2203 专用环境

## 当前执行位置（2026-09-19）

数据集下载、续传、解压、读取核验、预处理、派生数据生成，以及模型权重下载、特征提取、训练/评估和缓存，全部在 `server2203` 完成。不得在 Windows 本地下载后再上传。远端不可用时报告阻塞，不回退到本地计算。

本目录只保存环境定义与版本文本。服务器数据、权重和特征张量不回传；仅同步轻量代码、配置、来源/许可/路径清单、汇总结果与报告。下列环境是历史已建环境的记录：先检查能否复用；若确需新依赖，使用本项目专用环境，不修改无关科研环境。

## 既有环境与入口

- 环境名：`auditory_m6a_public_001`
- 远端项目根：`/home/fanyu/auditory_simulation_m6a`
- 创建入口：`scripts/bootstrap_2203.sh`
- 自检入口：`scripts/remote_selfcheck.py`
- Conda 基础：`environment/m6a_public_2203.yml`，只从 `conda-forge` 创建 Python 3.11 与 pip
- Python 包：`environment/requirements_2203.txt`，固定首轮直接依赖版本
- 开发工具：`environment/dev-requirements_2203.txt` 与 `environment/dev-requirements_2203_pip.txt`，仅安装到本项目环境，用于 pytest、Ruff 与 mypy 审计
- 已解析开发工具：`environment/resolved_dev_packages_2203.txt`

首次部署时新建本项目专用环境；后续任务先检查并复用兼容的既有专用环境，不得激活后修改 `auditory_demos` 或其他科研环境。环境解析后的实际包版本以 `python -m pip list --format=freeze` 的非哈希文本记录为准；该记录只能声称版本清单，不构成密码学完整性证明。

本地仓库不保存 Conda 包缓存、模型权重或数据。代码同步到 2203 时使用普通文件复制，不在远端初始化第二个 Git 工作区。

开发工具安装入口：

```text
conda install --override-channels -c conda-forge -n auditory_m6a_public_001 \
  --file environment/dev-requirements_2203.txt
conda run -n auditory_m6a_public_001 python -m pip install \
  -r environment/dev-requirements_2203_pip.txt
```

`types-jsonschema` 与运行时 `jsonschema 4.25.1` 主版本/次版本匹配；`scipy-stubs 1.17.1.5` 对应运行时 SciPy 1.17 系列。本项目不对 jsonschema 或 scipy 全局关闭 mypy。MNE 与 SoundFile 当前未提供本项目可直接采用的类型声明，因此只在两处延迟导入上使用精确的 `import-untyped` 抑制，不影响其余模块检查。
