# PUB-01-S01 实际执行进度

更新时间：2026-09-19。当前 primary subtask：`PUB-01-S01`，执行中，尚未完成许可/配对/时轴验收。旧 M6A 工作已结束，不作为本轮任务。

## 首次重新连接服务器的实际结果

通过 `ssh -o ClearAllForwardings=yes -o BatchMode=yes -o ConnectTimeout=15 server2203` 执行只读盘点，退出码 0：

- hostname：`nercn`。
- 项目根：`/home/fanyu/auditory_simulation_m6a`。
- `df -h`：所在卷 3.5T，已用 2.8T，余量 702G，使用率 81%。
- `du -sh data/ds004703`：14G。
- `find data/ds004703 -name '*.edf' -type f | wc -l`：11。
- 项目根下 `find -maxdepth 4 -iname '*sparr*'` 无输出；仅说明该深度未发现相应命名路径，不证明服务器全局不存在。

已重新读取磁盘最新 README、AGENTS、CURRENT_TASK、PUB-01 TASK/SUBTASKS、S01 与 S06 任务书。当前仅执行 S01；S06 等协调者核验后切换并记录。既有 14G 数据只作为待核验输入，不算本轮下载，也不构成许可/身份/配对 PASS。

## 下一步与边界

核验官方版本、许可和访问范围，建立可追溯身份/配对/时轴清单。未知字段保留 UNKNOWN；全部数据读取和下载在 2203。本地仅轻量产物；不训练、不接触患者/STN、不做 hash 审计、不改历史归档或根级管理文档、不提交或 push。

## 阶段交接（同日）

SparrKULee 官方 API 已实际读取：V3.1，4,142 public 文件，135,967,030,049 bytes，196 restricted；公开 README 实际下载读取成功。许可/身份阶段门见 LICENSE_IDENTITY_GATE.md，协调者明确批准公开分支转 S06，无需等待受限分支。

当前 primary 切换为 **PUB-01-S06**；S01 完整审计未验收，保持 REVIEW，不解锁模型。ds004703 新审计已读11个EDF头与首秒、270个音频头、438个事件连续块；初版按文件名存在多个音频候选，全部保留 HOLD，不冒称配对通过。后续需要依据实验脚本定位实际刺激目录。

失败记录：S01 脚本首版出现多余右括号 SyntaxError，修复后服务器 py_compile 与实际审计成功。S06 初次启动使用了错误 Python 绝对路径，nohup 未运行；日志保留于远端 launch_20260919.log，正在查询真实环境路径后重新启动，不计作下载。
