# S01 许可与身份阶段门（2026-09-19）

此记录不是 S01 全部验收；pairing/timing/split 仍待审核。

## SparrKULee：公开分支 PASS，受限分支 HOLD

服务器实际获取官方 RDR API：<https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/K3VSND>。
版本 3.1；许可 CC-BY-NC-4.0；4,142 公开文件，共 135,967,030,049 字节；196 受限文件，不请求、不重试。
官方 README 数据文件 API 137074 已在服务器无需凭据实际读取成功。逐文件清单与许可元数据见同目录 CSV/JSON。

非商业科研可在许可范围内复制、处理；共享时保留作者/来源/许可与修改说明，不将衍生数据默认视为无条件可商用。许可正文：<https://creativecommons.org/licenses/by-nc/4.0/legalcode.en>。访问限制不因许可开放而解除；完整受限内容等待用户获准访问方式，不搜索凭据或代发邮件。

官方代码 README 明确公开 derivatives 可用；旧 downloader deprecated 是维护状态，不等于禁止公开 API。来源：<https://github.com/exporl/auditory-eeg-dataset>。

协调者已独立核验本地官方清单，明确批准公开部分转 S06 下载/QC。702G 可用空间大于约126.63GiB公开载荷；持续监测并预留解压空间。

## ds004703：非商业只读审计可继续；再分发条件 HOLD

现存版本目录 v1.1.0 与 dataset_description 中 DOI `10.18112/openneuro.ds004703.v1.1.0` 一致；官方当前 description 与 README 已在服务器获取。来源：<https://openneuro.org/datasets/ds004703/versions/1.1.0>，<https://raw.githubusercontent.com/OpenNeuroDatasets/ds004703/master/README>。

metadata License 为 CC0，但 README 要求不得商业使用（含商业 ML 训练），不得重识别。保守同时遵守，不裁定法律冲突已解决，不发布可商用 artifact。当前 master 文本不是不可变快照验证，版本身份进一步对账仍待完成。

## 执行切换

S01 已完成本阶段许可/公开访问门，完整审计未完成。按协调者明确授权，下一 primary 为 S06 公开 SparrKULee 下载与初步 QC；S01 的 benchmark 准入仍 HOLD。禁止由下载成功直接解锁 S02。
