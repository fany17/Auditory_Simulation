# D14工程attempt01保留与v2修正

2026-11-02。attempt01 PREPARE PID1686331在正式训练前失败：非零checkpoint输出检查使用eval模式，混record scalar loss autograd前漏切train，cuDNN报 `RNN backward can only be called in training mode`。正式fit/optimizer更新/encoder/head前向均0；没有科学结果或模型选择。

旧input/PLAN/PROTOCOL/PREPARE.log/PREPARE_LAUNCH原样移入server新ROOT的 `engineering_attempt01/`，创建明确映射；无覆盖或删除。按实际到达的报错行和冻结源码执行路径重建工程NN窗362：H的first64/next64/tail38有157有效窗各正反两次314，再mixed batch一次48。该数是失败后代码路径重建，不是成功成本日志；CPU/wall/峰资源未落盘，标UNKNOWN。失败工程成本与后续成功工程另外计账。

v2仅在同一混record梯度检查前调用model.train（原模型无dropout），使cuDNN可反传；新增固定60排列每个64/尾38 batch至少一可学习窗检查；新protocol额外绑定本amendment与原D13三seed SUMMARY。正式epoch permutation SHA已绑定protocol。保持所有科学条件/总61,710,828正式NN窗/710100更新与16,384工程上限，不改变loss/权重/结构/预算，不复跑任何旧评估。source重新冻结并完成工程后仍须协调独立执行审阅。
