# 传输工程接续v2：科学适配不变

2026-11-02，正式新数据模型预测/分数/学习模型拟合均0。v1已成功处理的载荷和输出不废弃；原source、PREPROCESSING_PROTOCOL、日志及计时保留。该接续仅处理传输瓶颈，不是阴性结果后的科学重试。

相同已验证canary的固定prefix实测：2MiB Range9.360秒，8MiB40.028秒，206/Content-Range/字节均与已固定完整canary匹配，同一S3version。更大块没有每字节收益，因此保留2MiB；四网络流的实际速率可能令19.4GB超原4h阶段上限。

唯一变化为网络worker4→8、BLAS2→1，DSP/prepare_record另用Semaphore限制最多2并发。保持同一数据版本、128→64空间/滤波/参考/重采样/标量gain/共同长度/目标/mask/模型/统计规则；GPU仍0，RSS仍8GiB、ROOT3GiB、全SS07 NPY28GiB、free20GiB，每对象30min/最多3次原传输重试，新stage4h。账本另列原v1耗时、v2耗时与自v1起总elapsed，不把总预算伪称4h。

BLAS1用原首canary完整重算，最终FP32校准输入最大差4.768e−7，满足原工程1e−6，非bitwise一致。未看任何模型成绩。复用的v1记录保留原BLAS2来源，其余用新BLAS1；这不是另一个科学适配候选，四角色最终共享同一套冻结输入。实际差及此数值限制必须保留。

接续前逐项核原17个source/metadata/norm pin、119 TSV的Git blob和20个已生成target的原SHA。仅在核UID/argv/start_ticks/无children后终止自有PID1379762，并确认全部线程退出；其他作业不动。原已登记processed逐个完整SHA/shape/float32/C-order/finite、来源/共同mask/原A_source核验后复用。不覆写；未登记孤立/不完整文件只移动到本ROOT明确备份目录，不永久删除。交接数量及精确记录写PREPROCESSING_TRANSPORT_HANDOFF。

新源码/交接/数学桥及本修订另冻结为PREPROCESSING_PROTOCOL_V2。最终预测前还需完整实际数据/mask/model/statistics freeze和协调审阅；本接续不等于正式评估已完成。
