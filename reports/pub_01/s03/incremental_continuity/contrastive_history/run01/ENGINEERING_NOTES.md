# D26 工程版本与计量

- V01 实际完成 4 次对比更新和 6 次监督工程更新。进入 Ccap 工程时，包络尾部超出冻结共同支持，布尔索引失败。此前神经路径的全部断言通过。墙钟、总 CPU 与逐梯度数值未写入终态收据，记缺失。[V01 日志](ENGINEERING_V01.txt) · [已到达阶段](NEURAL_V01_REACHED.json)
- V02 只补做 seed 1024 的四次 CL 更新并即时记录两段 token 梯度。head 与 FT 没有重跑。Ccap 另用 12 条既有合法 TRAIN 缓存核验切片、正仿射与首四窗回退，0 次神经前向或更新。工程权重均未进入正式链。[梯度](GRADIENT_ENGINEERING_V02.json) · [读出](CCAP_ENGINEERING_V02.json)
- V03 固定初版完整协议，并以无前向、无优化的恢复测试确认模型、Adam、StepLR、CPU/CUDA RNG 和游标能从已保存边界恢复。V04 源传输时有文件为空，0 次前向及更新，未形成正式门。V05 使用原 V03 实物重新施加计量修订。[恢复及几何](RESUME_AND_COVERAGE_CHECK.json) · [最终协议](PROTOCOL_V05.json)
- 正式 head 对每臂逐相位独立记录缓存当前与历史实际窗口、wall 和 CPU，优化及缓存 VAL 另外计时。队列写每 worker 从进程生成至观察退出的 wall，worker 同时报告自身 wall 和 CPU。配对等待与并行作业的时段可以重叠。
