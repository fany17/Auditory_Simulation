# D6实际结果独立审阅

27套新闭式模型+18套D4原样复用、45候选及3P*评分已正常完成。无新TRAIN载荷、PCA/normalization拟合、NN训练、encoder/head前向、GPU或test/DTU推理；无失败重跑/矩阵扩大。

- **Technical：PASS_WITH_LIMITATIONS。** 协调独立完整审阅driver/core，核对52来源SHA/12输入指纹、45候选共同49人/752块/30500窗、共享λ及fallback；全部27组新系数有限、形状/参数数、P-only零Z、S/U band、S32严格共享、展开正则的实际正规方程/截距通过。独立从缓存输入+保存系数重建所选9个新模型，并另写Hann/NumPy相关评分，逐块r最大差4.44e−16；secondary MSE/分解重算≤3.44e−16。完整REPORT与三张选定PNG实际审阅，presentation02第三图与原数值一致。
- **Acceptance：PASS_WITH_LIMITATION（本轮归因交付）。** 科学状态CURRENT_READOUT_ATTRIBUTION_SUPPORTED_ON_REUSED_SOURCE_VALIDATION_ONLY。P-only已明显提升，latent映射还有开发增量；解除同支持共享不是选定收益来源，完整latent C相对U32稳定正向，平方增益依赖正则。一致性不是准确度替代。本轮未建立纯loss因果、额外历史贡献、独立泛化或新方法发表充分性。
- **Parent consistency：PASS。** 同一SS07公开EEG强baseline/连续输出研究；D4/D5原结论不改，所有数据/权重/Gram/系数/预测仅2203；无患者/STN、新数据、其他节点、远端Git或PR合并。

原样[数值audit](COORDINATOR_NUMERICAL_AUDIT.json) SHA54a778a9eb99880c491621fcf1dadb83738692d22e8d653e95ba55f0fd7e86d3与[最终review](COORDINATOR_RESULT_REVIEW.json)保留；protocol8dbbef41a5c914a56f5eb01ca2b8943fd86c3ba5f95b139d2d6b53a877d62575。独立重建复用冻结latent/P，并非独立raw EEG/encoder预处理或外部实验室复现；对应本轮代数读出修改已足够，不追加NN重放。

S32在D4时间坐标标准化空间共享，S/U有共同dense全窗P分支，native encoder本身看全3秒；C增加latent支持也增容。P-only是监督再读出，MSE与旧Pearson差同时含solver/参数化/正则；未完成严格单因素loss实验。重复已知被试/部分同书VAL不当独立test，三seed点差不称显著、等价或生理机制。

实际48次完整VAL评分/连续性1461648对次是同集重复；共享上游非零成本，stage和与总wall口径分开。旧三图原位保留，第三图仅改善细小MSE差距与刻度，不改模型/数据/统计。

完整[REPORT](REPORT.md)及全部λ/七个配对/逐seed逐人逐块/成本/三图中文解释交付。本D6有限归因完成，SS07/S03/PUB及科学goal继续ACTIVE。后续先给有界假说与强P-only/C/Ccap控制及确认性证据缺口，由协调审定；不扩大D4-D6、自动优化低overlap或以弱原head作唯一baseline。
