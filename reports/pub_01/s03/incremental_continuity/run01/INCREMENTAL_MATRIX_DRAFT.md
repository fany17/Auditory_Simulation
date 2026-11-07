# SS07 增量精确矩阵（已协调批准并冻结；保留历史文件名）

2026-11-01。执行权威为服务器冻结INCREMENTAL_PROTOCOL.json；本地保存同名副本。协调端已独立读取实际代码和工程证据并同意18条件×3seed排队，不再处于待批准状态。此文不授权读取test，不将synthetic检查当真实性能。实现pub_01_s03_ss07_incremental.py已在2203通过17组synthetic模型/控制检查：关闭/零支路恒等0；未来latent扰动/排除重叠上一窗0；奇数分块max2.39e-7；首步末投影/次步内部梯度有效。v2的single-call跨窗梯度不能单独证明跨forward；另TRAINER_ENGINEERING_CHECK.json用实际两次forward证明不断开状态时前块输入梯度非零、显式detach后为None，detach前后前向数值完全相同。native backbone仍完整共享每seed正式新encoder，下面参数只指增量+原head，不是换backbone。

## 1. 候选与控制（首轮合计18条件、每个3seed）

主候选每类仅width16/32两个配置，tau固定30s（M1固定3/12/48/192s多尺度），残差scale固定.1，其余native head/Adam不变。

| 条件 | 确切变化 | 数量 |
|---|---|---:|
| M1 | 四EMA历史尺度，过去条件8×8 latent mixing残差；当前native表示保留 | 2 |
| M2 | GRU快状态+可学慢EMA（tau范围3..300s）→8×8 latent混合 | 2 |
| M3 | 同M2 + stopgrad下一非重叠窗口48维teacher预测，lambda .01，train-only teacher whitening | 2 |
| M4 | 同快慢历史条件，逐channel对4/8/16sample latent shift的signed残差系数；不称生理时延 | 2 |
| M5 | 历史生成8×6→8×192先验，当前latent减prior为创新，历史条件8×8校正残差 | 2 |
| Native head | 原257参数head，同screen预算/初始化 | 1 |
| M2 static active-capacity | width16主模型增量5584参数匹配静态非线性adapter（非未用padding） | 1 |
| M2 current-only | 同width16全部核心参数，用当前窗6个子bin而非过去；recurrent权重实际激活 | 1 |
| M4 static alignment | 原head+学习24个逐channel非均匀shift系数，不是uniform fixed | 1 |
| M3 current-teacher | 各width相同额外teacher/predictor算力，重建当前而非未来，区分预测与一般aux学习 | 2 |
| Output prior adaptation | width16，对过去EEG-only baseline预测的6bin GRU先验添加时变残差；无真包络输入，DECAF-inspired非逐项复现 | 1 |
| M5 prior-only | width16，原head只接历史latent预测；unused correction参数不冒充active容量 | 1 |

控制不可作为额外创新框架挑赢家。对最终选定主候选再按其family/width确定**已预设规则**的static/current active匹配控制（同inference active参数±1%），原native P完整head、output-prior、M4最优静态/或M3同成本控制；不得用M2控制泛称其他family均已控制。

## 2. 两阶段研究预算（正式候选不继承pilot权重）

Screen：上述18×3seed（1024/2026/31415），每个完整426train，每epoch同原252454窗/3945个global64更新；max12epoch/patience4，Adam.001、无wd/clip、shuffle=false。保持原每3s负Pearson；每epoch按全部30500 val窗选点。所有配置结束后，以原Hann/60s→person的val macro r在三seed上平均排序；先选各family width，负结果也保留。

Full：五family各选出的一个width都从头重训原head+module，**各3seed完整max250epoch/patience10**，同P native目标/顺序/实际预算与encoder；不是只让最好单seed/一个框架跑全量。原P五seed完整基线继续。再按完整train/val选择主候选，最终主候选及其预设controls用五seed1024/2026/31415/42/12345（已预先冻结，不依显著性扩seed）。历史/容量控制不得以看test后的变化挑选。

强控制：另保留原完整P的encoder微调对照，拟原P best encoder+head，EEG lr1e-5/head1e-3、最多5epoch/patience3、same global64/native3s目标/原train-val；成本单列。其精确矩阵另在test前冻结；如果主框架也微调，必须同预算。不能只战胜较弱冻结head就称超过最强同协议模型。此控制不阻断当前P full。

M3额外aux损失/48维target访问数、梯度参数、FLOPs/time单列，teacher仅相应train数据拟合统计。M3本身不加独立额外epoch；current-teacher同lambda/次数/参数。若以后增加额外SSL，须新增明确同成本控制后协调冻结。

## 3. 精确时间/状态与原batch契约

所有current latent仍为原native encoder的完整3s表征，EEG[s:s+192]/包络[s:s+192]。主历史只含i−2/i−4等原始支持end≤当前窗start的已完成窗口；奇偶两条流分别更新。原global batch64包含最多多个record部分，模型按record切片/reset后拼回64个预测，**整个64只一次原Pearson loss/optimizer step**；不能把record尾另成step增加预算。record/phase/epoch均reset，step间detach保留；同step内不detach，奇数子切片仍按record-local parity而非batch parity。

M3 student after-window-i最多读取[i*96,i*96+192)及过去；future teacher取i+2支持[i*96+192,i*96+384)，不重叠、同record尾不足mask，不能用相邻overlap teacher。训练可用future目标，不作为inference输入。teacher current-control明确当前重建目标，不混称预测。

初始两窗无纯历史时照常评分，不丢。Hann融合原full-record grids/covered mask保留，各sample input支持并集不扩原baseline未来上限。每窗true/wrong历史重建，固定当前EEG/native latent；同人错故事/同故事错人/顺序打乱区分内容和身份贡献，跨人只离线干预。

## 4. 选择、证据与资源

仅train/val筛选，五类结果/失败都登记；test计划在全选择/seed/统计冻结后一次统一评估。旧test已暴露，所有结果内部探索；找到更高点估计并不等于goal或投稿证据完成。matched差值、刺激依赖/多模型选择与history机制/效率需独立复核，外部确认缺口保留。

复用每seed一个train/val latent cache，不为18条件复制cache；GPU/RAM可共享读并限制并发。baseline五seed缓存8.69GB，25GiB cap/20GiB reserve不变。best/latest有界、所有历史保留。先真实工程pilot测每step/内存；若不可执行先向协调报告工程问题，不因val/test负分更改预算。

## 5. 已落地的接续与诊断

SCREEN_QUEUED.json记录三个真实screen driver；只等待本seed正式pretrain完成及train/val缓存，不等待native head训练结束。FULL_QUEUE_PROTOCOL.json冻结独立selector/controller源；控制器必须验证54个真实completed（每项49人/752个固定有效块、原3s验证选点、完整epochs×3945更新），不平均缺失或失败seed，再完成五family各3seed full。主候选按full验证均值及预设tie规则锁定，补预定2seed与同预算匹配controls；screen权重不继承。失败需明确review，保留所有产物，不自动重试。metadata fixtures仅存engineering_history，科学选择器不读取该目录。

独立history wrapper保持每窗当前z和head不变，可重建真历史、固定供体历史、打乱历史、逐窗reset、训练均值内容盲以及older-outside-fused-support。后者使用ctx_before(i−1)，最晚历史源i−3的原EEG支持结束不晚于该预测实际Hann融合baseline输入并集起点；同时改变recency和奇偶流相位，不能解释为纯延迟或单一机制因果效应。6family非零残差与冻结forward误差0，15种首尾/尾差支持情况及EMA0恒等/gap reset通过；仍无真实性能含义。
