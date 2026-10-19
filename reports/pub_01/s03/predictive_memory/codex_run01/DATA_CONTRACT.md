# SS05 prospective INTERNAL EEG输入契约

仅2203公开根/home/fanyu/auditory_simulation_m6a；新根pub_01/s03_history_memory_20261019/codex_run01。不读患者/STN，也不读旧SS03 test信号、成绩或checkpoint。

## 实际分组及来源限制

旧SS03 cache manifest仅取phase=train的426记录，其余193旧test仅按元数据身份排除，未读payload。逐record/subject/story匹配S02实际split_manifest的source_family/family_mapping_basis，得到37个family。它们曾进入SS03训练，因此只称SS05前瞻INTERNAL family holdout，不是未知外部数据。

真实family排序后用numpy RNG20261019一次置换，前7holdout、后7validation、其余23train；在新拟合前write-once保存split_frozen.json，未按成绩挑组。

| Phase | Families | Records | Subjects | 60s目标段 |
|---|---:|---:|---:|---:|
| train | 23 | 353 | 70 | 4427 |
| validation | 7 | 26 | 17 | 308 |
| holdout | 7 | 47 | 21 | 正式freeze后按相同规则处理信号 |

validation：audiobook_10/8/11、podcast_19/7/11/17。holdout：podcast_24/26/16/3/20、audiobook_12/13。三组family/record无交集，人可跨phase，故非新被试泛化。完整真实ID和映射在split证据，不随机拆相邻窗口。

## 必要防泄漏修正

旧SS02 train-only统计覆盖原train池，含新划出的validation/holdout，不能称SS05 train-only。现只在新353train记录、20448565时间sample上流式重拟合64channel mean/std，另存server-only newtrain_normalization.npz；旧统计和SS02/SS03不改。normalization_contract.json列全部fit IDs。新旧mean差L2=.00819725、std相对L2差.0282174，仅作来源审计，不拿旧统计拟合新模型。所有phase只用新train统计；目标不在val/test另拟尺度，发布方历史预处理限制保留。

## 连续窗口、状态及lag

EEG定向time×64，按新统计归一化；目标为同record既有公开包络。读取时严格检查finite/shape/common support。3s无重叠=192samples，20个当前窗组成60s目标段。共享前端输入20past+20current共40连续窗；每个当前query只递归自己的20past+当前，共21token完整BPTT，在上下文左边界清零。无跨query/不连续片段/record/person/session状态传递，不宣称任意长记录持久记忆。

为wrong_time提供完整更早历史，所有组统一从record第40窗即120s开始评分，之后每20窗一段；不满60s尾段不评分。train/val379记录无时长排除，holdout按冻结规则处理并记录。当前窗内可用完整3s EEG，跨窗因果而非sample级实时。

lag0/8/16/32只用五组各100步后的完整validation选择。平均r=.086648/.092460/.063559/.031155，固定8samples：EEG=x[t:t+192]，target=env[t−8:t−8+192]，即EEG比目标晚125ms。不是test寻峰或生理潜伏期证明。physical_sync仍UNKNOWN。

primary：60s串接Pearson→record等权平均→subject等权平均→subject-macro；不将3s窗/三个seed当独立被试。新split、非重叠/lag/warmup均异于SS02/SS03，禁止直接用旧r排名。3s-window Pearson损失不直接校准跨窗mean/scale，这也是历史效应解释的目标函数限制。

## 同当前EEG的固定历史条件

全部条件复用同一个当前窗编码和目标，仅改过去token：true前20窗；reset空状态；wrong_record另一录音（优先异人/异family）20连续窗；wrong_time本record更早60–120s的20窗；固定seed打乱顺序；按当前EEG-token相似度最高/最低的单个历史token换成同一donor对应token。donor/顺序规则测试前冻结，不用target选事件。

高/低相似性不是EEG语义相关/无关事件真标签。wrong_time把更早窗作为context而不插时间空档，测错时内容而非额外60s遗忘；真正插空白间隔的结果仅在合成probe单列。历史反事实不改变当前3s EEG，也不把音频ID、候选、声音或标签送进模型。

新holdout payload在FORMAL_PROTOCOL冻结且15次训练/验证全部完成后才读取；每selected模型一次test包含全部预定条件。所有raw/cache/normalizer/包络/weights及逐窗数组只在2203；本地仅代码/config、公开身份、汇总CSV/JSON、PNG和Markdown。
