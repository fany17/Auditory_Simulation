# M6A-PUBLIC-004 — Auditory–Neural Representation Alignment and Transfer

状态：`READY_TO_EXECUTE`

Owner：`Auditory_Simulation`

患者/STN 数据：`FORBIDDEN`

上游：`M6A-PUBLIC-002` frozen pretrained baseline；`M6A-PUBLIC-003` completed execution / ready for review。

下游 consumer：`STN_Decoding_Encoding / M6B-STN-REP-001`。

---

## 1. 科学目标

本任务建立一套可以从公开数据学习、随后冻结并迁移到患者 STN 的 auditory-aligned neural representation。

核心问题不是“哪个网络最高分”，而是：

> 在不读取任何患者/STN 数据的前提下，能否从公开 EEG 与公开 intracranial 数据中学习一个对声音时间结构和事件信息敏感、跨被试/跨模态可迁移、且能以小样本适配新神经记录模态的表示空间？

本任务必须直接服务后续 STN few-shot adaptation 和写入目标选择。

---

## 2. 数据路线

### Stage 0 — Dataset / stimulus / feature audit

Primary EEG dataset：`SparrKULee`。

Primary public intracranial transfer dataset：`OpenNeuro ds004703`。

本阶段先完成：

- license / access / download manifest；
- subject / session / recording inventory；
- audio file identity 与 neural recording 对应；
- sampling rate、channel count、reference、event/timing metadata；
- story/clip/speaker identity；
- train/validation/test grouping keys；
- 已知缺失、坏通道、无同步或无法配对 recording；
- 音频与神经数据的真实时间对齐可用性。

禁止为了“先跑起来”随机切相邻时间点。

### Stage 1 — Audio representation bank

至少包括三组：

#### A. interpretable acoustic baseline

- waveform envelope；
- onset strength / event onset；
- log-mel / cochleagram；
- F0（适用时）；
- spectral flux；
- modulation / rhythm features。

#### B. frozen speech/audio SSL

第一轮优先：

- wav2vec2；
- HuBERT。

允许使用 M6A-002 已冻结的其他模型作为补充 probe，但禁止因为 004 再扩充模型名单。

#### C. explicit timing/rhythm features

- onset probability；
- event interval；
- local rhythm rate；
- phase / relative event phase（若刺激允许定义）；
- jitter / deviation regressor；
- event surprisal（只有在定义可审计时）。

每个 pretrained model 必须逐层提取；不得预先把某层命名为“节律层”“高级层”或“STN 层”。

---

## 3. Neural encoder 设计

目标是**跨 channel-count / modality 可适配**，而不是为一个固定 64-channel EEG 数据集写死。

### 必做 baseline

1. linear/ridge/TRF neural encoding baseline；
2. simple temporal CNN/TCN baseline；
3. channel-agnostic neural encoder。

### 推荐 channel-agnostic 结构

```text
per-channel signal
    ↓ shared temporal stem
per-channel temporal tokens
    ↓ channel pooling / attention
shared temporal latent
    ↓ temporal TCN / Transformer / SSM
neural embedding z_N(t)
```

必须允许：

- EEG 多通道；
- sEEG 不规则 contact 数；
- 后续 STN 2–4 bipolar channel 的 low-parameter adapter。

禁止把 scalp topography 作为唯一不可替换的输入结构。

### 可选 comparator

EEG foundation model（如可稳定复现）只能作为 comparator / initialization ablation，不得替代主路线。

---

## 4. Audio–neural alignment

对齐对象：

```text
audio x_A → frozen audio encoder f_A → z_A(t)
neural x_N → neural encoder f_N → z_N(t)
```

神经延迟必须显式处理，不假设 `t_audio = t_neural`。

至少实现两类目标：

### A. Encoding objective

使用 ridge/TRF/banded ridge：

```text
z_A(t-lag) → neural signal / neural latent
```

用于得到可解释的 held-out encoding performance 和 feature/layer baseline。

### B. Contrastive / retrieval objective

正样本：真实配对 audio–neural window。

负样本必须来自：

- 同 recording 的时间错位；
- 其他 clip/story；
- 其他 subject（在适用评估中）。

禁止使用容易被 story identity、recording identity 或 block artifact 直接识别的泄漏负样本。

可使用 InfoNCE / CLIP-style loss，但实现细节必须在报告中可审计。

---

## 5. 数据拆分与泄漏控制

至少报告：

1. within-subject held-out clip/story；
2. held-out subject；
3. held-out recording/session；
4. public EEG → public sEEG transfer；
5. few-shot adaptation learning curve。

所有标准化、PCA、layer selection、feature selection 和模型选择只能在训练折内部完成。

禁止：

- random adjacent-time split；
- 同一故事相邻片段分散到 train/test 后声称泛化；
- test subject 参与 normalization fitting；
- 用 test 结果决定 audio layer。

---

## 6. 关键对照

至少比较：

### Representation

- acoustic handcrafted；
- wav2vec2 layers；
- HuBERT layers；
- explicit timing/rhythm features；
- combined feature families。

### Neural encoder

- ridge/TRF；
- temporal CNN/TCN from scratch；
- channel-agnostic encoder from scratch；
- auditory-specific pretrained encoder；
- optional generic EEG pretrained comparator。

### Transfer

- sEEG from scratch；
- EEG-pretrained → sEEG frozen readout；
- EEG-pretrained → sEEG small adapter；
- EEG-pretrained → sEEG larger finetune（仅作上界，若资源允许）。

核心不是单点最优，而是**data efficiency / transfer gain**。

---

## 7. 主要指标

### Encoding

- held-out correlation；
- held-out R² / explained variance；
- unique variance over envelope/onset baseline。

### Match–mismatch / retrieval

- balanced accuracy；
- ROC-AUC；
- Recall@K / median rank；
- permutation null。

### Transfer

- zero-shot performance；
- few-shot performance at fixed fractions；
- samples/minutes required to reach a fixed performance；
- transfer gain over from-scratch；
- negative transfer when present。

### Reliability

- split-half；
- across recording/session；
- across subject distribution，不只报告 grand mean。

---

## 8. Stage gates

### Gate A — SparrKULee pipeline valid

必须满足：

- timing/auditory pairing audit complete；
- no leakage in split；
- interpretable acoustic baseline reproducible；
- at least one neural encoder has stable held-out performance above permutation null；
- held-out-subject result reported regardless of sign。

若 held-out subject 完全失败，不得跳过并只保留 within-subject 结果。

### Gate B — auditory-specific pretraining value

比较 from-scratch 与 auditory-specific pretraining。

通过标准不是必须“显著更高”，而是形成可审计结论：

- positive transfer；或
- no transfer；或
- negative transfer。

阴性同样可以进入下一步，但必须改变下游预期。

### Gate C — public intracranial transfer

在 `ds004703` 评估：

- zero/few-shot transfer；
- modality/channel mismatch；
- audio-layer stability；
- adapter parameter count vs performance。

不得把一个 subject/recording 的 preliminary result 写成跨模态成功。

### Gate D — artifact freeze

只有 Gate A–C 全部完成并人工审核后，才允许发布 `ART-AUDREP-v1` candidate。

---

## 9. `ART-AUDREP-v1` 交付要求

必须包含：

- manifest + semantic version；
- source commit / config / seed；
- frozen audio feature/model specification；
- neural encoder architecture；
- frozen weights；
- preprocessing and timing specification；
- embedding dimensions / frame step / layer semantics；
- portable transform/runtime；
- tiny public/synthetic canary；
- benchmark table；
- few-shot transfer curves；
- known failures / unsupported cases；
- schema + validator。

不得包含：

- 患者/STN 数据；
- 患者派生 embedding；
- 私有临床 metadata；
- 以患者结果选择过的 model/layer。

---

## 10. 论文定位

M6A-004 本身首先是公共方法/预训练节点。最终包含 STN 的主体论文在私有 `STN_Decoding_Encoding` 完成。

M6A 必须提供足够独立的公共结果，使得后续能够回答：

> STN 上的增益究竟来自“公共 auditory-specific neural pretraining”，还是来自患者数据本身的重新拟合。

---

## 11. 明确禁止

- 不继续堆新的 pretrained audio architecture；
- 不读取任何 STN/patient 数据；
- 不在少量 public sEEG 上端到端训练超大模型；
- 不只报告最佳 subject；
- 不把 EEG→sEEG transfer 自动外推成 EEG→STN 已成功；
- 不把高相关 representation 解释成脑区功能同源；
- 不把模型 layer 先验命名为节律/语义层；
- 不通过反复更换 split、窗口或负样本来追求阳性。

---

## 12. 完成定义

`M6A-PUBLIC-004 = COMPLETE` 需要：

1. SparrKULee 数据审计与 leak-safe benchmark；
2. acoustic + wav2vec2 + HuBERT layerwise baseline；
3. channel-agnostic neural encoder；
4. encoding + contrastive/retrieval 两类目标；
5. held-out subject；
6. ds004703 public intracranial transfer；
7. few-shot transfer curve；
8. negative/failed conditions retained；
9. `ART-AUDREP-v1` candidate bundle；
10. independent consumer cross-test 尚未通过前，不得声称 frozen/accepted。
