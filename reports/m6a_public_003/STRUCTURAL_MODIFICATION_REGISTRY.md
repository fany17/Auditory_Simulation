# M6A-PUBLIC-003 Structural Modification Registry

本 registry 只覆盖本轮三组：downsampling、RF growth/multiscale/decoupling、explicit change/event branch。所有模型均为小型、从零训练的 1D temporal models；不含 pretrained backbone。

| family | modification | computational meaning | temporal-resolution effect | RF effect | parameter/compute change | matched control | falsifiable endpoint | status |
|---|---|---|---|---|---|---|---|---|
| downsampling | early vs late stride | same total factor 4, move stride timing | intermediate frame step changes | RF changes as a consequence of stride timing | exact same convolutional parameterization | early vs late | seen-rate, omission, onset metrics | RUN + RF_CONFOUND noted |
| RF growth | uniform/local dilation | retain local windows | 1 ms output step | slow RF growth | exact schedule-parameter match | exponential/delayed | RF table and task metrics | RUN |
| RF growth | exponential dilation | grow context 1,2,4,8 | 1 ms output step | fast RF growth | exact schedule-parameter match | uniform/delayed | RF table and task metrics | RUN |
| RF growth | delayed-growth dilation | keep early layers local | 1 ms output step | delayed RF growth | exact schedule-parameter match | uniform/exponential | RF table and task metrics | RUN |
| RF growth | parallel multiscale k=3/7/15 | concurrent temporal windows then 1x1 fusion | 1 ms output step | max branch RF per block | width-adjusted; must be checked | uniform_local | RF table and task metrics | RUN/CONFOUNDED if outside ±10% |
| RF/downsampling | stride-coupled vs dilation-decoupled | same target RF, change source of RF | 4 ms vs 1 ms final step | matched theoretical RF | exact convolutional parameterization | pairwise | fast-event metrics at matched RF | RUN |
| RF diagnostic | kernel 3/7/15/31 | change direct local window | 1 ms output step | direct kernel growth | width-adjusted; confound retained if >±10% | kernel_3 | RF and performance diagnostic | RUN/CONFOUNDED if needed |
| event branch | explicit Δx branch | expose first temporal difference | same final resolution | same branch RF | branch width 11 vs baseline width 16 | ordinary second raw branch | onset/omission/phase metrics | RUN |

参数匹配状态以 `m6a_public_003_model_parameters.csv` 为准；任何 `CONFOUNDED` 比较只能作为工程描述，不能被写成单一结构因果证据。
