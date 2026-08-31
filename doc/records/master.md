# master.md master 域交付与验收

> 状态：现行
> 类型：T3 流水（只追加）｜写者：维护者｜读者：全员
> 范围：master 域交付记录与验收证据（现行唯一域；未来分支域各自建文件，规则见 records/README.md）
> 关联：doc/ledgers/roadmap.md（权威状态）；doc/records/shared.md（共享面登记）
> 更新：2026-08-19

## 待办区

| 行号引用 | 待办 | 状态 |
|---|---|---|
| （空） | | |

## 交付记录

> 条目模板：`### <日期> <主题>` + 决策（为什么）/ 动作（改了什么）/ 验证（测试与门禁证据）/ 待手动验收

### 2026-08-19 M0 交付：九项承诺全绿（S1-S7）

**决策**：M0 收缩为九项承诺（doc/archive/m0-spec.md）；CLI 手写参数解析不引入 clap（易踩坑 8 依赖克制）；benchmark 对齐并入 run_benchmark.py 新增 `--mode sooboost`（可复现入 CI）。

**动作**：
- S1 workspace：sooboost-core + sooboost-cli（edition 2024，arrow 59.2，红线 7 `#![forbid(unsafe_code)]`）。
- S2 数据层 data/（Dataset + MissingPolicy，红线 2 缺失唯一来源）；S3 分箱 binning/（排序精确分位，D4）；S4 损失 loss/（L2 + binary logloss，D3 泛型 Loss trait）；S5 树内核 tree/（SoA Tree + 单线程直方图 + level-wise TreeBuilder，D7 标量叶子）；S6 训练层 boosting/（fit GBDT 循环 + Booster + TrainingContext 红线 4 + 确定性测试入 CI）。
- S7 CLI + 对齐：`sooboost-cli train` 输出与 correctness 档同构；4 数据集生成 sooboost_predictions.csv + sooboost_metrics.json。

**验证**：cargo test --workspace 41 测试全绿（含确定性逐位一致测试）；clippy --all-targets 0 警告；cargo fmt --check 干净。benchmark --mode sooboost 4/4 数据集超过 RandomForest、贴近 HistGradientBoosting（详见 doc/ledgers/roadmap.md 对齐表）。

**待手动验收**：M1 立项范围确认（并行/序列化/类别特征/基准门禁）——已归档 m0-spec 至 doc/archive/m0-spec.md 并修复全部引用。

### 2026-08-19 M1 交付：五项工作流全绿（M1-1 ~ M1-5）

**决策**：M1 按风险递进推进（序列化 → 目标函数 → 并行 → 类别特征 → 门禁）；模型格式升 VERSION 2（追加类别编码段）；热替换用 `RwLock<Arc<..>>`（std 无 `Arc::swap`，contracts §1.2 勘误）；fuzz 以 proptest「任意字节不崩溃」入 CI（cargo-fuzz 正式化留 M2 前置）。

**动作**：
- M1-1 model/：显式小端字节格式（magic/版本/树/bin 表/loss/元数据/checksum/类别编码段）+ roundtrip + HotSwappable。
- M1-2 loss/：huber(pseudo)/quantile/poisson/gamma/tweedie + boosting/multiclass.rs 原生 softmax；修复分箱 off-by-one（BUG-2026-08-19-01）。
- M1-3 tree/builder.rs：特征级 rayon 并行直方图，任意线程数逐位一致。
- M1-4 data/target_stats.rs + dataset 类别列支持：ordered TS（D9），OOV/null/高基数边界。
- M1-5 scripts/gate.ps1 + .github/workflows/ci.yml + benchmark --gate。

**验证**：cargo test --workspace 84 测试全绿（含并行确定性、序列化 roundtrip、proptest fuzz 占位、类别契约）；clippy --all-targets -- -D warnings 0 警告；cargo fmt --check 干净；scripts/gate.ps1 全 6 步通过；benchmark --mode sooboost --gate 4/4 数据集 OK（线性 R² 0.957 / 非线性 0.375 / AUC 0.948 / 加州 0.829，均贴近 HGB）。

**待手动验收**：M2 立项（条件分布树 + Flow Matching，D7 实验 crate 隔离）；cargo-fuzz 正式化。

### 2026-08-19 M2 立项

**决策**：用户确认 M2 两条研究方向都做（A 条件分布树 / B Flow Matching，M2 内分 tranche），并把 M1 遗留的 cargo-fuzz 正式化纳入前置（M2-0）。规格 doc/archive/m2-spec.md；参照 UnmaskingTrees/BaltoBot（arXiv:2407.05593, TMLR 2025）与 ForestDiffusion（arXiv:2309.09968）。

**动作**：roadmap M2 置「进行中」；plans/README 登记 m2-spec；新实验 crate `sooboost-experiments`（D7 风险隔离，`#![forbid(unsafe_code)]`）。

**验证**：待 M2 各 tranche 交付。

**验收**：M2-A/B 方向均保留在实验 crate；M3 继续做质量硬化，不回灌核心向量叶子。

### 2026-08-19 M2 交付：研究原型与生成门禁全绿（M2-0 ~ M2-D）

**决策**：按 D7 保持研究隔离，新增 workspace crate `sooboost-experiments`；不引入多输出/向量叶子。M2-C 验证核心现有 `Booster::predict_row`、`Dataset::from_record_batch/from_csv_bytes` 已足够支撑两条原型，未新增研究专用核心 API。

**动作**：
- M2-0：新增独立 cargo-fuzz crate（nightly + libFuzzer）与 `model_deserialize` / `csv_parse` targets；`Dataset::from_csv_bytes` 作为 CSV 字节路径并入 stable proptest 守护。
- M2-A：`UnmaskingImputer` 按可预测性顺序迭代解掩；`BaltoBot` 平衡中位数条件树 + 叶内 GBDT 均值模型 + 经验残差分布采样。
- M2-B：`ForestFlow` 每特征独立 GBDT 拟合 conditional flow-matching 速度场，z-score 归一化，Euler 采样/条件填补，同 seed 逐位一致。
- M2-D：新增 `sooboost-experiments` benchmark binary；`benchmark --mode gen` 输出边际误差、相关矩阵 MAE、C2ST AUC、填补 RMSE；`scripts/gate.ps1` 与 `.github/workflows/ci.yml` 扩为 8 步 gate。

**验证**：cargo-fuzz `model_deserialize` 61s / 10,858,679 runs、`csv_parse` 61s / 99,321 runs，均 0 crash；cargo test --workspace **91 tests** 全绿；clippy `--workspace --all-targets -- -D warnings` 0 警告；fmt clean；M2 8 步 gate 全过；gen gate 4/4 数据集通过有限性/结构/C2ST/依赖感知填补门槛。

**研究结论**：条件分布树与 Flow Matching 均有正向原型证据；强依赖数据填补优于均值（binary +47.4%、California +40.9%），独立特征数据不把均值基线改善强行作为门槛。California 生成边际误差仍偏高，后续如继续应先做标定/混合类型/采样质量改进，不直接提升为稳定 API。

**待手动验收**：下一里程碑立项；M2 研究原型保持实验 crate，不作为生产 API 发布。

### 2026-08-19 M3 交付：ForestFlow 质量硬化（B-03）

**决策**：按 D7 继续隔离研究实现；边际校准与点填补逻辑只进入 `sooboost-experiments` / benchmark，不改变核心模型格式和稳定 API。随机 `impute` 语义保留，RMSE benchmark 改用显式的 `impute_mean` 点估计。

**动作**：
- `ForestFlow` 每列改为经验分位数正态化，生成结果用经验分位数插值反变换，改善偏态边际。
- 每个训练行采 4 个分层时间流匹配样本，按特征派生 seed；积分器改为二阶中点法。
- 新增 `ForestFlow::impute_mean`，CLI 增加 `--imputation-samples`（默认 4）；benchmark 指标写入 `imputation_samples`。
- 新增经验正态变换 roundtrip 与 Monte Carlo 点填补逐位确定性测试。

**验证**：`scripts/gate.ps1` 8/8 全过；workspace **93 tests** 全绿；clippy `-D warnings`、fmt clean；`benchmark --mode gen --gate` 4/4 通过。California 生成均值误差 `1.027 → 0.049`、std 误差 `4.158 → 0.057`、相关 MAE `0.088 → 0.062`、C2ST `0.803 → 0.620`；4 次样本均值点填补增益 `+23.3%`；binary 点填补增益 `+72.4%`。

**限制**：弱相关 synthetic 数据点填补仍略差于均值基线（约 `-7.7%` / `-10.8%`），门禁按无依赖场景允许不超过 `1.5x` 基线；生成结果仍属研究原型，不作为生产质量承诺。

**待手动验收**：下一里程碑方向；M3 方案已归档至 `doc/archive/m3-forest-flow-quality.md`。

### 2026-09-01 M5 阶段一交付：公共 API 门面 + 端到端 example + README

**决策**：M4 能力审计确认核心算法已可用、但 API 需要用户自行拼装 `fit(ds, &params, loss, &ctx)` 四个概念且要面对三套独立错误类型。门面的定位是**收口而非替代**——底层 `fit` / `Booster<L>` / `Dataset` 保持一等公民，架构 D3 的 monomorphized `Loss`（编译期内联）不变，门面只压平「选目标 / 配参数 / 传上下文」并把错误收敛为单一枚举。此决策保证「好用」不以牺牲既有设计红线为代价。

**动作**：
- 新增 `sooboost-core/src/api.rs`（M5 门面层）：
  - `Error`：统一枚举，收敛 `DataError` / `BoostingError` / `ModelError` / `io::Error`，并新增 `InvalidParam` / `FeatureCountMismatch` / `RowPredictUnsupportedWithCategorical` 三类门面专属错误；原始错误保留在变体内可 `source()` 下钻。
  - `Objective`（`SquaredError` / `BinaryLogLoss`）+ 扁平 `Config`（不再要求用户知道 `max_depth` 属于分裂契约还是分箱契约）；`GradientBoosting::regressor() / classifier()` → builder → `fit`。
  - `predict` / `raw_scores`（二分类下为 logit）/ `predict_row`（单行在线推断，缺失以 `f64::NAN` 表达并按 `MissingPolicy` 解释）。
  - `to_bytes` / `from_bytes` / `save` / `load`：`from_bytes` 依据 contracts §1.2 的校验顺序（checksum 先于损失名校验）做目标自动探测——只有字节本身合法、仅目标不同才会落到 `LossMismatch`，截断/checksum 失败一律原样上抛（红线 6）。
  - 入口参数校验：`n_estimators` / `learning_rate` / `min_samples_leaf` / `max_bins` / `max_categories` / `categorical_alpha` 非法时在 `fit` 前显式报错（NaN 也拦，易踩坑 5 静默错误纪律）。
- `Booster::learning_rate()` 由 `pub(crate)` 提升为公开只读访问器，供门面在载入模型时回填持久化配置（`n_estimators` / `learning_rate` / `max_bins`）；树参数与 seed 不属于模型格式，载入后取默认值并已在文档注明。
- 新增 `sooboost-core/examples/california_housing.rs`：完整回路（读 CSV → 训练 → 预测 → R²/MAE → 存盘载入复核 → 单行预测），数据路径沿用「从 `CARGO_MANIFEST_DIR` 向上查找含 `benchmark/` 的目录」，与集成测试同一思路，对 crate 嵌套深度不敏感。
- 新增根 `README.md`（定位 / 快速开始 / 精度对标表 / 取舍 / 结构 / 开发命令 / 路线图指针）；`sooboost-core/Cargo.toml` 补 description / repository / homepage / documentation / keywords / categories / readme。

**验证**：
- `cargo test --workspace`：**106 测试全绿**（59 单测 + 38 集成 + 8 实验 + 1 doctest），其中门面新增 12 项（默认值对齐、线性拟合、概率区间、logit 一致性、存读逐位一致、目标探测、非法参数拦截、单行与批量一致、特征数不符、同 seed 同字节、错误统一、篡改字节必报错）。
- `cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings`：全绿（修掉 4 处告警：`neg_cmp_op_on_partial_ord` 改为显式 `is_finite` + 正比较、`needless_range_loop` 改迭代器、`type_complexity` 提 type alias）。
- example（release）：california_housing **R² 0.8404 / MAE 30052**，200 轮 lr 0.1，训练 1.08s（16,512×8），存读后预测逐位一致。
- `cargo package -p sooboost-core` 通过（49 文件），README 被正规化打入包内 `sooboost-core-0.1.0/README.md`。
- benchmark：`sooboost --gate` / `gen --gate` 均 exit 0，4/4 数据集对标 sklearn HGB 差 ≤0.05。

**限制**：当前精度基线**仅对标 sklearn `HistGradientBoosting`**，尚未对标 XGBoost / LightGBM / CatBoost；README 已如实标注此边界，不宣称达到三巨头水平。门面暂不覆盖多分类（`MulticlassBooster` 仍需直接调用）。

**待手动验收**：M5 剩余项——对标 XGBoost/LightGBM/CatBoost（≥3 真实数据集）、推送远程触发真实 GitHub Actions、发布 crates.io 0.1.0。
