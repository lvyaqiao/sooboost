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

**限制**：门面暂不覆盖多分类（`MulticlassBooster` 仍需直接调用）；对标三巨头使用统一预算（200 轮/lr 0.1），非逐库调参后的最优成绩，结论限定于「同一梯队」而非全面领先。

**2026-09-01 补充交付（对标三巨头 + CI 真跑绿）**：
- 推送 master，GitHub Actions **首次真实运行全绿**（run 33463341389，gate job 13 步 3m56s）——M4 的「CI 干净 clone 全绿」出口正式关闭。
- 新增 `benchmark/compare_giants.py` + `benchmark/giants_comparison.{json,md}`：统一预算（200 轮/lr 0.1/seed 42）在 3 个**真实**数据集上对标 XGBoost 3.4.1 / LightGBM 4.7.0 / CatBoost 1.2.10 / sklearn HGB 1.9.0：

| 数据集 | 指标 | sooboost | XGBoost | LightGBM | CatBoost | HGB |
|---|---|---|---|---|---|---|
| california_housing | R² | 0.8403 | 0.8410 | **0.8466** | 0.8243 | 0.8427 |
| diabetes | R² | 0.3877 | 0.3594 | 0.3494 | **0.4521** | 0.3408 |
| breast_cancer | AUC | 0.9950 | 0.9937 | 0.9891 | **0.9970** | 0.9904 |

- 结论（如实）：sooboost 三集**全部前二**，与三巨头同一梯队（<1% 差距）；CatBoost 小数据集领先属有序提升等算法差异。速度：快于 HGB/CatBoost，慢于 LightGBM/XGBoost（SIMD/leaf-wise 差距 → M6 性能门禁方向）。README 已如实更新（替换原「仅对标 HGB」边界声明）。

**待手动验收**：无——**M5 出口已关闭（2026-09-01）**：crates.io 0.1.0 发布上线（`cargo publish` 成功，线上确认 [crates.io/crates/sooboost-core](https://crates.io/crates/sooboost-core)，keywords/categories/README 渲染就位），README 快速开始改为 `sooboost-core = "0.1.0"`。下一里程碑：M6 硬化与差异化（早停 + CV + 多分类 + 特征重要度 + 性能门禁），待立项。

### 2026-09-01 M6 一期交付：早停 + K 折交叉验证 + 特征重要度（模型格式 v3）

**决策**：M5 出口关闭后按路线图推进 M6 硬化。一期聚焦「训练控制与可解释性」三件套（早停 / CV / 特征重要度），多分类硬化与性能门禁留二期；为承载 gain/cover 统计，模型格式从 v2 升级至 v3，序列化同步扩展并保持对旧格式读取路径的明确校验。

**动作**：
- 早停：`Booster::fit` 拆为 `fit`（原签名不变）+ `fit_with_early_stopping` + 共用 `fit_impl`；`EarlyStoppingConfig { eval_set, rounds }`，每轮以 `Loss::value` 均值在验证集评估，patience 轮无改善即停并回滚最优轮权重；`boosting::Error` 新增 `EarlyStoppingStopped { best_iteration, rounds }`。`EvalState` 采用拥有式克隆列（`Float64Array` clone 为 Arc 共享零拷贝）规避自引用借用。
- 特征重要度底座：`Tree` 补 `split_gains: Vec<f64>` / `covers: Vec<f64>`，`NodeBuf` 分裂时写入 gain；`model/format.rs` `VERSION = 3`，io 序列化/反序列化同步两个新数组。
- 新建 `sooboost-core/src/metrics.rs`：`r2_score` / `auc`（CV 与评估共用）。
- `Dataset` 补 `slice_rows(offset, length)`（arrow RecordBatch 零拷贝切片）与 `concatenate_rows(&others)`；`data::Error` 新增 `RowSliceOutOfBounds` / `ConcatSchemaMismatch`。
- 门面（api.rs）：`early_stopping(eval_set, rounds)` builder、`cross_validate(ds, k)` 返回 `CvResult { fold_scores, mean, metric }`（回归 r2 / 分类 auc）、`feature_importances()` 输出 gain / cover / frequency 三口径；`Error` 新增 `Metric` 变体。
- CLI：`--eval <path>`（验证集）+ `--early-stopping <rounds>`，`--early-stopping` 未配 `--eval` 时显式报错。

**验证**：
- `cargo test --workspace`：**116 测试全绿**（M5 时 106 → 新增早停/CV/重要度/切片/拼接/metrics 单测，CV 测试数据用确定性交错排列避免连续分块外推问题）。
- `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings`：干净（修掉 `!(x > 0.0)` 模式、`map_clone`、`is_some_and` 等 lint）。
- 双基准门禁：`sooboost --gate` / `gen --gate` 均 exit 0。
- CLI 早停冒烟（california_housing，请求 2000 轮）：**第 798 轮早停，3.83s vs 全轮 8.18s（省 53% 训练时间）**，测试口径为「早停模型泛化不劣于全轮模型」。

**限制**：多分类硬化/校准与「真实基准固化进 CI 性能门禁」未做，留 M6 二期；早停评估每轮全量 `Loss::value` 计算，未做增量优化；速度与 LightGBM/XGBoost 的 SIMD/leaf-wise 差距仍在。

**待手动验收**：无——一期改动全部提交推送，CI 以 GitHub Actions 实跑为准。二期候选：多分类硬化/校准、性能门禁固化进 CI。

### 2026-09-01 M6 二期交付：softmax 多分类 + 真实基准性能门禁（M6 收口）

**决策**：一期（早停/CV/重要度）收口后完成 M6 剩余两项。多分类接入门面采用「收口而非替代」一致策略——底层 `MulticlassBooster`/`fit_multiclass` 保持一等公民，门面新增 `Fitted::Multiclass` 变体；为承载多分类，模型格式 v3→v4（loss 名后加 `num_classes` 头），破坏 v2/v3 读取已在 README 显著声明（0.1.0 为 v2，pre-1.0 可接受）。性能门禁不在 CI 安装三巨头（体积/稳定性），以 giants_comparison.json 记录值减安全余量为下限，sooboost 自测跌破即视为回归。

**动作**：
- 模型格式 v4（format.rs）：`num_classes u32` 头 + 多分类 K 个 init score、树类主序平铺；`MULTICLASS_LOSS_NAME = "multiclass_softmax"`。
- io.rs 重构：共享 `parse`（magic→版本→checksum→头→树→分箱表→类别段→元数据）+ 标量/多分类双路反序列化；`LossMismatch` 语义保持（仅目标不同的合法字节才报，门面按 回归→二分类→多分类 探测）；多分类拒绝 `has_categorical=1` 与「树数不能被类别数整除」。
- `MulticlassBooster`：`serialize`/`deserialize`/`feature_importances`（跨全部类别聚合后归一化）/`raw_logits_row`/`num_trees`/`learning_rate`/`from_parts`。
- 门面（api.rs）：`GradientBoosting::multiclass_classifier(n_classes)`；`predict_classes`/`predict_proba`（softmax 行和为 1）/`raw_logits`；`predict`/`predict_row` 对多分类输出 argmax 类别；`num_classes()`；标量专属操作（`raw_scores` 等）在多分类上显式报错（`Error::UnsupportedForObjective`）。入口校验：类别数 <2、早停 × 多分类、非整数/越界标签（`InvalidLabel`）均显式报错。
- `metrics::accuracy`（非法类别标签显式报错）接入 CV，多分类指标名 "accuracy"。
- 性能门禁：新增 `benchmark/run_real_gate.py`（import compare_giants 复用同口径/同数据加载，杜绝两套实现漂移），CI 新增第 9 步「Real-dataset performance gate」。
- README：状态行（v4 破坏性声明）、多分类用法节、路线图 M6 收口。

**验证**：
- `cargo test --workspace`：**124 测试全绿**（一期 116 → 新增 8 项多分类测试：可分数据拟合+softmax 行和+argmax 一致、存读逐位一致+目标探测、单行=批量、非法标签/越界/类别数<2 拦截、早停显式拒绝、CV accuracy、重要度三口径、标量专属操作报错）。
- fmt / clippy `--all-targets -D warnings` 干净。
- 三道基准门禁全过：`sooboost --gate`（对齐 HGB，差 -0.0068 ≤0.05）、`gen --gate`（ForestFlow 4/4）、`run_real_gate.py`（california R² 0.8403 / diabetes R² 0.3877 / breast_cancer AUC 0.9950，与 giants_comparison.json 记录**逐位一致**——数据划分 seed 42 固定，交叉验证了确定性）。

**限制**：多分类不支持类别特征与早停（显式报错）；校准（温度缩放/Platt）未做；模型格式 v4 不读 v2/v3 旧文件（0.1.0 用户需重训）；速度差距（SIMD/leaf-wise）延续一期结论。

**待手动验收**：无——CI 以 GitHub Actions 实跑为准（三道基准门禁已全部固化进 CI）。**M6 出口关闭（2026-09-01）**。下一里程碑未立项；候选：crates.io 0.2.0 发版（多分类 + v4 格式）、多分类校准、速度优化。

### 2026-09-01 补充交付：crates.io 0.2.0 发布上线（多分类 + 模型格式 v4）

**动作**：workspace 版本 0.1.0→0.2.0（仅 sooboost-core 发布，cli/experiments 保持 path 依赖）；`cargo publish --dry-run` 通过后正式上传；README 状态行与快速开始同步到 0.2.0。

**验证**：crates.io API 线上确认——version 3124213、num 0.2.0、yanked=false、crate_size 75,927B、37 个 Rust 文件/4,632 行、edition 2024、README 随包渲染。发布者 lvyaqiao。

**意义**：M6 全部成果（早停/CV/特征重要度/softmax 多分类/模型格式 v4）随 0.2.0 可被 `cargo add sooboost-core@0.2.0` 使用；0.1.x 导出的模型文件需重训（v4 不读旧格式，README 已声明）。

### 2026-09-01 M7 交付：多分类早停 + 温度缩放校准（M7 收口）

**决策**：0.2.0 发布后从远期清单中捞起「多分类校准 + 多分类早停」立项 M7。两个设计原则：① 早停**完全镜像标量语义**——每轮在验证集评估多分类 logloss，patience 轮无改善即 break 并把树集合回滚到最优轮（`truncate(best_round + 1)`），不引入错误变体，与标量 `fit_with_early_stopping` 行为一致；② 温度 T **不写入模型格式**——post-hoc 校准参数由调用方持有并显式传入预测，避免为不入格式的派生量再做 v4→v5 破坏性 bump（0.2.0 用户零迁移成本）。

**动作**：
- `multiclass.rs` 重构训练入口：`fit_multiclass` / `fit_multiclass_with_early_stopping` 双入口收敛到共享 `fit_impl`（早停为 `Option<&EarlyStopping>`）。验证 logits 逐类列增量累加（每棵类树完成后只更新该类列），logloss = `-mean ln softmax(logits)[true]`，零额外前向遍历。入口校验：rounds==0 → `InvalidEarlyStopping`、空验证集 → `EmptyDataset`、特征数不符 → `EvalSetFeatureMismatch`、非法标签 → `InvalidClassLabel`。
- `MulticlassBooster` 新增 `best_iteration()`（每类轮数，回滚后 < n_estimators）与 `eval_history()`（每轮验证 logloss，学习曲线）；`from_parts` 反序列化路径同步回填 `best_iteration`。
- 温度缩放：`calibrate_temperature(ds)`——200 点对数均匀粗网格 ∈ [0.05, 20] + 黄金分割 80 迭代，**完全确定**（红线 3，无随机性）；`predict_proba_with_temperature(ds, t)` = `softmax(logits/T)`；非正/非有限温度 → 新增 `BoostingError::InvalidTemperature`。
- 门面（api.rs）：`early_stopping` builder 对多分类生效（移除 M6 的 `UnsupportedForObjective` 拒绝路径）；`fit` 的 MulticlassSoftmax 分支按有无早停分派；`calibrate_temperature` / `predict_proba_with_temperature` 门面方法（非多分类 → `UnsupportedForObjective`）；`best_iteration()` / `eval_history()` 委托。
- README：多分类节补早停与温度校准用法，早停节标注「三类目标通用」，路线图加 M7 行。

**验证**：
- `cargo test --workspace`：**126 测试全绿**（124 → 新增 2 项）：
  - `multiclass_early_stopping_stops_before_max_rounds`：正交网格验证集 + patience 5，断言每类树数 < 200、`best_iteration == num_trees()`（门面 num_trees 对多分类语义为每类棵数，M6 起即如此）、历史最小值位置 +1 == best_iteration、概率行和为 1、**早停模型验证 NLL ≤ 全量 200 轮模型 + 1e-9**（早停的真实价值断言）。
  - `temperature_calibration_is_deterministic_and_improves_nll`：两次校准 T 完全一致（红线 3）、T 为正有限、T=1 与 `predict_proba` 恒等（1e-12）、**校准后 NLL ≤ T=1 NLL + 1e-9**、温度 0.0 → `InvalidTemperature`、回归目标 → `UnsupportedForObjective { operation: "calibrate_temperature" }`。
- fmt / clippy `--all-targets -D warnings` 干净。
- 三道基准门禁全过：`run_real_gate.py`（0.8403 / 0.3877 / 0.9950）、`sooboost --gate`（对齐 HGB，差 -0.0068）、`gen --gate`。

**限制**：温度校准为 post-hoc 一维搜索，不改变 logits 本身；多分类仍不支持类别特征（显式报错）；早停回滚为整批树回滚（与标量一致，不做部分树裁剪）。

**待手动验收**：无——CI 以 GitHub Actions 实跑为准。**M7 出口关闭（2026-09-01）**。下一里程碑未立项；候选：多分类类别特征（ordered TS 扩展到 K 类）、速度优化（SIMD/leaf-wise）、WASM/C ABI。

### 2026-09-01 M8 交付：多分类类别特征（ordered TS 复用，模型格式零变更）

**决策**：M7 后从远期清单捞起「多分类类别特征」立项 M8。核心判断：标量 D9 管线把类别特征经 ordered TS 数值化后完全复用数值建树管线，且 v4 格式的类别编码段布局与目标无关——因此多分类接入**不需要任何格式变更**，只需解除 M6 时「多分类路 has_categorical 恒 0」的自我限制。统计量取标签均值（整数标签的平滑 TS）：实现零成本且对类别→类的强信号可分；CatBoost 式 per-class 每类统计量需把一个类别特征扩成 K 列并扩展编码段布局，留远期并记入 roadmap。

**动作**：
- `multiclass.rs`：`fit_impl` 开头镜像标量的类别检测 + `max_categories` 校验 + `compute_ordered_ts`（ctx 的 seed 终于在多分类路生效——TS permutation 防泄漏）+ `resolve_to_dataset`；早停验证集用训练编码解析（OOV → 先验、null → 缺失）；`MulticlassBooster` 新增 `encoding`/`cat_features` 字段、`categorical_encoding()` 公开访问器、`resolve()` 推断集解析；`raw_logits` 先 resolve（`predict_proba`/`predict`/`predict_classes`/`calibrate_temperature` 自动受益）。
- `io.rs`：`serialize_multiclass` 按标量同布局写编码段；`deserialize_multiclass` 解除 `has_categorical` 拒绝，`Parsed.has_categorical` 字段因失去唯一读者删除；`format.rs` 布局注释同步。
- `api.rs`：`predict_row` 对类别多分类模型显式报错（`RowPredictUnsupportedWithCategorical`）；`multiclass_classifier` 文档更新（类别特征 M8 已支持）。
- README：状态行（M6–M8）、多分类节补类别特征说明、路线图 M8 行。

**验证**：
- `cargo test --workspace`：**127 测试全绿**（新增 2 项）：
  - `multiclass_categorical_fit_predict_and_roundtrip`：类别强信号（a/b/c→0/1/2）训练集全部分类正确；OOV 类别（"zzz"）与 null 类别推断不崩溃、概率有限且行和为 1；`to_bytes`/`from_bytes` roundtrip 预测逐位一致 + 再序列化字节一致 + 目标探测仍为 MulticlassSoftmax；`predict_row` 显式报错。
  - `multiclass_categorical_deterministic_same_seed`：同 seed 两次训练序列化字节逐位一致（TS permutation 由 seed 派生，红线 3 在多分类类别路成立）。
- fmt / clippy `--all-targets -D warnings` 干净。
- 三道基准门禁全过：`run_real_gate.py`（0.8403 / 0.3877 / 0.9950）、`sooboost --gate`（对齐 HGB，差 -0.0068）、`gen --gate`。

**限制**：TS 统计量为标签均值，弱于 CatBoost per-class 每类统计量（后者留远期）；`predict_row` 对类别模型不支持（`&[f64]` 承载不了类别键，标量同类模型同语义）。

**计数校正**：M7 记录的「126 测试」实为 125（笔误多计 1）；本里程碑实测总数 127（80 lib + 38 integration + 8 experiments + 1 doctest）。

**待手动验收**：无——CI 以 GitHub Actions 实跑为准。**M8 出口关闭（2026-09-01）**。下一里程碑未立项；候选：速度优化（SIMD/leaf-wise）、WASM/C ABI、crates.io 发版节奏随下一个用户可见功能再定。
