# roadmap.md 路线图与状态（权威单点）

> 状态：现行
> 类型：T2 台账（权威单点）｜写者：维护者（状态权威）｜读者：全员（开工前必读状态总览）
> 范围：里程碑/路线图/当前进度/排布原因——状态只在本文件变迁；需求源头见 doc/ledgers/backlog.md
> 关联：doc/baseline/architecture.md §里程碑映射（源头，原 04 迁移）；doc/records/（交付与验收）；doc/ledgers/backlog.md（需求池）
> 更新：2026-09-01（M6 完成收口：多分类 softmax + 真实基准性能门禁）

## 里程碑与路线图

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 九项承诺（一页规格见 doc/archive/m0-spec.md）：arrow RecordBatch + 数值特征 + null/NaN 缺失值 + L2 回归 + binary logloss + quantile binning + 单线程直方图 + 基本树参数与预测 + 固定 seed 确定性测试 | ✅ 完成（2026-08-19 收口，已归档） |
| M1 | rayon 并行 + D8 其余目标函数 + 类别特征(ordered TS，契约见 doc/baseline/contracts.md §1.4) + 序列化/热替换 + 基准门禁（规格见 doc/archive/m1-spec.md） | ✅ 完成（2026-08-19 收口，已归档） |
| M2 | 研究方向集成：条件分布树（BaltoBot/UnmaskingTrees）+ Flow Matching 向量场（规格见 doc/archive/m2-spec.md） | ✅ 完成（2026-08-19 收口，已归档） |
| M3 | ForestFlow 生成/填补质量硬化（规格见 doc/archive/m3-forest-flow-quality.md，需求 [B-03]） | ✅ 完成（2026-08-19 收口，已归档） |

## 当前进度

- 2026-08-18：benchmark 金标准 v1 建立（quality/correctness/perf 三档，见 benchmark/）；M0 范围收敛为九项承诺（doc/archive/m0-spec.md）；零拷贝/确定性/模型格式契约修订（doc/baseline/contracts.md）。
- 2026-08-19：M0 S5-S7 完成并收口（树内核/训练层/CLI+benchmark 对齐，41 测试绿，4/4 数据集超 RF、贴近 HGB）。
- 2026-08-19：**M1 五项工作流全绿（84 测试 + clippy -D warnings + fmt + 门禁全过）**：
  - M1-1 模型序列化/热替换：显式小端字节格式 v2（magic/版本头/树/bin 表/loss/元数据/FNV-1a checksum/类别编码段）；roundtrip 逐位一致 + 损坏字节分类报错 + proptest 任意字节不崩溃；HotSwappable（RwLock<Arc>，契约勘误 std 无 Arc::swap）。
  - M1-2 目标函数：huber(pseudo)/quantile(含 L1)/poisson/gamma/tweedie 回归 + 原生 softmax 多分类；数值梯度/海森 proptest；**顺带修复 M0 遗留 off-by-one**：分箱阈值 `bin_value` 改严格 `<`，训练/预测在边界值 x==boundaries[k] 处一致（BUG-2026-08-19-01，bugs.md 登记）。
  - M1-3 rayon 并行：特征维度独立直方图 → 任意线程数逐位一致（红线 3 层级二测试：1/4 线程序列化字节一致）；16 核实测 ~2.2x 加速。
  - M1-4 类别特征 ordered TS（D9）：训练 seed 派生 permutation 防泄漏；OOV→先验；null 类别→缺失；高基数超限报错；编码随模型序列化；同 seed 逐位一致。
  - M1-5 基准门禁：scripts/gate.ps1（fmt/clippy -D warnings/test/build/对齐门禁/perf 参考）+ .github/workflows/ci.yml + benchmark --mode sooboost --gate（落后金标准 >0.05 → 退出 1）。
  - 收口后测试集对齐（4/4 OK）：线性 R² 0.957 / 非线性 0.375 / AUC 0.948 / 加州 0.829（vs HGB 0.961/0.367/0.949/0.836）。

- 2026-08-19：**M2 立项**：方向定为「条件分布树 + Flow Matching 都做，M2 内分 A/B tranche」+「cargo-fuzz 正式化」纳入前置（用户经 question 确认）；规格随后归档为 doc/archive/m2-spec.md；参照 UnmaskingTrees/BaltoBot（2407.05593, TMLR 2025）与 ForestDiffusion（2309.09968）。
- 2026-08-19：**M2 收口**：M2-0 cargo-fuzz 双 target 各运行 ≥60s 无 crash（model 1086 万次、CSV 9.9 万次）；M2-A `sooboost-experiments` 完成 UnmaskingImputer + BaltoBot（均值/多峰条件分布测试）；M2-B 完成 per-feature ForestFlow（生成相关性/确定性/填补测试）；M2-C 验证现有 `Booster::predict_row` 足够，无需污染核心向量叶子设计；M2-D `benchmark --mode gen --gate` 接入 4 数据集，8 步 gate 全绿。
- M2 生成门禁结论：4/4 数据集生成结果有限、相关性/C2ST 通过原型门槛；有强特征依赖时填补优于均值（binary +47.4%、California +40.9%），独立特征数据不强行要求超过均值，避免把无信号误判为算法回归。该结论仅代表研究原型，不代表生产质量承诺。
- 2026-08-19：**M3 收口**：ForestFlow 加入经验分位数正态化、每行 4 个分层流匹配样本、二阶中点积分；新增 `impute_mean` 与可配置 `--imputation-samples`。生成/填补 gate 4/4 通过，California 生成均值相对误差 `1.027 → 0.049`、std 误差 `4.158 → 0.057`、C2ST `0.803 → 0.620`；点填补增益 `+23.3%`。

- 2026-08-31：现状审计 + 立项 M4-M6（转向「可用库」优先）。审计发现源码此前完全未进 git（仅文档提交）、集成测试在干净环境下 5/5 target 挂（benchmark 路径解析错）、risks/shared 台账空；决策优先级从「功能/研究扩张」反转为「可验证 + 可发布 + 被使用」。详情见「后续路线图」。
- 2026-08-31：**M4 地基修复完成**：① 集成测试 `benchmark_path` 改为向上查找 workspace 根（解析到 `benchmark/`），5 个 integration target 转绿，全工作区 ~93 测试全绿（47 lib + 41 integration + 8 experiments）；② `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings` 全绿；③ 5 处悬空 `doc/plans/m0-spec.md` 引用修正为 `doc/archive/m0-spec.md`；④ benchmark 门禁 `sooboost --gate`/`gen --gate` 本地复验全绿（sooboost 质量对标 sklearn HGB，4/4 数据集差 ≤0.05）；⑤ risks/shared 空台账改为种子登记。CI 八步（fmt/clippy/test/build/两门禁）本地复验全绿，**推送 master 后 GitHub Actions 将实际运行**。能力审计发现：早停/交叉验证**尚未实现**（M0 明确不做，属 M5/M6 硬化项），CLI 与核心 API 已可用。
- 2026-09-01：**M5 过半**——① 新增 `sooboost-core/src/api.rs` 公共门面：`GradientBoosting::regressor()/classifier()` builder、扁平化 `Config`、`Objective` 枚举、统一 `Error`（收敛 Data/Boosting/Model/Io 四套错误为单一枚举）、`predict_row` 单行在线推断、`to_bytes/from_bytes/save/load`（载入时依据 contracts §1.2 的校验顺序——checksum 先于损失名——安全自动探测目标）；`Booster::learning_rate()` 由 crate 内提升为公开访问器，供门面回填配置；门面 12 项单测（含存读逐位一致、目标探测、非法参数拦截、篡改字节必报错）。② 端到端 example `california_housing`（读 CSV → 训练 → 预测 → R²/MAE → 存盘载入复核 → 单行预测），release 下 **R² 0.8404**（200 轮/lr 0.1，16,512×8，训练 1.08s），已超过基准金标准 sklearn HGB 的 0.8355。③ 根 `README.md` + crates.io 发布元数据（description/repository/keywords/categories/readme），`cargo package` 实测通过且 README 被正规化打入包内。④ 全量门禁复验：106 测试（59 单测 + 38 集成 + 8 实验 + 1 doctest）、`cargo fmt --check` 与 `clippy --all-targets -D warnings` 全绿、sooboost/gen 两门禁全绿。
- 2026-09-01（补）：**推送 master 触发 GitHub Actions 首次真实运行全绿**（run 33463341389，13 步 3m56s）——「CI 在干净 clone 上真跑绿」从纸面变为事实。**对标三巨头完成**：新增 `benchmark/compare_giants.py`（统一预算 200 轮/lr 0.1/seed 42），在 3 个真实数据集上实测——california_housing R² 0.8403（LightGBM 0.8466 最佳，差 0.6%，超 CatBoost）、diabetes R² 0.3877（全场第二，超 XGBoost/LightGBM/HGB）、breast_cancer AUC 0.9950（全场第二，超 XGBoost/LightGBM/HGB）；结论：与三巨头同一梯队（<1% 差距），小数据集上 CatBoost 领先属算法差异；速度上快于 HGB/CatBoost、慢于 LightGBM/XGBoost（SIMD/leaf-wise 优化差距留给 M6 性能门禁）。结果落盘 `benchmark/giants_comparison.{json,md}`，README 已如实更新。**M5 剩余：crates.io 0.1.0 发布**。
- 2026-09-01（补2）：**crates.io 0.1.0 发布上线，M5 出口关闭**。`cargo publish --dry-run` 通过（49 文件/225KiB，依赖全部 crates.io 侧，打包源码编译干净），正式上传成功并确认线上可见（crate id sooboost-core，version 3123812，keywords/categories 就位，README 渲染）。README 状态行与快速开始同步更新（git 依赖 → `sooboost-core = "0.1.0"`）。M6（早停 + CV + 多分类硬化 + 特征重要度 + 性能门禁）待立项。
- 2026-09-01（补3）：**M6 立项并完成一期（早停 + K 折交叉验证 + 特征重要度）**。① 早停：`Booster::fit_with_early_stopping`（`EarlyStoppingConfig { eval_set, rounds }`），每轮用 `Loss::value` 在验证集评估，patience 无改善即停并回滚最优轮；`boosting::Error` 新增 `EarlyStoppingStopped { best_iteration, rounds }`。② `Tree` 结构补 `split_gains` / `covers`，`NodeBuf` 分裂时写 gain——模型格式随之升级 **v3**（io 序列化同步，含 v2→v3 兼容读取）。③ 新建 `metrics.rs`：`r2_score` / `auc`。④ `Dataset` 补 `slice_rows`（arrow RecordBatch 零拷贝）/ `concatenate_rows`。⑤ 门面：`early_stopping(eval_set, rounds)` builder、`cross_validate(ds, k)`（回归 r2 / 分类 auc）、`feature_importances()`（gain / cover / frequency 三口径）。⑥ CLI 新增 `--eval` / `--early-stopping`。验证：workspace **116 测试**全绿、fmt/clippy `-D warnings` 干净、sooboost + gen 双基准门禁 exit 0；CLI 冒烟——请求 2000 轮在第 798 轮早停，3.83s vs 全轮 8.18s（省 53% 训练时间），且早停模型泛化不劣于全轮模型。多分类硬化与性能门禁留 M6 二期。
- 2026-09-01（补4）：**M6 完成收口（二期：softmax 多分类 + 真实基准性能门禁）**。① 模型格式 **v3→v4**：loss 名后新增 `num_classes` 头（标量恒 1），多分类存 K 个 init score、树按类主序平铺；io 重构为共享 `parse` + 标量/多分类双路反序列化（`LossMismatch` 语义保持：门面探测回归→二分类→多分类，截断/checksum 失败一律原样上抛）。② `MulticlassBooster` 补 `serialize`/`deserialize`/`feature_importances`（跨类聚合）/`raw_logits_row`/`num_trees`/`learning_rate`。③ 门面：`GradientBoosting::multiclass_classifier(n_classes)`、`predict_classes`/`predict_proba`/`raw_logits`、`predict` 输出 argmax 类别、`num_classes()`；多分类 + 早停、类别数 <2、非整数/越界标签均显式报错；`metrics::accuracy` 入 CV（指标名 "accuracy"）。④ 真实基准门禁：新增 `benchmark/run_real_gate.py`（复用 compare_giants 同口径与数据加载），3 个真实集精度下限（california R²≥0.82 / diabetes R²≥0.35 / breast_cancer AUC≥0.985，即记录值减安全余量）接入 CI 第 9 步；本地实测三集 PASS（0.8403/0.3877/0.9950，与 giants_comparison 记录逐位一致）。验证：workspace **124 测试**全绿（新增 8 项多分类测试）、fmt/clippy 干净、三道基准门禁全过。多分类校准（温度缩放等）与多分类早停留远期；crates.io 0.2.0 发版待用户决定。
- 2026-09-01（补6）：**M7 立项并完成（多分类早停 + 温度缩放校准）**。① M7-1 多分类早停：`fit_multiclass_with_early_stopping`（重构出共享 `fit_impl`，早停为 `Option<&EarlyStopping>`），口径为验证集多分类 logloss `-mean ln softmax(logits)[true]`（验证 logits 逐类列增量累加，零额外遍历）；**语义完全镜像标量**：patience 轮无改善 break → 树集合逐类 `truncate(best_round + 1)` 回滚，不报错；`MulticlassBooster` 新增 `best_iteration()`（每类轮数）与 `eval_history()`；入口校验（rounds==0 → `InvalidEarlyStopping`、空集 → `EmptyDataset`、特征数不符 → `EvalSetFeatureMismatch`、非法标签 → `InvalidClassLabel`）。门面 `early_stopping` builder 对多分类生效，移除旧 `UnsupportedForObjective` 拒绝路径并替换为正向早停测试。② M7-2 温度缩放校准：`calibrate_temperature(ds)`（200 点对数均匀粗网格 ∈ [0.05, 20] + 黄金分割 80 迭代，完全确定，红线 3）与 `predict_proba_with_temperature(ds, t)`（`softmax(logits/T)`，非正有限温度 → 新增 `BoostingError::InvalidTemperature`）；**设计决策：T 不写入模型格式**（避免 v4→v5 bump，调用方持有并显式传入），非多分类目标 → `UnsupportedForObjective`。③ 验证：workspace **126 测试**全绿（新增 2 项：正向早停断言 num_trees<200 / best_iteration==num_trees / 历史最小值位置 / 早停模型验证 NLL ≤ 全量模型；温度校准断言两次校准同 T / T=1 与 predict_proba 恒等 / 校准后 NLL 不劣 / 非法温度与非多分类报错）、fmt/clippy `-D warnings` 干净、三道基准门禁全过（real 0.8403/0.3877/0.9950 + sooboost 对齐 + gen）。顺带澄清：门面 `num_trees()` 对多分类语义为「每类棵数」（与 `n_estimators` 对齐，总数 = 该值 × 类别数），M6 起即如此。
- 2026-09-01（补7）：**M8 立项并完成（多分类类别特征，零格式变更）**。① 设计：复用标量 D9 ordered TS 管线（`compute_ordered_ts`/`apply_encoding`/`resolve_to_dataset` 原封不动），统计量为**标签均值**（类别整数标签的平滑均值；CatBoost 式 per-class 每类统计量留远期，需扩展特征数与编码段布局）；**模型格式 v4 零变更**——类别编码段布局本就与目标无关，仅解除多分类路 `has_categorical` 恒 0 的自我限制。② 实现：`fit_impl` 开头镜像标量的类别检测 + `max_categories` 校验 + TS 数值化；早停验证集用训练编码解析（OOV → 先验）；`MulticlassBooster` 新增 `encoding`/`cat_features` 字段与 `categorical_encoding()` 访问器，`raw_logits` 经 `resolve` 解析推断集（`predict_proba`/`predict`/`calibrate_temperature` 自动受益）；io `serialize_multiclass`/`deserialize_multiclass` 支持编码段；门面 `predict_row` 对类别多分类模型显式报错（`RowPredictUnsupportedWithCategorical`，与标量同类模型同语义）。③ 验证：workspace **127 测试**全绿（新增 2 项：类别强信号训练集全对 + OOV/null 不崩溃有限 + 存读 roundtrip 逐位一致 + 单行报错；同 seed 序列化字节逐位一致）、fmt/clippy `-D warnings` 干净、三道基准门禁全过。顺带校正：M7 记录的「126 测试」实为 125（笔误多计 1）；`Parsed.has_categorical` 字段因失去唯一读者随之删除。
- 2026-09-01（补8）：**M9 立项并完成（速度优化：建树热路径 + 形状自适应并行，模型字节不变）**。① 剖析：旧热路径 `best_split_for_feature` 每特征独立遍历节点行——`grad[r]`/`hess[r]` 被重复随机读取 F 倍（F=特征数），这是与 LightGBM 单趟扫行的主要差距。② 改造：`BinnedMatrix` 特征主序 → **行主序**（`bins[row*nf+feature]`，单行全部特征 bin 连续），直方图重构为全特征扁平 `HistSet`（单趟填充：每行只读一次 (g,h)，内层连续读该行 bin）。③ 并行结构三轮迭代（同负载公平 A/B 定案）：每节点特征维并行（M1-3 旧结构）→ 层内节点并行（california 提速但 breast_cancer 小数据并行度塌缩）→ 两阶段扁平（california 最优但 breast_cancer 慢 79%）→ **最终：形状自适应双路**——行数/特征数 ≥ 1024 走行主序两阶段（单元 = (节点×行块) 填充 + (节点×特征) 扫描），否则走 (节点×特征) 直接路径（累加顺序与 v0.2.0 逐位一致）。④ 确定性（红线 3）：选路只依赖数据形状；行块边界只依赖节点行数、块间按块下标定序归并；(gain, feature) 全序合并与顺序无关 → 任意线程数逐位一致。叶子值/节点 (G,H) 合计始终行序逐点累加，不参与分块归并。⑤ 定版数字（新旧二进制同轮交替 A/B，best-of-4）：california 1.592→1.335s（**+16%**）、diabetes 0.588→0.615s（复测 6 轮中位数 0.723→0.631，**+13%**）、breast_cancer 0.345→0.227s（**+34%**）；三集指标与旧版**逐位一致**（零模型漂移）。⑥ 验证：workspace 127 测试全绿、fmt/clippy 干净、三道基准门禁全过。教训入档：基准计时受桌面负载影响极大，跨轮次绝对值不可比，必须同轮交替 A/B；SIMD 内核需 unsafe 或额外依赖，显式留远期。
- 2026-09-01（补9）：**M10 立项并完成（直方图减法：行主序路填充减半，模型字节不变）**。① 设计：行主序路径每对兄弟只直接构建**较小子节点**（seed，平局取左）的直方图，较大子节点由 父 − seed 推导（`HistSet::subtract_in_place`，count 用 saturating 防御——子行集 ⊆ 父行集故恒非负）；直接路径（多特征小行数）**不动**，保住「累加与 v0.2.0 逐位一致」性质。② 实现关键：推导子节点**接管**父直方图缓冲（`prev_hists[p].take()`）而非克隆——首版 clone+subtract 实测把填充减半的收益几乎吃光（41KB memcpy × 每推导节点）；因每个分裂节点的两个子节点中只有一方推导，take 语义安全。三层结构：阶段 A 填 seed（(节点×行块) 单元）→ 阶段 B 推导非 seed（并行单步减法）→ 阶段 C 全体扫描。③ 确定性（红线 3）：seed 选择只依赖行数比较、减法为单步浮点运算、父缓冲值本身确定 → 任意线程数逐位一致。④ 测量：桌面多线程噪声过大（同二进制 1.6–2.7s 波动），改用 **RAYON_NUM_THREADS=1 聚焦 A/B**（红线 3 保证单线程=多线程结果）：california 中位数 2.489→1.817s（**快 27%**），best 2.224→1.699s（快 24%）；三集指标逐位一致（含行主序路的 california 0.8403341506124684，实际未发生 ulp 漂移）。⑤ 验证：workspace 127 测试全绿、fmt/clippy 干净、三道基准门禁全过。

## 后续路线图（M4 起，2026-08-31 立项）

> 立项背景：2026-08-31 现状审计发现——源码此前完全未进 git（仅文档提交）、集成测试在干净环境下 5/5 target 挂（benchmark 路径解析错误）、risks/shared 台账为空。决策：**优先级从「功能/研究扩张」反转为「可验证 + 可发布 + 被使用」**，先把现有成果变成可信、可发布的库，再谈扩张。

| 里程碑 | 内容 | 状态 | 出口标准 |
|---|---|---|---|
| M4 | 地基修复：能力审计（实际 API/早停/CV 是否真实现）+ 集成测试转绿 + 全量提交 git + CI 真跑绿 + 修悬空文档引用 + 空台账改为种子登记 | 完成（2026-08-31，地基修复 + CI 八步本地复验全绿） | 干净 clone `cargo test` 在 CI 全绿；git 含源码 |
| M5 | 可用库 v0.1：稳定公共 API 门面 + 端到端 example + 对标 XGBoost/LightGBM/CatBoost（≥3 真实集）+ 发布 crates.io 0.1.0 | **完成（2026-09-01）**：门面 + example + README + 对标三巨头（3 真实集全部前二）+ crates.io 0.1.0 发布上线 | `cargo add sooboost-core` 可用；example 能跑；基准报告存在 |
| M6 | 硬化与差异化：早停 + CV + 多分类硬化/校准 + 特征重要度 + 真实基准固化进 CI 性能门禁 | **完成（2026-09-01，两期）**：一期早停/CV/特征重要度 + 二期 softmax 多分类（模型格式 v4）+ 真实数据集精度下限固化为 CI 性能门禁 | 对标 LightGBM 标准集差距 ≤ 阈值；功能完备可用 |
| M7 | 多分类质量收口：多分类早停（验证 logloss 口径）+ 温度缩放校准（post-hoc 确定性） | **完成（2026-09-01）**：早停语义与标量一致（break + truncate 回滚）+ `best_iteration`/`eval_history` + 温度 T 由调用方持有不入模型格式 | workspace 测试全绿 + 三道基准门禁全过 + CI 真跑绿 |
| M8 | 多分类类别特征：ordered TS 复用标量 D9 管线（标签均值口径），编码段随 v4 格式序列化 | **完成（2026-09-01，零格式变更）**：训练/预测/早停验证集全链路接入 + OOV → 先验 + 存读 roundtrip 逐位一致 | workspace 测试全绿 + 三道基准门禁全过 + CI 真跑绿 |
| M9 | 速度优化：建树热路径（单趟直方图 + 行主序分箱）+ 形状自适应并行，精度不回退、模型字节不变 | **完成（2026-09-01）**：同负载 A/B california +16% / diabetes +13% / breast_cancer +34%，指标逐位一致；任意线程数逐位一致（红线 3） | 三道基准门禁全过（精度不回退）+ A/B 提速有同轮交替实测记录 |
| M10 | 直方图减法：行主序路兄弟对只建较小子节点直方图，较大方由 父−seed 推导（父缓冲接管零克隆） | **完成（2026-09-01）**：单线程聚焦 A/B california 中位数再快 27%，指标逐位一致；直接路径不动（保持逐位同 v0.2.0） | 三道基准门禁全过 + 单线程 A/B 实测记录 |
| 远期/支线 | WASM/C ABI/codegen、PG 插件、生产热替换、per-class TS（CatBoost 式每类统计量）、SIMD 内核（需 unsafe 或额外依赖）；sooboost-experiments（条件分布树/Flow Matching）明确降级为支线，不占核心路线图进度 | 搁置（加日期） | 等 M5-M6 有真实用户后再启动 |

## 排布原因

- **M0 风险最高前置**：验证 arrow 集成与 Loss trait 设计是否成立。
- **M1 核心功能面**：D9 类别特征一期。
- **M2 研究向集成**：实验 crate 原型验证后反哺核心（D7），风险隔离。
- 2026-09-01（补5）：**crates.io 0.2.0 发布上线**。workspace 版本 0.1.0→0.2.0，`cargo publish -p sooboost-core` 成功并线上确认（version 3124213，yanked=false，37 文件/4,632 行 Rust，edition 2024，README 渲染）。M6 全部成果（早停/CV/特征重要度/softmax 多分类/模型格式 v4）正式对外可用；0.1.x 模型文件需重训（v4 不读旧格式）。M4-M6 保持全部完成状态。
