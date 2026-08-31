# roadmap.md 路线图与状态（权威单点）

> 状态：现行
> 类型：T2 台账（权威单点）｜写者：维护者（状态权威）｜读者：全员（开工前必读状态总览）
> 范围：里程碑/路线图/当前进度/排布原因——状态只在本文件变迁；需求源头见 doc/ledgers/backlog.md
> 关联：doc/baseline/architecture.md §里程碑映射（源头，原 04 迁移）；doc/records/（交付与验收）；doc/ledgers/backlog.md（需求池）
> 更新：2026-08-31（审计现状 + 立项 M4-M6 转向「可用库」优先）

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
- 2026-09-01：**M5 过半**——① 新增 `sooboost-core/src/api.rs` 公共门面：`GradientBoosting::regressor()/classifier()` builder、扁平化 `Config`、`Objective` 枚举、统一 `Error`（收敛 Data/Boosting/Model/Io 四套错误为单一枚举）、`predict_row` 单行在线推断、`to_bytes/from_bytes/save/load`（载入时依据 contracts §1.2 的校验顺序——checksum 先于损失名——安全自动探测目标）；`Booster::learning_rate()` 由 crate 内提升为公开访问器，供门面回填配置；门面 12 项单测（含存读逐位一致、目标探测、非法参数拦截、篡改字节必报错）。② 端到端 example `california_housing`（读 CSV → 训练 → 预测 → R²/MAE → 存盘载入复核 → 单行预测），release 下 **R² 0.8404**（200 轮/lr 0.1，16,512×8，训练 1.08s），已超过基准金标准 sklearn HGB 的 0.8355。③ 根 `README.md` + crates.io 发布元数据（description/repository/keywords/categories/readme），`cargo package` 实测通过且 README 被正规化打入包内。④ 全量门禁复验：106 测试（59 单测 + 38 集成 + 8 实验 + 1 doctest）、`cargo fmt --check` 与 `clippy --all-targets -D warnings` 全绿、sooboost/gen 两门禁全绿。**未完成**：对标 XGBoost/LightGBM/CatBoost（目前**只对标了 sklearn HGB**，README 已如实标注此边界，不宣称达到三巨头水平）；crates.io 0.1.0 发布（待推送远程后执行）。

## 后续路线图（M4 起，2026-08-31 立项）

> 立项背景：2026-08-31 现状审计发现——源码此前完全未进 git（仅文档提交）、集成测试在干净环境下 5/5 target 挂（benchmark 路径解析错误）、risks/shared 台账为空。决策：**优先级从「功能/研究扩张」反转为「可验证 + 可发布 + 被使用」**，先把现有成果变成可信、可发布的库，再谈扩张。

| 里程碑 | 内容 | 状态 | 出口标准 |
|---|---|---|---|
| M4 | 地基修复：能力审计（实际 API/早停/CV 是否真实现）+ 集成测试转绿 + 全量提交 git + CI 真跑绿 + 修悬空文档引用 + 空台账改为种子登记 | 完成（2026-08-31，地基修复 + CI 八步本地复验全绿） | 干净 clone `cargo test` 在 CI 全绿；git 含源码 |
| M5 | 可用库 v0.1：稳定公共 API 门面 + 端到端 example + 对标 XGBoost/LightGBM/CatBoost（≥3 真实集）+ 发布 crates.io 0.1.0 | 进行中（门面 + example + README + 发布元数据已完成；**对标三巨头与 crates.io 0.1.0 发布待办**） | `cargo add sooboost-core` 可用；example 能跑；基准报告存在 |
| M6 | 硬化与差异化：早停 + CV + 多分类硬化/校准 + 特征重要度 + 真实基准固化进 CI 性能门禁 | 待立项 | 对标 LightGBM 标准集差距 ≤ 阈值；功能完备可用 |
| 远期/支线 | WASM/C ABI/codegen、PG 插件、生产热替换；sooboost-experiments（条件分布树/Flow Matching）明确降级为支线，不占核心路线图进度 | 搁置（加日期） | 等 M5-M6 有真实用户后再启动 |

## 排布原因

- **M0 风险最高前置**：验证 arrow 集成与 Loss trait 设计是否成立。
- **M1 核心功能面**：D9 类别特征一期。
- **M2 研究向集成**：实验 crate 原型验证后反哺核心（D7），风险隔离。
