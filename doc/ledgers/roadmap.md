# roadmap.md 路线图与状态（权威单点）

> 状态：现行
> 类型：T2 台账（权威单点）｜写者：维护者（状态权威）｜读者：全员（开工前必读状态总览）
> 范围：里程碑/路线图/当前进度/排布原因——状态只在本文件变迁；需求源头见 doc/ledgers/backlog.md
> 关联：doc/baseline/architecture.md §里程碑映射（源头，原 04 迁移）；doc/records/（交付与验收）；doc/ledgers/backlog.md（需求池）
> 更新：2026-08-19（M3 质量硬化收口）

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

## 排布原因

- **M0 风险最高前置**：验证 arrow 集成与 Loss trait 设计是否成立。
- **M1 核心功能面**：D9 类别特征一期。
- **M2 研究向集成**：实验 crate 原型验证后反哺核心（D7），风险隔离。
