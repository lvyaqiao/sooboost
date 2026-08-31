# architecture.md 架构设计

> 状态：现行
> 类型：T1 基线（ADR 式）｜写者：维护者｜读者：全员（架构出处）
> 范围：设计红线（7 条）、分层/模块边界、关键设计决策 D1~D10、架构原因、ADR 区段（重构观察与授权）
> 关联：doc/research/GBDT三巨头功能-工程-架构-对错评价.md（调研依据）；doc/ledgers/bugs.md（BUG 族定义）；doc/baseline/contracts.md（契约落地）；doc/baseline/tech.md（选型）
> 更新：2026-08-13（编号制废除，原 03 迁移）

## 设计红线

以下红线是防错机制，全部来源于三巨头调研结论（`doc/research/GBDT三巨头功能-工程-架构-对错评价.md` §4.4 洞察：做对的共性 = 用机制防错；做错的共性 = 复制实现）。任何代码改动不得突破红线；突破即架构变更，须在本文件 §ADR 区段追加观察项并获授权（见 doc/README.md §重构管线）。

1. **单一内核，禁止复制实现**：同一算法（分箱/直方图/分裂搜索/loss）只能有一份实现；并行、SIMD 等变体通过泛型特化组合，绝不复制粘贴。（对治 BUG 族 1：LightGBM treelearner 60% 修复密度、XGBoost updater 副本漂移、CatBoost GPU 双轨）
2. **缺失值/NaN 语义唯一定义点**：在数据层一次性定义（arrow 原生 null 位图 + NaN→missing 可配置），全库唯一来源，禁止各层自行解释。（对治 BUG 族 4：三家 700+ 条 missing/nan 提交）
3. **结果逐位可复现（三层定义）**：分箱 sketch 带种子、并行聚合确定顺序。（对治 BUG 族 3：LightGBM 加权分位数两年两次修复）
   - 层级一（同平台、同编译器、同线程配置）：相同输入 + 相同种子 → 模型与预测逐位一致；
   - 层级二（同平台、不同线程数）：采用确定性归约（固定分块、固定归约树）→ 逐位一致；
   - 层级三（不同 CPU/编译器/优化级别）：允许极小浮点误差，仅要求指标一致，跨平台不做逐位承诺。
   - 注：FMA 等编译器优化会破坏层级一/二，相关路径必须显式控制（禁止或确定化），不得依赖运气。
4. **零全局可变状态**：所有运行时状态经 `TrainingContext` 显式传递；禁止 static mut / 全局配置。（对治 BUG 族 5：LightGBM #4705/#5102 全局线程双 issue）
5. **模型格式从第一天带版本契约**：版本头 + 结构化的模型头 + 树数组 + bin 表 + loss 配置 + 元数据；格式演进必须显式升版本，禁止隐式兼容。（对治 BUG 族 6：CatBoost 零拷贝加载静默损坏）
6. **五类测试入 CI**：确定性测试（逐位）+ 属性测试（proptest）+ 快照回归（canondata 思路）+ fuzz（cargo-fuzz，三家全无，本项目差异化机会）+ 基准门禁（训练/推理 benchmark 回归门禁，LightGBM 无性能门禁的教训）。
7. **`no_unsafe` 默认红线**：除 SIMD 与零拷贝边界（需 `#![deny(unsafe_op_in_unsafe_fn)]` + 逐处审查注释）外，禁止 `unsafe`。

## 最新架构设计

### 分层（依赖单向向下，禁止反向依赖）

```
┌─────────────────────────────────────────────────────┐
│ 部署面：Rust 库 API + 模型热替换（Arc 原子 swap）      │
│   （C ABI / WASM / codegen：远期可选，当前不做）       │
├─────────────────────────────────────────────────────┤
│ 训练层：boosting 循环、采样、早停/CV/回调、objective   │
├─────────────────────────────────────────────────────┤
│ 树内核（单一内核）：binning → 直方图 → 分裂搜索        │
│   泛型特化：串行 / rayon 并行 / SIMD（编译期组合）     │
├─────────────────────────────────────────────────────┤
│ 数据层：arrow RecordBatch 零拷贝视图 + 缺失值语义唯一  │
│   定义点 + bin 表 + 权重/group/类别元数据              │
└─────────────────────────────────────────────────────┘
```

### Workspace（core + cli，无 pyo3；Python 绑定不做或极靠后）

| crate | 职责 |
|---|---|
| `sooboost-core` | 数据层、binning、树内核、loss、boosting、model 序列化与热替换 |
| `sooboost-cli` | 训练 CLI（arrow/parquet 读取，演示与回归测试用） |

### 模块划分（sooboost-core 内）

```
sooboost-core/
├── data/       # RecordBatch 视图、缺失值语义唯一定义、类别/权重/group 元数据
├── binning/    # 确定性 quantile sketch、bin 表（随模型序列化）
├── tree/       # 推理优先树表示（SoA 紧凑数组、叶子支持向量输出预留）
│               # TreeBuilder<HistAlgo, SplitAlgo> 泛型内核
├── loss/       # Loss trait（monomorphized）+ 内置 losses + 注册表
├── boosting/   # GBDT 循环、采样、早停/CV/回调、TrainingContext
├── model/      # 版本化模型格式、JSON/紧凑二进制序列化、Arc 热替换
└── lib.rs      # 公共 API（稳定契约，对标 XGBoost C API 思路）
```

### 关键设计决策 D1~D10

| # | 决策 | 内容 | 依据 |
|---|---|---|---|
| D1 | 不定义自己的 DMatrix | `Dataset` 直接封装 arrow `RecordBatch`，零拷贝读 Parquet/Polars/DuckDB；类别列用 `DictionaryArray`；缺失值统一映射 arrow null 位图 | doc/baseline/product.md 核心卖点；族 4 |
| D2 | 单一内核多后端 | `TreeBuilder<HistAlgo, SplitAlgo>` 编译期组合 exact/quantile-hist/approx × 串行/rayon/SIMD，零运行时工厂 | 族 1；LightGBM 工厂硬编码教训 |
| D3 | Loss 用 monomorphized trait | 训练内核泛型于 `Loss` → 编译期内联零回调；用户 crate 实现 trait 即插件；内置 losses 走注册表 | doc/baseline/product.md；XGBoost 31 处注册宏长处 |
| D4 | 分箱确定性 | sketch 带种子、并行聚合排序后分位数；bin 表序列化进模型，训练/预测共用 | 族 3 |
| D5 | Context 显式贯穿 | `TrainingContext { rng, threads, verbosity, … }` 显式传入，无全局态；Send/Sync 编译期防 data race | 族 5；XGBoost Context 147 处贯穿长处 |
| D6 | 模型格式版本化 | 版本头 + 模型头 + 树数组 + bin 表 + loss 配置 + 元数据；JSON（调试）+ 紧凑二进制（mmap 热加载）双格式 | 族 6；XGBoost 三格式长处 |
| D7 | 推理优先树表示 | 树存 SoA 紧凑数组（thresholds/子索引/叶子值连续），缓存友好；**核心仅支持标量叶子**；研究方向（条件分布树/Flow Matching）放独立实验 crate，原型验证后再反哺核心 trait | 避免研究方向提前污染核心设计 |
| D8 | 目标函数排布（分期） | M0：L2 回归 + binary logloss；M1：回归(L1/quantile/huber/poisson/gamma/tweedie)+多分类+排序；二期：生存(cox/aft)+多输出 | doc/research/GBDT三巨头功能-工程-架构-对错评价.md §1.1；LightGBM 面为基线 |
| D9 | 类别特征一期实现 | ordered target statistics（仿 CatBoost）；**明确不做 EFB/GOSS**（微软专利规避 + 纯 CPU 收益有限） | doc/research/GBDT三巨头功能-工程-架构-对错评价.md §1.2；专利风险 |
| D10 | 分布式二期 | 预留 `aggregator` 接口位，不搞 rabit 式全局通信层 | doc/research/GBDT三巨头功能-工程-架构-对错评价.md §3；XGBoost collective 独立层思路 |

### 里程碑映射

| 里程碑 | 内容 | 排布原因 |
|---|---|---|
| M0 | 九项承诺（一页规格见 doc/archive/m0-spec.md）：arrow RecordBatch 数据层 + 数值特征 + null/NaN 缺失值 + L2 回归 + binary logloss + quantile binning + 单线程直方图 + 基本树参数与预测 + 固定 seed 确定性测试 | 风险最高前置：验证 arrow 集成与 Loss trait 设计是否成立；刻意排除类别/权重/排序/并行/SIMD/自定义 Loss/热替换 |
| M1 | rayon 并行 + D8 其余目标函数 + 类别特征(ordered TS，契约见 doc/baseline/contracts.md §1.4) + 序列化/热替换 + 基准门禁 | 核心功能面；D9 类别特征一期 |
| M2 | 研究方向集成：条件分布树（BaltoBot/UnmaskingTrees）/ Flow Matching 向量场 | D7 已留位，风险隔离 |
| M3 | ForestFlow 生成/填补质量硬化（经验边际变换、采样积分、点填补评估） | 不改变核心 API，继续实验隔离 |

## 架构设计的原因

- **不做 EFB/GOSS**：微软专利风险（规避"轻度魔改 XGBOOST"式的法律雷区）；且纯 CPU 下 EFB 收益有限、复杂度高，与"工程做扎实"目标冲突。
- **无 pyo3**：doc/baseline/product.md 的核心诉求是"Rust 生态内闭环、不引入 Python 摩擦"；绑定层是三巨头最大的维护负担源（族 7，LightGBM API 兼容修复占比 31%），跳过即避开最大压力源。
- **部署面仅 Rust API**：当前排期聚焦训练与模型库能力；C ABI/WASM/codegen 是远期可选项，不影响核心架构（序列化格式先行设计，后续可直接导出）。
- **不搞自己的 DMatrix**：arrow 已是表格数据的事实标准底座；自造 DMatrix = 复制 XGBoost 的数据层 bug 面（稀疏格式、外存、索引一致性），且破坏零拷贝卖点。
- **泛型特化取代工厂**：LightGBM 的 `CreateTreeLearner` 硬编码 10+ 分支是族 1 的结构性根源；Rust 泛型让"新增变体"= 新增 trait 实现，不修改既有代码——这正是"加机制而非加实现"（doc/research/GBDT三巨头功能-工程-架构-对错评价.md §4.4）。
- **测试五件套**：三巨头共同短板（无 fuzz、无覆盖率、LightGBM 无性能门禁）就是本项目的差异化空间；"轻实现、重验证"（工程调研结论）。
- **红线即防错机制**：七条红线全部对应调研中"做对的共性"（稳定契约、schema、插件注册、快照回归）——用机制替代人肉，消灭打地鼠。


## ADR 区段（重构观察与授权）

> 重构记录机制（2026-08-13 起，替代原 06-架构重构依据编号文档，历史见 doc/archive/06-架构重构依据.md）：

1. **观察项**：谁想重构谁追加——### ADR-<序号> <主题> + 动机/方案/影响面/状态（观察中→已授权→已收口）
2. **授权**：触碰红线/契约/模型格式须维护者授权（见 doc/README.md §重构管线 ②）
3. **收口**：重构完成后状态标 ✅ 并指向 doc/records/ 交付条目与 doc/ledgers/roadmap.md 相关行

（暂无观察项）
