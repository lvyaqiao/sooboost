# product.md 产品需求与边界

> 状态：现行
> 类型：T1 基线｜写者：维护者｜读者：全员
> 范围：产品愿景、需求与目标、验收标准与边界（由「想做的.md」与「01-需求与目标.md」合并而来）
> 关联：doc/ledgers/backlog.md（需求池）；doc/baseline/tech.md（技术选型）；doc/baseline/architecture.md（架构）
> 更新：2026-08-13

## 需求与目标

本软件需要具备的功能及其分析，是对「想做的.md」的软件工程翻译，以及讨论的各种相关内容。翻译进行中——功能面正式立项见 doc/ledgers/backlog.md，架构与里程碑见 doc/baseline/architecture.md。

## 愿景（想做的.md 原文）

我是重度 rust 用户。

有时候，写代码会用到 xgboost，因为 GBDT 处理表格数据哪怕现在也还是比神经网络好用。

各种「表格神经网络」，说实话是有点华而不实的。

而各种「轻度魔改 XGBOOST」，很多时候又有问题，也不好用，没有重工业的投入与优化，科研玩具不可用。

但是 rust 的生态在这个地方实在是太薄弱了。

我引入一个 python，训练完之后把模型放进去其实摩擦很大。

用 C++ 的编译工具链的问题就更大了，很令人头大。

此外，我觉得 rust 的语言特性是适合做这些事情的。

还有 C++ 依赖 gcc / clang 的自动向量化（Auto-vectorization）。但因为直方图构建涉及非连续内存写（Gather/Scatter），自动向量化极易失效，往往需要手写复杂的 AVX-512 / NEON 汇编或 Intrinsics。

并且用 Rust 训练好模型后，可以把模型转译为一个几百 KB、无任何依赖的超小型二进制库（甚至是纯 C ABI 头文件），或者直接编译为 WASM。在智能手机、网关设备、浏览器前端、甚至是 SQL 数据库插件（如 PostgreSQL Extension）内部直接毫秒级运行 GBDT 推理。甚至高并发线上服务可以做到零延迟、零锁竞争地在运行时热替换新的 GBDT 模型。

还有一些新的理念，例如 NeurIPS 提出的 UnmaskingTrees / BaltoBot 思想。它不是用神经网络去生成数据，而是把 GBDT 改造为非参数的条件概率分布树。

还有将 GBDT 作为底层弱学习器，用 Boosting 树去拟合 Flow Matching 的向量场（Vector Field）。

不需要 GPU，仅用 CPU 和极致优化的 Rust GBDT 就能实现毫秒级的表格数据合成与高质量缺失值填补（Imputation）。

因此，我想做一个好用的 rust 梯度提升树。

不搞新的学术发明，但是学术界最新的，真正的好的思想一定要吸收拿来用。

并且最主要的是把工程给做扎实了。

例如，不要再定义自己的 DMatrix。直接基于 arrow-rs，让 Rust GBDT 能够直接零拷贝（Zero-Copy）读取 Parquet、Polars DataFrame 或者 DuckDB 内存数据。

插件化可替换 Loss（Monomorphized Traits）：允许用户用 Rust 写几行代码定义 Loss，编译期通过泛型展开内联（Inlining），做到零回调开销。

## 验收标准与边界

### M0 验收标准（一页规格见 doc/archive/m0-spec.md）

功能面（九项承诺，缺一不可）：
1. arrow `RecordBatch` 为唯一数据入参（数值列）；
2. 数值特征（无类别/无权重/无 group）；
3. null/NaN 缺失值语义唯一定义并生效（红线 2）；
4. L2 回归（squared error）端到端可训练、可预测；
5. binary logloss 端到端可训练、可预测；
6. quantile binning 确定性分箱（带 seed，红线 3 层级一）；
7. 单线程直方图 + 分裂搜索（无并行/SIMD）；
8. 基本树参数（max_depth / learning_rate / n_estimators / min_samples_leaf 等）与预测接口；
9. 固定 seed 确定性测试：相同输入 + 相同种子 → 逐位一致。

质量门（可测量）：
- 在 benchmark 质量基准（RandomForest 通用基线 + HistGradientBoosting GBDT 基线，见 benchmark/）上，相同数据与预算下达目标指标——M0 目标为**持平 RandomForest、接近 HistGradientBoosting**（"全面超过 RandomForest"不作为合同承诺）。

边界（M0 明确不做）：
- 类别特征、权重、group、排序目标（M1 及之后）；
- rayon 并行、SIMD（M1）；
- 自定义 Loss 运行时注册（M0 仅内置 L2 / binary logloss）；
- 模型序列化/热替换/模型格式（M1）；
- 多分类、L1/quantile/huber/poisson/gamma/tweedie、生存、多输出（M1/二期）。
