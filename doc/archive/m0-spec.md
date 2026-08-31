# m0-spec.md M0 一页可执行规格

> 状态：现行
> 类型：T1 变体（方案 = 契约候选）｜写者：维护者｜读者：M0 实施人（开工前必读）
> 范围：M0 的九项承诺、数据契约、树与分裂参数、验收标准、测试面、明确不做清单
> 关联：doc/baseline/architecture.md §红线/D1-D8；doc/baseline/contracts.md §1.1-1.3；benchmark/；doc/ledgers/roadmap.md
> 更新：2026-08-18（M0 范围收缩决议落地）

## 1. 目标

用最小闭环验证三条架构假设是否成立：
1. arrow `RecordBatch` 作为数据底座可用（红线 2 缺失值语义唯一定义点）；
2. `Loss` monomorphized trait 设计成立（D3）；
3. quantile binning 确定性可复现（D4，红线 3 层级一）。

## 2. 九项承诺（缺一不可）

| # | 承诺 | 验收方式 |
|---|---|---|
| 1 | arrow `RecordBatch` 唯一数据入参（数值列） | 集成测试：从 CSV → RecordBatch → 训练 |
| 2 | 数值特征（无类别/权重/group） | — |
| 3 | null/NaN 缺失值语义唯一定义并生效 | 契约测试：缺失样本走指定子树，NaN 策略可配置且全库一致 |
| 4 | L2 回归端到端 | benchmark correctness 对齐（predictions.csv 对比） |
| 5 | binary logloss 端到端 | 同上 |
| 6 | quantile binning 确定性（带 seed） | 确定性测试：同输入同种子 → bin 表逐位一致 |
| 7 | 单线程直方图 + 分裂搜索（无并行/SIMD） | 与基准门禁无关；单线程路径断言 |
| 8 | 基本树参数 + 预测接口 | API 契约测试：max_depth / learning_rate / n_estimators / min_samples_leaf |
| 9 | 固定 seed 确定性测试入 CI | 红线 3 层级一：模型与预测逐位一致 |

## 3. 数据契约（contracts.md §1.1 落地）

- 入参：arrow `RecordBatch`（数值列）；首版从 CSV 构建（benchmark/ 数据），parquet 零拷贝进入为可测量目标。
- 缺失值：arrow null 位图 = 缺失；NaN 视为缺失与否由训练配置决定，确定后全库一致（红线 2）。
- schema 校验：列名/类型不一致显式报错，禁止静默降级。

## 4. 树与分裂参数（首版固定）

- 树：binary 树，逐层生长（level-wise）或逐叶生长（leaf-wise）——M0 选 level-wise（实现最简，直方图复用友好）。
- 分裂：单线程直方图遍历（每特征扫描 bin 边界），无并行无 SIMD。
- 参数：`max_depth`（默认 6）、`learning_rate`（默认 0.1）、`n_estimators`（默认 100）、`min_samples_leaf`（默认 5）、`min_split_gain`（默认 0.0）、`subsample` 与特征采样 M0 不做。
- 叶子值：L2 → 残差均值；binary logloss → log-odds（一阶牛顿步近似，与 GBDT 惯例一致）。
- 初始化：L2 → 全局均值；binary logloss → log(p/(1-p))。

## 5. 验收标准

功能面 = 第二节九项承诺全绿。

质量门（product.md 验收标准落地）：
- benchmark quality 档：同数据同预算持平 RandomForest、接近 HistGradientBoosting（指标见 benchmark/<dataset>/metrics.json）。
- benchmark correctness 档：预测曲线与 HistGradientBoosting 定性一致（非逐位——算法实现不同，不作逐位对齐承诺）。

## 6. 测试面（入 CI）

| 类别 | 内容 |
|---|---|
| 确定性 | 同输入同种子 → bin 表/模型/预测逐位一致（层级一） |
| 属性测试 | 直方图 bin 单调性（bin 边界单调递增）；树深度 ≤ max_depth |
| 集成 | CSV → RecordBatch → 训练 → 预测 回路 |
| 对齐 | 与 benchmark correctness 输出对比（指标与预测分布） |

## 7. 明确不做（M0 边界）

- 类别特征（M1，契约见 contracts.md §1.4）、权重、group、排序
- rayon 并行、SIMD（M1）
- 自定义 Loss 运行时注册（仅内置 L2 / binary logloss）
- 模型序列化/热替换/模型格式（M1）
- 多分类、L1/quantile/huber/poisson/gamma/tweedie、生存、多输出
- fuzz（首版无外部字节流输入，M1 加模型解析时引入）

## 8. 收口条件

九项承诺全绿 + 质量门达标 + 确定性测试入 CI → 状态标 ✅，方案文件归档 doc/archive/，M1 立项。
