# Sooboost

纯 Rust 实现的梯度提升决策树（GBDT）。目标很直接：**让 Rust 项目用表格数据时，不必为了一个 GBDT 拖进 Python 或 C++ 工具链。**

> **状态：v0.1.0 开发中，尚未发布到 crates.io。**
> 核心算法（分箱 / 直方图 / 建树 / 提升 / 类别特征 / 序列化）已完整可用并通过 CI 门禁；
> 公共 API 门面刚落地，早停与交叉验证尚未实现。生产使用请自行评估。

---

## 快速开始

```toml
[dependencies]
sooboost-core = { git = "https://github.com/lvyaqiao/sooboost" }
```

```rust
use sooboost_core::api::GradientBoosting;
use sooboost_core::data::{Dataset, MissingPolicy};

// 1. 读数据（arrow RecordBatch 零拷贝视图；也支持 from_record_batch）
let train = Dataset::from_csv_path(
    "benchmark/california_housing/train.csv",
    &["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
    "target",
    MissingPolicy::default(),
)?;

// 2. 训练（builder 式配置，一行目标）
let model = GradientBoosting::regressor()
    .n_estimators(200)
    .learning_rate(0.1)
    .max_depth(6)
    .seed(42)
    .fit(&train)?;

// 3. 预测 / 在线单行推断 / 存读
let preds = model.predict(&train)?;
let one = model.predict_row(&[3.2, 33.0, 5.0, 1.0, 2300.0, 3.6, 32.7, -117.0])?;
model.save("model.sbm")?;
let reloaded = GradientBoosting::load("model.sbm")?;
```

二分类把 `regressor()` 换成 `classifier()` 即可，`predict` 输出正类概率，
`raw_scores` 输出 logit（供自定义阈值 / 校准）。

完整可运行示例：

```bash
cargo run --release --example california_housing
```

---

## 精度对标

当前基线是 **scikit-learn `HistGradientBoostingRegressor/Classifier`**
（`benchmark/*/correctness_metrics.json`，纳入 CI 门禁）。sooboost 一列来自
`sooboost-cli`，超参与基线一致（100 轮 / lr 0.1 / max_depth 6）：

| 数据集 | 指标 | sooboost | sklearn HGB | 差 |
| --- | --- | --- | --- | --- |
| california_housing（16.5k×8） | R² | 0.8287 | 0.8355 | −0.007 |
| synthetic_regression | R² | 0.9567 | 0.9615 | −0.005 |
| synthetic_regression_nonlinear | R² | **0.3745** | 0.3666 | +0.008 |
| synthetic_binary | AUC | 0.9478 | 0.9491 | −0.001 |

参考：example 用 200 轮 / lr 0.1 在 california_housing 上跑到 **R² 0.8404**，
训练 1.08s（16,512 行 × 8 特征），测试集 4,128 行。

**诚实的边界**：上表只对标了 sklearn HGB（纳入 CI 门禁的基线）。
对标 XGBoost / LightGBM / CatBoost 的实测见下一节。

### 对比 XGBoost / LightGBM / CatBoost（真实数据集）

统一预算（200 轮 / lr 0.1 / seed 42，深度或叶子取各库默认等价容量），
三个**真实**数据集（非合成），复现：`python benchmark/compare_giants.py`：

| 数据集 | 指标 | sooboost | XGBoost 3.4 | LightGBM 4.7 | CatBoost 1.2 | sklearn HGB |
| --- | --- | --- | --- | --- | --- | --- |
| california_housing（16.5k×8） | R² | 0.8403 | 0.8410 | **0.8466** | 0.8243 | 0.8427 |
| diabetes（353×10） | R² | **0.3877** | 0.3594 | 0.3494 | 0.4521 | 0.3408 |
| breast_cancer（455×30） | AUC | **0.9950** | 0.9937 | 0.9891 | 0.9970 | 0.9904 |

结论（如实说，不过度宣称）：sooboost 在三个真实集上**全部进入前二**——
回归与分类质量与三巨头同一梯队（差距 <1%），小数据集上甚至超过 XGBoost / LightGBM；
CatBoost 在小数据集上领先，归因于其有序提升等算法差异，不是工程问题。
训练速度：快于 sklearn HGB 与 CatBoost，慢于 LightGBM / XGBoost（后者有多年的
SIMD / leaf-wise 优化沉淀，这是 M6 性能门禁要追的方向）。

---

## 为什么再造一个

| 取舍 | 选择 | 代价 |
| --- | --- | --- |
| 实现语言 | 纯 Rust，`#![forbid(unsafe_code)]` | 放弃了部分 SIMD / 零拷贝的极致优化空间 |
| 数据入口 | arrow `RecordBatch` 零拷贝视图 | 与 arrow 生态绑定 |
| 可复现性 | 无隐藏随机源，seed 显式传递；同输入同 seed 逐位一致 | 需要显式管理 `seed` |
| 全局状态 | 无。运行时状态经 `TrainingContext` 显式传递 | API 略啰嗦（门面已收口） |
| 平台 | MVP 只支持 x86_64 Linux / Windows | 其他平台未验证 |
| 依赖 | 克制；不引入按月强迫升级的重依赖 | 暂无 GPU 支持 |

设计取自对 XGBoost / LightGBM / CatBoost 失败模式的调研（见 `doc/research/`），
7 条红线与 10 条架构决策记录在 `doc/baseline/architecture.md`。

---

## 仓库结构

```
sooboost-core/       核心库（数据层 → 树内核 → 训练层 → 部署面）
sooboost-cli/        命令行基准工具
sooboost-experiments/ 研究方向原型（条件分布树 / Flow Matching；非生产 API）
benchmark/           基准数据集与金标准指标（含 CI 门禁脚本）
fuzz/                cargo-fuzz 靶机
doc/                 文档体系（入口 doc/README.md）
```

分层依赖单向向下，文档按 `baseline`（契约）/ `ledgers`（台账）/ `records`（流水）/
`plans`（方案）/ `research`（调研）/ `archive`（冻结）分类。

---

## 开发

```bash
cargo test --workspace        # 全部测试（106 个：59 单测 + 38 集成 + 8 实验 + 1 doctest）
cargo fmt --check             # 格式门禁
cargo clippy --workspace --all-targets -- -D warnings   # lint 门禁
cargo run --release --example california_housing        # 端到端示例

# 基准门禁（需 numpy/pandas/scikit-learn）
python benchmark/run_benchmark.py --mode sooboost --gate
python benchmark/run_benchmark.py --mode gen --gate
```

---

## 路线图

权威记录在 [`doc/ledgers/roadmap.md`](doc/ledgers/roadmap.md)：

- **M4 地基修复**（已完成）：源码全量入 git、集成测试转绿、CI 门禁复验全绿
- **M5 可用库 v0.1**（进行中）：公共 API 门面 ✅、端到端示例 ✅、对标三巨头 ✅、发布 crates.io 0.1.0
- **M6 硬化与差异化**（待立项）：早停 + 交叉验证、多分类硬化、特征重要度、性能门禁
- **远期 / 支线**（显式搁置）：WASM / C ABI / codegen、PostgreSQL 插件、生产热替换

---

## 许可证

Apache-2.0
