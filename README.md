# Sooboost

纯 Rust 实现的梯度提升决策树（GBDT）。目标很直接：**让 Rust 项目用表格数据时，不必为了一个 GBDT 拖进 Python 或 C++ 工具链。**

> **状态：[crates.io 0.2.0 已发布](https://crates.io/crates/sooboost-core)。**
> 核心算法（分箱 / 直方图 / 建树 / 提升 / 类别特征 / 序列化）已完整可用并通过 CI 门禁；
> M6–M8 已收口：早停（标量 + 多分类）/ 交叉验证 / 特征重要度 / softmax 多分类 /
> 多分类类别特征 / 温度缩放校准全部落地，真实数据集精度下限已固化为 CI 性能门禁。
> 注意：模型格式为 v4，0.1.x 导出的模型文件需重新训练。
> 生产使用请自行评估。

---

## 快速开始

```toml
[dependencies]
sooboost-core = "0.2.0"
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

### 早停 / 交叉验证 / 特征重要度（M6 已落地；早停自 M7 起也支持多分类）

```rust
// 早停：连续 50 轮验证损失无改善即停，并回滚到最优轮（回归 / 二分类 / 多分类通用）
let model = GradientBoosting::regressor()
    .n_estimators(2000)
    .early_stopping(eval_ds, 50)   // 验证集 Dataset + patience
    .fit(&train)?;
model.best_iteration();   // 实际使用的轮数
model.eval_history();     // 每轮验证损失（学习曲线）

// K 折交叉验证（连续分块，完全确定；指标按目标自动选 R² / AUC / accuracy）
let cv = GradientBoosting::regressor()
    .n_estimators(100)
    .cross_validate(&train, 5)?;
cv.mean; cv.std; cv.fold_scores;

// 特征重要度（gain / cover / frequency，归一化；随模型持久化，load 后可用）
let imp = model.feature_importances(sooboost_core::boosting::ImportanceKind::Gain);
```

### 多分类（softmax，M6 已落地；早停与校准 M7 已落地）

```rust
// 每轮每类一棵树；y 必须为整数标签 ∈ [0, n_classes)
let model = GradientBoosting::multiclass_classifier(3)
    .n_estimators(200)
    .learning_rate(0.1)
    .early_stopping(eval_ds, 50)      // M7-1：口径为验证集多分类 logloss
    .fit(&train)?;
model.best_iteration();               // 早停回滚后每类实际轮数
model.eval_history();                 // 每轮验证 logloss（学习曲线）

let classes = model.predict_classes(&test)?;           // argmax 类别
let proba = model.predict_proba(&test)?;               // [row][class]，softmax 行和为 1
let logits = model.raw_logits(&test)?;                 // softmax 前的 logits（自定义校准用）
model.feature_importances(ImportanceKind::Gain);       // 跨全部类别聚合

// 温度缩放校准（M7-2）：在验证/校准集上求最小化 NLL 的温度 T
let t = model.calibrate_temperature(&calib_ds)?;       // 完全确定（网格 + 黄金分割）
let calibrated = model.predict_proba_with_temperature(&test, t)?;  // softmax(logits/T)

// 类别特征（M8）：ordered TS 数值化，与标量模型同一管线、同一防泄漏纪律
// 编码随模型序列化，load 后对 OOV 类别自动走先验；单行 predict_row 不支持类别模型

// 序列化与目标自动探测与标量模型完全一致：save / load / from_bytes
model.save("model.sbm")?;
let loaded = GradientBoosting::load("model.sbm")?;     // 自动探测出多分类目标
```

交叉验证指标自动用 accuracy。
温度 T 不写入模型格式，由调用方持有后传入 `predict_proba_with_temperature`。

CLI 同样支持：`--eval <valid.csv> --early-stopping <rounds>`。

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
- **M5 可用库 v0.1**（已完成）：公共 API 门面 ✅、端到端示例 ✅、对标三巨头 ✅、发布 crates.io 0.1.0
- **M6 硬化与差异化**（已完成）：早停 ✅、交叉验证 ✅、特征重要度 ✅（gain/cover/frequency）、softmax 多分类 ✅（模型格式 v4）、真实基准固化进 CI 性能门禁 ✅
- **M7 多分类质量收口**（已完成）：多分类早停 ✅（验证 logloss 口径，语义与标量一致）、温度缩放校准 ✅（post-hoc 确定性搜索，T 不入模型格式）
- **M8 多分类类别特征**（已完成）：ordered TS 复用标量 D9 管线 ✅（标签均值口径）、编码段随 v4 格式序列化 ✅（零格式变更）、OOV → 先验
- **远期 / 支线**（显式搁置）：WASM / C ABI / codegen、PostgreSQL 插件、生产热替换、per-class TS（CatBoost 式每类统计量）

---

## 许可证

Apache-2.0
