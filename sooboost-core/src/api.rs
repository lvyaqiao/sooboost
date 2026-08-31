//! 公共 API 门面（M5：可用库 v0.1）。
//!
//! 门面的职责是**收口**，不是替代：底层 [`crate::boosting::fit`] / `Booster<L>` /
//! `Dataset` 仍是一等公民（架构 D3 的 monomorphized `Loss` 保持不变），门面只把
//! 「选目标 → 配参数 → 传训练上下文」压成一层，并把数据/训练/模型三套错误收敛为
//! 单一 [`Error`]。
//!
//! 典型用法：
//!
//! ```no_run
//! use sooboost_core::api::GradientBoosting;
//! use sooboost_core::data::{Dataset, MissingPolicy};
//!
//! let train = Dataset::from_csv_path(
//!     "benchmark/california_housing/train.csv",
//!     &["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
//!     "target",
//!     MissingPolicy::default(),
//! )?;
//! let model = GradientBoosting::regressor()
//!     .n_estimators(200)
//!     .learning_rate(0.1)
//!     .fit(&train)?;
//! let preds = model.predict(&train)?;
//! model.save("model.sbm")?;
//! # Ok::<(), sooboost_core::api::Error>(())
//! ```
//!
//! 设计约束（与红线一致）：
//! - 红线 4：无全局状态，seed 经 [`Config::seed`] 显式注入 `TrainingContext`；
//! - 红线 3：同数据同配置同 seed → 模型与预测逐位一致（门面不引入任何随机源）；
//! - 红线 6：读外部字节（`from_bytes` / `load`）的失败一律显式返回，不静默降级；
//! - 红线 2：缺失语义仍由 `data::missing` 单点定义，门面只转发策略。

use std::path::Path;

use crate::boosting::booster::Booster;
use crate::boosting::{BoostingError, BoostingParams, TrainingContext, fit};
use crate::data::missing::is_missing_value;
use crate::data::{DataError, Dataset, MissingPolicy};
use crate::loss::{BinaryLogLoss, Loss, SquaredError};
use crate::model::ModelError;
use crate::tree::TreeParams;

/// sooboost 统一错误类型。
///
/// 门面把数据层 [`DataError`] / 训练层 [`BoostingError`] / 模型层 [`ModelError`]
/// 收敛为一类，调用方只需 match 一个枚举；原始错误保留在变体内，`source()` 可下钻。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// 数据层错误（schema 校验 / CSV 解析 / 缺失语义）。
    #[error("数据错误: {0}")]
    Data(#[from] DataError),
    /// 训练层错误（分箱 / 建树 / 类别基数超限）。
    #[error("训练错误: {0}")]
    Boosting(#[from] BoostingError),
    /// 模型层错误（magic / 版本 / checksum / 结构 / 损失名）。
    #[error("模型错误: {0}")]
    Model(#[from] ModelError),
    /// 文件系统 IO 错误（`save` / `load`）。
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    /// 参数非法（在 `fit` 入口校验，避免静默产出无意义模型）。
    #[error("参数非法: {field} = {value}（{reason}）")]
    InvalidParam {
        /// 字段名。
        field: &'static str,
        /// 实际值（已格式化）。
        value: String,
        /// 为什么非法。
        reason: &'static str,
    },
    /// 单行预测的特征数与模型不符。
    #[error("特征数不匹配：模型期望 {expected}，实际 {got}")]
    FeatureCountMismatch {
        /// 模型训练时的特征数。
        expected: usize,
        /// 传入的特征数。
        got: usize,
    },
    /// 含类别特征的模型不支持单行预测。
    ///
    /// 类别值必须经训练期编码解析（OOV → 先验），`&[f64]` 无法承载该语义；
    /// 与其静默给出错误结果，不如显式报错（易踩坑 5 的纪律）。
    #[error("单行预测不支持含类别特征的模型：类别需经编码解析，请用 predict(&Dataset)")]
    RowPredictUnsupportedWithCategorical,
}

/// 训练目标。
///
/// 门面层的枚举；底层仍是 monomorphized `Loss`（架构 D3），编译期内联不变。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Objective {
    /// 回归：平方误差（L2）。`predict` 输出原始分数（恒等 transform）。
    #[default]
    SquaredError,
    /// 二分类：对数损失。`predict` 输出正类概率（sigmoid transform）。
    BinaryLogLoss,
}

impl Objective {
    /// 目标名称（与模型头中的损失名一致，contracts §1.2）。
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Objective::SquaredError => SquaredError.name(),
            Objective::BinaryLogLoss => BinaryLogLoss.name(),
        }
    }
}

/// 训练配置（门面层扁平参数，映射到 `BoostingParams` + `TreeParams`）。
///
/// 之所以扁平化：用户不该为了设 `max_depth` 先知道参数分属分箱契约还是分裂契约。
#[derive(Debug, Clone)]
pub struct Config {
    /// 提升轮数（决策树棵数）。
    pub n_estimators: usize,
    /// 学习率（每棵树预测累加的缩放）。
    pub learning_rate: f64,
    /// 树最大深度（根 = 0）。
    pub max_depth: usize,
    /// 叶子最少样本数。
    pub min_samples_leaf: usize,
    /// 最小分裂增益（不超过则不分裂）。
    pub min_split_gain: f64,
    /// L2 正则 λ。
    pub reg_lambda: f64,
    /// 分箱数量上限（数据层契约，binning::DEFAULT_MAX_BINS）。
    pub max_bins: usize,
    /// 类别特征基数上限（超限报错而非静默截断，contracts §1.4）。
    pub max_categories: usize,
    /// ordered TS smoothing α（contracts §1.4）。
    pub categorical_alpha: f64,
    /// 确定性随机种子（红线 3/红线 4：显式传入，无全局随机源）。
    pub seed: u64,
    /// 缺失值策略（红线 2 单点定义）。
    pub missing_policy: MissingPolicy,
}

impl Default for Config {
    fn default() -> Self {
        let p = BoostingParams::default();
        let t = p.tree_params;
        Self {
            n_estimators: p.n_estimators,
            learning_rate: p.learning_rate,
            max_depth: t.max_depth,
            min_samples_leaf: t.min_samples_leaf,
            min_split_gain: t.min_split_gain,
            reg_lambda: t.reg_lambda,
            max_bins: p.max_bins,
            max_categories: p.max_categories,
            categorical_alpha: p.categorical_alpha,
            seed: 0,
            missing_policy: MissingPolicy::default(),
        }
    }
}

impl Config {
    /// 入口校验：非法参数在 `fit` 前显式报错（红线 6，不静默降级）。
    fn validate(&self) -> Result<(), Error> {
        if self.n_estimators == 0 {
            return Err(Error::InvalidParam {
                field: "n_estimators",
                value: self.n_estimators.to_string(),
                reason: "至少 1 轮",
            });
        }
        // 显式写 is_finite + 正比较：NaN 既不是 >0 也不是非法比较的结果，必须被拦下
        // （易踩坑 5：这类错误不崩溃、只静默产出错误模型）。
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidParam {
                field: "learning_rate",
                value: self.learning_rate.to_string(),
                reason: "必须为正的有限值",
            });
        }
        if self.min_samples_leaf == 0 {
            return Err(Error::InvalidParam {
                field: "min_samples_leaf",
                value: self.min_samples_leaf.to_string(),
                reason: "至少 1",
            });
        }
        if self.max_bins < 2 {
            return Err(Error::InvalidParam {
                field: "max_bins",
                value: self.max_bins.to_string(),
                reason: "至少 2（1 个分箱边界需要 2 个 bin）",
            });
        }
        if self.max_categories == 0 {
            return Err(Error::InvalidParam {
                field: "max_categories",
                value: self.max_categories.to_string(),
                reason: "至少 1",
            });
        }
        if !self.categorical_alpha.is_finite() || self.categorical_alpha < 0.0 {
            return Err(Error::InvalidParam {
                field: "categorical_alpha",
                value: self.categorical_alpha.to_string(),
                reason: "必须非负且有限",
            });
        }
        Ok(())
    }

    /// 映射到底层参数（分箱参数与树参数分离，D4）。
    fn boosting_params(&self) -> BoostingParams {
        BoostingParams {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_bins: self.max_bins,
            tree_params: TreeParams {
                max_depth: self.max_depth,
                min_samples_leaf: self.min_samples_leaf,
                min_split_gain: self.min_split_gain,
                reg_lambda: self.reg_lambda,
            },
            max_categories: self.max_categories,
            categorical_alpha: self.categorical_alpha,
        }
    }
}

/// 训练完成的模型（门面产物）。
///
/// 内部按目标持有单态 `Booster`，预测时无虚派发；对外表现为一个类型。
#[derive(Debug)]
pub struct GradientBoosting {
    objective: Objective,
    config: Config,
    fitted: Fitted,
}

#[derive(Debug)]
enum Fitted {
    Regression(Booster<SquaredError>),
    Binary(Booster<BinaryLogLoss>),
}

impl GradientBoosting {
    /// 回归（平方误差）构造器。
    #[must_use]
    pub fn regressor() -> GradientBoostingBuilder {
        GradientBoostingBuilder::new(Objective::SquaredError)
    }

    /// 二分类（对数损失）构造器。
    #[must_use]
    pub fn classifier() -> GradientBoostingBuilder {
        GradientBoostingBuilder::new(Objective::BinaryLogLoss)
    }

    /// 训练目标。
    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// 树棵数。
    #[must_use]
    pub fn num_trees(&self) -> usize {
        match &self.fitted {
            Fitted::Regression(b) => b.num_trees(),
            Fitted::Binary(b) => b.num_trees(),
        }
    }

    /// 特征数（由模型自带分箱表确定）。
    #[must_use]
    pub fn num_features(&self) -> usize {
        match &self.fitted {
            Fitted::Regression(b) => b.bin_table().num_features(),
            Fitted::Binary(b) => b.bin_table().num_features(),
        }
    }

    /// 训练配置。
    ///
    /// 注意：`fit` 产出的模型为完整配置；由 [`Self::load`] / [`Self::from_bytes`]
    /// 载入的模型**仅回填持久化字段**（轮数/学习率/分箱数），树结构参数与 seed
    /// 不属于模型格式（contracts §1.2），取默认值。预测不受影响。
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// 最终预测：回归 → 原值；二分类 → 正类概率。
    pub fn predict(&self, ds: &Dataset) -> Result<Vec<f64>, Error> {
        Ok(match &self.fitted {
            Fitted::Regression(b) => b.predict(ds)?,
            Fitted::Binary(b) => b.predict(ds)?,
        })
    }

    /// 原始分数（`init + Σ lr·tree`，未经 `transform`）。
    ///
    /// 二分类下即 logit，供自定义阈值/校准使用。
    pub fn raw_scores(&self, ds: &Dataset) -> Result<Vec<f64>, Error> {
        Ok(match &self.fitted {
            Fitted::Regression(b) => b.raw_scores(ds)?,
            Fitted::Binary(b) => b.raw_scores(ds)?,
        })
    }

    /// 单行预测（在线推断路径，无需构造 `Dataset`）。
    ///
    /// 缺失以 `f64::NAN` 表示，按 [`Config::missing_policy`] 解释
    /// （红线 2：语义仍由 `data::missing` 单点定义）。
    pub fn predict_row(&self, values: &[f64]) -> Result<f64, Error> {
        if values.len() != self.num_features() {
            return Err(Error::FeatureCountMismatch {
                expected: self.num_features(),
                got: values.len(),
            });
        }
        // 类别特征必须经训练期编码解析，`&[f64]` 承载不了 → 显式报错而非错算。
        let has_categorical = match &self.fitted {
            Fitted::Regression(b) => b.categorical_encoding().is_some(),
            Fitted::Binary(b) => b.categorical_encoding().is_some(),
        };
        if has_categorical {
            return Err(Error::RowPredictUnsupportedWithCategorical);
        }
        let policy = self.config.missing_policy;
        let is_missing: Vec<bool> = values
            .iter()
            .map(|&v| is_missing_value(v, v.is_nan(), policy))
            .collect();
        Ok(match &self.fitted {
            Fitted::Regression(b) => b.predict_row(values, &is_missing),
            Fitted::Binary(b) => b.predict_row(values, &is_missing),
        })
    }

    /// 序列化为字节（contracts §1.2 显式布局 + checksum）。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.fitted {
            Fitted::Regression(b) => b.serialize(),
            Fitted::Binary(b) => b.serialize(),
        }
    }

    /// 由字节恢复模型，目标自动探测。
    ///
    /// 安全依据：contracts §1.2 的校验顺序是 magic → 版本 → checksum → 结构 →
    /// 损失名，因此只有「字节本身合法、仅目标不同」才会落到 `LossMismatch`；
    /// 截断 / checksum 失败等一律原样上抛（红线 6）。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        match Booster::deserialize(bytes, SquaredError) {
            Ok(b) => {
                return Ok(Self::from_fitted(
                    Objective::SquaredError,
                    Fitted::Regression(b),
                ));
            }
            Err(ModelError::LossMismatch { .. }) => {}
            Err(e) => return Err(Error::Model(e)),
        }
        let b = Booster::deserialize(bytes, BinaryLogLoss)?;
        Ok(Self::from_fitted(
            Objective::BinaryLogLoss,
            Fitted::Binary(b),
        ))
    }

    /// 保存到文件。
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    /// 从文件加载（同 [`Self::from_bytes`]，目标自动探测）。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    fn from_fitted(objective: Objective, fitted: Fitted) -> Self {
        let config = match &fitted {
            Fitted::Regression(b) => config_from_booster(b),
            Fitted::Binary(b) => config_from_booster(b),
        };
        Self {
            objective,
            config,
            fitted,
        }
    }
}

/// 由模型自身可观测字段回填配置（学习率/轮数/分箱数持久化，其余取默认）。
fn config_from_booster<L: Loss>(b: &Booster<L>) -> Config {
    Config {
        n_estimators: b.num_trees(),
        learning_rate: b.learning_rate(),
        max_bins: b.bin_table().max_bins(),
        ..Config::default()
    }
}

/// [`GradientBoosting`] 的 builder。
#[derive(Debug, Clone)]
pub struct GradientBoostingBuilder {
    objective: Objective,
    config: Config,
}

impl GradientBoostingBuilder {
    #[must_use]
    fn new(objective: Objective) -> Self {
        Self {
            objective,
            config: Config::default(),
        }
    }

    /// 提升轮数。
    #[must_use]
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.config.n_estimators = n;
        self
    }

    /// 学习率。
    #[must_use]
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// 树最大深度（根 = 0）。
    #[must_use]
    pub fn max_depth(mut self, d: usize) -> Self {
        self.config.max_depth = d;
        self
    }

    /// 叶子最少样本数。
    #[must_use]
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.config.min_samples_leaf = n;
        self
    }

    /// 最小分裂增益。
    #[must_use]
    pub fn min_split_gain(mut self, g: f64) -> Self {
        self.config.min_split_gain = g;
        self
    }

    /// L2 正则 λ。
    #[must_use]
    pub fn reg_lambda(mut self, l: f64) -> Self {
        self.config.reg_lambda = l;
        self
    }

    /// 分箱数量上限。
    #[must_use]
    pub fn max_bins(mut self, n: usize) -> Self {
        self.config.max_bins = n;
        self
    }

    /// 类别特征基数上限。
    #[must_use]
    pub fn max_categories(mut self, n: usize) -> Self {
        self.config.max_categories = n;
        self
    }

    /// ordered TS smoothing α。
    #[must_use]
    pub fn categorical_alpha(mut self, a: f64) -> Self {
        self.config.categorical_alpha = a;
        self
    }

    /// 确定性随机种子（红线 3/红线 4）。
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// 缺失值策略。
    #[must_use]
    pub fn missing_policy(mut self, policy: MissingPolicy) -> Self {
        self.config.missing_policy = policy;
        self
    }

    /// 训练。
    pub fn fit(self, ds: &Dataset) -> Result<GradientBoosting, Error> {
        self.config.validate()?;
        let params = self.config.boosting_params();
        let ctx = TrainingContext::new(self.config.seed);
        let fitted = match self.objective {
            Objective::SquaredError => Fitted::Regression(fit(ds, &params, SquaredError, &ctx)?),
            Objective::BinaryLogLoss => Fitted::Binary(fit(ds, &params, BinaryLogLoss, &ctx)?),
        };
        Ok(GradientBoosting::from_fitted(self.objective, fitted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn dataset(x: Vec<f64>, y: Vec<f64>) -> Dataset {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(x)),
                Arc::new(Float64Array::from(y)),
            ],
        )
        .expect("构造测试 batch");
        Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .expect("构造 Dataset")
    }

    /// y = 2x 的训练集（100 点，无噪声 → GBDT 应能逼近）。
    fn linear_data() -> Dataset {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        dataset(x, y)
    }

    #[test]
    fn default_config_matches_underlying_defaults() {
        let c = Config::default();
        let p = BoostingParams::default();
        assert_eq!(c.n_estimators, p.n_estimators);
        assert_eq!(c.learning_rate, p.learning_rate);
        assert_eq!(c.max_bins, p.max_bins);
        assert_eq!(c.max_depth, p.tree_params.max_depth);
        assert_eq!(c.min_samples_leaf, p.tree_params.min_samples_leaf);
        assert_eq!(c.seed, 0, "默认 seed 必须显式为 0（红线 4 无隐藏状态）");
    }

    #[test]
    fn regressor_fits_linear_function() {
        let ds = linear_data();
        let model = GradientBoosting::regressor()
            .n_estimators(60)
            .learning_rate(0.2)
            .fit(&ds)
            .expect("训练成功");
        assert_eq!(model.objective(), Objective::SquaredError);
        assert_eq!(model.num_trees(), 60);
        assert_eq!(model.num_features(), 1);

        let preds = model.predict(&ds).expect("预测成功");
        assert_eq!(preds.len(), 100);
        // 端点处允许偏差，中段应相当贴近 y = 2x。
        for (i, &pred) in preds.iter().enumerate().take(80).skip(20) {
            let expected = 2.0 * i as f64;
            assert!(
                (pred - expected).abs() < 2.0,
                "x={i} 预测 {pred} 偏离真值 {expected}"
            );
        }
    }

    #[test]
    fn classifier_outputs_probabilities_in_unit_range() {
        // y = 1 当 x >= 50，否则 0。
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 50.0 { 1.0 } else { 0.0 })
            .collect();
        let ds = dataset(x, y);

        let model = GradientBoosting::classifier()
            .n_estimators(50)
            .learning_rate(0.3)
            .fit(&ds)
            .expect("训练成功");
        assert_eq!(model.objective(), Objective::BinaryLogLoss);

        let probs = model.predict(&ds).expect("预测成功");
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "概率越界: {p}");
        }
        assert!(
            probs[90] > 0.9 && probs[5] < 0.1,
            "正负类概率未拉开: p(90)={} p(5)={}",
            probs[90],
            probs[5]
        );
    }

    #[test]
    fn classifier_raw_scores_are_logits() {
        let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 30.0 { 1.0 } else { 0.0 })
            .collect();
        let ds = dataset(x, y);
        let model = GradientBoosting::classifier()
            .n_estimators(30)
            .learning_rate(0.3)
            .fit(&ds)
            .expect("训练成功");

        let raw = model.raw_scores(&ds).expect("raw_scores 成功");
        let probs = model.predict(&ds).expect("predict 成功");
        for (r, p) in raw.iter().zip(probs.iter()) {
            let sigmoid = 1.0 / (1.0 + (-r).exp());
            assert!(
                (sigmoid - p).abs() < 1e-12,
                "raw→sigmoid 与 predict 不一致: {r} vs {p}"
            );
        }
    }

    #[test]
    fn save_load_roundtrip_is_bitwise_identical() {
        let ds = linear_data();
        let model = GradientBoosting::regressor()
            .n_estimators(30)
            .learning_rate(0.1)
            .max_depth(4)
            .seed(7)
            .fit(&ds)
            .expect("训练成功");

        let path = std::env::temp_dir().join(format!("sooboost_api_{}.sbm", std::process::id()));
        model.save(&path).expect("保存成功");
        let loaded = GradientBoosting::load(&path).expect("加载成功");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.objective(), model.objective());
        assert_eq!(loaded.num_trees(), model.num_trees());
        // 逐位一致（红线 3）：预测结果必须完全相同，不能只是接近。
        let before = model.predict(&ds).expect("预测成功");
        let after = loaded.predict(&ds).expect("预测成功");
        assert_eq!(before, after, "存读后预测必须逐位一致");
        // 持久化字段回填正确。
        assert_eq!(loaded.config().n_estimators, 30);
        assert!((loaded.config().learning_rate - 0.1).abs() < 1e-15);
    }

    #[test]
    fn loaded_objective_is_detected_not_assumed() {
        let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 30.0 { 1.0 } else { 0.0 })
            .collect();
        let ds = dataset(x, y);
        let model = GradientBoosting::classifier()
            .n_estimators(20)
            .fit(&ds)
            .expect("训练成功");

        let loaded = GradientBoosting::from_bytes(&model.to_bytes()).expect("加载成功");
        assert_eq!(
            loaded.objective(),
            Objective::BinaryLogLoss,
            "载入时必须探测出二分类目标，而非默认回归"
        );
    }

    #[test]
    fn invalid_params_are_rejected_before_training() {
        let ds = linear_data();
        // 各 setter 闭包类型不同，装箱为 trait object 才能放进同一集合。
        type Setter = Box<dyn Fn(GradientBoostingBuilder) -> GradientBoostingBuilder>;
        let cases: Vec<(Setter, &'static str)> = vec![
            (Box::new(|b| b.n_estimators(0)), "n_estimators"),
            (Box::new(|b| b.learning_rate(0.0)), "learning_rate"),
            (Box::new(|b| b.min_samples_leaf(0)), "min_samples_leaf"),
            (Box::new(|b| b.max_bins(1)), "max_bins"),
            (Box::new(|b| b.max_categories(0)), "max_categories"),
            (Box::new(|b| b.categorical_alpha(-1.0)), "categorical_alpha"),
        ];
        for (f, field) in cases {
            let err = f(GradientBoosting::regressor())
                .fit(&ds)
                .expect_err("非法参数必须报错");
            assert!(
                matches!(&err, Error::InvalidParam { field: got, .. } if *got == field),
                "期望 {field} 报错，实际: {err}"
            );
        }
    }

    #[test]
    fn row_predict_matches_dataset_predict() {
        let ds = linear_data();
        let model = GradientBoosting::regressor()
            .n_estimators(20)
            .learning_rate(0.2)
            .fit(&ds)
            .expect("训练成功");

        let batch_preds = model.predict(&ds).expect("批量预测成功");
        for i in [0usize, 1, 37, 99] {
            let row_pred = model.predict_row(&[i as f64]).expect("单行预测成功");
            assert!(
                (row_pred - batch_preds[i]).abs() < 1e-12,
                "行 {i} 单行 {row_pred} 与批量 {} 不一致",
                batch_preds[i]
            );
        }
    }

    #[test]
    fn row_predict_rejects_feature_count_mismatch() {
        let ds = linear_data();
        let model = GradientBoosting::regressor()
            .n_estimators(5)
            .fit(&ds)
            .expect("训练成功");
        let err = model
            .predict_row(&[1.0, 2.0])
            .expect_err("特征数不符必须报错");
        assert!(matches!(
            err,
            Error::FeatureCountMismatch {
                expected: 1,
                got: 2
            }
        ));
    }

    #[test]
    fn determinism_same_config_same_bytes() {
        let ds = linear_data();
        let build = || {
            GradientBoosting::regressor()
                .n_estimators(25)
                .learning_rate(0.15)
                .seed(42)
                .fit(&ds)
                .expect("训练成功")
        };
        // 红线 3：同数据同配置同 seed → 逐位一致。
        assert_eq!(build().to_bytes(), build().to_bytes());
    }

    #[test]
    fn dataset_construction_error_is_unified() {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![1.0])),
            ],
        )
        .expect("构造 batch");
        let err = Dataset::from_record_batch(batch, &["nope"], "target", MissingPolicy::default())
            .expect_err("列名不存在必须报错");
        let unified: Error = err.into();
        assert!(
            matches!(unified, Error::Data(DataError::ColumnNotFound(ref n)) if n == "nope"),
            "应统一为 Error::Data，实际: {unified}"
        );
    }

    #[test]
    fn corrupted_model_bytes_are_reported_not_ignored() {
        let ds = linear_data();
        let model = GradientBoosting::regressor()
            .n_estimators(5)
            .fit(&ds)
            .expect("训练成功");
        let mut bytes = model.to_bytes();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff; // 破坏 checksum 区
        let err = GradientBoosting::from_bytes(&bytes).expect_err("篡改字节必须报错");
        assert!(
            matches!(err, Error::Model(_)),
            "应统一为 Error::Model，实际: {err}"
        );
    }
}
