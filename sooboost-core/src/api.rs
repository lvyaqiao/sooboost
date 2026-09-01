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
use crate::boosting::multiclass::MulticlassBooster;
use crate::boosting::{
    BoostingError, BoostingParams, EarlyStopping, ImportanceKind, TrainingContext, fit,
    fit_multiclass, fit_with_early_stopping,
};
use crate::data::missing::is_missing_value;
use crate::data::{DataError, Dataset, MissingPolicy};
use crate::loss::{BinaryLogLoss, Loss, SquaredError};
use crate::metrics;
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
    /// 评估指标退化或输入非法（交叉验证内计算 R²/AUC 时，M6-2）。
    #[error("指标错误: {0}")]
    Metric(#[from] crate::metrics::MetricsError),
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
    /// 该操作仅特定目标支持（如概率输出仅分类目标、早停暂不支持多分类）。
    #[error("操作不支持当前目标：{operation}（{reason}）")]
    UnsupportedForObjective {
        /// 操作名。
        operation: &'static str,
        /// 为什么不支持。
        reason: &'static str,
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
    /// 多分类：softmax（M6-5a）。每轮每类一棵树，`predict` 输出 argmax 类别。
    MulticlassSoftmax,
}

impl Objective {
    /// 目标名称（与模型头中的损失名一致，contracts §1.2）。
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Objective::SquaredError => SquaredError.name(),
            Objective::BinaryLogLoss => BinaryLogLoss.name(),
            Objective::MulticlassSoftmax => crate::model::format::MULTICLASS_LOSS_NAME,
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
    Multiclass(MulticlassBooster),
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

    /// 多分类（softmax）构造器（M6-5a）。
    ///
    /// 每轮对每类建一棵树（共 `n_estimators × n_classes` 棵）；`y` 必须为
    /// 整数标签 ∈ [0, n_classes)，非整数或越界在 `fit` 时显式报错。
    /// 多分类暂不支持类别特征与早停（显式报错，不静默降级）。
    #[must_use]
    pub fn multiclass_classifier(n_classes: usize) -> GradientBoostingBuilder {
        GradientBoostingBuilder::new_multiclass(n_classes)
    }

    /// 训练目标。
    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// 类别数（多分类模型返回 `Some(k)`；回归/二分类为 `None`）。
    #[must_use]
    pub fn num_classes(&self) -> Option<usize> {
        match &self.fitted {
            Fitted::Multiclass(m) => Some(m.n_classes()),
            _ => None,
        }
    }

    /// 树棵数（多分类为每类棵数，与 `n_estimators` 对齐；总数 = 该值 × 类别数）。
    #[must_use]
    pub fn num_trees(&self) -> usize {
        match &self.fitted {
            Fitted::Regression(b) => b.num_trees(),
            Fitted::Binary(b) => b.num_trees(),
            Fitted::Multiclass(m) => m.num_trees_per_class(),
        }
    }

    /// 特征数（由模型自带分箱表确定）。
    #[must_use]
    pub fn num_features(&self) -> usize {
        match &self.fitted {
            Fitted::Regression(b) => b.bin_table().num_features(),
            Fitted::Binary(b) => b.bin_table().num_features(),
            Fitted::Multiclass(m) => m.bin_table().num_features(),
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

    /// 特征重要度（归一化到和为 1；M6-3）。
    ///
    /// 数据源为树节点持久化的 gain/cover（v4 格式），load 后同样可用。
    /// 多分类跨全部类别的树聚合。极端退化（无任何分裂）时返回全 0。
    #[must_use]
    pub fn feature_importances(&self, kind: ImportanceKind) -> Vec<f64> {
        match &self.fitted {
            Fitted::Regression(b) => b.feature_importances(kind),
            Fitted::Binary(b) => b.feature_importances(kind),
            Fitted::Multiclass(m) => m.feature_importances(kind),
        }
    }

    /// 实际使用的提升轮数（早停回滚后可能小于请求的 n_estimators；
    /// 多分类为每类轮数）。
    #[must_use]
    pub fn best_iteration(&self) -> usize {
        match &self.fitted {
            Fitted::Regression(b) => b.best_iteration(),
            Fitted::Binary(b) => b.best_iteration(),
            Fitted::Multiclass(m) => m.num_trees_per_class(),
        }
    }

    /// 早停验证集损失历史（每轮一个值；未启用早停则为空切片）。
    #[must_use]
    pub fn eval_history(&self) -> &[f64] {
        match &self.fitted {
            Fitted::Regression(b) => b.eval_history(),
            Fitted::Binary(b) => b.eval_history(),
            Fitted::Multiclass(_) => &[],
        }
    }

    /// 最终预测：回归 → 原值；二分类 → 正类概率；多分类 → argmax 类别（f64）。
    pub fn predict(&self, ds: &Dataset) -> Result<Vec<f64>, Error> {
        Ok(match &self.fitted {
            Fitted::Regression(b) => b.predict(ds)?,
            Fitted::Binary(b) => b.predict(ds)?,
            Fitted::Multiclass(m) => m.predict(ds)?.into_iter().map(|c| c as f64).collect(),
        })
    }

    /// 多分类预测类别（argmax；并列取小类）。
    ///
    /// 仅多分类模型可用，其余目标显式报错。
    pub fn predict_classes(&self, ds: &Dataset) -> Result<Vec<usize>, Error> {
        match &self.fitted {
            Fitted::Multiclass(m) => Ok(m.predict(ds)?),
            _ => Err(Error::UnsupportedForObjective {
                operation: "predict_classes",
                reason: "仅多分类模型有类别输出；回归用 predict，二分类用阈值化概率",
            }),
        }
    }

    /// 多分类各类别概率矩阵 `probs[row][class]`（softmax，行和为 1）。
    ///
    /// 仅多分类模型可用；二分类概率用 `predict`，回归无概率。
    pub fn predict_proba(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, Error> {
        match &self.fitted {
            Fitted::Multiclass(m) => Ok(m.predict_proba(ds)?),
            _ => Err(Error::UnsupportedForObjective {
                operation: "predict_proba",
                reason: "仅多分类模型输出类别概率矩阵；二分类概率用 predict",
            }),
        }
    }

    /// 多分类各类别 logits 矩阵 `logits[row][class]`（softmax 前）。
    ///
    /// 仅多分类模型可用，供自定义校准；二分类 logit 用 `raw_scores`。
    pub fn raw_logits(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, Error> {
        match &self.fitted {
            Fitted::Multiclass(m) => Ok(m.raw_logits(ds)?),
            _ => Err(Error::UnsupportedForObjective {
                operation: "raw_logits",
                reason: "仅多分类模型输出 logits 矩阵；标量原始分数用 raw_scores",
            }),
        }
    }

    /// 原始分数（`init + Σ lr·tree`，未经 `transform`）。
    ///
    /// 二分类下即 logit，供自定义阈值/校准使用；多分类请用 [`Self::raw_logits`]。
    pub fn raw_scores(&self, ds: &Dataset) -> Result<Vec<f64>, Error> {
        Ok(match &self.fitted {
            Fitted::Regression(b) => b.raw_scores(ds)?,
            Fitted::Binary(b) => b.raw_scores(ds)?,
            Fitted::Multiclass(_) => {
                return Err(Error::UnsupportedForObjective {
                    operation: "raw_scores",
                    reason: "多分类的原始分数是矩阵，请用 raw_logits",
                });
            }
        })
    }

    /// 单行预测（在线推断路径，无需构造 `Dataset`）。
    ///
    /// 缺失以 `f64::NAN` 表示，按 [`Config::missing_policy`] 解释
    /// （红线 2：语义仍由 `data::missing` 单点定义）。
    /// 多分类返回 argmax 类别（与 [`Self::predict`] 一致）。
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
            Fitted::Multiclass(_) => false,
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
            Fitted::Multiclass(m) => {
                let logits = m.raw_logits_row(values, &is_missing);
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0.0, |(i, _)| i as f64)
            }
        })
    }

    /// 序列化为字节（contracts §1.2 显式布局 + checksum）。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.fitted {
            Fitted::Regression(b) => b.serialize(),
            Fitted::Binary(b) => b.serialize(),
            Fitted::Multiclass(m) => m.serialize(),
        }
    }

    /// 由字节恢复模型，目标自动探测（标量回归 → 二分类 → 多分类）。
    ///
    /// 安全依据：contracts §1.2 的校验顺序是 magic → 版本 → checksum →
    /// 损失名/类别数 → 结构，因此只有「字节本身合法、仅目标不同」才会落到
    /// `LossMismatch`；截断 / checksum 失败等一律原样上抛（红线 6）。
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
        match Booster::deserialize(bytes, BinaryLogLoss) {
            Ok(b) => {
                return Ok(Self::from_fitted(
                    Objective::BinaryLogLoss,
                    Fitted::Binary(b),
                ));
            }
            Err(ModelError::LossMismatch { .. }) => {}
            Err(e) => return Err(Error::Model(e)),
        }
        let m = MulticlassBooster::deserialize(bytes)?;
        Ok(Self::from_fitted(
            Objective::MulticlassSoftmax,
            Fitted::Multiclass(m),
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
            Fitted::Regression(b) => {
                config_from_booster(b.num_trees(), b.learning_rate(), b.bin_table().max_bins())
            }
            Fitted::Binary(b) => {
                config_from_booster(b.num_trees(), b.learning_rate(), b.bin_table().max_bins())
            }
            Fitted::Multiclass(m) => config_from_booster(
                m.num_trees_per_class(),
                m.learning_rate(),
                m.bin_table().max_bins(),
            ),
        };
        Self {
            objective,
            config,
            fitted,
        }
    }
}

/// 由模型自身可观测字段回填配置（学习率/轮数/分箱数持久化，其余取默认）。
fn config_from_booster(n_estimators: usize, learning_rate: f64, max_bins: usize) -> Config {
    Config {
        n_estimators,
        learning_rate,
        max_bins,
        ..Config::default()
    }
}

/// [`GradientBoosting`] 的 builder。
#[derive(Debug, Clone)]
pub struct GradientBoostingBuilder {
    objective: Objective,
    config: Config,
    early_stopping: Option<EarlyStopping>,
    /// 多分类类别数（仅 `Objective::MulticlassSoftmax` 时为 Some）。
    n_classes: Option<usize>,
}

impl GradientBoostingBuilder {
    #[must_use]
    fn new(objective: Objective) -> Self {
        Self {
            objective,
            config: Config::default(),
            early_stopping: None,
            n_classes: None,
        }
    }

    #[must_use]
    fn new_multiclass(n_classes: usize) -> Self {
        Self {
            objective: Objective::MulticlassSoftmax,
            config: Config::default(),
            early_stopping: None,
            n_classes: Some(n_classes),
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

    /// 启用早停（M6-1）：每轮在 `eval_set` 上评估损失，连续 `rounds` 轮无改善则停，
    /// 树集合回滚到最优轮（见 [`GradientBoosting::best_iteration`] / `eval_history`）。
    ///
    /// 验证集特征数必须与训练集一致；类别特征用训练集学到的编码解析。
    /// 暂不支持多分类（`multiclass_classifier` + 本方法在 `fit` 时显式报错）。
    #[must_use]
    pub fn early_stopping(mut self, eval_set: Dataset, rounds: usize) -> Self {
        self.early_stopping = Some(EarlyStopping { eval_set, rounds });
        self
    }

    /// 训练。
    pub fn fit(self, ds: &Dataset) -> Result<GradientBoosting, Error> {
        self.config.validate()?;
        let params = self.config.boosting_params();
        let ctx = TrainingContext::new(self.config.seed);
        let fitted = match self.objective {
            Objective::SquaredError => Fitted::Regression(match &self.early_stopping {
                None => fit(ds, &params, SquaredError, &ctx)?,
                Some(es) => fit_with_early_stopping(ds, &params, SquaredError, &ctx, es)?,
            }),
            Objective::BinaryLogLoss => Fitted::Binary(match &self.early_stopping {
                None => fit(ds, &params, BinaryLogLoss, &ctx)?,
                Some(es) => fit_with_early_stopping(ds, &params, BinaryLogLoss, &ctx, es)?,
            }),
            Objective::MulticlassSoftmax => {
                let n_classes = self.n_classes.unwrap_or_default();
                if n_classes < 2 {
                    return Err(Error::InvalidParam {
                        field: "n_classes",
                        value: n_classes.to_string(),
                        reason: "多分类至少需要 2 个类别",
                    });
                }
                if self.early_stopping.is_some() {
                    return Err(Error::UnsupportedForObjective {
                        operation: "early_stopping",
                        reason: "多分类早停尚未实现（每轮需对全部类别联合评估 softmax 损失）",
                    });
                }
                Fitted::Multiclass(fit_multiclass(ds, &params, n_classes, &ctx)?)
            }
        };
        Ok(GradientBoosting::from_fitted(self.objective, fitted))
    }

    /// K 折交叉验证（M6-2）。
    ///
    /// 折切分为**连续分块**（无 shuffle，完全确定性；对有序数据请先自行打乱再建
    /// `Dataset`）。指标按目标自动选择：回归 → R²（[`crate::metrics::r2_score`]），
    /// 二分类 → AUC（[`crate::metrics::roc_auc`]），多分类 → accuracy（M6-5a）。
    ///
    /// 返回逐折得分与汇总（均值 + 样本标准差）。训练用 `self` 的全部配置
    /// （含早停——若设置，逐折训练集内部不再二次切分早停验证集）。
    pub fn cross_validate(self, ds: &Dataset, k: usize) -> Result<CvResult, Error> {
        self.config.validate()?;
        let n = ds.num_rows();
        if k < 2 {
            return Err(Error::InvalidParam {
                field: "k",
                value: k.to_string(),
                reason: "至少 2 折",
            });
        }
        if k > n {
            return Err(Error::InvalidParam {
                field: "k",
                value: k.to_string(),
                reason: "折数不能超过样本数",
            });
        }
        let metric_name = match self.objective {
            Objective::SquaredError => "r2",
            Objective::BinaryLogLoss => "auc",
            Objective::MulticlassSoftmax => "accuracy",
        };

        // 连续分块：前 n % k 折各多 1 行（与 sklearn KFold 一致的确定性切法）
        let base = n / k;
        let rem = n % k;
        let mut offset = 0usize;
        let mut fold_scores = Vec::with_capacity(k);
        for fold in 0..k {
            let len = base + usize::from(fold < rem);
            let eval = ds.slice_rows(offset, len)?;
            // 折内训练集 = 其余行；两次 slice 拼接
            let train = match offset {
                0 => ds.slice_rows(offset + len, n - offset - len)?,
                o if offset + len == n => ds.slice_rows(0, o)?,
                o => {
                    let left = ds.slice_rows(0, o)?;
                    let right = ds.slice_rows(o + len, n - o - len)?;
                    left.concatenate_rows(&right)?
                }
            };
            let model = self.clone().fit(&train)?;
            let preds = model.predict(&eval)?;
            let y: Vec<f64> = eval.target_values().values().to_vec();
            let score = match self.objective {
                Objective::SquaredError => metrics::r2_score(&y, &preds)?,
                Objective::BinaryLogLoss => metrics::roc_auc(&y, &preds)?,
                Objective::MulticlassSoftmax => metrics::accuracy(&y, &preds)?,
            };
            fold_scores.push(score);
            offset += len;
        }

        let k_f = k as f64;
        let mean = fold_scores.iter().sum::<f64>() / k_f;
        let var = fold_scores
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f64>()
            / (k_f - 1.0);
        Ok(CvResult {
            fold_scores,
            mean,
            std: var.sqrt(),
            metric: metric_name,
        })
    }
}

/// 交叉验证结果（M6-2）。
#[derive(Debug, Clone)]
pub struct CvResult {
    /// 逐折得分（长度 = k）。
    pub fold_scores: Vec<f64>,
    /// 得分均值。
    pub mean: f64,
    /// 得分样本标准差（自由度 k−1）。
    pub std: f64,
    /// 指标名（`"r2"` / `"auc"` / `"accuracy"`）。
    pub metric: &'static str,
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

    // -- M6-1 早停 ----------------------------------------------------------

    /// 过拟合场景：训练集仅 8 个点（x=0..7, y=x），验证集 100 点连续分布。
    /// 高学习率下训练损失很快到 0，后续轮在验证集上无改善 → 早停应触发。
    fn overfit_pair() -> (Dataset, Dataset) {
        let train = dataset(
            (0..8).map(|i| i as f64).collect(),
            (0..8).map(|i| i as f64).collect(),
        );
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 2.0).collect();
        let eval = dataset(x.clone(), x.to_vec());
        (train, eval)
    }

    #[test]
    fn early_stopping_stops_before_max_rounds() {
        let (train, eval) = overfit_pair();
        let model = GradientBoosting::regressor()
            .n_estimators(200)
            .learning_rate(0.9)
            .max_depth(6)
            .seed(3)
            .early_stopping(eval.clone(), 5)
            .fit(&train)
            .expect("训练成功");
        assert!(
            model.num_trees() < 200,
            "早停必须提前停止，实际树数 {}",
            model.num_trees()
        );
        assert_eq!(model.best_iteration(), model.num_trees());
        // 历史覆盖停止前的全部轮数（≥ 回滚后树数）
        assert!(model.eval_history().len() >= model.num_trees());
        // 最优轮即历史中的最小损失处
        let best = model
            .eval_history()
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("历史非空");
        assert_eq!(best + 1, model.num_trees());

        // 早停的真实价值断言：回滚模型在验证集上的损失不劣于跑满 200 轮的模型
        let eval_loss = |m: &GradientBoosting| -> f64 {
            let preds = m.predict(&eval).expect("预测成功");
            let y: Vec<f64> = eval.target_values().values().to_vec();
            y.iter()
                .zip(preds.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                / y.len() as f64
        };
        let full = GradientBoosting::regressor()
            .n_estimators(200)
            .learning_rate(0.9)
            .max_depth(6)
            .seed(3)
            .fit(&train)
            .expect("全量训练成功");
        assert!(
            eval_loss(&model) <= eval_loss(&full) + 1e-9,
            "早停验证损失 {} 不应劣于全量 {}",
            eval_loss(&model),
            eval_loss(&full)
        );
    }

    #[test]
    fn early_stopping_rounds_zero_is_rejected() {
        let (train, eval) = overfit_pair();
        let err = GradientBoosting::regressor()
            .n_estimators(10)
            .early_stopping(eval, 0)
            .fit(&train)
            .expect_err("patience=0 必须报错");
        assert!(matches!(
            err,
            Error::Boosting(BoostingError::InvalidEarlyStopping(_))
        ));
    }

    // -- M6-3 特征重要度 ------------------------------------------------------

    /// f0 承载全部信号（y = 2·f0），f1 为常数 → gain 重要度应完全偏向 f0。
    fn two_features_signal_and_noise() -> Dataset {
        let n = 120;
        let f0: Vec<f64> = (0..n).map(|i| (i % 40) as f64).collect();
        let f1 = vec![7.5; n];
        let y: Vec<f64> = f0.iter().map(|&v| 2.0 * v).collect();
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("f1", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(f0)),
                Arc::new(Float64Array::from(f1)),
                Arc::new(Float64Array::from(y)),
            ],
        )
        .expect("构造测试 batch");
        Dataset::from_record_batch(batch, &["f0", "f1"], "target", MissingPolicy::default())
            .expect("构造 Dataset")
    }

    #[test]
    fn feature_importances_gain_identifies_signal() {
        let ds = two_features_signal_and_noise();
        let model = GradientBoosting::regressor()
            .n_estimators(40)
            .learning_rate(0.3)
            .fit(&ds)
            .expect("训练成功");
        for kind in [
            ImportanceKind::Gain,
            ImportanceKind::Cover,
            ImportanceKind::Frequency,
        ] {
            let imp = model.feature_importances(kind);
            assert_eq!(imp.len(), 2);
            let total: f64 = imp.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "{kind:?} 归一化和应为 1: {total}"
            );
            assert!(
                imp[0] > imp[1],
                "{kind:?}: 信号特征 f0 重要度必须高于常数特征 f1: {imp:?}"
            );
        }
    }

    #[test]
    fn feature_importances_survive_roundtrip() {
        // v3 格式持久化 gain/cover：load 后重要度必须可用且与原模型一致
        let ds = two_features_signal_and_noise();
        let model = GradientBoosting::regressor()
            .n_estimators(30)
            .seed(9)
            .fit(&ds)
            .expect("训练成功");
        let loaded = GradientBoosting::from_bytes(&model.to_bytes()).expect("载入成功");
        for kind in [ImportanceKind::Gain, ImportanceKind::Cover] {
            assert_eq!(
                model.feature_importances(kind),
                loaded.feature_importances(kind),
                "{kind:?}: 存读后重要度必须逐位一致"
            );
        }
    }

    // -- M6-2 交叉验证 --------------------------------------------------------

    #[test]
    fn cross_validate_is_deterministic_and_complete() {
        // 连续分块的折切分对有序数据会外推（这正是文档要求先打乱的原因），
        // 故用确定性交错排列（x = (i·37) mod 100）让每折训练集覆盖全值域。
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
        let ds = dataset(x, y);
        let cv = || {
            GradientBoosting::regressor()
                .n_estimators(20)
                .seed(42)
                .cross_validate(&ds, 5)
                .expect("CV 成功")
        };
        let a = cv();
        let b = cv();
        assert_eq!(a.fold_scores.len(), 5);
        assert_eq!(a.metric, "r2");
        assert_eq!(a.fold_scores, b.fold_scores, "同 seed 同数据必须逐位一致");
        let mean: f64 = a.fold_scores.iter().sum::<f64>() / 5.0;
        assert!((a.mean - mean).abs() < 1e-12);
        // 交错数据的 R² 应为正（模型确实学到了信号）
        assert!(a.mean > 0.9, "交错线性数据 CV R² 应接近 1，实际 {}", a.mean);
    }

    #[test]
    fn cross_validate_k_bounds_are_validated() {
        let ds = linear_data();
        for k in [0, 1, 101] {
            let err = GradientBoosting::regressor()
                .cross_validate(&ds, k)
                .expect_err("非法 k 必须报错");
            assert!(matches!(err, Error::InvalidParam { field: "k", .. }));
        }
    }

    // -- M6-5a 多分类 ---------------------------------------------------------

    #[test]
    fn cross_validate_classifier_uses_auc() {
        // 交错排列让每折验证集都含正负两类（连续分块 + 有序标签会使整折单类）
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 50.0 { 1.0 } else { 0.0 })
            .collect();
        let ds = dataset(x, y);
        let cv = GradientBoosting::classifier()
            .n_estimators(30)
            .cross_validate(&ds, 4)
            .expect("CV 成功");
        assert_eq!(cv.metric, "auc");
        for &s in &cv.fold_scores {
            assert!((0.0..=1.0).contains(&s), "AUC 越界: {s}");
        }
        assert!(cv.mean > 0.9, "可分数据 CV AUC 应接近 1，实际 {}", cv.mean);
    }

    /// 3 类可分数据：x ∈ 0..99，class = ⌊x/33⌋（单特征阈值可分）。
    fn three_class_data() -> Dataset {
        let x: Vec<f64> = (0..99).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v / 33.0).floor()).collect();
        dataset(x, y)
    }

    #[test]
    fn multiclass_fits_separable_data() {
        let ds = three_class_data();
        let model = GradientBoosting::multiclass_classifier(3)
            .n_estimators(30)
            .learning_rate(0.3)
            .fit(&ds)
            .expect("训练成功");
        assert_eq!(model.objective(), Objective::MulticlassSoftmax);
        assert_eq!(model.num_classes(), Some(3));
        assert_eq!(model.num_trees(), 30, "num_trees 为每类棵数");
        assert_eq!(model.num_features(), 1);

        let classes = model.predict_classes(&ds).expect("类别预测成功");
        let hits = classes
            .iter()
            .zip(ds.target_values().values())
            .filter(|&(&c, &y)| c as f64 == y)
            .count();
        assert!(
            hits as f64 / classes.len() as f64 > 0.9,
            "可分数据准确率应 >0.9，实际 {hits}/{}",
            classes.len()
        );

        // 概率矩阵：行和为 1，argmax 与 predict_classes 一致
        let proba = model.predict_proba(&ds).expect("概率成功");
        assert_eq!(proba.len(), 99);
        for (row, &c) in proba.iter().zip(classes.iter()) {
            assert_eq!(row.len(), 3);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9, "行和应为 1");
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("非空");
            assert_eq!(argmax, c, "softmax argmax 应与 predict_classes 一致");
        }

        // predict（f64 口径）与 predict_classes 一致
        let preds = model.predict(&ds).expect("predict 成功");
        for (p, &c) in preds.iter().zip(classes.iter()) {
            assert_eq!(*p, c as f64);
        }
    }

    #[test]
    fn multiclass_save_load_roundtrip_is_bitwise_identical() {
        let ds = three_class_data();
        let model = GradientBoosting::multiclass_classifier(3)
            .n_estimators(15)
            .seed(5)
            .fit(&ds)
            .expect("训练成功");
        let loaded = GradientBoosting::from_bytes(&model.to_bytes()).expect("载入成功");
        assert_eq!(
            loaded.objective(),
            Objective::MulticlassSoftmax,
            "载入时必须探测出多分类目标"
        );
        assert_eq!(loaded.num_classes(), Some(3));
        let before = model.predict_proba(&ds).expect("概率成功");
        let after = loaded.predict_proba(&ds).expect("概率成功");
        assert_eq!(before, after, "存读后概率矩阵必须逐位一致");
        assert_eq!(
            model.feature_importances(ImportanceKind::Gain),
            loaded.feature_importances(ImportanceKind::Gain),
            "存读后重要度必须逐位一致"
        );
    }

    #[test]
    fn multiclass_predict_row_matches_batch() {
        let ds = three_class_data();
        let model = GradientBoosting::multiclass_classifier(3)
            .n_estimators(20)
            .fit(&ds)
            .expect("训练成功");
        let batch = model.predict(&ds).expect("批量成功");
        for i in [0usize, 40, 80] {
            let row = model.predict_row(&[i as f64]).expect("单行成功");
            assert_eq!(row, batch[i], "行 {i} 单行与批量 argmax 不一致");
        }
        // 特征数不符 → 显式报错
        assert!(matches!(
            model.predict_row(&[1.0, 2.0]),
            Err(Error::FeatureCountMismatch {
                expected: 1,
                got: 2
            })
        ));
    }

    #[test]
    fn multiclass_invalid_labels_and_arity_rejected() {
        // 非整数标签
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let bad = dataset(x.clone(), {
            let mut y: Vec<f64> = (0..20).map(|i| (i / 10) as f64).collect();
            y[0] = 1.5;
            y
        });
        let err = GradientBoosting::multiclass_classifier(2)
            .n_estimators(5)
            .fit(&bad)
            .expect_err("非整数标签必须报错");
        assert!(
            matches!(
                err,
                Error::Boosting(BoostingError::Data(DataError::InvalidLabel { .. }))
            ),
            "实际: {err}"
        );
        // 标签越界（n_classes=2，但 y 含标签 2）
        let out_of_range = dataset(x.clone(), x.iter().map(|&v| (v / 8.0).floor()).collect());
        let err = GradientBoosting::multiclass_classifier(2)
            .n_estimators(5)
            .fit(&out_of_range)
            .expect_err("标签越界必须报错");
        assert!(matches!(
            err,
            Error::Boosting(BoostingError::Data(DataError::InvalidLabel { .. }))
        ));
        // 类别数 < 2
        let ds = dataset(x, vec![0.0; 20]);
        for k in [0usize, 1] {
            let err = GradientBoosting::multiclass_classifier(k)
                .n_estimators(5)
                .fit(&ds)
                .expect_err("类别数 <2 必须报错");
            assert!(matches!(
                err,
                Error::InvalidParam {
                    field: "n_classes",
                    ..
                }
            ));
        }
    }

    #[test]
    fn multiclass_rejects_early_stopping_explicitly() {
        let ds = three_class_data();
        let (eval_x, eval_y): (Vec<f64>, Vec<f64>) = (0..99)
            .map(|i| {
                let x = i as f64;
                (x, (x / 33.0).floor())
            })
            .unzip();
        let eval = dataset(eval_x, eval_y);
        let err = GradientBoosting::multiclass_classifier(3)
            .n_estimators(10)
            .early_stopping(eval, 3)
            .fit(&ds)
            .expect_err("多分类早停必须显式报错");
        assert!(
            matches!(
                err,
                Error::UnsupportedForObjective {
                    operation: "early_stopping",
                    ..
                }
            ),
            "实际: {err}"
        );
    }

    #[test]
    fn multiclass_cross_validate_uses_accuracy() {
        // 确定性交错排列让每折覆盖全部 3 类（连续分块 + 有序标签会整折单类）
        let n = 99usize;
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v / 33.0).floor()).collect();
        let ds = dataset(x, y);
        let cv = GradientBoosting::multiclass_classifier(3)
            .n_estimators(20)
            .cross_validate(&ds, 3)
            .expect("CV 成功");
        assert_eq!(cv.metric, "accuracy");
        for &s in &cv.fold_scores {
            assert!((0.0..=1.0).contains(&s), "accuracy 越界: {s}");
        }
        assert!(
            cv.mean > 0.8,
            "可分数据 CV accuracy 应很高，实际 {}",
            cv.mean
        );
    }

    #[test]
    fn multiclass_feature_importances_identifies_signal() {
        // f0 决定类别（3 类分界在 f0），f1 常数 → gain 重要度偏向 f0
        let n = 120;
        let f0: Vec<f64> = (0..n).map(|i| (i % 60) as f64).collect();
        let f1 = vec![3.25; n];
        let y: Vec<f64> = f0.iter().map(|&v| (v / 20.0).floor().min(2.0)).collect();
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("f1", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(f0)),
                Arc::new(Float64Array::from(f1)),
                Arc::new(Float64Array::from(y)),
            ],
        )
        .expect("构造测试 batch");
        let ds =
            Dataset::from_record_batch(batch, &["f0", "f1"], "target", MissingPolicy::default())
                .expect("构造 Dataset");

        let model = GradientBoosting::multiclass_classifier(3)
            .n_estimators(30)
            .learning_rate(0.3)
            .fit(&ds)
            .expect("训练成功");
        for kind in [
            ImportanceKind::Gain,
            ImportanceKind::Cover,
            ImportanceKind::Frequency,
        ] {
            let imp = model.feature_importances(kind);
            assert_eq!(imp.len(), 2);
            assert!(
                (imp.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "{kind:?} 归一化和应为 1"
            );
            assert!(
                imp[0] > imp[1],
                "{kind:?}: 信号特征 f0 重要度必须高于常数特征 f1: {imp:?}"
            );
        }
    }

    #[test]
    fn scalar_only_operations_error_on_multiclass() {
        let ds = three_class_data();
        let reg = GradientBoosting::regressor()
            .n_estimators(2)
            .fit(&ds)
            .expect("reg");
        let err = reg.predict_proba(&ds).expect_err("回归无概率");
        assert!(
            matches!(
                err,
                Error::UnsupportedForObjective {
                    operation: "predict_proba",
                    ..
                }
            ),
            "实际: {err}"
        );
        let mc = GradientBoosting::multiclass_classifier(3)
            .n_estimators(5)
            .fit(&ds)
            .expect("mc");
        let err = mc.raw_scores(&ds).expect_err("多分类无标量 raw_scores");
        assert!(
            matches!(
                err,
                Error::UnsupportedForObjective {
                    operation: "raw_scores",
                    ..
                }
            ),
            "实际: {err}"
        );
    }
}
