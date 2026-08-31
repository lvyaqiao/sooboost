//! 提升训练参数（m0-spec §4 首版固定）。

use crate::binning::DEFAULT_MAX_BINS;
use crate::tree::TreeParams;

/// GBDT 训练参数。
///
/// 分箱参数 `max_bins` 与树参数 `tree_params` 拆开：分箱是数据层契约
/// （BinTable 训练/预测共用，D4），树参数是分裂契约（TreeBuilder）。
#[derive(Debug, Clone, Copy)]
pub struct BoostingParams {
    /// 提升轮数（决策树棵数）。
    pub n_estimators: usize,
    /// 每棵树的学习率缩放（预测累加 pred += lr·tree）。
    pub learning_rate: f64,
    /// 分箱数量上限（传给 BinTable，见 binning::DEFAULT_MAX_BINS）。
    pub max_bins: usize,
    /// 树参数（max_depth / min_samples_leaf / min_split_gain / reg_lambda）。
    pub tree_params: TreeParams,
    /// 类别特征基数上限（超限报错而非静默截断，contracts §1.4）。
    pub max_categories: usize,
    /// ordered TS smoothing α（contracts §1.4：默认值写入契约）。
    pub categorical_alpha: f64,
}

impl Default for BoostingParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_bins: DEFAULT_MAX_BINS,
            tree_params: TreeParams::default(),
            max_categories: 1000,
            categorical_alpha: 1.0,
        }
    }
}
