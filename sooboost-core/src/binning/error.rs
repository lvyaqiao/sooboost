//! 分箱层错误类型：显式传播（易踩坑 10）。

use crate::data::DataError;

/// 分箱错误。
#[derive(Debug, thiserror::Error)]
pub enum BinningError {
    /// 边界数量超过 max_bins-1 上限。
    #[error("特征 {feature} 边界数 {got} 超过上限 max_bins-1（max_bins={max_bins}）")]
    TooManyBoundaries {
        feature: usize,
        got: usize,
        max_bins: usize,
    },

    /// 边界未严格升序。
    #[error("特征 {feature} 的边界未严格升序")]
    BoundariesNotSorted { feature: usize },

    /// 数据层错误。
    #[error("数据层错误: {0}")]
    Data(#[from] DataError),
}
