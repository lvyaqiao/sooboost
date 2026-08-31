//! 训练层错误：向上聚合数据层/分箱层/树内核错误。

use crate::binning::BinningError;
use crate::data::DataError;
use crate::tree::TreeError;

/// 训练层统一错误类型（易踩坑 10：显式传播，不吞错误）。
#[derive(Debug, thiserror::Error)]
pub enum BoostingError {
    #[error("数据层错误: {0}")]
    Data(#[from] DataError),
    #[error("分箱错误: {0}")]
    Binning(#[from] BinningError),
    #[error("建树错误: {0}")]
    Tree(#[from] TreeError),
}
