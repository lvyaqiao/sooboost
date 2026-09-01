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
    /// 早停验证集特征数与训练集不一致（M6-1）。
    #[error("早停验证集特征数 {eval} 与训练集 {train} 不一致")]
    EvalSetFeatureMismatch {
        /// 训练集特征数。
        train: usize,
        /// 验证集特征数。
        eval: usize,
    },
    /// 早停参数非法（patience 为 0 会使第 2 轮必然停止）。
    #[error("早停参数非法: {0}")]
    InvalidEarlyStopping(&'static str),
}
