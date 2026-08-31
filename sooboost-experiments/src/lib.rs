//! sooboost-experiments：研究方向实验 crate（架构 D7 风险隔离）。
//!
//! 原则（doc/plans/m2-spec.md）：
//! - 研究先在此原型验证，验证后仅反哺被证明的最小原语到核心；
//! - `#![forbid(unsafe_code)]`（红线 7）；库代码禁 unwrap/expect（易踩坑 10）；
//! - 采样固定 seed（红线 3），同 seed 逐位一致；
//! - 只依赖 workspace 已有依赖（易踩坑 8）。
//!
//! 模块：
//! - `unmasking`：UnmaskingTrees 迭代单特征解掩 GBDT 填补（M2-A，arXiv:2407.05593）
//! - `baltobot`：BaltoBot 平衡树 of boosted 回归器 → 非参条件分布（M2-A，arXiv:2407.05593）
//! - `forest_flow`：GBDT×Flow Matching 向量场（M2-B，ForestDiffusion arXiv:2309.09968 思路）

#![forbid(unsafe_code)]

pub mod baltobot;
pub mod dataset_util;
pub mod forest_flow;
pub mod unmasking;

/// 实验 crate 统一错误类型（错误显式传播，易踩坑 10）。
#[derive(Debug, thiserror::Error)]
pub enum ExperimentError {
    #[error("arrow 错误: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("数据错误: {0}")]
    Data(#[from] sooboost_core::data::DataError),
    #[error("提升错误: {0}")]
    Boosting(#[from] sooboost_core::boosting::BoostingError),
    #[error("参数/输入不合法: {0}")]
    InvalidInput(String),
    #[error("实验内部状态缺失: {0}")]
    Missing(&'static str),
}
