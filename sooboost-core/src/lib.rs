//! sooboost-core：纯 Rust GBDT 核心库。
//!
//! 架构见 doc/baseline/architecture.md：
//! 分层（依赖单向向下）——部署面 → 训练层 → 树内核 → 数据层。
//! M0 范围见 doc/plans/m0-spec.md：数据层 + binning + 单线程直方图 + L2/binary logloss。

// 红线 7：no_unsafe 默认红线。M0 无 SIMD/零拷贝边界，全库禁止 unsafe；
// 将来引入 SIMD/零拷贝边界时改为 deny(unsafe_op_in_unsafe_fn) + 逐处审查注释。
#![forbid(unsafe_code)]

pub mod binning;
pub mod boosting;
pub mod data;
pub mod loss;
pub mod model;
pub mod tree;
