//! 树内核：binning → 直方图 → 分裂搜索（架构 D2/D7）。
//!
//! M0 范围（doc/archive/m0-spec.md §4/§7）：
//! - 单线程直方图 + level-wise 逐层生长，无并行无 SIMD；
//! - 核心仅标量叶子（D7：研究方向放独立实验 crate）；
//! - 分裂/gain 公式对 L2 与 binary logloss 通用（牛顿步，λ 正则）。
//!
//! 红线 1：单一内核，变体（并行/SIMD）在 M1 以泛型组合扩展，不复制实现。

pub mod builder;
pub mod error;
pub mod histogram;
pub mod model;

pub use builder::{TreeBuilder, TreeParams};
pub use error::TreeError;
pub use model::Tree;
