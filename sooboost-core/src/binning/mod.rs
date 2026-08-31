//! 分箱层：确定性 quantile binning + bin 表（架构 D4）。
//!
//! M0 实现（doc/archive/m0-spec.md §2 承诺 6）：排序精确分位，无 seed 也天然确定
//! （红线 3 层级一）；M1 换带 seed 的 quantile sketch。
//! 语义：非缺失值 x 落入 bin = 边界中小于 x 的个数（边界为含上界，与树阈值
//! `x <= threshold` 一致）；缺失值恒为 `MISSING_BIN`。

pub mod bintable;
pub mod error;
pub mod matrix;

pub use bintable::{BinTable, DEFAULT_MAX_BINS, MISSING_BIN};
pub use error::BinningError;
pub use matrix::BinnedMatrix;
