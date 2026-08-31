//! 数据层：arrow RecordBatch 之上的薄封装（架构 D1）。
//!
//! M0 范围（doc/archive/m0-spec.md §3）：
//! - 入参形态：arrow RecordBatch（数值列）；
//! - 缺失值语义唯一定义点（红线 2）：本模块是 null/NaN 语义的唯一来源；
//! - schema 校验：类型/列名不一致显式报错。

pub mod dataset;
pub mod error;
pub mod missing;
pub mod target_stats;

pub use dataset::Dataset;
pub use error::DataError;
pub use missing::MissingPolicy;
pub use target_stats::{
    CategoricalEncoding, apply_encoding, compute_ordered_ts, resolve_to_dataset,
};
