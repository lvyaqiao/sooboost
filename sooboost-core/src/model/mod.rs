//! 模型层：版本化模型格式 + 序列化 + 热替换（架构 D6，contracts §1.2）。
//!
//! - `format.rs`：显式小端字节布局规格 + 读写原语 + FNV-1a checksum；
//! - `io.rs`：Booster ↔ 字节（校验 magic/版本/checksum/结构/loss 名）；
//! - `hot_swap.rs`：`HotSwappable`（Arc::swap 原子发布，单写者场景）。
//!
//! 明确不依赖 serde/bincode/postcard 默认布局（contracts §1.2 字节布局显式规格）。

pub mod error;
pub mod format;
pub mod hot_swap;
pub mod io;

pub use error::ModelError;
pub use hot_swap::HotSwappable;
