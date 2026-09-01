//! 训练层：GBDT 提升循环与训练产物（架构 分层中的「训练层」）。
//!
//! M0 范围（doc/archive/m0-spec.md §2 承诺 4/5/8/9）：
//! - 仅内置 L2（SquaredError）与 binary logloss（BinaryLogLoss）两种损失；
//! - `fit`：分箱 → init 偏置 → 迭代「梯度/海森 → 建树 → 学习率累加」；
//! - `TrainingContext`：显式传递的运行时状态载体（红线 4，零全局状态）；
//! - 确定性：无随机路径，同输入同 seed → 模型与预测逐位一致（红线 3 层级一）。
//!
//! M0 明确不做（m0-spec §7）：subsample、特征采样、early stopping、
//! 模型序列化（M1）、多分类、其他损失。

pub mod booster;
pub mod context;
pub mod error;
pub mod multiclass;
pub mod params;

pub use booster::{Booster, EarlyStopping, ImportanceKind, fit, fit_with_early_stopping};
pub use context::TrainingContext;
pub use error::BoostingError;
pub use multiclass::{MulticlassBooster, fit_multiclass, fit_multiclass_with_early_stopping};
pub use params::BoostingParams;
