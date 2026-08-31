//! 损失函数层（架构 D3）。
//!
//! M0（doc/archive/m0-spec.md）：`Loss` 为 monomorphized 泛型 trait，训练内核
//! 编译期内联（零回调）；内置 L2 与 binary logloss。
//! M1（doc/plans/m1-spec.md M1-2，D8）：回归面扩展 huber/quantile/poisson/gamma/tweedie。

pub mod binary_logloss;
pub mod gamma;
pub mod huber;
pub mod poisson;
pub mod quantile;
pub mod squared_error;
pub mod r#trait;
pub mod tweedie;

pub use binary_logloss::BinaryLogLoss;
pub use gamma::GammaLoss;
pub use huber::HuberLoss;
pub use poisson::PoissonLoss;
pub use quantile::QuantileLoss;
pub use squared_error::SquaredError;
pub use r#trait::Loss;
pub use tweedie::TweedieLoss;
