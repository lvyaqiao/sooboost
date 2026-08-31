//! Poisson 回归损失（D8：M1 回归面之一；log link）。
//!
//! 原始分数 q = log(μ)，μ = exp(q)。loss = exp(q) − y·q（Poisson 负对数似然去常数）；
//! grad = exp(q) − y；hess = exp(q)。适用于非负计数目标 y ≥ 0。

use super::r#trait::Loss;

/// Poisson 回归损失（log link）。
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonLoss;

impl Loss for PoissonLoss {
    fn name(&self) -> &'static str {
        "poisson"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        pred.exp() - y * pred
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        (y.iter().sum::<f64>() / y.len() as f64).ln()
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        pred.exp() - y
    }

    fn hessian(&self, _y: f64, pred: f64) -> f64 {
        pred.exp()
    }

    fn transform(&self, raw: f64) -> f64 {
        raw.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const H: f64 = 1e-5;

    fn numeric_gradient(y: f64, p: f64) -> f64 {
        (PoissonLoss.value(y, p + H) - PoissonLoss.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(y: f64, p: f64) -> f64 {
        (PoissonLoss.gradient(y, p + H) - PoissonLoss.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(y in 0.0f64..1e3, p in -10.0f64..10.0) {
            prop_assert!((PoissonLoss.gradient(y, p) - numeric_gradient(y, p)).abs() < 1e-3);
        }
        #[test]
        fn hessian_matches_numeric(y in 0.0f64..1e3, p in -10.0f64..10.0) {
            prop_assert!((PoissonLoss.hessian(y, p) - numeric_hessian(y, p)).abs() < 1e-3);
        }
    }

    #[test]
    fn init_is_log_mean_and_transform_exp() {
        assert!((PoissonLoss.init_score(&[1.0, 2.0, 3.0]) - 2.0f64.ln()).abs() < 1e-12);
        assert!((PoissonLoss.transform(0.0) - 1.0).abs() < 1e-12);
        assert_eq!(PoissonLoss.name(), "poisson");
    }
}
