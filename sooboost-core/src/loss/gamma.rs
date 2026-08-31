//! Gamma 回归损失（D8：M1 回归面之一；log link，inverse-scale 形状）。
//!
//! 原始分数 q = log(μ)，μ = exp(q)。loss = y·exp(−q) + q（Gamma 单位形状负对数似然）；
//! grad = 1 − y·exp(−q)；hess = y·exp(−q)。适用于严格正目标 y > 0。

use super::r#trait::Loss;

/// Gamma 回归损失（log link，单位形状）。
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaLoss;

impl Loss for GammaLoss {
    fn name(&self) -> &'static str {
        "gamma"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        y * (-pred).exp() + pred
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        (y.iter().sum::<f64>() / y.len() as f64).ln()
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        1.0 - y * (-pred).exp()
    }

    fn hessian(&self, y: f64, pred: f64) -> f64 {
        y * (-pred).exp()
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
        (GammaLoss.value(y, p + H) - GammaLoss.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(y: f64, p: f64) -> f64 {
        (GammaLoss.gradient(y, p + H) - GammaLoss.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(y in 1e-3f64..1e3, p in -10.0f64..10.0) {
            prop_assert!((GammaLoss.gradient(y, p) - numeric_gradient(y, p)).abs() < 1e-3);
        }
        #[test]
        fn hessian_matches_numeric(y in 1e-3f64..1e3, p in -10.0f64..10.0) {
            prop_assert!((GammaLoss.hessian(y, p) - numeric_hessian(y, p)).abs() < 1e-3);
        }
    }

    #[test]
    fn init_is_log_mean_and_transform_exp() {
        assert!((GammaLoss.init_score(&[1.0, 2.0, 3.0]) - 2.0f64.ln()).abs() < 1e-12);
        assert!((GammaLoss.transform(0.0) - 1.0).abs() < 1e-12);
        assert_eq!(GammaLoss.name(), "gamma");
    }
}
