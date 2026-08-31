//! Tweedie 回归损失（D8：M1 回归面之一；log link，power p ∈ (1,2)）。
//!
//! 原始分数 q = log(μ)。梯度/海森（LightGBM 同款，rho = variance power）：
//!   grad = −y·exp((1−p)·q) + exp((2−p)·q)
//!   hess = −y·(1−p)·exp((1−p)·q) + (2−p)·exp((2−p)·q)
//! p=1 → Poisson；p=2 → Gamma。适用于 y ≥ 0（复合泊松-伽马）。

use super::r#trait::Loss;

/// Tweedie 回归损失，`power ∈ (1,2)`。
#[derive(Debug, Clone, Copy)]
pub struct TweedieLoss {
    pub power: f64,
}

impl Default for TweedieLoss {
    fn default() -> Self {
        Self { power: 1.5 }
    }
}

impl Loss for TweedieLoss {
    fn name(&self) -> &'static str {
        "tweedie"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        let p = self.power;
        // 负对数似然（常数项略去）
        -y * ((1.0 - p) * pred).exp() / (1.0 - p) + ((2.0 - p) * pred).exp() / (2.0 - p)
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        (y.iter().sum::<f64>() / y.len() as f64).ln()
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        let p = self.power;
        -y * ((1.0 - p) * pred).exp() + ((2.0 - p) * pred).exp()
    }

    fn hessian(&self, y: f64, pred: f64) -> f64 {
        let p = self.power;
        -y * (1.0 - p) * ((1.0 - p) * pred).exp() + (2.0 - p) * ((2.0 - p) * pred).exp()
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

    fn numeric_gradient(l: &TweedieLoss, y: f64, p: f64) -> f64 {
        (l.value(y, p + H) - l.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(l: &TweedieLoss, y: f64, p: f64) -> f64 {
        (l.gradient(y, p + H) - l.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(power in 1.05f64..1.95, y in 0.0f64..1e3, p in -10.0f64..10.0) {
            let l = TweedieLoss { power };
            prop_assert!((l.gradient(y, p) - numeric_gradient(&l, y, p)).abs() < 1e-2);
        }
        #[test]
        fn hessian_matches_numeric(power in 1.05f64..1.95, y in 0.0f64..1e3, p in -10.0f64..10.0) {
            let l = TweedieLoss { power };
            prop_assert!((l.hessian(y, p) - numeric_hessian(&l, y, p)).abs() < 1e-2);
        }
    }

    #[test]
    fn init_is_log_mean_and_transform_exp() {
        let l = TweedieLoss::default();
        assert!((l.init_score(&[1.0, 2.0, 3.0]) - 2.0f64.ln()).abs() < 1e-12);
        assert!((l.transform(0.0) - 1.0).abs() < 1e-12);
        assert_eq!(l.name(), "tweedie");
    }
}
