//! Pseudo-Huber 回归损失（D8：M1 回归面之一）。
//!
//! 采用 XGBoost `reg:pseudohubererror` 同款平滑近似（而非分段 huber），
//! 处处二次可导 → hess 恒正，牛顿步无除零/死区。delta → 大时逼近 L2；
//! delta → 小时逼近 L1（更抗离群点）。

use super::r#trait::Loss;

/// Pseudo-Huber 损失，`delta > 0` 控制鲁棒性。
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    pub delta: f64,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

/// 常用鲁棒回归损失，MSE 可解释性。
impl Loss for HuberLoss {
    fn name(&self) -> &'static str {
        "huber"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        let d = (y - pred) / self.delta;
        self.delta * self.delta * ((1.0 + d * d).sqrt() - 1.0)
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        y.iter().sum::<f64>() / y.len() as f64
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        let r = pred - y;
        let t = 1.0 + (r / self.delta) * (r / self.delta);
        r / t.sqrt()
    }

    fn hessian(&self, y: f64, pred: f64) -> f64 {
        let r = pred - y;
        let t = 1.0 + (r / self.delta) * (r / self.delta);
        1.0 / (t * t.sqrt())
    }

    fn transform(&self, raw: f64) -> f64 {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const H: f64 = 1e-5;

    fn numeric_gradient(l: &HuberLoss, y: f64, p: f64) -> f64 {
        (l.value(y, p + H) - l.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(l: &HuberLoss, y: f64, p: f64) -> f64 {
        (l.gradient(y, p + H) - l.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(delta in 0.1f64..10.0, y in -50.0f64..50.0, p in -50.0f64..50.0) {
            let l = HuberLoss { delta };
            prop_assert!((l.gradient(y, p) - numeric_gradient(&l, y, p)).abs() < 1e-3);
        }
        #[test]
        fn hessian_matches_numeric(delta in 0.1f64..10.0, y in -50.0f64..50.0, p in -50.0f64..50.0) {
            let l = HuberLoss { delta };
            prop_assert!((l.hessian(y, p) - numeric_hessian(&l, y, p)).abs() < 1e-3);
        }
        #[test]
        fn hessian_positive(delta in 0.1f64..10.0, y in -50.0f64..50.0, p in -50.0f64..50.0) {
            let l = HuberLoss { delta };
            prop_assert!(l.hessian(y, p) > 0.0);
        }
    }

    #[test]
    fn init_is_mean_and_transform_identity() {
        let l = HuberLoss::default();
        assert!((l.init_score(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
        assert_eq!(l.transform(7.0), 7.0);
        assert_eq!(l.name(), "huber");
    }
}
