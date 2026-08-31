//! 二分类 logloss 损失（m0-spec §2 承诺 5）。
//! 原始分数 p（对数几率），概率 = sigmoid(p)；
//! grad = sigmoid(p) − y；hess = sigmoid(p)·(1−sigmoid(p))；
//! init = ln(mean(y)/(1−mean(y)))（带 1e-12 截断防除零）。

use super::r#trait::Loss;

/// 二分类 logloss 损失。
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryLogLoss;

impl Loss for BinaryLogLoss {
    fn name(&self) -> &'static str {
        "binary_logloss"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        let p = sigmoid(pred);
        -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let p = (y.iter().sum::<f64>() / y.len() as f64).clamp(1e-12, 1.0 - 1e-12);
        (p / (1.0 - p)).ln()
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        sigmoid(pred) - y
    }

    fn hessian(&self, _y: f64, pred: f64) -> f64 {
        let s = sigmoid(pred);
        s * (1.0 - s)
    }

    fn transform(&self, raw: f64) -> f64 {
        sigmoid(raw)
    }
}

/// 数值稳定的 sigmoid：对 |x| 大时避免 exp 溢出。
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const H: f64 = 1e-6;

    fn numeric_gradient(y: f64, p: f64) -> f64 {
        (BinaryLogLoss.value(y, p + H) - BinaryLogLoss.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(y: f64, p: f64) -> f64 {
        (BinaryLogLoss.gradient(y, p + H) - BinaryLogLoss.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(y in 0i8..=1i8, p in -10.0f64..10.0) {
            let y = y as f64;
            prop_assert!((BinaryLogLoss.gradient(y, p) - numeric_gradient(y, p)).abs() < 1e-4);
        }
        #[test]
        fn hessian_matches_numeric(y in 0i8..=1i8, p in -10.0f64..10.0) {
            let y = y as f64;
            prop_assert!((BinaryLogLoss.hessian(y, p) - numeric_hessian(y, p)).abs() < 1e-4);
        }
        #[test]
        fn sigmoid_is_between_0_and_1(x in -30.0f64..30.0) {
            let s = sigmoid(x);
            prop_assert!(s > 0.0 && s < 1.0);
        }
        #[test]
        fn sigmoid_is_symmetric(x in -20.0f64..20.0) {
            prop_assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn init_is_log_odds_of_mean() {
        let s = BinaryLogLoss;
        assert!((s.init_score(&[0.0, 0.0, 1.0]) - 0.5f64.ln()).abs() < 1e-12);
        assert_eq!(s.init_score(&[]), 0.0);
        // 全 0 / 全 1 不崩溃（截断）
        assert!(s.init_score(&[0.0, 0.0]).is_finite());
        assert!(s.init_score(&[1.0, 1.0]).is_finite());
    }

    #[test]
    fn transform_maps_zero_to_half() {
        assert!((BinaryLogLoss.transform(0.0) - 0.5).abs() < 1e-12);
        assert!(BinaryLogLoss.transform(10.0) > 0.999);
        assert!(BinaryLogLoss.transform(-10.0) < 0.001);
    }
}
