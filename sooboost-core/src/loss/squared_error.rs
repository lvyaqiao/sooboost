//! L2 回归损失（squared error，m0-spec §2 承诺 4）。
//! loss = 0.5·(y − p)²；grad = p − y；hess = 1；init = 均值；transform = 恒等。

use super::r#trait::Loss;

/// L2 平方误差损失。
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredError;

impl Loss for SquaredError {
    fn name(&self) -> &'static str {
        "squared_error"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        0.5 * (y - pred) * (y - pred)
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        y.iter().sum::<f64>() / y.len() as f64
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        pred - y
    }

    fn hessian(&self, _y: f64, _pred: f64) -> f64 {
        1.0
    }

    fn transform(&self, raw: f64) -> f64 {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // 平方损失是二次函数，中心差分无截断误差；h 取大可抑制相消误差。
    const H: f64 = 1e-4;

    fn numeric_gradient(y: f64, p: f64) -> f64 {
        (SquaredError.value(y, p + H) - SquaredError.value(y, p - H)) / (2.0 * H)
    }

    fn numeric_hessian(y: f64, p: f64) -> f64 {
        (SquaredError.gradient(y, p + H) - SquaredError.gradient(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(y in -1e3f64..1e3, p in -1e3f64..1e3) {
            prop_assert!((SquaredError.gradient(y, p) - numeric_gradient(y, p)).abs() < 1e-4);
        }
        #[test]
        fn hessian_is_one(y in -1e3f64..1e3, p in -1e3f64..1e3) {
            prop_assert!((SquaredError.hessian(y, p) - 1.0).abs() < 1e-12);
            prop_assert!((SquaredError.hessian(y, p) - numeric_hessian(y, p)).abs() < 1e-4);
        }
    }

    #[test]
    fn init_is_mean_and_transform_is_identity() {
        assert_eq!(SquaredError.init_score(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(SquaredError.init_score(&[]), 0.0);
        assert_eq!(SquaredError.transform(7.5), 7.5);
        assert_eq!(SquaredError.name(), "squared_error");
    }
}
