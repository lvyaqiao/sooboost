//! Quantile（pinball）回归损失（D8：M1 回归面之一；alpha=0.5 即 L1/MAE）。
//!
//! loss = max(alpha·(y−p), (alpha−1)·(y−p))，分段线性。
//! 二阶导处处为 0 → hess 返回常数 1（牛顿步退化为梯度下降，与 LightGBM
//! quantile 特殊处理同思路）；预测为条件分位数。

use super::r#trait::Loss;

/// Quantile 回归损失，`alpha ∈ (0,1)` 为目标分位。
#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    pub alpha: f64,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self { alpha: 0.5 } // 即 L1 回归
    }
}

impl Loss for QuantileLoss {
    fn name(&self) -> &'static str {
        "quantile"
    }

    fn value(&self, y: f64, pred: f64) -> f64 {
        let r = y - pred;
        if r >= 0.0 {
            self.alpha * r
        } else {
            (self.alpha - 1.0) * r
        }
    }

    fn init_score(&self, y: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        let mut s = y.to_vec();
        s.sort_by(f64::total_cmp);
        // 样本分位数（线性插值，alpha∈[0,1]）
        let n = s.len();
        let pos = self.alpha * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            s[lo]
        } else {
            let frac = pos - lo as f64;
            s[lo] * (1.0 - frac) + s[hi] * frac
        }
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        if y >= pred {
            -self.alpha
        } else {
            1.0 - self.alpha
        }
    }

    fn hessian(&self, _y: f64, _pred: f64) -> f64 {
        1.0 // 分段线性二阶为零；常数使牛顿步退化为一阶
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

    fn numeric_gradient(l: &QuantileLoss, y: f64, p: f64) -> f64 {
        (l.value(y, p + H) - l.value(y, p - H)) / (2.0 * H)
    }

    proptest! {
        #[test]
        fn gradient_matches_numeric(alpha in 0.05f64..0.95, y in -50.0f64..50.0, p in -50.0f64..50.0) {
            // 避开 y==p 折点
            prop_assume!((y - p).abs() > 1e-3);
            let l = QuantileLoss { alpha };
            prop_assert!((l.gradient(y, p) - numeric_gradient(&l, y, p)).abs() < 1e-3);
        }
        #[test]
        fn value_is_nonnegative(alpha in 0.05f64..0.95, y in -50.0f64..50.0, p in -50.0f64..50.0) {
            let l = QuantileLoss { alpha };
            prop_assert!(l.value(y, p) >= -1e-12);
        }
    }

    #[test]
    fn init_is_sample_quantile() {
        let l = QuantileLoss { alpha: 0.5 };
        assert!((l.init_score(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-12);
        assert!((l.init_score(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
        let lo = QuantileLoss { alpha: 0.0 };
        assert!((lo.init_score(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
        let hi = QuantileLoss { alpha: 1.0 };
        assert!((hi.init_score(&[1.0, 2.0, 3.0]) - 3.0).abs() < 1e-12);
        assert_eq!(l.transform(7.0), 7.0);
        assert_eq!(l.name(), "quantile");
    }
}
