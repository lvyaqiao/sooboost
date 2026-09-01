//! 评估指标（M6-2）：R²（回归）与 ROC-AUC（二分类）。
//!
//! 设计约束：
//! - 确定性（红线 3）：AUC 的秩计算使用稳定排序，并列取平均秩（Mann-Whitney U）；
//! - 显式退化（易踩坑 5/10）：退化输入（空输入 / 单一类 / 零方差）显式报错，
//!   不返回 NaN 静默毒化下游比较。

/// 指标计算错误。
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// 输入为空或长度不一致。
    #[error("指标输入非法：y 长度 {y_len} 与 pred 长度 {pred_len} 必须一致且非空")]
    LengthMismatch {
        /// 真值长度。
        y_len: usize,
        /// 预测长度。
        pred_len: usize,
    },
    /// AUC 需要正负两类都出现。
    #[error("AUC 需要正负两类样本（正类 {pos} 个，负类 {neg} 个）")]
    NeedsBothClasses {
        /// 正类样本数。
        pos: usize,
        /// 负类样本数。
        neg: usize,
    },
    /// R² 的真值方差为 0（全部相等），无法定义解释方差。
    #[error("R² 退化：真值方差为 0（全部相等）")]
    DegenerateVariance,
    /// 类别标签非法（带小数或负值，无法解释为类别）。
    #[error("类别标签非法：{value}（必须为非负整数）")]
    InvalidClassLabel {
        /// 实际值。
        value: f64,
    },
}

/// 决定系数 R² = 1 − SS_res / SS_tot。
///
/// 真值方差为 0 时显式报错（不静默返回 0 或 1）。
pub fn r2_score(y: &[f64], pred: &[f64]) -> Result<f64, MetricsError> {
    if y.is_empty() || y.len() != pred.len() {
        return Err(MetricsError::LengthMismatch {
            y_len: y.len(),
            pred_len: pred.len(),
        });
    }
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let ss_tot: f64 = y.iter().map(|&v| (v - mean) * (v - mean)).sum();
    if ss_tot == 0.0 {
        return Err(MetricsError::DegenerateVariance);
    }
    let ss_res: f64 = y
        .iter()
        .zip(pred.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok(1.0 - ss_res / ss_tot)
}

/// ROC-AUC（Mann-Whitney U 统计量，并列概率取平均秩）。
///
/// 标签语义：`y > 0.5` 视为正类（与门面二分类的 0/1 标签约定一致）。
pub fn roc_auc(y: &[f64], prob: &[f64]) -> Result<f64, MetricsError> {
    if y.is_empty() || y.len() != prob.len() {
        return Err(MetricsError::LengthMismatch {
            y_len: y.len(),
            pred_len: prob.len(),
        });
    }
    let pos = y.iter().filter(|&&v| v > 0.5).count();
    let neg = y.len() - pos;
    if pos == 0 || neg == 0 {
        return Err(MetricsError::NeedsBothClasses { pos, neg });
    }

    // 按 prob 升序稳定排序；并列组共享平均秩（秩从 1 起）。
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| prob[a].total_cmp(&prob[b]));

    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < order.len() {
        // 找到并列组的边界
        let mut j = i + 1;
        while j < order.len() && prob[order[j]] == prob[order[i]] {
            j += 1;
        }
        // 平均秩 = (起秩 + 止秩) / 2，秩从 1 起
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for &idx in &order[i..j] {
            if y[idx] > 0.5 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }

    // U = R_pos − n_pos(n_pos+1)/2；AUC = U / (n_pos·n_neg)
    let n_pos = pos as f64;
    let n_neg = neg as f64;
    let u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0;
    Ok(u / (n_pos * n_neg))
}

/// 多分类准确率（预测类别与真值精确匹配的比例；M6-5a）。
///
/// `pred` 为 argmax 类别（非负整数标签）；带小数或负值视为非法输入显式报错。
pub fn accuracy(y: &[f64], pred: &[f64]) -> Result<f64, MetricsError> {
    if y.is_empty() || y.len() != pred.len() {
        return Err(MetricsError::LengthMismatch {
            y_len: y.len(),
            pred_len: pred.len(),
        });
    }
    let mut hits = 0usize;
    for (&t, &p) in y.iter().zip(pred.iter()) {
        if p.fract() != 0.0 || p < 0.0 {
            return Err(MetricsError::InvalidClassLabel { value: p });
        }
        if t == p {
            hits += 1;
        }
    }
    Ok(hits as f64 / y.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2_perfect_and_known_values() {
        let y = [1.0, 2.0, 3.0, 4.0];
        // 完美预测 → 1
        assert!((r2_score(&y, &y).expect("r2") - 1.0).abs() < 1e-12);
        // 预测均值 → 0
        let mean_pred = [2.5; 4];
        assert!(r2_score(&y, &mean_pred).expect("r2").abs() < 1e-12);
        // 常数真值 → 显式报错（不静默）
        assert!(matches!(
            r2_score(&[3.0, 3.0], &[1.0, 2.0]),
            Err(MetricsError::DegenerateVariance)
        ));
        // 长度不符 → 显式报错
        assert!(matches!(
            r2_score(&[1.0, 2.0], &[1.0]),
            Err(MetricsError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn auc_known_values() {
        // 完美分离 → 1
        let y = [0.0, 0.0, 1.0, 1.0];
        let p = [0.1, 0.2, 0.8, 0.9];
        assert!((roc_auc(&y, &p).expect("auc") - 1.0).abs() < 1e-12);
        // 完全反向 → 0
        let p_rev = [0.9, 0.8, 0.2, 0.1];
        assert!(roc_auc(&y, &p_rev).expect("auc").abs() < 1e-12);
        // 全同分（并列平均秩）→ 0.5
        let p_same = [0.5; 4];
        assert!((roc_auc(&y, &p_same).expect("auc") - 0.5).abs() < 1e-12);
        // 单一类 → 显式报错
        assert!(matches!(
            roc_auc(&[1.0, 1.0], &[0.1, 0.9]),
            Err(MetricsError::NeedsBothClasses { pos: 2, neg: 0 })
        ));
    }

    #[test]
    fn auc_handles_ties_with_average_rank() {
        // 手工案例：4 样本，1 个并列组横跨正负类
        // prob=[0.5, 0.5, 0.1, 0.9]，y=[1, 0, 0, 1]
        // 秩：0.1→1, 0.5 组平均秩 (2+3)/2=2.5, 0.9→4
        // R_pos = 2.5 + 4 = 6.5；U = 6.5 − 2·3/2 = 3.5；AUC = 3.5/4 = 0.875
        let y = [1.0, 0.0, 0.0, 1.0];
        let p = [0.5, 0.5, 0.1, 0.9];
        assert!((roc_auc(&y, &p).expect("auc") - 0.875).abs() < 1e-12);
    }
}
