//! BinTable：每特征的升序边界集合 + 构建/查询。
//!
//! 语义（与 BinnedMatrix 共享）：非缺失值 x → `bin = partition_point(b < x)`
//! 即边界中小于 x 的个数，范围 `0..=num_bins-1`；缺失值 → `MISSING_BIN`。
//! 边界为所属 bin 的**含上界**（x == boundaries[k] 落入 bin k），与树预测
//! `x <= threshold → 左子树` 严格一致（2026-08-19 修复 off-by-one）。

use arrow::array::Array;
use serde::{Deserialize, Serialize};

use crate::data::dataset::Dataset;
use crate::data::missing::MissingPolicy;

use super::error::BinningError;
use super::matrix::BinnedMatrix;

/// 非缺失 bin 数量上限（对齐 LightGBM 默认 num_bins=255）。
pub const DEFAULT_MAX_BINS: usize = 255;

/// 缺失值专用 bin id（u16::MAX，与任何非缺失 bin id 不相交）。
pub const MISSING_BIN: u16 = u16::MAX;

/// 分箱表：每特征的升序边界。
///
/// 序列化后随模型保存（M1 模型格式，D6）；训练与预测共用同一张表（D4）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinTable {
    num_features: usize,
    max_bins: usize,
    boundaries: Vec<Vec<f64>>,
}

impl BinTable {
    /// 由已校验的边界构造（边界须严格升序、数量 ≤ max_bins-1）。
    pub fn new(boundaries: Vec<Vec<f64>>, max_bins: usize) -> Result<Self, BinningError> {
        let num_features = boundaries.len();
        for (feature, b) in boundaries.iter().enumerate() {
            if b.len() > max_bins.saturating_sub(1) {
                return Err(BinningError::TooManyBoundaries {
                    feature,
                    got: b.len(),
                    max_bins,
                });
            }
            if b.windows(2).any(|w| w[0] >= w[1]) {
                return Err(BinningError::BoundariesNotSorted { feature });
            }
        }
        Ok(Self {
            num_features,
            max_bins,
            boundaries,
        })
    }

    /// 由 Dataset 构建：排序精确分位 → BinTable + 全量 BinnedMatrix。
    pub fn build_from_dataset(
        ds: &Dataset,
        max_bins: usize,
    ) -> Result<(Self, BinnedMatrix), BinningError> {
        let mut boundaries = Vec::with_capacity(ds.num_features());
        for f in 0..ds.num_features() {
            let vals = ds.feature_values(f)?;
            let policy = ds.missing_policy();
            let mut collected = Vec::with_capacity(ds.num_rows());
            for r in 0..ds.num_rows() {
                if !is_missing(vals.value(r), vals.is_null(r), policy) {
                    collected.push(vals.value(r));
                }
            }
            boundaries.push(compute_quantile_boundaries(&collected, max_bins));
        }
        let table = Self::new(boundaries, max_bins)?;
        let matrix = BinnedMatrix::build(ds, &table)?;
        Ok((table, matrix))
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn max_bins(&self) -> usize {
        self.max_bins
    }

    /// 第 `feature` 特征的升序边界（可能为空，空 = 该特征无非缺失样本）。
    pub fn boundaries(&self, feature: usize) -> &[f64] {
        &self.boundaries[feature]
    }

    /// 第 `feature` 特征的有效非缺失 bin 数（= 边界数 + 1，含缺失 bin 为 +2）。
    pub fn num_bins(&self, feature: usize) -> usize {
        self.boundaries[feature].len() + 1
    }

    pub fn missing_bin(&self) -> u16 {
        MISSING_BIN
    }

    /// 非缺失值 → bin id（调用方须先排除缺失，否则结果无意义但确定）。
    /// 严格 `<`：边界值为所属 bin 的含上界，保证与树阈值 `x <= threshold` 一致。
    pub fn bin_value(&self, feature: usize, value: f64) -> u16 {
        self.boundaries[feature].partition_point(|b| *b < value) as u16
    }
}

/// 核心算法：升序排序后按等频取分位边界，去重相邻相等边界。
/// 确定性由 `f64::total_cmp` 排序保证（红线 3 层级一，NaN 也确定）。
/// 返回严格升序（可能为空）的边界向量。
pub fn compute_quantile_boundaries(values: &[f64], max_bins: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let max_b = max_bins.saturating_sub(1);
    let mut out = Vec::with_capacity(max_b);
    for k in 1..=max_b {
        // 用 u128 防 k*n 溢出；末位钳制到 n-1
        let p = ((k as u128 * n as u128) / max_bins as u128).min(n as u128 - 1) as usize;
        let v = sorted[p];
        if out.last().is_none_or(|last| *last < v) {
            out.push(v);
        }
    }
    out
}

/// 值级缺失判断（转发到数据层唯一来源）。
pub(crate) fn is_missing(value: f64, is_null: bool, policy: MissingPolicy) -> bool {
    crate::data::missing::is_missing_value(value, is_null, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// 边界严格升序（或为空）且数量 ≤ max_bins-1。
        #[test]
        fn boundaries_strictly_increasing(values in prop::collection::vec(any::<f64>(), 0..500), max_bins in 2usize..64) {
            let b = compute_quantile_boundaries(&values, max_bins);
            assert!(b.len() <= max_bins.saturating_sub(1));
            assert!(b.windows(2).all(|w| w[0] < w[1]));
            // 每个边界都是输入值之一（等频取点）
            for v in &b {
                assert!(values.contains(v));
            }
        }
    }

    #[test]
    fn empty_values_give_no_boundaries() {
        assert!(compute_quantile_boundaries(&[], 255).is_empty());
    }

    #[test]
    fn boundaries_within_data_range() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let b = compute_quantile_boundaries(&values, 255);
        assert!(b.len() <= 254, "边界数 ≤ max_bins-1");
        assert!(b.windows(2).all(|w| w[0] < w[1]), "严格升序");
        assert!(
            b.iter().all(|v| *v >= 0.0 && *v <= 999.0),
            "边界都在数据范围内"
        );
    }

    #[test]
    fn bin_value_is_order_preserving() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.5).collect();
        let b = compute_quantile_boundaries(&values, 64);
        let table = BinTable::new(vec![b], 64).expect("构造 BinTable");
        let mut prev = 0;
        for i in 0..1000 {
            let bin = table.bin_value(0, (i as f64) * 0.5);
            assert!(bin >= prev, "bin 单调不降");
            prev = bin;
        }
    }
}
