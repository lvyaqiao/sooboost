//! 分箱结果矩阵：特征主序的 u16 bin id 表，供直方图构建（S5）消费。

use arrow::array::Array;

use crate::data::dataset::Dataset;

use super::bintable::{BinTable, MISSING_BIN, is_missing};
use super::error::BinningError;

/// 分箱后的训练矩阵（特征主序：`bins[feature * num_rows + row]`）。
///
/// 只读缓存友好布局；本结构是 binning 的派生产物，不随模型序列化
/// （模型只存 BinTable，训练/预测各自分箱）。
#[derive(Debug, Clone)]
pub struct BinnedMatrix {
    num_features: usize,
    num_rows: usize,
    bins: Vec<u16>,
}

impl BinnedMatrix {
    /// 第 `feature` 特征、第 `row` 行的 bin id（缺失值 = `MISSING_BIN`）。
    pub fn bin(&self, feature: usize, row: usize) -> u16 {
        self.bins[feature * self.num_rows + row]
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// 按 BinTable 对 Dataset 全量分箱。
    pub fn build(ds: &Dataset, table: &BinTable) -> Result<Self, BinningError> {
        debug_assert_eq!(ds.num_features(), table.num_features());
        let num_features = ds.num_features();
        let num_rows = ds.num_rows();
        let mut bins = vec![0u16; num_features * num_rows];
        let policy = ds.missing_policy();
        for f in 0..num_features {
            let vals = ds.feature_values(f)?;
            let base = f * num_rows;
            for r in 0..num_rows {
                let value = vals.value(r);
                bins[base + r] = if is_missing(value, vals.is_null(r), policy) {
                    MISSING_BIN
                } else {
                    table.bin_value(f, value)
                };
            }
        }
        Ok(Self {
            num_features,
            num_rows,
            bins,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, MissingPolicy};
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn tiny_dataset() -> Dataset {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("f1", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, f64::NAN])),
                Arc::new(Float64Array::from(vec![
                    Some(5.0),
                    None,
                    Some(7.0),
                    Some(8.0),
                    Some(9.0),
                ])),
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 1.0])),
            ],
        )
        .expect("构造 batch");
        Dataset::from_record_batch(batch, &["f0", "f1"], "target", MissingPolicy::default())
            .expect("构造 Dataset")
    }

    #[test]
    fn missing_rows_map_to_missing_bin() {
        let ds = tiny_dataset();
        let (table, matrix) = BinTable::build_from_dataset(&ds, 16).expect("分箱");
        assert_eq!(matrix.num_features(), 2);
        assert_eq!(matrix.num_rows(), 5);
        // f0 行 4 是 NaN → MISSING_BIN
        assert_eq!(matrix.bin(0, 4), MISSING_BIN);
        // f1 行 1 是 null → MISSING_BIN
        assert_eq!(matrix.bin(1, 1), MISSING_BIN);
        // 其余非缺失
        for (f, r) in [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (1, 4),
        ] {
            assert_ne!(matrix.bin(f, r), MISSING_BIN);
        }
        assert!(table.boundaries(0).windows(2).all(|w| w[0] < w[1]));
        assert!(table.boundaries(1).windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn keep_nan_treats_nan_as_value() {
        let ds = tiny_dataset();
        let (_, matrix) = BinTable::build_from_dataset(&ds, 16).expect("分箱");
        // 默认 TreatNanAsMissing：f0 行4 = NaN → missing
        assert_eq!(matrix.bin(0, 4), MISSING_BIN);

        // KeepNan：NaN 不是缺失，应落入某个非缺失 bin
        let ds2 =
            Dataset::from_record_batch(ds_batch(), &["f0", "f1"], "target", MissingPolicy::KeepNan)
                .expect("构造 Dataset");
        let (_, matrix2) = BinTable::build_from_dataset(&ds2, 16).expect("分箱");
        assert_ne!(matrix2.bin(0, 4), MISSING_BIN);
    }

    fn ds_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("f1", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, f64::NAN])),
                Arc::new(Float64Array::from(vec![
                    Some(5.0),
                    None,
                    Some(7.0),
                    Some(8.0),
                    Some(9.0),
                ])),
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 1.0])),
            ],
        )
        .expect("构造 batch")
    }
}
