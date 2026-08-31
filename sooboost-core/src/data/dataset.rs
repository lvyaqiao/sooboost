//! 数据集：arrow RecordBatch 零拷贝视图（架构 D1）。
//!
//! M0 范围（doc/archive/m0-spec.md §3）：数值特征 + 单个数值 target；
//! 缺失值语义一律经本模块查询（红线 2 唯一定义点）。

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, DictionaryArray, Float64Array};
use arrow::datatypes::{DataType, SchemaRef, UInt16Type, UInt32Type};
use arrow::record_batch::RecordBatch;

use super::error::DataError;
use super::missing::MissingPolicy;

/// 训练数据集。
///
/// 构造时做 schema 校验（列存在 / 特征全 Float64 或 Dictionary / 特征与 target 不重叠），
/// 之后特征读取为零拷贝借用，无中间副本；target 若为非 Float64 数值类型，
/// 构造时显式 cast 一次为 Float64 并持有（contracts.md §1.1 允许的一次显式转换）。
///
/// 类别特征（M1-4，D9）：以 `DictionaryArray` 进入，特征读取经 `categorical_key`
/// 取类别键（u32），缺失经 `is_missing` 统一判定（红线 2）。
#[derive(Debug, Clone)]
pub struct Dataset {
    batch: RecordBatch,
    feature_indices: Vec<usize>,
    feature_names: Vec<String>,
    feature_categorical: Vec<bool>,
    target: Float64Array,
    target_name: String,
    missing_policy: MissingPolicy,
}

impl Dataset {
    /// 从 RecordBatch 构造；校验失败显式返回错误（禁止静默降级）。
    pub fn from_record_batch(
        batch: RecordBatch,
        feature_names: &[&str],
        target_name: &str,
        missing_policy: MissingPolicy,
    ) -> Result<Self, DataError> {
        let schema = batch.schema();
        let mut feature_indices = Vec::with_capacity(feature_names.len());
        let mut feature_categorical = Vec::with_capacity(feature_names.len());
        for name in feature_names {
            check_feature_column(&schema, name)?;
            let idx = column_index(&schema, name)?;
            feature_indices.push(idx);
            feature_categorical.push(is_dictionary(&schema, name));
        }
        let target_index = column_index(&schema, target_name)?;
        if feature_indices.contains(&target_index) {
            return Err(DataError::FeatureTargetOverlap(target_name.to_string()));
        }
        if batch.num_rows() == 0 {
            return Err(DataError::EmptyDataset);
        }
        let target_col = batch.column(target_index);
        let target = coerce_numeric_to_f64(target_col, target_name)?;
        Ok(Self {
            batch,
            feature_indices,
            feature_names: feature_names.iter().map(|s| s.to_string()).collect(),
            feature_categorical,
            target,
            target_name: target_name.to_string(),
            missing_policy,
        })
    }

    /// 从 CSV 读取：读全量批次后合并为单个 RecordBatch（M0 数据形态约定为
    /// benchmark/ 金标准数据集 train.csv / test.csv）。
    pub fn from_csv_path(
        path: impl AsRef<Path>,
        feature_names: &[&str],
        target_name: &str,
        missing_policy: MissingPolicy,
    ) -> Result<Self, DataError> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_csv_bytes(&bytes, feature_names, target_name, missing_policy)
    }

    /// 从 CSV 字节流读取（M2-0 fuzz 目标 csv_parse 共用；红线 6 读入外部字节路径）。
    pub fn from_csv_bytes(
        bytes: &[u8],
        feature_names: &[&str],
        target_name: &str,
        missing_policy: MissingPolicy,
    ) -> Result<Self, DataError> {
        let format = arrow_csv::reader::Format::default().with_header(true);
        let (inferred_schema, _) = format.infer_schema(Cursor::new(bytes), Some(10_000))?;
        let schema = Arc::new(inferred_schema);
        let reader = arrow_csv::ReaderBuilder::new(Arc::clone(&schema))
            .with_header(true)
            .build(Cursor::new(bytes))?;
        let batches: Result<Vec<RecordBatch>, _> = reader.collect();
        let batches = batches?;
        let batch = arrow::compute::concat_batches(&schema, &batches)?;
        Self::from_record_batch(batch, feature_names, target_name, missing_policy)
    }

    // -- 元信息 ----------------------------------------------------------

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn num_features(&self) -> usize {
        self.feature_indices.len()
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// 第 `feature_idx` 个特征是否为类别特征（Dictionary 列）。
    pub fn feature_is_categorical(&self, feature_idx: usize) -> bool {
        self.feature_categorical[feature_idx]
    }

    /// 类别特征在第 `row` 行的类别键（字典索引，u32）。
    ///
    /// - 返回 `None` = 该行该特征缺失（arrow null，红线 2）；
    /// - 键是字典数组的 key（与字典值一一对应），供 ordered TS 分组。
    pub fn categorical_key(
        &self,
        row: usize,
        feature_idx: usize,
    ) -> Result<Option<u32>, DataError> {
        let col = self.batch.column(self.feature_indices[feature_idx]);
        dict_row_key(col, row).map_err(|_| DataError::UnsupportedDictionary {
            name: self.feature_names[feature_idx].clone(),
            ty: col.data_type().clone(),
        })
    }

    /// 类别特征键 → 值的映射（训练期统一字典，OOV 在预测期经 CategoricalEncoding 处理）。
    /// 返回 `None` 若该特征非类别。
    pub fn categorical_dictionary_len(
        &self,
        feature_idx: usize,
    ) -> Result<Option<usize>, DataError> {
        if !self.feature_categorical[feature_idx] {
            return Ok(None);
        }
        let col = self.batch.column(self.feature_indices[feature_idx]);
        dict_values_len(col)
            .map(Some)
            .ok_or_else(|| DataError::UnsupportedDictionary {
                name: self.feature_names[feature_idx].clone(),
                ty: col.data_type().clone(),
            })
    }

    pub fn target_name(&self) -> &str {
        &self.target_name
    }

    pub fn missing_policy(&self) -> MissingPolicy {
        self.missing_policy
    }

    // -- 零拷贝取值 ------------------------------------------------------

    /// 第 `feature_idx` 个特征列的数值视图（零拷贝借用；类别特征请用 categorical_key）。
    pub fn feature_values(&self, feature_idx: usize) -> Result<&Float64Array, DataError> {
        if self.feature_categorical[feature_idx] {
            return Err(DataError::CategoricalFeatureNotNumeric(
                self.feature_names[feature_idx].clone(),
            ));
        }
        self.float64_column(
            self.feature_indices[feature_idx],
            &self.feature_names[feature_idx],
        )
    }

    /// target 列数值视图（已统一为 Float64）。
    pub fn target_values(&self) -> &Float64Array {
        &self.target
    }

    /// 缺失查询（红线 2 唯一定义点）：null 位图 + 按策略的 NaN。
    /// 类别特征缺失 = null（字典值非浮点，NaN 语义不适用）。
    pub fn is_missing(&self, row: usize, feature_idx: usize) -> Result<bool, DataError> {
        if self.feature_categorical[feature_idx] {
            let col = self.batch.column(self.feature_indices[feature_idx]);
            return Ok(col.is_null(row));
        }
        let values = self.feature_values(feature_idx)?;
        Ok(super::missing::is_missing_value(
            values.value(row),
            values.is_null(row),
            self.missing_policy,
        ))
    }

    fn float64_column(&self, idx: usize, name: &str) -> Result<&Float64Array, DataError> {
        self.batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| DataError::UnsupportedType {
                name: name.to_string(),
                ty: self.batch.column(idx).data_type().clone(),
            })
    }
}

fn column_index(schema: &SchemaRef, name: &str) -> Result<usize, DataError> {
    schema
        .index_of(name)
        .map_err(|_| DataError::ColumnNotFound(name.to_string()))
}

fn check_feature_column(schema: &SchemaRef, name: &str) -> Result<(), DataError> {
    let field = schema
        .field_with_name(name)
        .map_err(|_| DataError::ColumnNotFound(name.to_string()))?;
    match field.data_type() {
        DataType::Float64 => Ok(()),
        DataType::Dictionary(k, _) if is_supported_dict_key(k) => Ok(()),
        other => Err(DataError::UnsupportedType {
            name: name.to_string(),
            ty: other.clone(),
        }),
    }
}

/// 支持 UInt16/UInt32 键的字典（M1-4；UInt32 为 arrow 常见默认，UInt16 兼顾小字典）。
fn is_supported_dict_key(k: &DataType) -> bool {
    matches!(k, DataType::UInt16 | DataType::UInt32)
}

fn is_dictionary(schema: &SchemaRef, name: &str) -> bool {
    matches!(
        schema.field_with_name(name).map(|f| f.data_type()),
        Ok(DataType::Dictionary(_, _))
    )
}

/// 字典列第 `row` 行的键（u32）。支持 UInt16/UInt32 键；null → None；类型不支持 → Err。
fn dict_row_key(col: &dyn Array, row: usize) -> Result<Option<u32>, ()> {
    if let Some(d) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
        let k = d.keys();
        return Ok(if k.is_null(row) {
            None
        } else {
            Some(k.value(row))
        });
    }
    if let Some(d) = col.as_any().downcast_ref::<DictionaryArray<UInt16Type>>() {
        let k = d.keys();
        return Ok(if k.is_null(row) {
            None
        } else {
            Some(k.value(row) as u32)
        });
    }
    Err(())
}

/// 字典列字典长度（类别基数）。支持 UInt16/UInt32 键；类型不支持 → None。
fn dict_values_len(col: &dyn Array) -> Option<usize> {
    if let Some(d) = col.as_any().downcast_ref::<DictionaryArray<UInt32Type>>() {
        return Some(d.values().len());
    }
    col.as_any()
        .downcast_ref::<DictionaryArray<UInt16Type>>()
        .map(|d| d.values().len())
}

/// 将数值列显式转换为 Float64（M0 允许的一次显式转换）。
/// 特征列仍强制 Float64；target 列放宽到全部数值类型（二分类 0/1 常为整型）。
fn coerce_numeric_to_f64(col: &dyn Array, name: &str) -> Result<Float64Array, DataError> {
    let dtype = col.data_type();
    let numeric = matches!(
        dtype,
        DataType::Float64
            | DataType::Float32
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    );
    if !numeric {
        return Err(DataError::UnsupportedType {
            name: name.to_string(),
            ty: dtype.clone(),
        });
    }
    if dtype == &DataType::Float64 {
        return col
            .as_any()
            .downcast_ref::<Float64Array>()
            .cloned()
            .ok_or_else(|| DataError::UnsupportedType {
                name: name.to_string(),
                ty: dtype.clone(),
            });
    }
    let casted = arrow::compute::cast(col, &DataType::Float64)?;
    casted
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| DataError::UnsupportedType {
            name: name.to_string(),
            ty: dtype.clone(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};

    fn batch_with(feature: Float64Array, target: Float64Array) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature), Arc::new(target)])
            .expect("构造测试 batch")
    }

    #[test]
    fn access_values_and_metadata() {
        let batch = batch_with(
            Float64Array::from(vec![1.0, 2.0, 3.0]),
            Float64Array::from(vec![10.0, 20.0, 30.0]),
        );
        let ds = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .expect("构造 Dataset");
        assert_eq!(ds.num_rows(), 3);
        assert_eq!(ds.num_features(), 1);
        assert_eq!(ds.feature_names(), &["f0".to_string()]);
        assert_eq!(ds.target_name(), "target");
        assert_eq!(ds.feature_values(0).unwrap().value(1), 2.0);
        assert_eq!(ds.target_values().value(2), 30.0);
    }

    #[test]
    fn column_not_found_is_error() {
        let batch = batch_with(
            Float64Array::from(vec![1.0]),
            Float64Array::from(vec![10.0]),
        );
        let err =
            Dataset::from_record_batch(batch, &["missing"], "target", MissingPolicy::default())
                .unwrap_err();
        assert!(matches!(err, DataError::ColumnNotFound(n) if n == "missing"));
    }

    #[test]
    fn non_float_column_is_error() {
        use arrow::array::Int64Array;
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Int64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![10.0])),
            ],
        )
        .expect("构造 batch");
        let err = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .unwrap_err();
        assert!(matches!(err, DataError::UnsupportedType { name, .. } if name == "f0"));
    }

    #[test]
    fn feature_target_overlap_is_error() {
        let batch = batch_with(
            Float64Array::from(vec![1.0]),
            Float64Array::from(vec![10.0]),
        );
        let err =
            Dataset::from_record_batch(batch, &["target"], "target", MissingPolicy::default())
                .unwrap_err();
        assert!(matches!(err, DataError::FeatureTargetOverlap(n) if n == "target"));
    }

    #[test]
    fn empty_dataset_is_error() {
        let batch = batch_with(
            Float64Array::from(Vec::<f64>::new()),
            Float64Array::from(Vec::<f64>::new()),
        );
        let err = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .unwrap_err();
        assert!(matches!(err, DataError::EmptyDataset));
    }

    #[test]
    fn null_is_always_missing() {
        let batch = batch_with(
            Float64Array::from(vec![Some(1.0), None]),
            Float64Array::from(vec![10.0, 20.0]),
        );
        let ds = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::KeepNan)
            .expect("构造 Dataset");
        assert!(!ds.is_missing(0, 0).unwrap());
        assert!(ds.is_missing(1, 0).unwrap());
    }

    #[test]
    fn from_csv_bytes_parses_valid_csv() {
        let csv = b"f0,target\n1.0,10\n2.5,20\n3.0,30\n";
        let ds = Dataset::from_csv_bytes(csv, &["f0"], "target", MissingPolicy::default())
            .expect("解析合法 CSV");
        assert_eq!(ds.num_rows(), 3);
        assert_eq!(ds.num_features(), 1);
        assert_eq!(ds.feature_values(0).unwrap().value(2), 3.0);
        assert_eq!(ds.target_values().value(1), 20.0);
    }

    #[test]
    fn from_csv_bytes_null_cell_is_missing() {
        let csv = b"f0,target\n,10\n2.0,20\n";
        let ds = Dataset::from_csv_bytes(csv, &["f0"], "target", MissingPolicy::default())
            .expect("解析含空格的 CSV");
        assert!(ds.is_missing(0, 0).unwrap());
        assert!(!ds.is_missing(1, 0).unwrap());
    }

    #[test]
    fn from_csv_bytes_rejects_garbage_without_panic() {
        // 任意垃圾字节必须显式报错，绝不 panic（红线 6 / 易踩坑 10）。
        for bytes in [
            &b""[..],
            &b"not a csv at all"[..],
            &b"f0,target\n\x00\xff\xfe"[..],
            &b"f0,f1,target\n1,2,3\n4,5\n"[..],
        ] {
            let _ = Dataset::from_csv_bytes(bytes, &["f0"], "target", MissingPolicy::default());
        }
    }

    #[test]
    fn nan_missing_depends_on_policy() {
        let batch = batch_with(
            Float64Array::from(vec![f64::NAN, 1.0]),
            Float64Array::from(vec![10.0, 20.0]),
        );
        let keep =
            Dataset::from_record_batch(batch.clone(), &["f0"], "target", MissingPolicy::KeepNan)
                .expect("构造 Dataset");
        assert!(!keep.is_missing(0, 0).unwrap(), "KeepNan 时 NaN 不是缺失");

        let treat =
            Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::TreatNanAsMissing)
                .expect("构造 Dataset");
        assert!(
            treat.is_missing(0, 0).unwrap(),
            "TreatNanAsMissing 时 NaN 是缺失"
        );
        assert!(!treat.is_missing(1, 0).unwrap());
    }

    proptest::proptest! {
        /// 任意字节喂 CSV 解析绝不 panic（红线 6 / M2-0 fuzz 的 stable 回归守护）。
        #[test]
        fn from_csv_bytes_arbitrary_bytes_never_panic(
            data in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512),
        ) {
            let _ = Dataset::from_csv_bytes(&data, &["f0"], "target", MissingPolicy::default());
        }
    }
}
