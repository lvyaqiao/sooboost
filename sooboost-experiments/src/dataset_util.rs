//! 数据集构造/子集/指标助手（实验 crate 内部便利层，错误显式传播）。

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use sooboost_core::data::{Dataset, MissingPolicy};

use crate::ExperimentError;

/// 由列构造 Dataset：`feature_columns` 为 `(name, values)`，target 单列。
/// 缺失以 NaN 传入（红线 2：TreatNanAsMissing 视为缺失）。
pub fn dataset_from_columns(
    feature_columns: &[(String, Vec<f64>)],
    target_name: &str,
    target: Vec<f64>,
) -> Result<Dataset, ExperimentError> {
    let mut fields = Vec::with_capacity(feature_columns.len() + 1);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(feature_columns.len() + 1);
    for (name, col) in feature_columns {
        fields.push(Field::new(name, DataType::Float64, true));
        arrays.push(Arc::new(Float64Array::from(col.clone())) as ArrayRef);
    }
    fields.push(Field::new(target_name, DataType::Float64, true));
    arrays.push(Arc::new(Float64Array::from(target)) as ArrayRef);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    let feature_names: Vec<&str> = feature_columns.iter().map(|(n, _)| n.as_str()).collect();
    Ok(Dataset::from_record_batch(
        batch,
        &feature_names,
        target_name,
        MissingPolicy::TreatNanAsMissing,
    )?)
}

/// 由特征列直接构造 Dataset（target 与特征同一矩阵列，BaltoBot 用）。
pub fn dataset_from_matrix(
    feature_names: &[String],
    matrix: &[Vec<f64>],
    target_idx: usize,
) -> Result<Dataset, ExperimentError> {
    let cols: Vec<(String, Vec<f64>)> = feature_names
        .iter()
        .enumerate()
        .map(|(f, n)| (n.clone(), matrix[f].clone()))
        .collect();
    dataset_from_columns(
        &cols,
        &feature_names[target_idx],
        matrix[target_idx].clone(),
    )
}

/// 取子集矩阵：`rows` 指定行索引；特征列值，缺失 → NaN。
pub fn subset_matrix(
    ds: &Dataset,
    rows: &[usize],
    feature_indices: &[usize],
) -> Result<Vec<Vec<f64>>, ExperimentError> {
    let mut out = Vec::with_capacity(feature_indices.len());
    for &f in feature_indices {
        let col = ds.feature_values(f)?;
        let mut v = Vec::with_capacity(rows.len());
        for &r in rows {
            if ds.is_missing(r, f)? {
                v.push(f64::NAN);
            } else {
                v.push(col.value(r));
            }
        }
        out.push(v);
    }
    Ok(out)
}

/// 全量特征矩阵（缺失 → NaN）。
pub fn full_matrix(ds: &Dataset) -> Result<Vec<Vec<f64>>, ExperimentError> {
    let n = ds.num_rows();
    let idx: Vec<usize> = (0..ds.num_features()).collect();
    subset_matrix(ds, &(0..n).collect::<Vec<_>>(), &idx)
}

/// 全量 target 列。
pub fn target_vec(ds: &Dataset) -> Result<Vec<f64>, ExperimentError> {
    Ok(ds.target_values().values().to_vec())
}

/// 行子集 target。
pub fn target_subset(ds: &Dataset, rows: &[usize]) -> Vec<f64> {
    let t = ds.target_values();
    rows.iter().map(|&r| t.value(r)).collect()
}

/// 判定系数 R²。
pub fn r2(actual: &[f64], pred: &[f64]) -> f64 {
    let n = actual.len();
    if n == 0 {
        return 0.0;
    }
    let mean = actual.iter().sum::<f64>() / n as f64;
    let ss_tot = actual.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>();
    if ss_tot == 0.0 {
        return 1.0;
    }
    let ss_res = actual
        .iter()
        .zip(pred)
        .map(|(a, p)| (a - p) * (a - p))
        .sum::<f64>();
    1.0 - ss_res / ss_tot
}

/// 均方根误差。
pub fn rmse(actual: &[f64], pred: &[f64]) -> f64 {
    let n = actual.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let mse = actual
        .iter()
        .zip(pred)
        .map(|(a, p)| (a - p) * (a - p))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

/// 列均值（仅观测值；全缺失 → NaN）。
pub fn column_means(ds: &Dataset) -> Result<Vec<f64>, ExperimentError> {
    let n = ds.num_rows();
    let mut means = Vec::with_capacity(ds.num_features());
    for f in 0..ds.num_features() {
        let col = ds.feature_values(f)?;
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for r in 0..n {
            if !ds.is_missing(r, f)? {
                sum += col.value(r);
                cnt += 1;
            }
        }
        means.push(if cnt > 0 { sum / cnt as f64 } else { f64::NAN });
    }
    Ok(means)
}

/// 校验每列均值皆有限（无全缺失列），否则显式报错。
pub fn ensure_finite_means(means: &[f64]) -> Result<(), ExperimentError> {
    if means.iter().any(|m| !m.is_finite()) {
        return Err(ExperimentError::InvalidInput(
            "存在整列缺失，无法用列均值初始化".into(),
        ));
    }
    Ok(())
}

/// 把 NaN 视为缺失后逐列写入（构造 arrow batch 用）。
pub fn columns_to_batch(
    names: &[String],
    columns: &[Vec<f64>],
) -> Result<RecordBatch, ExperimentError> {
    let fields: Vec<Field> = names
        .iter()
        .map(|n| Field::new(n, DataType::Float64, true))
        .collect();
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|c| Arc::new(Float64Array::from(c.clone())) as ArrayRef)
        .collect();
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
}
