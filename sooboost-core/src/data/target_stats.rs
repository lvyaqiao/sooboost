//! Ordered target statistics（D9 类别特征一期，仿 CatBoost 顺序原则防泄漏）。
//!
//! 原理（contracts §1.4 防泄漏）：
//! - 训练期：用训练 seed 派生一个行 permutation，对每个样本用**该样本之前**（permutation
//!   序）同类别样本的 target 估计值（有序 TS），禁止使用全量 target 均值 → 防泄漏；
//! - 预测期：用每类别的平滑均值映射（全量统计，此时无泄漏风险）；OOV（训练未见类别）
//!   → 全局先验，行为可预测不崩溃；
//! - null 类别 = 缺失，走缺失值语义（红线 2），不参与统计、解析为 null。
//!
//! 类别经 TS 转为数值后，复用现有数值分箱/建树管线（模型仍是数值树）。

use std::collections::HashMap;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::Rng;

use crate::boosting::TrainingContext;
use crate::data::{DataError, Dataset};

/// 类别特征编码（训练产物，随模型序列化；预测期把类别键映射为数值）。
#[derive(Debug, Clone)]
pub struct CategoricalEncoding {
    /// 每特征：类别键 → 平滑均值（预测期映射；OOV → prior）。
    maps: Vec<HashMap<u32, f64>>,
    /// 每特征：全局先验（y 均值）。
    priors: Vec<f64>,
    /// 每特征：smoothing α。
    alphas: Vec<f64>,
}

impl CategoricalEncoding {
    pub fn num_features(&self) -> usize {
        self.maps.len()
    }

    /// 类别键 → 数值（OOV 返回先验；调用方对缺失键须自行走缺失路径）。
    pub fn value(&self, feature: usize, key: u32) -> f64 {
        self.maps[feature]
            .get(&key)
            .copied()
            .unwrap_or(self.priors[feature])
    }

    pub fn prior(&self, feature: usize) -> f64 {
        self.priors[feature]
    }

    pub fn alpha(&self, feature: usize) -> f64 {
        self.alphas[feature]
    }

    /// 供序列化：每特征类别键-值对（已排序，保证确定性）。
    pub fn entries(&self, feature: usize) -> Vec<(u32, f64)> {
        let mut v: Vec<(u32, f64)> = self.maps[feature].iter().map(|(&k, &x)| (k, x)).collect();
        v.sort_by_key(|&(k, _)| k);
        v
    }

    pub(crate) fn from_parts(
        maps: Vec<HashMap<u32, f64>>,
        priors: Vec<f64>,
        alphas: Vec<f64>,
    ) -> Self {
        Self {
            maps,
            priors,
            alphas,
        }
    }
}

/// 解析后的类别特征数值（按类别特征顺序，行序）。
pub type ResolvedFeatures = Vec<Vec<Option<f64>>>;

/// 计算类别特征的 ordered TS，返回解析后的数值特征。
pub fn compute_ordered_ts(
    ds: &Dataset,
    cat_features: &[usize],
    alpha: f64,
    ctx: &TrainingContext,
) -> Result<(ResolvedFeatures, CategoricalEncoding), DataError> {
    let n = ds.num_rows();
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let prior = if n == 0 {
        0.0
    } else {
        y.iter().sum::<f64>() / n as f64
    };

    let mut rng = ctx.rng();
    let mut perm: Vec<usize> = (0..n).collect();
    shuffle(&mut perm, &mut rng);

    let mut resolved: Vec<Vec<Option<f64>>> = Vec::with_capacity(cat_features.len());
    let mut maps: Vec<HashMap<u32, f64>> = Vec::with_capacity(cat_features.len());

    for &f in cat_features {
        // 有序累计：category -> (sum_target, count)，在更新前取估计（防泄漏）
        let mut acc: HashMap<u32, (f64, u32)> = HashMap::new();
        let mut col = vec![None; n];
        for &r in &perm {
            let key = ds.categorical_key(r, f)?;
            let Some(k) = key else { continue }; // null 类别：缺失，不参与
            let (s, c) = acc.get(&k).copied().unwrap_or((0.0, 0));
            col[r] = Some(if c == 0 {
                prior
            } else {
                (s + alpha * prior) / (c as f64 + alpha)
            });
            acc.entry(k)
                .and_modify(|e| {
                    e.0 += y[r];
                    e.1 += 1;
                })
                .or_insert((y[r], 1));
        }
        resolved.push(col);
        // 预测期映射：平滑类别均值（全量统计，预测期无泄漏问题）
        let map: HashMap<u32, f64> = acc
            .into_iter()
            .map(|(k, (s, c))| {
                let v = (s + alpha * prior) / (c as f64 + alpha);
                (k, v)
            })
            .collect();
        maps.push(map);
    }

    let enc = CategoricalEncoding::from_parts(
        maps,
        vec![prior; cat_features.len()],
        vec![alpha; cat_features.len()],
    );
    Ok((resolved, enc))
}

/// 用已训练编码把（推断期）类别特征解析为数值（OOV → 先验；null → null）。
pub fn apply_encoding(
    ds: &Dataset,
    enc: &CategoricalEncoding,
    cat_features: &[usize],
) -> Result<ResolvedFeatures, DataError> {
    let n = ds.num_rows();
    let mut resolved: ResolvedFeatures = Vec::with_capacity(cat_features.len());
    for (fi, &f) in cat_features.iter().enumerate() {
        let mut col = vec![None; n];
        for (r, slot) in col.iter_mut().enumerate() {
            if ds.is_missing(r, f)? {
                continue; // null → 缺失
            }
            if let Some(k) = ds.categorical_key(r, f)? {
                *slot = Some(enc.value(fi, k));
            }
        }
        resolved.push(col);
    }
    Ok(resolved)
}

/// 由原始 Dataset + 类别特征解析结果，构造全数值的解析后 Dataset
/// （数值特征原样、类别特征用解析值；target 原样）。
pub fn resolve_to_dataset(
    ds: &Dataset,
    resolved_cat: &ResolvedFeatures,
) -> Result<Dataset, DataError> {
    use arrow::array::{ArrayRef, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let mut fields = Vec::with_capacity(ds.num_features() + 1);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(ds.num_features() + 1);
    let mut cat_idx = 0usize;

    for f in 0..ds.num_features() {
        if ds.feature_is_categorical(f) {
            // 该类别特征对应 resolved_cat[cat_idx]
            let arr = Float64Array::from(resolved_cat[cat_idx].clone());
            fields.push(Field::new(&ds.feature_names()[f], DataType::Float64, true));
            arrays.push(Arc::new(arr) as ArrayRef);
            cat_idx += 1;
        } else {
            let col = ds.feature_values(f)?;
            fields.push(Field::new(&ds.feature_names()[f], DataType::Float64, true));
            arrays.push(Arc::new(col.clone()) as ArrayRef);
        }
    }
    fields.push(Field::new(ds.target_name(), DataType::Float64, false));
    arrays.push(Arc::new(ds.target_values().clone()) as ArrayRef);

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays).map_err(DataError::Arrow)?;
    let names: Vec<&str> = ds.feature_names().iter().map(|s| s.as_str()).collect();
    Dataset::from_record_batch(batch, &names, ds.target_name(), ds.missing_policy())
}

/// Fisher-Yates 洗牌（确定性：同 seed 同序列，红线 3）。
fn shuffle(slice: &mut [usize], rng: &mut Xoshiro256PlusPlus) {
    for i in (1..slice.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        slice.swap(i, j);
    }
}
