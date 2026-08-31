//! 原生 softmax 多分类（D8：M1 多分类一期）。
//!
//! 每轮提升对每个类建一棵树：梯度 g_k = p_k − 1{y=k}，海森用对角线近似
//! h_k = p_k·(1−p_k)（标准 GBDT 多分类）。预测 = 各类原始分数累加 → softmax。
//!
//! 注意：与 `Booster<L>`（标量损失）分离；本类型自带训练与预测。

use arrow::array::{Array, Float64Array};

use crate::binning::BinTable;
use crate::data::missing::is_missing_value;
use crate::data::{DataError, Dataset, MissingPolicy};
use crate::tree::{Tree, TreeBuilder, TreeParams};

use super::context::TrainingContext;
use super::error::BoostingError;
use super::params::BoostingParams;

/// 多分类训练产物（每类一棵树序列）。
#[derive(Debug)]
pub struct MulticlassBooster {
    n_classes: usize,
    /// `trees[class][tree_idx]`
    trees: Vec<Vec<Tree>>,
    table: BinTable,
    /// 每类初始 logit（类先验 log）。
    init_scores: Vec<f64>,
    learning_rate: f64,
}

impl MulticlassBooster {
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn num_trees_per_class(&self) -> usize {
        self.trees.first().map_or(0, |v| v.len())
    }

    pub fn init_scores(&self) -> &[f64] {
        &self.init_scores
    }

    /// 模型自包含的分箱表（与 Booster 一致，供序列化/热替换）。
    pub fn bin_table(&self) -> &BinTable {
        &self.table
    }

    /// 类别概率矩阵 `probs[row][class]`（softmax）。
    pub fn predict_proba(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, BoostingError> {
        let raw = self.raw_logits(ds)?;
        Ok(raw.iter().map(|row| softmax(row)).collect())
    }

    /// 预测类别（argmax；并列取小类）。
    pub fn predict(&self, ds: &Dataset) -> Result<Vec<usize>, BoostingError> {
        let proba = self.predict_proba(ds)?;
        Ok(proba
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect())
    }

    /// 每行原始 logits（init + Σ lr·tree）。
    pub fn raw_logits(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, BoostingError> {
        let n = ds.num_rows();
        let cols = feature_columns(ds)?;
        let policy = ds.missing_policy();
        let mut out = vec![self.init_scores.clone(); n];
        for k in 0..self.n_classes {
            for tree in &self.trees[k] {
                for (r, row) in out.iter_mut().enumerate() {
                    row[k] += self.learning_rate * predict_row(tree, &cols, r, policy);
                }
            }
        }
        Ok(out)
    }
}

/// 拟合多分类模型。`y` 必须为整数标签 ∈ [0, n_classes)。
pub fn fit_multiclass(
    ds: &Dataset,
    params: &BoostingParams,
    n_classes: usize,
    ctx: &TrainingContext,
) -> Result<MulticlassBooster, BoostingError> {
    let _ = ctx;
    if n_classes < 2 {
        return Err(BoostingError::Data(DataError::InvalidMulticlassClasses(
            n_classes,
        )));
    }
    let n = ds.num_rows();
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let labels = to_labels(&y, n_classes)?;

    let (table, matrix) = BinTable::build_from_dataset(ds, params.max_bins)?;
    let cols = feature_columns(ds)?;
    let policy = ds.missing_policy();
    let tree_params: TreeParams = params.tree_params;

    // 类先验 logit
    let mut counts = vec![0usize; n_classes];
    for &l in &labels {
        counts[l] += 1;
    }
    let total = n.max(1) as f64;
    let init_scores: Vec<f64> = counts
        .iter()
        .map(|&c| ((c as f64 / total).clamp(1e-12, 1.0 - 1e-12)).ln())
        .collect();

    // pred[k][i]
    let mut pred: Vec<Vec<f64>> = (0..n_classes).map(|k| vec![init_scores[k]; n]).collect();
    let mut trees: Vec<Vec<Tree>> = (0..n_classes).map(|_| Vec::new()).collect();
    let builder = TreeBuilder::new(tree_params);

    let mut logits = vec![0.0f64; n_classes];
    let mut grad = vec![0.0f64; n];
    let mut hess = vec![0.0f64; n];
    for _ in 0..params.n_estimators {
        for k in 0..n_classes {
            for i in 0..n {
                for (c, p) in logits.iter_mut().enumerate() {
                    *p = pred[c][i];
                }
                let probs = softmax(&logits);
                let is_target = (labels[i] == k) as u8 as f64;
                grad[i] = probs[k] - is_target;
                hess[i] = probs[k] * (1.0 - probs[k]);
            }
            let tree = builder.build(&matrix, &table, &grad, &hess)?;
            for (r, p) in pred[k].iter_mut().enumerate() {
                *p += params.learning_rate * predict_row(&tree, &cols, r, policy);
            }
            trees[k].push(tree);
        }
    }

    Ok(MulticlassBooster {
        n_classes,
        trees,
        table,
        init_scores,
        learning_rate: params.learning_rate,
    })
}

/// 每行 softmax（数值稳定，减 max）。
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// 校验标签 ∈ [0, n_classes) 且为整数。
fn to_labels(y: &[f64], n_classes: usize) -> Result<Vec<usize>, BoostingError> {
    y.iter()
        .map(|&v| {
            if v.fract() != 0.0 || !(0.0..n_classes as f64).contains(&v) {
                Err(BoostingError::Data(DataError::InvalidLabel {
                    value: v,
                    n_classes,
                }))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

fn predict_row(tree: &Tree, cols: &[&Float64Array], row: usize, policy: MissingPolicy) -> f64 {
    tree.predict_one(|f| {
        let col = cols[f];
        let v = col.value(row);
        (v, is_missing_value(v, col.is_null(row), policy))
    })
}

fn feature_columns(ds: &Dataset) -> Result<Vec<&Float64Array>, BoostingError> {
    let mut cols = Vec::with_capacity(ds.num_features());
    for f in 0..ds.num_features() {
        cols.push(ds.feature_values(f)?);
    }
    Ok(cols)
}
