//! BaltoBot：平衡树 of boosted 回归器 → 非参条件分布（M2-A，arXiv:2407.05593）。
//!
//! 简化原型（连续 target）：
//! - 把 conditioning 特征空间用平衡中位数切分递归分成叶（depth 上限 / 叶最小样本）；
//! - 每叶：拟合 boosted 回归器（叶内条件均值）+ 保存训练残差直方图；
//! - 采样：叶内条件均值 + 从残差直方图抽一个残差 → 即 P(y | x) 的非参样本
//!   （无需对条件分布做参数假设，可捕获多峰）。
//!
//! 验证目标：条件均值跟踪真实 E[y|x]；多峰条件分布能被采出双峰。

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::Rng;
use sooboost_core::boosting::{BoostingParams, TrainingContext, fit};
use sooboost_core::data::Dataset;
use sooboost_core::loss::SquaredError;

use crate::ExperimentError;
use crate::dataset_util;

/// 特征 × 行 矩阵视图（NaN = 缺失）。
type Matrix = Vec<Vec<f64>>;

/// 平衡条件分布树的内部节点。
#[derive(Debug)]
struct CondNode {
    /// 内部节点：切分特征 + 阈值（<= 阈值 → 左子树）。
    split_feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<CondNode>>,
    right: Option<Box<CondNode>>,
    /// 叶节点：条件均值模型 + 残差直方图。
    model: Option<sooboost_core::boosting::Booster<SquaredError>>,
    residuals: Option<Vec<f64>>,
}

/// 训练完成的 BaltoBot 条件分布模型。
#[derive(Debug)]
pub struct BaltoBot {
    root: CondNode,
    n_features: usize,
    cond_indices: Vec<usize>,
}

impl BaltoBot {
    /// 训练。`target_idx` 为目标特征（建模 P(该特征 | 其余条件特征)）；
    /// `cond_indices` 为参与条件化的特征索引（其余特征不参与）。
    pub fn fit(
        ds: &Dataset,
        target_idx: usize,
        cond_indices: &[usize],
        params: &BoostingParams,
        ctx: &TrainingContext,
        max_depth: usize,
        min_leaf: usize,
    ) -> Result<Self, ExperimentError> {
        if target_idx >= ds.num_features() {
            return Err(ExperimentError::InvalidInput("target 特征索引越界".into()));
        }
        if max_depth == 0 {
            return Err(ExperimentError::InvalidInput("max_depth 须 > 0".into()));
        }
        if cond_indices.iter().any(|&f| f >= ds.num_features()) {
            return Err(ExperimentError::InvalidInput("条件特征索引越界".into()));
        }
        if cond_indices.contains(&target_idx) || cond_indices.is_empty() {
            return Err(ExperimentError::InvalidInput(
                "target 不得同时作为条件特征，且至少需要一个条件特征".into(),
            ));
        }
        let matrix = dataset_util::full_matrix(ds)?;
        let names = ds.feature_names().to_vec();
        let nf = ds.num_features();
        let build = BuildContext {
            names: &names,
            target_idx,
            cond_indices,
            params,
            ctx,
            max_depth,
            min_leaf,
        };
        let root = build_node(&matrix, &build, 0)?;
        Ok(Self {
            root,
            n_features: nf,
            cond_indices: cond_indices.to_vec(),
        })
    }

    /// 条件均值：给定完整特征向量，返回叶内回归器预测。
    pub fn conditional_mean(
        &self,
        values: &[f64],
        is_missing: &[bool],
    ) -> Result<f64, ExperimentError> {
        if values.len() != self.n_features || is_missing.len() != self.n_features {
            return Err(ExperimentError::InvalidInput("特征向量长度不符".into()));
        }
        let leaf = traverse(&self.root, values, is_missing)?;
        let model_values: Vec<f64> = self.cond_indices.iter().map(|&f| values[f]).collect();
        let model_missing: Vec<bool> = self.cond_indices.iter().map(|&f| is_missing[f]).collect();
        leaf.model
            .as_ref()
            .map(|m| m.predict_row(&model_values, &model_missing))
            .ok_or(ExperimentError::Missing("叶模型缺失"))
    }

    /// 非参条件采样：条件均值 + 叶残差直方图抽样。
    pub fn sample(
        &self,
        values: &[f64],
        is_missing: &[bool],
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<f64, ExperimentError> {
        let mean = self.conditional_mean(values, is_missing)?;
        let leaf = traverse(&self.root, values, is_missing)?;
        let residuals = leaf
            .residuals
            .as_ref()
            .ok_or(ExperimentError::Missing("叶残差直方图缺失"))?;
        let k = (rng.next_u64() as usize) % residuals.len();
        Ok(mean + residuals[k])
    }

    /// 批量采样（TrainingContext::rng() 固定 seed → 同 seed 逐位一致，红线 3）。
    pub fn sample_many(
        &self,
        values: &[f64],
        is_missing: &[bool],
        n: usize,
        ctx: &TrainingContext,
    ) -> Result<Vec<f64>, ExperimentError> {
        let mut rng = ctx.rng();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.sample(values, is_missing, &mut rng)?);
        }
        Ok(out)
    }
}

struct BuildContext<'a> {
    names: &'a [String],
    target_idx: usize,
    cond_indices: &'a [usize],
    params: &'a BoostingParams,
    ctx: &'a TrainingContext,
    max_depth: usize,
    min_leaf: usize,
}

fn build_node(
    matrix: &Matrix,
    build: &BuildContext<'_>,
    depth: usize,
) -> Result<CondNode, ExperimentError> {
    let n = matrix.first().map(|c| c.len()).unwrap_or(0);
    if depth >= build.max_depth || n < build.min_leaf {
        return build_leaf(matrix, build);
    }
    // 选方差最大的条件特征切分（区分度）；阈值 = 中位数（平衡）。
    let split_feature = *build
        .cond_indices
        .iter()
        .max_by(|&&a, &&b| variance_of(matrix, a).total_cmp(&variance_of(matrix, b)))
        .ok_or(ExperimentError::InvalidInput("无条件特征".into()))?;
    let threshold = median_of(matrix, split_feature);
    if !threshold.is_finite() {
        return build_leaf(matrix, build);
    }
    let (left_rows, right_rows) = partition_rows(matrix, split_feature, threshold);
    if left_rows.is_empty() || right_rows.is_empty() {
        return build_leaf(matrix, build);
    }
    let left = build_node(&project(matrix, &left_rows), build, depth + 1)?;
    let right = build_node(&project(matrix, &right_rows), build, depth + 1)?;
    Ok(CondNode {
        split_feature: Some(split_feature),
        threshold: Some(threshold),
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
        model: None,
        residuals: None,
    })
}

fn build_leaf(matrix: &Matrix, build: &BuildContext<'_>) -> Result<CondNode, ExperimentError> {
    // 叶内条件均值模型：cond 特征 → target 特征（缺失 → NaN）。
    let cols: Vec<(String, Vec<f64>)> = build
        .cond_indices
        .iter()
        .map(|&g| (build.names[g].clone(), matrix[g].clone()))
        .collect();
    let target = matrix[build.target_idx].clone();
    let sub = dataset_util::dataset_from_columns(&cols, &build.names[build.target_idx], target)?;
    let model = fit(&sub, build.params, SquaredError, build.ctx)?;
    let pred = model.predict(&sub)?;
    let truth = dataset_util::target_vec(&sub)?;
    let residuals: Vec<f64> = truth.iter().zip(&pred).map(|(a, p)| a - p).collect();
    Ok(CondNode {
        split_feature: None,
        threshold: None,
        left: None,
        right: None,
        model: Some(model),
        residuals: Some(residuals),
    })
}

/// 按行子集投影矩阵（特征 × 行 → 特征 × 子集行）。
fn project(matrix: &Matrix, rows: &[usize]) -> Matrix {
    matrix
        .iter()
        .map(|col| rows.iter().map(|&r| col[r]).collect())
        .collect()
}

fn variance_of(matrix: &Matrix, f: usize) -> f64 {
    let vals: Vec<f64> = matrix[f]
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if vals.len() < 2 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64
}

fn median_of(matrix: &Matrix, f: usize) -> f64 {
    let mut vals: Vec<f64> = matrix[f]
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2]
}

fn partition_rows(matrix: &Matrix, f: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (r, &v) in matrix[f].iter().enumerate() {
        if v.is_finite() && v <= threshold {
            left.push(r);
        } else {
            right.push(r);
        }
    }
    (left, right)
}

fn traverse<'a>(
    node: &'a CondNode,
    values: &[f64],
    is_missing: &[bool],
) -> Result<&'a CondNode, ExperimentError> {
    match (node.split_feature, node.threshold) {
        (Some(f), Some(t)) => {
            let go_left = !is_missing[f] && values[f] <= t;
            let child = if go_left {
                node.left.as_ref()
            } else {
                node.right.as_ref()
            };
            child
                .ok_or(ExperimentError::Missing("树节点子指针缺失"))
                .and_then(|c| traverse(c, values, is_missing))
        }
        _ => Ok(node),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;

    fn build_ds(x: Vec<f64>, y: Vec<f64>) -> Dataset {
        // target 特征以普通特征列进入；Dataset 的 target 用哑列（避免重叠校验）。
        let n = y.len();
        dataset_util::dataset_from_columns(
            &[("x".to_string(), x), ("target".to_string(), y)],
            "__dummy__",
            vec![0.0; n],
        )
        .unwrap()
    }

    #[test]
    fn conditional_mean_tracks_true() {
        let rows = 500;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for _ in 0..rows {
            let xi = (rng.next_u64() % 1000) as f64 / 1000.0;
            let e = ((rng.next_u64() % 1000) as f64 - 500.0) / 1000.0;
            x.push(xi);
            y.push(5.0 + 3.0 * xi + 0.2 * e);
        }
        let ds = build_ds(x, y);
        let params = BoostingParams::default();
        let ctx = TrainingContext::new(1);
        let bot = BaltoBot::fit(&ds, 1, &[0], &params, &ctx, 4, 16).unwrap();
        for xi in [0.2, 0.5, 0.8] {
            let samples = bot
                .sample_many(&[xi, 0.0], &[false, false], 300, &TrainingContext::new(9))
                .unwrap();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let true_mean = 5.0 + 3.0 * xi;
            assert!(
                (mean - true_mean).abs() < 0.35,
                "x={xi} 采样均值 {mean:.3} 应接近真实条件均值 {true_mean:.3}"
            );
        }
    }

    #[test]
    fn multimodal_conditional_is_bimodal() {
        // y|x 与 x 无关但自身双峰（0 或 10）：采样应呈双峰。
        let rows = 400;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for _ in 0..rows {
            x.push((rng.next_u64() % 1000) as f64 / 1000.0);
            let mode = rng.next_u64() % 2;
            let noise = ((rng.next_u64() % 100) as f64 - 50.0) / 100.0;
            y.push(if mode == 0 { noise } else { 10.0 + noise });
        }
        let ds = build_ds(x, y);
        let params = BoostingParams::default();
        let ctx = TrainingContext::new(2);
        let bot = BaltoBot::fit(&ds, 1, &[0], &params, &ctx, 3, 24).unwrap();
        let samples = bot
            .sample_many(&[0.5, 0.0], &[false, false], 800, &TrainingContext::new(10))
            .unwrap();
        let near_zero = samples.iter().filter(|&&s| s < 5.0).count();
        let near_ten = samples.iter().filter(|&&s| s > 5.0).count();
        assert!(
            near_zero > 100 && near_ten > 100,
            "采样应双峰：低峰 {near_zero} 高峰 {near_ten}（共 {})",
            samples.len()
        );
    }
}
