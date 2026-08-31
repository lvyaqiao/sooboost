//! UnmaskingTrees：迭代单特征逐步解掩的 GBDT 填补（M2-A，arXiv:2407.05593）。
//!
//! 原理（原型简化）：
//! - 训练：对每个特征 f，用「其余特征」训练 GBDT 回归模型预测 f（仅用 f 已观测的行）；
//! - 填补：按可预测性（验证 R²）从高到低排序，逐个特征用当前其他特征值预测解掩，
//!   多轮迭代直到收敛（缺失位逐步被填）。
//!
//! 验证目标：在相关特征合成的 MCAR 缺失上，填补 RMSE 显著优于列均值填充。

use sooboost_core::boosting::{Booster, BoostingParams, TrainingContext, fit};
use sooboost_core::data::Dataset;
use sooboost_core::loss::SquaredError;

use crate::ExperimentError;
use crate::dataset_util;

/// 训练完成的迭代解掩填补器。
#[derive(Debug)]
pub struct UnmaskingImputer {
    feature_names: Vec<String>,
    /// 每个特征一个模型：其余特征 → 该特征。
    models: Vec<Option<Booster<SquaredError>>>,
    /// 可预测性（验证 R²；观测行过少 → 0）。
    predictability: Vec<f64>,
    /// 列均值（缺失初始化 + 无模型特征的填充值）。
    col_means: Vec<f64>,
    n_rows: usize,
}

impl UnmaskingImputer {
    /// 训练：逐特征建模型并估计可预测性。
    pub fn fit(
        ds: &Dataset,
        params: &BoostingParams,
        ctx: &TrainingContext,
    ) -> Result<Self, ExperimentError> {
        let nf = ds.num_features();
        if nf == 0 {
            return Err(ExperimentError::InvalidInput("无特征可填补".into()));
        }
        let n = ds.num_rows();
        if n == 0 {
            return Err(ExperimentError::InvalidInput("空数据集".into()));
        }
        let feature_names = ds.feature_names().to_vec();
        let means = dataset_util::column_means(ds)?;
        dataset_util::ensure_finite_means(&means)?;

        let mut models = Vec::with_capacity(nf);
        let mut predictability = Vec::with_capacity(nf);
        for f in 0..nf {
            let mut observed = Vec::new();
            for r in 0..n {
                if !ds.is_missing(r, f)? {
                    observed.push(r);
                }
            }
            if observed.len() < MIN_OBSERVED {
                // 观测行过少：退化用列均值填充，不建模型。
                models.push(None);
                predictability.push(0.0);
                continue;
            }
            // 子数据集：其余特征 → 特征 f（行 = f 已观测）。
            let others: Vec<usize> = (0..nf).filter(|&g| g != f).collect();
            let mut cols = Vec::with_capacity(nf);
            for &g in &others {
                let col = ds.feature_values(g)?;
                let mut v = Vec::with_capacity(observed.len());
                for &r in &observed {
                    v.push(if ds.is_missing(r, g)? {
                        f64::NAN
                    } else {
                        col.value(r)
                    });
                }
                cols.push((feature_names[g].clone(), v));
            }
            let target_col = ds.feature_values(f)?;
            let target: Vec<f64> = observed.iter().map(|&r| target_col.value(r)).collect();
            let sub = dataset_util::dataset_from_columns(&cols, &feature_names[f], target)?;
            let model = fit(&sub, params, SquaredError, ctx)?;
            // 可预测性：观测行上 in-sample R²（原型用 in-sample，足够排序）。
            let pred = model.predict(&sub)?;
            predictability.push(dataset_util::r2(&dataset_util::target_vec(&sub)?, &pred));
            models.push(Some(model));
        }
        Ok(Self {
            feature_names,
            models,
            predictability,
            col_means: means,
            n_rows: n,
        })
    }

    /// 填补：返回完整特征矩阵（行 × 特征）。
    pub fn impute(
        &self,
        ds: &Dataset,
        passes: usize,
        ctx: &TrainingContext,
    ) -> Result<Vec<Vec<f64>>, ExperimentError> {
        let nf = ds.num_features();
        if ds.num_rows() != self.n_rows || ds.feature_names() != self.feature_names.as_slice() {
            return Err(ExperimentError::InvalidInput(
                "训练/推断数据集不一致".into(),
            ));
        }
        let mut cur = dataset_util::full_matrix(ds)?;
        // 缺失位用列均值初始化。
        for (f, col) in cur.iter_mut().enumerate().take(nf) {
            for value in col.iter_mut().take(ds.num_rows()) {
                if value.is_nan() {
                    *value = self.col_means[f];
                }
            }
        }
        // 按可预测性从高到低解掩。
        let mut order: Vec<usize> = (0..nf).collect();
        order.sort_by(|&a, &b| {
            self.predictability[b]
                .partial_cmp(&self.predictability[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for _ in 0..passes {
            for &f in &order {
                let Some(model) = &self.models[f] else {
                    continue; // 无模型特征（均值已填）。
                };
                // 预测数据集：其余特征当前值。
                let mut cols = Vec::with_capacity(nf);
                for (g, current) in cur.iter().enumerate().take(nf) {
                    if g == f {
                        continue;
                    }
                    cols.push((self.feature_names[g].clone(), current.clone()));
                }
                let placeholder = self.col_means[f];
                let pred_ds = dataset_util::dataset_from_columns(
                    &cols,
                    &self.feature_names[f],
                    vec![placeholder; ds.num_rows()],
                )?;
                let pred = model.predict(&pred_ds)?;
                for r in 0..ds.num_rows() {
                    if ds.is_missing(r, f)? {
                        cur[f][r] = pred[r];
                    }
                }
            }
            let _ = ctx; // seed 保留给后续确定性扩展
        }
        Ok(cur)
    }
}

/// 观测行数下限：低于此不建模型（防过拟合 + 提速）。
const MIN_OBSERVED: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::{Rng, SeedableRng};

    fn correlated_data(rows: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        // 潜在变量 z 驱动三个相关特征。
        let mut features: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(rows)).collect();
        let mut z = Vec::with_capacity(rows);
        for _ in 0..rows {
            let zi = (rng.next_u64() % 1000) as f64 / 1000.0;
            z.push(zi);
            for (f, noise) in features.iter_mut().zip([0.05, 0.05, 0.05]) {
                let e = (rng.next_u64() % 1000) as f64 / 1000.0 * noise;
                f.push(zi + e);
            }
        }
        (features, z)
    }

    fn mask(features: &mut [Vec<f64>], col: usize, prob: f64, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut truth = Vec::new();
        for v in features[col].iter_mut() {
            let m = (rng.next_u64() % 10000) as f64 / 10000.0 < prob;
            if m {
                truth.push(*v);
                *v = f64::NAN;
            } else {
                truth.push(f64::NAN);
            }
        }
        truth
    }

    #[test]
    fn imputation_beats_mean_fill() {
        let rows = 400;
        let (mut features, _z) = correlated_data(rows);
        let truth = mask(&mut features, 1, 0.3, 11);
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let target: Vec<f64> = (0..rows).map(|i| features[0][i] + features[1][i]).collect();
        let ds = dataset_util::dataset_from_columns(
            &[
                (names[0].clone(), features[0].clone()),
                (names[1].clone(), features[1].clone()),
                (names[2].clone(), features[2].clone()),
            ],
            "target",
            target,
        )
        .unwrap();

        let params = BoostingParams::default();
        let ctx = TrainingContext::new(42);
        let imputer = UnmaskingImputer::fit(&ds, &params, &ctx).unwrap();
        let imputed = imputer.impute(&ds, 2, &ctx).unwrap();

        // 仅评估被掩的 f1 行。
        let (mut imp_pred, mut imp_true) = (Vec::new(), Vec::new());
        for r in 0..rows {
            if !truth[r].is_nan() {
                imp_true.push(truth[r]);
                imp_pred.push(imputed[1][r]);
            }
        }
        let imp_rmse = dataset_util::rmse(&imp_true, &imp_pred);

        // 基线：列均值填充。
        let mean_fill_rmse = {
            let m = dataset_util::column_means(&ds).unwrap()[1];
            dataset_util::rmse(&imp_true, &vec![m; imp_true.len()])
        };
        assert!(
            imp_rmse < mean_fill_rmse * 0.5,
            "填补 RMSE {imp_rmse:.4} 应显著优于均值填充 {mean_fill_rmse:.4}"
        );
    }
}
