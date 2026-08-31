//! ForestFlow：用 Boosting 树拟合 Flow Matching 向量场（M2-B，ForestDiffusion arXiv:2309.09968 思路）。
//!
//! 原理（per-feature 条件流匹配，规避多输出叶子 → 核心标量叶子即可复用）：
//! - 每特征独立回归模型，输入 = [自身噪声化值 x_t, 时间 t, 其余特征]，目标 = 速度 v = x - ε；
//! - 概率路径 x_t = (1-t)·ε + t·x（t=0 噪声，t=1 数据），v = x - ε = dx_t/dt；
//! - 采样：z 空间从 N(0,1) 出发，用中点积分从 t=0 推进到 t=1；
//! - 填补：观测特征固定为原值，只对流缺失特征积分（条件化采样）。
//! - 每列先做经验分位数正态化，降低 California 等偏态边际对流模型的压力。
//!
//! 确定性：全部噪声由 TrainingContext::rng() 派生（红线 3，同 seed 逐位一致）。

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::Rng;
use sooboost_core::boosting::{Booster, BoostingParams, TrainingContext, fit};
use sooboost_core::data::Dataset;
use sooboost_core::loss::SquaredError;

use crate::ExperimentError;
use crate::dataset_util;

/// 每特征的边际统计（经验分位数正态化，训练/采样/填补共用）。
#[derive(Debug, Clone)]
struct FeatureStats {
    std: f64,
    sorted_values: Vec<f64>,
}

/// 每个观测行采多个 (t, ε) 路径点，降低流匹配目标的蒙特卡洛噪声。
const FLOW_SAMPLES_PER_ROW: usize = 4;

/// 训练完成的 per-feature 流匹配生成器。
#[derive(Debug)]
pub struct ForestFlow {
    n_features: usize,
    stats: Vec<FeatureStats>,
    /// 每特征一个速度场模型；None = 无模型（常值/缺失过多，采样返均值）。
    models: Vec<Option<Booster<SquaredError>>>,
    /// 训练时的特征布局（供 predict_row 一致拼装）：[自身噪声化值, t, 其余特征…]。
    layouts: Vec<Vec<LayoutCol>>,
    n_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LayoutCol {
    SelfNoised,
    Time,
    Feature(usize),
}

impl ForestFlow {
    /// 训练。`n_steps` 为采样时的中点积分步数。
    pub fn fit(
        ds: &Dataset,
        params: &BoostingParams,
        ctx: &TrainingContext,
        n_steps: usize,
    ) -> Result<Self, ExperimentError> {
        let nf = ds.num_features();
        let n = ds.num_rows();
        if nf == 0 || n == 0 {
            return Err(ExperimentError::InvalidInput("空数据集".into()));
        }
        if n_steps == 0 {
            return Err(ExperimentError::InvalidInput("n_steps 须 > 0".into()));
        }
        let mut stats = Vec::with_capacity(nf);
        let mut zmatrix = Vec::with_capacity(nf);
        for f in 0..nf {
            let col = ds.feature_values(f)?;
            let mut vals = Vec::new();
            for r in 0..n {
                if !ds.is_missing(r, f)? {
                    vals.push(col.value(r));
                }
            }
            if vals.is_empty() {
                return Err(ExperimentError::InvalidInput(format!(
                    "特征 {} 全部缺失，无法训练流模型",
                    f
                )));
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
            let std = var.sqrt();
            let mut sorted_values = vals.clone();
            sorted_values.sort_by(|a, b| a.total_cmp(b));
            let stats_f = if std <= f64::EPSILON {
                FeatureStats {
                    std: 0.0,
                    sorted_values,
                }
            } else {
                FeatureStats { std, sorted_values }
            };
            // z 空间列（经验分位数正态化；缺失 → NaN）。
            let mut colz = Vec::with_capacity(n);
            for r in 0..n {
                colz.push(if ds.is_missing(r, f)? {
                    f64::NAN
                } else {
                    empirical_normal_score(&stats_f.sorted_values, col.value(r))
                });
            }
            zmatrix.push(colz);
            stats.push(stats_f);
        }

        // 每特征训练速度场模型。
        let mut models = Vec::with_capacity(nf);
        let mut layouts: Vec<Vec<LayoutCol>> = Vec::with_capacity(nf);
        for f in 0..nf {
            let layout: Vec<LayoutCol> = {
                let mut l = vec![LayoutCol::SelfNoised, LayoutCol::Time];
                for g in 0..nf {
                    if g != f {
                        l.push(LayoutCol::Feature(g));
                    }
                }
                l
            };
            if stats[f].std <= 0.0 {
                models.push(None);
                layouts.push(layout);
                continue;
            }
            let valid_rows = zmatrix[f].iter().filter(|value| value.is_finite()).count();
            if valid_rows == 0 {
                models.push(None);
                layouts.push(layout);
                continue;
            }
            let training_rows = valid_rows * FLOW_SAMPLES_PER_ROW;
            let mut rng = TrainingContext::new(
                ctx.rng_seed()
                    .wrapping_add((f as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            )
            .rng();
            // 训练样本：每行采多个分层 (t, ε)，覆盖整个时间轴并降低标签噪声。
            let mut cols_self = Vec::with_capacity(training_rows);
            let mut cols_t = Vec::with_capacity(training_rows);
            let mut cols_others: Vec<Vec<f64>> = vec![Vec::with_capacity(training_rows); nf];
            let mut target_v = Vec::with_capacity(training_rows);
            for (i, &z) in zmatrix[f].iter().enumerate() {
                if !z.is_finite() {
                    continue;
                }
                for sample in 0..FLOW_SAMPLES_PER_ROW {
                    let t = (sample as f64 + uniform01(&mut rng)) / FLOW_SAMPLES_PER_ROW as f64;
                    let eps = normal01(&mut rng);
                    let xt = (1.0 - t) * eps + t * z;
                    cols_self.push(xt);
                    cols_t.push(t);
                    target_v.push(z - eps);
                    for (g, values) in zmatrix.iter().enumerate() {
                        cols_others[g].push(values[i]);
                    }
                }
            }
            // 特征列：[自身, t, 其余…]
            let mut cols: Vec<(String, Vec<f64>)> = Vec::with_capacity(nf + 1);
            cols.push((format!("__x{}__", f), cols_self));
            cols.push(("__t__".to_string(), cols_t));
            for (g, values) in cols_others.iter().enumerate() {
                if g != f {
                    cols.push((format!("__z{}__", g), values.clone()));
                }
            }
            let sub = dataset_util::dataset_from_columns(&cols, "__v__", target_v)?;
            let model = fit(&sub, params, SquaredError, ctx)?;
            models.push(Some(model));
            layouts.push(layout);
        }

        Ok(Self {
            n_features: nf,
            stats,
            models,
            layouts,
            n_steps,
        })
    }

    /// 无条件生成 `count` 行（z 空间采样后反归一）。
    pub fn generate(
        &self,
        count: usize,
        ctx: &TrainingContext,
    ) -> Result<Vec<Vec<f64>>, ExperimentError> {
        let mut rng = ctx.rng();
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.integrate_row(None, &mut rng)?);
        }
        Ok(out)
    }

    /// 条件生成/填补：`row` 中 `mask` 为 false 的特征保持原值，仅流缺失特征。
    pub fn impute(
        &self,
        row: &[f64],
        observed_mask: &[bool],
        ctx: &TrainingContext,
    ) -> Result<Vec<f64>, ExperimentError> {
        if row.len() != self.n_features || observed_mask.len() != self.n_features {
            return Err(ExperimentError::InvalidInput("行长度不符".into()));
        }
        // 归一化观测特征。
        let mut z_fixed: Vec<Option<f64>> = vec![None; self.n_features];
        for f in 0..self.n_features {
            if observed_mask[f] {
                z_fixed[f] = Some(self.normalize(f, row[f]));
            }
        }
        let mut rng = ctx.rng();
        let out = self.integrate_row(Some(&z_fixed), &mut rng)?;
        // 观测特征还原为原值。
        let mut res = out;
        for f in 0..self.n_features {
            if observed_mask[f] {
                res[f] = row[f];
            }
        }
        Ok(res)
    }

    /// 条件期望的 Monte Carlo 点估计：平均多个条件样本，降低填补 RMSE 的采样方差。
    ///
    /// `impute` 保留单次随机填补语义；需要评估或使用点填补时调用本方法。
    pub fn impute_mean(
        &self,
        row: &[f64],
        observed_mask: &[bool],
        n_samples: usize,
        ctx: &TrainingContext,
    ) -> Result<Vec<f64>, ExperimentError> {
        if n_samples == 0 {
            return Err(ExperimentError::InvalidInput("n_samples 须 > 0".into()));
        }
        let mut sum = vec![0.0; self.n_features];
        for sample in 0..n_samples {
            let seed = ctx
                .rng_seed()
                .wrapping_add((sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let imputed = self.impute(row, observed_mask, &TrainingContext::new(seed))?;
            for (total, value) in sum.iter_mut().zip(imputed) {
                *total += value;
            }
        }
        let scale = 1.0 / n_samples as f64;
        for value in &mut sum {
            *value *= scale;
        }
        Ok(sum)
    }

    fn integrate_row(
        &self,
        fixed: Option<&[Option<f64>]>,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<Vec<f64>, ExperimentError> {
        let mut z: Vec<f64> = (0..self.n_features).map(|_| normal01(rng)).collect();
        if let Some(fx) = fixed {
            for f in 0..self.n_features {
                if let Some(v) = fx[f] {
                    z[f] = v;
                }
            }
        }
        for step in 0..self.n_steps {
            // 中点积分：比在区间起点使用中点时间的伪 Euler 更稳定。
            let dt = 1.0 / self.n_steps as f64;
            let t0 = step as f64 * dt;
            let t_mid = t0 + 0.5 * dt;
            let velocity_start = self.predict_velocity(&z, t0)?;
            let mut midpoint = z.clone();
            for f in 0..self.n_features {
                if self.models[f].is_some() {
                    midpoint[f] = z[f] + 0.5 * dt * velocity_start[f];
                } else {
                    // 无模型特征：收敛到归一化常值 0。
                    midpoint[f] = 0.0;
                }
            }
            self.apply_fixed(&mut midpoint, fixed);
            let velocity_mid = self.predict_velocity(&midpoint, t_mid)?;
            let mut newz = z.clone();
            for f in 0..self.n_features {
                if self.models[f].is_some() {
                    newz[f] = z[f] + dt * velocity_mid[f];
                } else {
                    newz[f] = 0.0;
                }
            }
            self.apply_fixed(&mut newz, fixed);
            z = newz;
        }
        Ok((0..self.n_features)
            .map(|f| self.denormalize(f, z[f]))
            .collect())
    }

    fn normalize(&self, f: usize, x: f64) -> f64 {
        empirical_normal_score(&self.stats[f].sorted_values, x)
    }

    fn denormalize(&self, f: usize, z: f64) -> f64 {
        empirical_inverse_normal(&self.stats[f].sorted_values, z)
    }

    fn predict_velocity(&self, z: &[f64], t: f64) -> Result<Vec<f64>, ExperimentError> {
        let mut velocity = Vec::with_capacity(self.n_features);
        for f in 0..self.n_features {
            let Some(model) = &self.models[f] else {
                velocity.push(0.0);
                continue;
            };
            let mut vals = Vec::with_capacity(self.layouts[f].len());
            let mut miss = vec![false; self.layouts[f].len()];
            for col in &self.layouts[f] {
                match col {
                    LayoutCol::SelfNoised => vals.push(z[f]),
                    LayoutCol::Time => vals.push(t),
                    LayoutCol::Feature(g) => {
                        let g = *g;
                        if z[g].is_nan() {
                            vals.push(0.0);
                            miss[vals.len() - 1] = true;
                        } else {
                            vals.push(z[g]);
                        }
                    }
                }
            }
            velocity.push(model.predict_row(&vals, &miss));
        }
        Ok(velocity)
    }

    fn apply_fixed(&self, z: &mut [f64], fixed: Option<&[Option<f64>]>) {
        if let Some(fx) = fixed {
            for f in 0..self.n_features {
                if let Some(v) = fx[f] {
                    z[f] = v;
                }
            }
        }
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

/// [0,1) 均匀随机数。
fn uniform01(rng: &mut Xoshiro256PlusPlus) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// 标准正态（Box-Muller）。
fn normal01(rng: &mut Xoshiro256PlusPlus) -> f64 {
    let u1 = uniform01(rng).max(1e-12);
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lower_bound(values: &[f64], target: f64) -> usize {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound(values: &[f64], target: f64) -> usize {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if values[mid] <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// 经验分布 → 标准正态分数，避免直接用 z-score 拟合严重偏态边际。
fn empirical_normal_score(sorted: &[f64], value: f64) -> f64 {
    if sorted.len() <= 1 {
        return 0.0;
    }
    let n = sorted.len() as f64;
    let lower = lower_bound(sorted, value);
    let upper = upper_bound(sorted, value);
    let p = ((lower + upper) as f64 / (2.0 * n)).clamp(0.5 / n, 1.0 - 0.5 / n);
    normal_quantile(p)
}

/// 标准正态分数 → 经验分布的线性分位数插值。
fn empirical_inverse_normal(sorted: &[f64], value: f64) -> f64 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let n = sorted.len() as f64;
    let position = (normal_cdf(value) * n - 0.5).clamp(0.0, n - 1.0);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn normal_cdf(value: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26; absolute error is below 7.5e-8.
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let polynomial =
        ((((1.330_274_429 * t - 1.821_255_978) * t + 1.781_477_937) * t - 0.356_563_782) * t
            + 0.319_381_530)
            * t;
    let tail = 0.398_942_280_401_432_7 * (-0.5 * x * x).exp() * polynomial;
    if sign > 0.0 { 1.0 - tail } else { tail }
}

fn normal_quantile(value: f64) -> f64 {
    // Peter J. Acklam's rational approximation.
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;
    let p = value.clamp(1e-12, 1.0 - 1e-12);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let numerator = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let denominator = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        return numerator / denominator;
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let numerator = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let denominator = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        return -numerator / denominator;
    }
    let q = p - 0.5;
    let r = q * q;
    let numerator = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
    let denominator = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;

    fn bivariate_gaussian(rows: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for _ in 0..rows {
            let e1 = normal01(&mut rng);
            let e2 = normal01(&mut rng);
            // x, y 相关（corr ≈ 0.8）。
            let xi = e1;
            let yi = 0.8 * e1 + 0.6 * e2;
            x.push(xi);
            y.push(yi);
        }
        vec![x, y]
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let na = a.iter().sum::<f64>() / a.len() as f64;
        let nb = b.iter().sum::<f64>() / b.len() as f64;
        let cov = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - na) * (y - nb))
            .sum::<f64>();
        let va = a.iter().map(|x| (x - na) * (x - na)).sum::<f64>().sqrt();
        let vb = b.iter().map(|y| (y - nb) * (y - nb)).sum::<f64>().sqrt();
        cov / (va * vb)
    }

    #[test]
    fn generation_preserves_correlation() {
        let rows = 600;
        let features = bivariate_gaussian(rows, 7);
        let ds = dataset_util::dataset_from_columns(
            &[
                ("f0".to_string(), features[0].clone()),
                ("f1".to_string(), features[1].clone()),
            ],
            "target",
            features[0].clone(),
        )
        .unwrap();
        let params = BoostingParams {
            n_estimators: 60,
            ..BoostingParams::default()
        };
        let ctx = TrainingContext::new(3);
        let ff = ForestFlow::fit(&ds, &params, &ctx, 20).unwrap();
        let samples = ff.generate(400, &TrainingContext::new(11)).unwrap();
        let (mut s0, mut s1) = (Vec::with_capacity(400), Vec::with_capacity(400));
        for s in &samples {
            s0.push(s[0]);
            s1.push(s[1]);
        }
        let corr = correlation(&s0, &s1);
        assert!(corr > 0.5, "生成样本应保留特征相关性，实测 corr={corr:.3}");
        // 边际 std 应接近 1（z 空间标准正态 → 反归一后 std≈std）。
        let mean0 = s0.iter().sum::<f64>() / s0.len() as f64;
        let std0 =
            (s0.iter().map(|v| (v - mean0) * (v - mean0)).sum::<f64>() / s0.len() as f64).sqrt();
        assert!(
            (std0 - 1.0).abs() < 0.4,
            "边际 std 应接近 1，实测 {std0:.3}"
        );
    }

    #[test]
    fn empirical_normal_transform_roundtrips() {
        let values: [f64; 5] = [-10.0, -1.0, 0.0, 2.0, 30.0];
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        for &value in &values {
            let z = empirical_normal_score(&sorted, value);
            let restored = empirical_inverse_normal(&sorted, z);
            assert!(
                (restored - value).abs() < 1e-4,
                "{value} -> {z} -> {restored}"
            );
        }
        for &value in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let roundtrip = normal_quantile(normal_cdf(value));
            assert!(
                (roundtrip - value).abs() < 1e-5,
                "normal roundtrip {value} -> {roundtrip}"
            );
        }
    }

    #[test]
    fn sampling_is_deterministic() {
        let features = bivariate_gaussian(200, 8);
        let ds = dataset_util::dataset_from_columns(
            &[
                ("f0".to_string(), features[0].clone()),
                ("f1".to_string(), features[1].clone()),
            ],
            "target",
            features[0].clone(),
        )
        .unwrap();
        let params = BoostingParams {
            n_estimators: 40,
            ..BoostingParams::default()
        };
        let ff = ForestFlow::fit(&ds, &params, &TrainingContext::new(4), 12).unwrap();
        let a = ff.generate(50, &TrainingContext::new(99)).unwrap();
        let b = ff.generate(50, &TrainingContext::new(99)).unwrap();
        assert_eq!(a, b, "同 seed 生成必须逐位一致（红线 3）");
    }

    #[test]
    fn imputation_beats_mean_fill() {
        let features = bivariate_gaussian(400, 9);
        let ds = dataset_util::dataset_from_columns(
            &[
                ("f0".to_string(), features[0].clone()),
                ("f1".to_string(), features[1].clone()),
            ],
            "target",
            features[0].clone(),
        )
        .unwrap();
        let params = BoostingParams {
            n_estimators: 60,
            ..BoostingParams::default()
        };
        let ctx = TrainingContext::new(5);
        let ff = ForestFlow::fit(&ds, &params, &ctx, 20).unwrap();
        // 掩掉 f1 的 30%（用 f0 条件填补）。
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
        let mut imp_pred = Vec::new();
        let mut imp_true = Vec::new();
        let mut baseline_pred = Vec::new();
        let mean_f1 = features[1].iter().sum::<f64>() / features[1].len() as f64;
        for (&f0, &f1) in features[0].iter().zip(&features[1]) {
            let masked = uniform01(&mut rng) < 0.3;
            if masked {
                let row = vec![f0, f64::NAN];
                let obs = vec![true, false];
                let imputed = ff.impute(&row, &obs, &TrainingContext::new(13)).unwrap();
                imp_pred.push(imputed[1]);
                imp_true.push(f1);
                baseline_pred.push(mean_f1);
            }
        }
        let ff_rmse = dataset_util::rmse(&imp_true, &imp_pred);
        let base_rmse = dataset_util::rmse(&imp_true, &baseline_pred);
        assert!(
            ff_rmse < base_rmse * 0.9,
            "流匹配填补 RMSE {ff_rmse:.4} 应至少比均值基线 {base_rmse:.4} 改善 10%"
        );
    }

    #[test]
    fn mean_imputation_is_deterministic() {
        let features = bivariate_gaussian(120, 14);
        let ds = dataset_util::dataset_from_columns(
            &[
                ("f0".to_string(), features[0].clone()),
                ("f1".to_string(), features[1].clone()),
            ],
            "target",
            features[0].clone(),
        )
        .unwrap();
        let ff = ForestFlow::fit(
            &ds,
            &BoostingParams::default(),
            &TrainingContext::new(15),
            12,
        )
        .unwrap();
        let row = [features[0][0], f64::NAN];
        let observed = [true, false];
        let a = ff
            .impute_mean(&row, &observed, 4, &TrainingContext::new(16))
            .unwrap();
        let b = ff
            .impute_mean(&row, &observed, 4, &TrainingContext::new(16))
            .unwrap();
        assert_eq!(a, b, "同 seed 的 Monte Carlo 点填补必须逐位一致");
    }
}
