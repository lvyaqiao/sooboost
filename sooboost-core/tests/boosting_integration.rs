//! 训练层集成测试：benchmark 金标准 CSV → 训练 → 预测 闭环。
//!
//! 守护（m0-spec §2 承诺 4/5/8/9）：
//! - L2 与 binary logloss 端到端可训练（学习发生）；
//! - 固定 seed 确定性测试入 CI（红线 3 层级一：逐位一致）；
//! - 树参数（n_estimators / max_depth）生效。

use std::path::PathBuf;

use sooboost_core::binning::DEFAULT_MAX_BINS;
use sooboost_core::boosting::{BoostingParams, TrainingContext, fit};
use sooboost_core::data::{Dataset, MissingPolicy};
use sooboost_core::loss::{BinaryLogLoss, Loss, SquaredError};
use sooboost_core::tree::TreeParams;

const SYNTH_REG_TRAIN: &str = "benchmark/synthetic_regression/train.csv";
const SYNTH_BIN_TRAIN: &str = "benchmark/synthetic_binary/train.csv";

fn benchmark_path(rel: &str) -> PathBuf {
    // Resolve the workspace root by walking up from CARGO_MANIFEST_DIR
    // until we find the directory that owns `benchmark/`. Robust to crate
    // nesting depth and works on a clean clone (CI) without assuming the
    // crate is exactly one level below the workspace root.
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while let Some(parent) = dir.parent() {
        if parent.join("benchmark").is_dir() {
            return parent.join(rel);
        }
        dir = parent.to_path_buf();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn feature_cols(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("f{i}")).collect()
}

fn load_synth_reg() -> Dataset {
    let features = feature_cols(10);
    Dataset::from_csv_path(
        benchmark_path(SYNTH_REG_TRAIN),
        &features.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 synthetic_regression train.csv")
}

fn load_synth_bin() -> Dataset {
    let features = feature_cols(10);
    Dataset::from_csv_path(
        benchmark_path(SYNTH_BIN_TRAIN),
        &features.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 synthetic_binary train.csv")
}

/// 秩无关的 AUC（O(n²)，仅测试用，n=1600 量级可接受）。
fn auc(labels: &[f64], scores: &[f64]) -> f64 {
    let n = labels.len();
    let n_pos = labels.iter().filter(|&&l| l > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut concordant = 0.0;
    for i in 0..n {
        if labels[i] > 0.5 {
            for j in 0..n {
                if labels[j] <= 0.5 && scores[i] > scores[j] {
                    concordant += 1.0;
                }
            }
        }
    }
    concordant / (n_pos * n_neg)
}

#[test]
fn l2_fit_reduces_mse_vs_constant() {
    let ds = load_synth_reg();
    let params = BoostingParams {
        n_estimators: 50,
        ..BoostingParams::default()
    };
    let booster = fit(&ds, &params, SquaredError, &TrainingContext::new(42)).expect("fit L2");
    assert_eq!(booster.num_trees(), 50);

    let y: Vec<f64> = ds.target_values().values().to_vec();
    let preds = booster.predict(&ds).expect("predict");
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let variance = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / y.len() as f64;
    let mse = y
        .iter()
        .zip(&preds)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / y.len() as f64;
    assert!(
        mse < 0.5 * variance,
        "训练 MSE {mse} 应显著低于常数基线方差 {variance}"
    );
}

#[test]
fn binary_fit_achieves_meaningful_auc() {
    let ds = load_synth_bin();
    let params = BoostingParams {
        n_estimators: 50,
        ..BoostingParams::default()
    };
    let booster = fit(&ds, &params, BinaryLogLoss, &TrainingContext::new(7)).expect("fit binary");
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let probs = booster.predict(&ds).expect("predict");

    let score = auc(&y, &probs);
    assert!(
        score > 0.7,
        "二分类训练后 AUC 应明显高于随机 0.5，实际 {score}"
    );
    // 概率应全部落在 (0,1)
    assert!(probs.iter().all(|p| *p > 0.0 && *p < 1.0));
}

#[test]
fn deterministic_same_seed_same_params_bitwise() {
    let ds = load_synth_bin();
    let params = BoostingParams {
        n_estimators: 15,
        ..BoostingParams::default()
    };
    let a = fit(&ds, &params, BinaryLogLoss, &TrainingContext::new(1234)).expect("fit A");
    let b = fit(&ds, &params, BinaryLogLoss, &TrainingContext::new(1234)).expect("fit B");

    let preds_a = a.raw_scores(&ds).expect("raw A");
    let preds_b = b.raw_scores(&ds).expect("raw B");
    assert_eq!(preds_a.len(), preds_b.len());
    // 红线 3 层级一：同输入同种子 → 逐位一致
    for (pa, pb) in preds_a.iter().zip(&preds_b) {
        assert_eq!(pa.to_bits(), pb.to_bits(), "逐位不一致");
    }
    assert_eq!(a.init_score().to_bits(), b.init_score().to_bits());
}

#[test]
fn parallel_deterministic_across_thread_counts() {
    // 红线 3 层级二：相同输入 + 相同种子 → 预测逐位一致（与线程数无关）。
    // 用自定义线程池（install 在指定池运行），绕开全局池初始化顺序问题。
    use rayon::ThreadPoolBuilder;
    let ds = load_synth_reg();
    let params = BoostingParams {
        n_estimators: 20,
        ..BoostingParams::default()
    };
    let pool1 = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool1");
    let pool4 = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool4");

    let r1 = pool1
        .install(|| fit(&ds, &params, SquaredError, &TrainingContext::new(7)).expect("fit 1 线程"));
    let r4 = pool4
        .install(|| fit(&ds, &params, SquaredError, &TrainingContext::new(7)).expect("fit 4 线程"));

    let p1 = r1.predict(&ds).expect("p1");
    let p4 = r4.predict(&ds).expect("p4");
    assert_eq!(p1.len(), p4.len());
    for (a, b) in p1.iter().zip(&p4) {
        assert_eq!(a.to_bits(), b.to_bits(), "1 线程与 4 线程预测应逐位一致");
    }
    // 模型字节也应逐位一致（含分箱表）
    assert_eq!(r1.serialize(), r4.serialize(), "不同线程数序列化应逐位一致");
}

#[test]
fn more_trees_improves_training_loss() {
    let ds = load_synth_reg();
    let small = BoostingParams {
        n_estimators: 10,
        ..BoostingParams::default()
    };
    let large = BoostingParams {
        n_estimators: 40,
        ..BoostingParams::default()
    };
    let loss = SquaredError;
    let b_small = fit(&ds, &small, loss, &TrainingContext::new(1)).expect("fit small");
    let b_large = fit(&ds, &large, loss, &TrainingContext::new(1)).expect("fit large");

    let y: Vec<f64> = ds.target_values().values().to_vec();
    let mean_loss = |booster: &sooboost_core::boosting::Booster<SquaredError>| -> f64 {
        let raw = booster.raw_scores(&ds).expect("raw");
        raw.iter()
            .zip(&y)
            .map(|(r, &yi)| loss.value(yi, *r))
            .sum::<f64>()
            / y.len() as f64
    };
    let l_small = mean_loss(&b_small);
    let l_large = mean_loss(&b_large);
    assert!(
        l_large <= l_small,
        "40 棵树训练损失 {l_large} 应不高于 10 棵树 {l_small}"
    );
}

#[test]
fn predict_row_matches_batch_predict() {
    let ds = load_synth_reg();
    let params = BoostingParams {
        n_estimators: 20,
        max_bins: 64,
        tree_params: TreeParams {
            max_depth: 4,
            ..TreeParams::default()
        },
        ..BoostingParams::default()
    };
    let booster = fit(&ds, &params, SquaredError, &TrainingContext::new(99)).expect("fit");
    let preds = booster.predict(&ds).expect("predict");

    for (r, &expected) in preds.iter().take(10).enumerate() {
        let mut values = vec![0.0; ds.num_features()];
        let mut is_missing = vec![false; ds.num_features()];
        for f in 0..ds.num_features() {
            let col = ds.feature_values(f).expect("取值");
            values[f] = col.value(r);
            is_missing[f] = ds.is_missing(r, f).expect("缺失查询");
        }
        let single = booster.predict_row(&values, &is_missing);
        assert_eq!(
            single.to_bits(),
            expected.to_bits(),
            "行 {r} 单行/批量不一致"
        );
    }
}

#[test]
fn max_bins_param_is_respected() {
    // 极小 max_bins 应产生明显更粗的分箱 → 更差或等价的训练拟合，但不崩溃
    let ds = load_synth_reg();
    let coarse = BoostingParams {
        n_estimators: 10,
        max_bins: 2,
        ..BoostingParams::default()
    };
    let booster = fit(&ds, &coarse, SquaredError, &TrainingContext::new(3)).expect("fit coarse");
    let preds = booster.predict(&ds).expect("predict");
    assert_eq!(preds.len(), ds.num_rows());
    assert!(preds.iter().all(|p| p.is_finite()), "预测应全有限");
    assert_eq!(booster.num_trees(), 10);
    // 未显式设置时默认值即 DEFAULT_MAX_BINS
    assert_eq!(BoostingParams::default().max_bins, DEFAULT_MAX_BINS);
}
