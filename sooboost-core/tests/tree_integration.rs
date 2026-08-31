//! 树集成测试：benchmark 数据上验证深度约束与训练损失改善。

use std::path::PathBuf;

use arrow::array::Array;

use sooboost_core::binning::{BinTable, DEFAULT_MAX_BINS};
use sooboost_core::data::{Dataset, MissingPolicy};
use sooboost_core::loss::{Loss, SquaredError};
use sooboost_core::tree::{TreeBuilder, TreeParams};

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

fn load(name: &str) -> Dataset {
    let feature_names: Vec<String> = (0..10).map(|i| format!("f{i}")).collect();
    Dataset::from_csv_path(
        benchmark_path(&format!("benchmark/{name}/train.csv")),
        &feature_names.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 train.csv")
}

/// L2 + init=均值 的梯度（单棵树第一轮，m0-spec §4 初始化）。
fn l2_first_grad(ds: &Dataset) -> (Vec<f64>, Vec<f64>) {
    let loss = SquaredError;
    let target = ds.target_values();
    let y: Vec<f64> = (0..ds.num_rows()).map(|r| target.value(r)).collect();
    let init = loss.init_score(&y);
    let grad: Vec<f64> = y.iter().map(|&yi| loss.gradient(yi, init)).collect();
    let hess: Vec<f64> = y.iter().map(|&yi| loss.hessian(yi, init)).collect();
    (grad, hess)
}

#[test]
fn tree_respects_max_depth() {
    for (name, depth) in [
        ("synthetic_regression", 3usize),
        ("synthetic_regression_nonlinear", 2),
    ] {
        let ds = load(name);
        let (table, matrix) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
        let (grad, hess) = l2_first_grad(&ds);
        let builder = TreeBuilder::new(TreeParams {
            max_depth: depth,
            ..TreeParams::default()
        });
        let tree = builder.build(&matrix, &table, &grad, &hess).expect("建树");
        assert!(
            tree.max_depth() <= depth,
            "{name}: max_depth {} > {depth}",
            tree.max_depth()
        );
    }
}

#[test]
fn single_tree_reduces_training_mse() {
    let ds = load("synthetic_regression");
    let (table, matrix) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    let target = ds.target_values();
    let y: Vec<f64> = (0..ds.num_rows()).map(|r| target.value(r)).collect();

    let loss = SquaredError;
    let init = loss.init_score(&y);
    let mse_before: f64 =
        y.iter().map(|&yi| loss.value(yi, init) * 2.0).sum::<f64>() / y.len() as f64;

    let (grad, hess) = l2_first_grad(&ds);
    let builder = TreeBuilder::new(TreeParams::default());
    let tree = builder.build(&matrix, &table, &grad, &hess).expect("建树");

    let mut mse_after = 0.0f64;
    for (r, &yi) in y.iter().enumerate() {
        // 用树预测该行：逐特征取值
        let row_pred = tree.predict_one(|f| {
            let col = ds.feature_values(f).expect("取值");
            let v = col.value(r);
            (v, col.is_null(r) || v.is_nan())
        });
        let full_pred = init + row_pred;
        mse_after += (yi - full_pred) * (yi - full_pred);
    }
    mse_after /= y.len() as f64;

    assert!(
        mse_after < mse_before,
        "单棵树后 MSE {mse_after} 应小于基线 {mse_before}"
    );
}

#[test]
fn tree_predictions_are_finite_on_all_rows() {
    let ds = load("synthetic_binary");
    let (table, matrix) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    let (grad, hess) = l2_first_grad(&ds);
    let builder = TreeBuilder::new(TreeParams::default());
    let tree = builder.build(&matrix, &table, &grad, &hess).expect("建树");

    for r in 0..ds.num_rows() {
        let pred = tree.predict_one(|f| {
            let col = ds.feature_values(f).expect("取值");
            let v = col.value(r);
            (v, col.is_null(r) || v.is_nan())
        });
        assert!(pred.is_finite(), "行 {r} 预测应为有限值，got {pred}");
    }
}
