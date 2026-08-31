//! 数据层集成测试：读 benchmark 金标准 CSV → Dataset。
//! 数据来源：benchmark/（run_benchmark.py 生成，checksum 见 dataset_meta.json）。

use std::path::PathBuf;

use sooboost_core::data::{DataError, Dataset, MissingPolicy};

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

#[test]
fn reads_synthetic_regression_train() {
    let feature_names: Vec<String> = feature_cols(10);
    let ds = Dataset::from_csv_path(
        benchmark_path(SYNTH_REG_TRAIN),
        &feature_names.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 train.csv");
    assert_eq!(ds.num_rows(), 1600);
    assert_eq!(ds.num_features(), 10);
    assert_eq!(ds.target_name(), "target");

    // 合成数据无缺失
    for i in 0..ds.num_features() {
        for r in 0..ds.num_rows() {
            assert!(!ds.is_missing(r, i).unwrap(), "行 {r} 特征 {i} 不应缺失");
        }
    }
}

#[test]
fn reads_synthetic_binary_train() {
    let feature_names: Vec<String> = feature_cols(10);
    let ds = Dataset::from_csv_path(
        benchmark_path(SYNTH_BIN_TRAIN),
        &feature_names.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 train.csv");
    assert_eq!(ds.num_rows(), 1600);
    let target = ds.target_values();
    for r in 0..ds.num_rows() {
        let v = target.value(r);
        assert!(
            (v == 0.0 || v == 1.0),
            "二分类 target 应为 0/1，行 {r} 为 {v}"
        );
    }
}

#[test]
fn missing_csv_errors_explicitly() {
    let err = Dataset::from_csv_path(
        benchmark_path("benchmark/does_not_exist.csv"),
        &["f0"],
        "target",
        MissingPolicy::default(),
    )
    .unwrap_err();
    assert!(matches!(err, DataError::Io(_)));
}
