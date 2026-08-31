//! 分箱集成测试：从 benchmark 数据构建 BinTable + BinnedMatrix。
//! 覆盖红线 3 层级一（确定性：两次构建逐位一致）。

use std::path::PathBuf;

use sooboost_core::binning::{BinTable, DEFAULT_MAX_BINS, MISSING_BIN};
use sooboost_core::data::{Dataset, MissingPolicy};

fn benchmark_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../")
        .join(rel)
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

#[test]
fn binning_is_deterministic() {
    let ds = load("synthetic_regression");
    let (t1, m1) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    let (t2, m2) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    for f in 0..ds.num_features() {
        assert_eq!(t1.boundaries(f), t2.boundaries(f), "特征 {f} 边界逐位一致");
    }
    for f in 0..m1.num_features() {
        for r in 0..m1.num_rows() {
            assert_eq!(m1.bin(f, r), m2.bin(f, r));
        }
    }
}

#[test]
fn binning_produces_valid_bins() {
    let ds = load("synthetic_regression_nonlinear");
    let (table, matrix) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    assert_eq!(matrix.num_rows(), ds.num_rows());
    assert_eq!(matrix.num_features(), ds.num_features());
    for f in 0..ds.num_features() {
        assert!(
            table.boundaries(f).windows(2).all(|w| w[0] < w[1]),
            "特征 {f} 边界严格升序"
        );
        for r in 0..ds.num_rows() {
            let b = matrix.bin(f, r);
            // 合成数据无缺失 → 恒为有效 bin，且小于该特征有效 bin 数
            assert_ne!(b, MISSING_BIN, "特征 {f} 行 {r} 不应缺失");
            assert!(
                (b as usize) < table.num_bins(f),
                "特征 {f} 行 {r} bin {} 越界（num_bins={}）",
                b,
                table.num_bins(f)
            );
        }
    }
}

#[test]
fn binning_handles_california_housing() {
    let feature_names: Vec<String> = (0..8).map(|i| format!("f{i}")).collect();
    let ds = Dataset::from_csv_path(
        benchmark_path("benchmark/california_housing/train.csv"),
        &feature_names.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("读取 train.csv");
    let (_table, matrix) = BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱");
    assert_eq!(matrix.num_rows(), ds.num_rows());
}
