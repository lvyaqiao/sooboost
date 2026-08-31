//! 端到端示例：California Housing 房价回归。
//!
//! 覆盖完整回路：读 CSV → 训练 → 预测 → 评估 → 存盘 → 载入 → 复核
//! （对应测试纪律中的「集成：训练→预测→序列化→再预测的端到端回路」）。
//!
//! 运行：
//!
//! ```text
//! cargo run --release --example california_housing
//! ```
//!
//! 建议带 `--release`：`cargo test` 只编译本示例（不执行），真正跑起来
//! 在 debug 下会慢一个数量级。

use std::path::{Path, PathBuf};
use std::time::Instant;

use sooboost_core::api::GradientBoosting;
use sooboost_core::data::{Dataset, MissingPolicy};

/// California Housing 的 8 个特征列（与 benchmark CSV 表头一致）。
const FEATURES: [&str; 8] = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"];
const TARGET: &str = "target";

/// 从 crate 目录向上找到含 `benchmark/` 的 workspace 根。
///
/// 与集成测试的 `benchmark_path` 同一思路：对 crate 嵌套深度不敏感，
/// 干净 clone 上同样成立（不依赖 CARGO_MANIFEST_DIR 的相对层级）。
fn workspace_benchmark_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let candidate = dir.join("benchmark");
        if candidate.is_dir() {
            return candidate;
        }
        if !dir.pop() {
            panic!(
                "未找到 benchmark/ 目录（从 {} 向上查找失败）",
                env!("CARGO_MANIFEST_DIR")
            );
        }
    }
}

fn load(path: &Path) -> Result<Dataset, Box<dyn std::error::Error>> {
    let ds = Dataset::from_csv_path(path, &FEATURES, TARGET, MissingPolicy::default())?;
    Ok(ds)
}

/// 决定系数 R²。
fn r2(y: &[f64], pred: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let ss_res: f64 = y
        .iter()
        .zip(pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

/// 平均绝对误差。
fn mae(y: &[f64], pred: &[f64]) -> f64 {
    y.iter()
        .zip(pred.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = workspace_benchmark_dir().join("california_housing");
    let train_path = data_dir.join("train.csv");
    let test_path = data_dir.join("test.csv");

    println!("== 读入数据 ==");
    let t0 = Instant::now();
    let train = load(&train_path)?;
    let test = load(&test_path)?;
    println!(
        "train: {} 行 × {} 特征 ({:.2?})\ntest : {} 行",
        train.num_rows(),
        train.num_features(),
        t0.elapsed(),
        test.num_rows()
    );

    println!("\n== 训练（200 轮 / lr=0.1 / max_depth=6）==");
    let t1 = Instant::now();
    let model = GradientBoosting::regressor()
        .n_estimators(200)
        .learning_rate(0.1)
        .max_depth(6)
        .seed(42)
        .fit(&train)?;
    println!(
        "完成：{} 棵树，耗时 {:.2?}",
        model.num_trees(),
        t1.elapsed()
    );

    println!("\n== 测试集评估 ==");
    let y_test: Vec<f64> = test.target_values().values().to_vec();
    let preds = model.predict(&test)?;
    println!("R²  = {:.4}", r2(&y_test, &preds));
    println!("MAE = {:.4}", mae(&y_test, &preds));

    println!("\n== 存盘 → 载入 → 复核（红线 3 逐位一致）==");
    let model_path = std::env::temp_dir().join("sooboost_california_housing.sbm");
    let bytes = model.to_bytes();
    model.save(&model_path)?;
    println!("模型 {} 字节 → {}", bytes.len(), model_path.display());

    let loaded = GradientBoosting::load(&model_path)?;
    let preds_loaded = loaded.predict(&test)?;
    assert_eq!(
        preds, preds_loaded,
        "存读后预测必须逐位一致（红线 3 可复现性）"
    );
    println!(
        "载入模型目标 = {:?}，树数 = {}",
        loaded.objective(),
        loaded.num_trees()
    );
    println!("复核通过：载入模型与训练态预测完全一致");

    println!("\n== 单行预测（在线推断路径）==");
    let first_row: Vec<f64> = (0..FEATURES.len())
        .map(|f| test.feature_values(f).expect("数值特征列").value(0))
        .collect();
    println!(
        "第 0 行：预测 {:.2}，真值 {:.2}",
        model.predict_row(&first_row)?,
        y_test[0]
    );

    let _ = std::fs::remove_file(&model_path);
    Ok(())
}
