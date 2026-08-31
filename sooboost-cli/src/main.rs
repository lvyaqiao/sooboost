//! sooboost-cli：训练 + 预测命令行工具。
//!
//! 用法（M0 范围，m0-spec §2 承诺 4/5/8/9 的 CLI 出口）：
//! ```text
//! sooboost-cli train \
//!   --train <train.csv> --test <test.csv> \
//!   --features f0,f1,f2 --target target \
//!   --task regression|binary --output <out.csv> \
//!   [--n-estimators 100] [--learning-rate 0.1] [--max-depth 6]
//!   [--min-samples-leaf 5] [--min-split-gain 0.0] [--max-bins 255] [--seed 42]
//! ```
//!
//! 输出 CSV 与 benchmark correctness 档同构：
//! - regression：`y_true,y_pred`
//! - binary：`y_true,y_pred,y_prob`（y_pred = y_prob ≥ 0.5）

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use sooboost_core::boosting::{BoostingParams, TrainingContext, fit};
use sooboost_core::data::{Dataset, MissingPolicy};
use sooboost_core::loss::{BinaryLogLoss, Loss, SquaredError};
use sooboost_core::tree::TreeParams;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    Regression,
    Binary,
}

#[derive(Debug)]
struct Args {
    train: PathBuf,
    test: PathBuf,
    features: Vec<String>,
    target: String,
    task: Task,
    output: PathBuf,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_split_gain: f64,
    max_bins: usize,
    seed: u64,
}

const USAGE: &str = "用法: sooboost-cli train --train <csv> --test <csv> --features f0,f1 --target target --task regression|binary --output <csv> [选项]";

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("错误: {msg}");
    eprintln!("{USAGE}");
    exit(2);
}

fn parse_task(s: &str) -> Task {
    match s {
        "regression" => Task::Regression,
        "binary" => Task::Binary,
        other => die(format!("未知任务类型 '{other}'（可选 regression|binary）")),
    }
}

fn parse_args() -> Args {
    let mut args = Args {
        train: PathBuf::new(),
        test: PathBuf::new(),
        features: Vec::new(),
        target: String::new(),
        task: Task::Regression,
        output: PathBuf::new(),
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 6,
        min_samples_leaf: 5,
        min_split_gain: 0.0,
        max_bins: 255,
        seed: 42,
    };
    let mut it = env::args().skip(1);
    let sub = it.next().unwrap_or_default();
    if sub != "train" {
        die(format!("未知子命令 '{sub}'（当前仅支持 train）"));
    }
    while let Some(key) = it.next() {
        let val = match it.next() {
            Some(v) => v,
            None => die(format!("参数 '{key}' 缺少值")),
        };
        match key.as_str() {
            "--train" => args.train = PathBuf::from(val),
            "--test" => args.test = PathBuf::from(val),
            "--features" => {
                args.features = val.split(',').map(str::to_string).collect();
                if args.features.is_empty() {
                    die("--features 不能为空");
                }
            }
            "--target" => args.target = val,
            "--task" => args.task = parse_task(&val),
            "--output" => args.output = PathBuf::from(val),
            "--n-estimators" => {
                args.n_estimators = val
                    .parse()
                    .unwrap_or_else(|_| die("--n-estimators 需为整数"))
            }
            "--learning-rate" => {
                args.learning_rate = val
                    .parse()
                    .unwrap_or_else(|_| die("--learning-rate 需为浮点数"))
            }
            "--max-depth" => {
                args.max_depth = val.parse().unwrap_or_else(|_| die("--max-depth 需为整数"))
            }
            "--min-samples-leaf" => {
                args.min_samples_leaf = val
                    .parse()
                    .unwrap_or_else(|_| die("--min-samples-leaf 需为整数"))
            }
            "--min-split-gain" => {
                args.min_split_gain = val
                    .parse()
                    .unwrap_or_else(|_| die("--min-split-gain 需为浮点数"))
            }
            "--max-bins" => {
                args.max_bins = val.parse().unwrap_or_else(|_| die("--max-bins 需为整数"))
            }
            "--seed" => args.seed = val.parse().unwrap_or_else(|_| die("--seed 需为整数")),
            other => die(format!("未知参数 '{other}'")),
        }
    }
    if args.train.as_os_str().is_empty() {
        die("缺少 --train");
    }
    if args.test.as_os_str().is_empty() {
        die("缺少 --test");
    }
    if args.features.is_empty() {
        die("缺少 --features");
    }
    if args.target.is_empty() {
        die("缺少 --target");
    }
    if args.output.as_os_str().is_empty() {
        die("缺少 --output");
    }
    args
}

fn load_dataset(path: &Path, features: &[String], target: &str) -> Dataset {
    Dataset::from_csv_path(
        path,
        &features.iter().map(String::as_str).collect::<Vec<_>>(),
        target,
        MissingPolicy::default(),
    )
    .unwrap_or_else(|e| die(format!("读取 {} 失败: {e}", path.display())))
}

fn main() {
    let args = parse_args();

    let train_ds = load_dataset(&args.train, &args.features, &args.target);
    let test_ds = load_dataset(&args.test, &args.features, &args.target);
    println!(
        "train: {} 行 × {} 特征；test: {} 行",
        train_ds.num_rows(),
        train_ds.num_features(),
        test_ds.num_rows()
    );

    let params = BoostingParams {
        n_estimators: args.n_estimators,
        learning_rate: args.learning_rate,
        max_bins: args.max_bins,
        tree_params: TreeParams {
            max_depth: args.max_depth,
            min_samples_leaf: args.min_samples_leaf,
            min_split_gain: args.min_split_gain,
            ..TreeParams::default()
        },
        ..BoostingParams::default()
    };
    let ctx = TrainingContext::new(args.seed);

    let t0 = Instant::now();
    let (loss_name, preds, header) = match args.task {
        Task::Regression => {
            let booster = fit(&train_ds, &params, SquaredError, &ctx)
                .unwrap_or_else(|e| die(format!("训练失败: {e}")));
            let preds = booster
                .predict(&test_ds)
                .unwrap_or_else(|e| die(format!("预测失败: {e}")));
            println!(
                "回归训练完成: 树 {} 棵，init={:.6}，耗时 {:.3}s",
                booster.num_trees(),
                booster.init_score(),
                t0.elapsed().as_secs_f64()
            );
            (booster.loss().name(), preds, "y_true,y_pred".to_string())
        }
        Task::Binary => {
            let booster = fit(&train_ds, &params, BinaryLogLoss, &ctx)
                .unwrap_or_else(|e| die(format!("训练失败: {e}")));
            let probs = booster
                .predict(&test_ds)
                .unwrap_or_else(|e| die(format!("预测失败: {e}")));
            println!(
                "二分类训练完成: 树 {} 棵，init={:.6}，耗时 {:.3}s",
                booster.num_trees(),
                booster.init_score(),
                t0.elapsed().as_secs_f64()
            );
            (
                booster.loss().name(),
                probs,
                "y_true,y_pred,y_prob".to_string(),
            )
        }
    };
    println!("损失: {loss_name}");

    // 写输出 CSV（与 benchmark correctness 档同构）
    let y_true: Vec<f64> = test_ds.target_values().values().to_vec();
    let mut w = BufWriter::new(
        File::create(&args.output)
            .unwrap_or_else(|e| die(format!("写 {} 失败: {e}", args.output.display()))),
    );
    writeln!(w, "{header}").expect("写表头");
    for (i, &y) in y_true.iter().enumerate() {
        match args.task {
            Task::Regression => writeln!(w, "{y},{}", preds[i]).expect("写行"),
            Task::Binary => {
                let prob = preds[i];
                let y_pred = if prob >= 0.5 { 1.0 } else { 0.0 };
                writeln!(w, "{y},{y_pred},{prob}").expect("写行");
            }
        }
    }
    w.flush().expect("flush");
    println!(
        "已写出预测 -> {}（{} 行）",
        args.output.display(),
        y_true.len()
    );
}
