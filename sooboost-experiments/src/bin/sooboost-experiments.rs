//! M2-D benchmark bridge：把实验 crate 的 ForestFlow 接到 benchmark/run_benchmark.py。
//!
//! 这是研究工具 CLI，不是稳定产品 API；参数解析保持手写，避免引入 clap。

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use sooboost_core::boosting::{BoostingParams, TrainingContext};
use sooboost_core::data::{Dataset, MissingPolicy};
use sooboost_experiments::forest_flow::ForestFlow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generate,
    Impute,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    train: PathBuf,
    test: Option<PathBuf>,
    features: Vec<String>,
    target: String,
    output: PathBuf,
    count: usize,
    feature_index: Option<usize>,
    seed: u64,
    steps: usize,
    imputation_samples: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let feature_refs: Vec<&str> = args.features.iter().map(String::as_str).collect();
    let train = Dataset::from_csv_path(
        &args.train,
        &feature_refs,
        &args.target,
        MissingPolicy::default(),
    )?;
    let params = BoostingParams {
        n_estimators: 60,
        ..BoostingParams::default()
    };
    let flow = ForestFlow::fit(
        &train,
        &params,
        &TrainingContext::new(args.seed),
        args.steps,
    )?;

    match args.mode {
        Mode::Generate => write_generated(
            &args.output,
            &args.features,
            &flow.generate(args.count, &TrainingContext::new(args.seed))?,
        )?,
        Mode::Impute => {
            let feature_index = args
                .feature_index
                .ok_or("--mode impute 需要 --feature-index")?;
            if feature_index >= args.features.len() {
                return Err("--feature-index 越界".into());
            }
            let test_path = args.test.as_ref().ok_or("--mode impute 需要 --test")?;
            let test = Dataset::from_csv_path(
                test_path,
                &feature_refs,
                &args.target,
                MissingPolicy::default(),
            )?;
            write_imputations(
                &args.output,
                &test,
                &flow,
                feature_index,
                args.seed,
                args.imputation_samples,
            )?;
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut it = env::args().skip(1);
    let mut mode = None;
    let mut train = None;
    let mut test = None;
    let mut features = None;
    let mut target = String::from("target");
    let mut output = None;
    let mut count = 256usize;
    let mut feature_index = None;
    let mut seed = 42u64;
    let mut steps = 20usize;
    let mut imputation_samples = 4usize;

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--mode" => mode = Some(parse_mode(&next_value(&mut it, &flag)?)?),
            "--train" => train = Some(PathBuf::from(next_value(&mut it, &flag)?)),
            "--test" => test = Some(PathBuf::from(next_value(&mut it, &flag)?)),
            "--features" => {
                let value = next_value(&mut it, &flag)?;
                let names: Vec<String> = value
                    .split(',')
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect();
                if names.is_empty() {
                    return Err("--features 不能为空".into());
                }
                features = Some(names);
            }
            "--target" => target = next_value(&mut it, &flag)?,
            "--output" => output = Some(PathBuf::from(next_value(&mut it, &flag)?)),
            "--count" => count = next_value(&mut it, &flag)?.parse()?,
            "--feature-index" => feature_index = Some(next_value(&mut it, &flag)?.parse()?),
            "--seed" => seed = next_value(&mut it, &flag)?.parse()?,
            "--steps" => steps = next_value(&mut it, &flag)?.parse()?,
            "--imputation-samples" => imputation_samples = next_value(&mut it, &flag)?.parse()?,
            "--help" | "-h" => {
                print_help();
                return Err("help requested".into());
            }
            _ => return Err(format!("未知参数: {flag}").into()),
        }
    }

    let mode = mode.ok_or("缺少 --mode")?;
    let train = train.ok_or("缺少 --train")?;
    let features = features.ok_or("缺少 --features")?;
    let output = output.ok_or("缺少 --output")?;
    if count == 0 || steps == 0 || imputation_samples == 0 {
        return Err("--count、--steps 和 --imputation-samples 须 > 0".into());
    }
    Ok(Args {
        mode,
        train,
        test,
        features,
        target,
        output,
        count,
        feature_index,
        seed,
        steps,
        imputation_samples,
    })
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| format!("参数 {flag} 缺少值").into())
}

fn parse_mode(value: &str) -> Result<Mode, Box<dyn Error>> {
    match value {
        "generate" => Ok(Mode::Generate),
        "impute" => Ok(Mode::Impute),
        _ => Err(format!("不支持的 mode: {value}").into()),
    }
}

fn write_generated(
    path: &PathBuf,
    feature_names: &[String],
    rows: &[Vec<f64>],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    write_record(&mut out, feature_names.iter().map(String::as_str))?;
    for row in rows {
        if row.len() != feature_names.len() {
            return Err("生成行宽度与特征数不符".into());
        }
        write_record(&mut out, row.iter().map(|value| value.to_string()))?;
    }
    Ok(())
}

fn write_imputations(
    path: &PathBuf,
    ds: &Dataset,
    flow: &ForestFlow,
    feature_index: usize,
    seed: u64,
    imputation_samples: usize,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    write_record(&mut out, ["row_index", "actual", "imputed"])?;
    let mut feature_cols = Vec::with_capacity(ds.num_features());
    for feature in 0..ds.num_features() {
        feature_cols.push(ds.feature_values(feature)?);
    }
    for row_index in 0..ds.num_rows() {
        let mut row = Vec::with_capacity(ds.num_features());
        let mut observed = Vec::with_capacity(ds.num_features());
        for (feature, col) in feature_cols.iter().enumerate() {
            let missing = ds.is_missing(row_index, feature)? || feature == feature_index;
            row.push(if missing {
                f64::NAN
            } else {
                col.value(row_index)
            });
            observed.push(!missing);
        }
        let context = TrainingContext::new(seed.wrapping_add(row_index as u64 + 1));
        let imputed = flow.impute_mean(&row, &observed, imputation_samples, &context)?;
        write_record(
            &mut out,
            [
                row_index.to_string(),
                feature_cols[feature_index].value(row_index).to_string(),
                imputed[feature_index].to_string(),
            ],
        )?;
    }
    Ok(())
}

fn write_record<I, S>(out: &mut impl Write, fields: I) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut first = true;
    for field in fields {
        if !first {
            write!(out, ",")?;
        }
        first = false;
        write!(out, "{}", field.as_ref())?;
    }
    writeln!(out)?;
    Ok(())
}

fn print_help() {
    eprintln!(
        "用法:\n  sooboost-experiments --mode generate --train <csv> --features f0,f1 --target target --output <csv> [--count 256] [--seed 42] [--steps 20]\n  sooboost-experiments --mode impute --train <csv> --test <csv> --features f0,f1 --target target --feature-index 0 --output <csv> [--seed 42] [--steps 20] [--imputation-samples 4]"
    );
}
