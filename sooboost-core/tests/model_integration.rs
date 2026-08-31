//! 模型序列化集成测试（contracts §1.2 契约落地）。
//!
//! 守护：save→load→预测逐位一致（roundtrip）；magic/版本/checksum/结构/loss 名
//! 错误显式分类；任意字节反序列化不崩溃（fuzz 占位，读入外部字节路径）。

use std::io::Write;
use std::path::PathBuf;

use proptest::prelude::*;

use sooboost_core::boosting::{BoostingParams, TrainingContext, fit};
use sooboost_core::data::{Dataset, MissingPolicy};
use sooboost_core::loss::{BinaryLogLoss, SquaredError};
use sooboost_core::model::format::{VERSION, fnv1a64};
use sooboost_core::model::{HotSwappable, ModelError};

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

fn load(rel: &str) -> Dataset {
    let features: Vec<String> = (0..10).map(|i| format!("f{i}")).collect();
    Dataset::from_csv_path(
        benchmark_path(rel),
        &features.iter().map(String::as_str).collect::<Vec<_>>(),
        "target",
        MissingPolicy::default(),
    )
    .expect("load csv")
}

fn fit_small(rel: &str) -> sooboost_core::boosting::Booster<SquaredError> {
    fit(
        &load(rel),
        &BoostingParams {
            n_estimators: 15,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(42),
    )
    .expect("fit")
}

#[test]
fn roundtrip_predictions_bitwise_identical() {
    let ds = load(SYNTH_REG_TRAIN);
    let model = fit_small(SYNTH_REG_TRAIN);
    let bytes = model.serialize();
    let back: sooboost_core::boosting::Booster<SquaredError> =
        sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError).expect("deserialize");

    let before = model.predict(&ds).expect("before");
    let after = back.predict(&ds).expect("after");
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.to_bits(), b.to_bits(), "roundtrip 预测应逐位一致");
    }
    assert_eq!(model.num_trees(), back.num_trees());
    assert_eq!(model.init_score().to_bits(), back.init_score().to_bits());
    // bin 表自包含且逐位一致
    assert_eq!(model.bin_table().max_bins(), back.bin_table().max_bins());
    for f in 0..model.bin_table().num_features() {
        assert_eq!(
            model.bin_table().boundaries(f),
            back.bin_table().boundaries(f)
        );
    }
}

#[test]
fn serialize_is_deterministic() {
    let m1 = fit_small(SYNTH_REG_TRAIN);
    let m2 = fit_small(SYNTH_REG_TRAIN);
    assert_eq!(
        m1.serialize(),
        m2.serialize(),
        "同输入同 seed → 序列化字节一致"
    );
}

#[test]
fn binary_roundtrip() {
    let ds = load(SYNTH_BIN_TRAIN);
    let model = fit(
        &ds,
        &BoostingParams {
            n_estimators: 10,
            ..BoostingParams::default()
        },
        BinaryLogLoss,
        &TrainingContext::new(1),
    )
    .expect("fit binary");
    let bytes = model.serialize();
    let back = sooboost_core::boosting::Booster::deserialize(&bytes, BinaryLogLoss)
        .expect("deserialize binary");
    let before = model.predict(&ds).expect("before");
    let after = back.predict(&ds).expect("after");
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn save_load_file_roundtrip() {
    let dir = std::env::temp_dir().join("sooboost_model_test");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join(format!("m1_{}.sb", std::process::id()));
    let model = fit_small(SYNTH_REG_TRAIN);
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(&model.serialize()).expect("write");

    let bytes = std::fs::read(&path).expect("read");
    std::fs::remove_file(&path).expect("cleanup");
    let back = sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError)
        .expect("deserialize from file");
    assert_eq!(back.num_trees(), model.num_trees());
}

#[test]
fn bad_magic_is_invalid_magic() {
    let mut bytes = fit_small(SYNTH_REG_TRAIN).serialize();
    bytes[0] = b'X';
    let err = sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError).unwrap_err();
    assert!(matches!(err, ModelError::InvalidMagic));
}

#[test]
fn unsupported_version_is_explicit() {
    let mut bytes = fit_small(SYNTH_REG_TRAIN).serialize();
    bytes[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
    let err = sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError).unwrap_err();
    assert!(matches!(
        err,
        ModelError::UnsupportedVersion { version, supported } if version == VERSION + 1 && supported == VERSION
    ));
}

#[test]
fn corrupted_checksum_detected() {
    let mut bytes = fit_small(SYNTH_REG_TRAIN).serialize();
    // 翻转主体中间一个字节（避开 magic/版本/checksum 位置）
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    let err = sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError).unwrap_err();
    assert!(matches!(err, ModelError::ChecksumFailed { .. }));
}

#[test]
fn half_truncated_fails_checksum_integrity() {
    let bytes = fit_small(SYNTH_REG_TRAIN).serialize();
    // 半截文件先被 checksum（尾部校验）拦截——完整性优先于结构解析
    let cut = &bytes[..bytes.len() / 2];
    let err = sooboost_core::boosting::Booster::deserialize(cut, SquaredError).unwrap_err();
    assert!(matches!(err, ModelError::ChecksumFailed { .. }));
}

#[test]
fn truncated_body_with_valid_checksum_is_truncated() {
    // 主体截断 + 重新计算合法 checksum → 结构解析遇到缓冲不足 → Truncated
    let full = fit_small(SYNTH_REG_TRAIN).serialize();
    let body = &full[..full.len() / 2];
    let mut crafted = body.to_vec();
    crafted.extend_from_slice(&fnv1a64(body).to_le_bytes());
    let err = sooboost_core::boosting::Booster::deserialize(&crafted, SquaredError).unwrap_err();
    assert!(matches!(err, ModelError::Truncated));
}

#[test]
fn loss_name_mismatch_is_explicit() {
    let bytes = fit_small(SYNTH_REG_TRAIN).serialize(); // SquaredError 模型
    let err = sooboost_core::boosting::Booster::deserialize(&bytes, BinaryLogLoss).unwrap_err();
    assert!(matches!(
        err,
        ModelError::LossMismatch { expected, found }
            if expected == "binary_logloss" && found == "squared_error"
    ));
}

#[test]
fn hot_swappable_roundtrip_serves_requests() {
    let ds = load(SYNTH_REG_TRAIN);
    let h = HotSwappable::new(fit_small(SYNTH_REG_TRAIN));
    let arc = h.load();
    assert_eq!(arc.num_trees(), 15);
    let preds = arc.predict(&ds).expect("predict via snapshot");
    assert_eq!(preds.len(), ds.num_rows());

    let updated = fit(
        &ds,
        &BoostingParams {
            n_estimators: 30,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(42),
    )
    .expect("fit updated");
    h.publish(updated);
    let fresh = h.load();
    assert_eq!(fresh.num_trees(), 30, "发布后新模型生效");
}

proptest! {
    /// fuzz 占位：任意字节输入不得 panic（读入外部字节路径，红线 6 操作化）。
    #[test]
    fn arbitrary_bytes_never_panic(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        let _ = sooboost_core::boosting::Booster::<SquaredError>::deserialize(&data, SquaredError);
        let _ = sooboost_core::boosting::Booster::<BinaryLogLoss>::deserialize(&data, BinaryLogLoss);
    }
}
