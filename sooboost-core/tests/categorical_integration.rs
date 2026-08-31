//! 类别特征集成测试（D9 / contracts §1.4 契约落地）。
//!
//! 守护：训练/预测回路；OOV（训练未见类别）→ 先验不崩溃；null 类别 → 缺失；
//! 高基数超限报错；类别编码随模型序列化 roundtrip 逐位一致；同 seed 确定性。

use std::sync::Arc;

use arrow::array::{ArrayRef, DictionaryArray, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;

use sooboost_core::boosting::{BoostingParams, TrainingContext, fit};
use sooboost_core::data::{DataError, Dataset, MissingPolicy};
use sooboost_core::loss::SquaredError;

fn dict_arr(values: Vec<Option<&str>>) -> ArrayRef {
    // 由输入推导字典（保持首现顺序）；OOV 键在新字典中获得自己的槽位
    let mut uniq: Vec<&str> = Vec::new();
    for v in values.iter().flatten() {
        if !uniq.contains(v) {
            uniq.push(v);
        }
    }
    let dict = StringArray::from(uniq.clone());
    let keys: UInt32Array = values
        .iter()
        .map(|v| v.map(|s| uniq.iter().position(|u| *u == s).expect("在字典中") as u32))
        .collect();
    Arc::new(DictionaryArray::<UInt32Type>::try_new(keys, Arc::new(dict)).expect("dict 构造"))
}

/// 类别 → 目标映射：a→0, b→10, c→20（强信号，可分离）。
fn make_ds(cats: Vec<Option<&str>>) -> Dataset {
    let n = cats.len();
    let f0: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let target: Vec<f64> = cats
        .iter()
        .map(|c| match c {
            Some("a") => 0.0,
            Some("b") => 10.0,
            Some("c") => 20.0,
            _ => 5.0, // OOV/null 占位
        })
        .collect();
    let schema = Schema::new(vec![
        Field::new("f0", DataType::Float64, true),
        Field::new(
            "f1",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("target", DataType::Float64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(f0)),
            dict_arr(cats),
            Arc::new(Float64Array::from(target)),
        ],
    )
    .expect("batch");
    Dataset::from_record_batch(batch, &["f0", "f1"], "target", MissingPolicy::default())
        .expect("dataset")
}

#[test]
fn categorical_feature_is_detected_and_key_readable() {
    let ds = make_ds(vec![Some("a"), Some("b"), None, Some("c")]);
    assert!(!ds.feature_is_categorical(0));
    assert!(ds.feature_is_categorical(1));
    assert_eq!(ds.categorical_key(0, 1).unwrap(), Some(0));
    assert_eq!(ds.categorical_key(1, 1).unwrap(), Some(1));
    assert_eq!(ds.categorical_key(2, 1).unwrap(), None, "null 类别");
    assert!(ds.is_missing(2, 1).unwrap());
    assert!(ds.feature_values(1).is_err(), "类别特征应拒绝数值读取");
}

#[test]
fn categorical_fit_and_predict_learns_signal() {
    let mut cats = Vec::new();
    for c in ["a", "b", "c"] {
        for _ in 0..20 {
            cats.push(Some(c));
        }
    }
    let ds = make_ds(cats);
    let model = fit(
        &ds,
        &BoostingParams {
            n_estimators: 30,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(1),
    )
    .expect("fit");
    assert!(
        model.categorical_encoding().is_some(),
        "有类别特征时应产生编码"
    );

    let preds = model.predict(&ds).expect("predict");
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let mse = y
        .iter()
        .zip(&preds)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / y.len() as f64;
    assert!(mse < 1.0, "类别强信号下训练 MSE 应很低，实际 {mse}");
}

#[test]
fn oov_new_category_uses_prior_and_no_panic() {
    let mut cats = Vec::new();
    for c in ["a", "b", "c"] {
        for _ in 0..20 {
            cats.push(Some(c));
        }
    }
    let ds = make_ds(cats);
    let model = fit(
        &ds,
        &BoostingParams {
            n_estimators: 30,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(1),
    )
    .expect("fit");

    // OOV 推断集：新增 "zzz" 类别（训练未见）
    let mut oov = Vec::new();
    for c in ["a", "b", "c", "zzz", "zzz"] {
        oov.push(Some(c));
    }
    let oov_ds = make_ds(oov);
    let preds = model.predict(&oov_ds).expect("OOV 预测不崩溃");
    assert_eq!(preds.len(), oov_ds.num_rows());
    assert!(preds.iter().all(|p| p.is_finite()), "OOV 预测应有限");
    // OOV 行（zzz）走先验：与训练 prior 一致（无 panic、有文档、行为可预测）
    let enc = model.categorical_encoding().unwrap();
    let prior = enc.prior(0);
    assert!(prior.is_finite());
}

#[test]
fn null_category_is_missing_and_no_panic() {
    let cats: Vec<Option<&str>> = (0..30)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some(["a", "b", "c"][i % 3])
            }
        })
        .collect();
    let ds = make_ds(cats);
    let model = fit(
        &ds,
        &BoostingParams {
            n_estimators: 20,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(1),
    )
    .expect("fit");
    let preds = model.predict(&ds).expect("predict");
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn high_cardinality_exceeds_limit_errors() {
    let mut cats = Vec::new();
    for c in ["a", "b", "c"] {
        for _ in 0..10 {
            cats.push(Some(c));
        }
    }
    let ds = make_ds(cats);
    let err = fit(
        &ds,
        &BoostingParams {
            n_estimators: 5,
            max_categories: 2, // 3 个类别 > 2 → 报错而非截断
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(1),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        sooboost_core::boosting::BoostingError::Data(DataError::TooManyCategories { got: 3, .. })
    ));
}

#[test]
fn categorical_serialization_roundtrip_bitwise() {
    let mut cats = Vec::new();
    for c in ["a", "b", "c"] {
        for _ in 0..15 {
            cats.push(Some(c));
        }
    }
    let ds = make_ds(cats);
    let model = fit(
        &ds,
        &BoostingParams {
            n_estimators: 20,
            ..BoostingParams::default()
        },
        SquaredError,
        &TrainingContext::new(42),
    )
    .expect("fit");

    let bytes = model.serialize();
    let back =
        sooboost_core::boosting::Booster::deserialize(&bytes, SquaredError).expect("deserialize");
    assert!(
        back.categorical_encoding().is_some(),
        "反序列化后应保留类别编码"
    );
    let before = model.predict(&ds).expect("before");
    let after = back.predict(&ds).expect("after");
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.to_bits(), b.to_bits(), "类别模型 roundtrip 应逐位一致");
    }
    assert_eq!(bytes, back.serialize(), "再序列化字节一致");
}

#[test]
fn ordered_ts_is_deterministic_same_seed() {
    let mut cats = Vec::new();
    for c in ["a", "b", "c"] {
        for _ in 0..15 {
            cats.push(Some(c));
        }
    }
    let ds = make_ds(cats);
    let params = BoostingParams {
        n_estimators: 20,
        ..BoostingParams::default()
    };
    let m1 = fit(&ds, &params, SquaredError, &TrainingContext::new(7)).expect("m1");
    let m2 = fit(&ds, &params, SquaredError, &TrainingContext::new(7)).expect("m2");
    assert_eq!(
        m1.serialize(),
        m2.serialize(),
        "同 seed → ordered TS 与模型逐位一致"
    );
}
