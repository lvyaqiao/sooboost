//! C ABI 集成测试（M12）。
//!
//! 进程内直调 `extern "C"` 函数（cdylib crate 同时产出 rlib，测试无需加载
//! 动态库）；raw 指针解引用是唯一的 unsafe，与真实宿主语言的用法一致。
//! 所有断言口径：返回码、last_error 非空、数值结果与 Rust 门面同 seed 逐位一致。

use std::ffi::{CStr, CString, c_char};

use sooboost_ffi::{
    SbsModel, sbs_deserialize, sbs_last_error, sbs_model_free, sbs_model_num_classes,
    sbs_model_num_features, sbs_model_num_trees, sbs_predict, sbs_predict_proba, sbs_serialize,
    sbs_train, sbs_version,
};

/// 行主序测试数据：y = 2·x（100 点，无噪声）。
fn linear_data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    (x, y)
}

fn cjson(s: &str) -> CString {
    CString::new(s).expect("合法 JSON 字符串")
}

/// 通用训练入口：返回模型句柄（调用方负责 free）。
fn train(
    params: &str,
    data: &[f64],
    n_rows: usize,
    n_features: usize,
    labels: &[f64],
) -> *mut SbsModel {
    let json = cjson(params);
    let mut model: *mut SbsModel = std::ptr::null_mut();
    let rc = sbs_train(
        json.as_ptr(),
        data.as_ptr(),
        n_rows as i64,
        n_features as i64,
        labels.as_ptr(),
        &mut model,
    );
    assert_eq!(rc, 0, "训练应成功: {params}");
    assert!(!model.is_null());
    model
}

fn last_error() -> String {
    let mut buf = [0 as c_char; 1024];
    sbs_last_error(buf.as_mut_ptr(), buf.len() as i64);
    // SAFETY：sbs_last_error 已写入 NUL 结尾。
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn version_string_is_static_and_nonempty() {
    // SAFETY：返回静态存储期指针（头文件契约）。
    let v = unsafe { CStr::from_ptr(sbs_version()) };
    let s = v.to_string_lossy();
    assert!(s.starts_with("sooboost-ffi "), "版本串格式异常: {s}");
    assert!(s.len() > "sooboost-ffi 0.".len());
}

#[test]
fn regression_train_predict_matches_expected_function() {
    let (x, y) = linear_data();
    let data: Vec<f64> = x.iter().flat_map(|v| [*v]).collect();
    let model = train(
        r#"{"task":"regression","n_estimators":60,"learning_rate":0.2,"seed":7}"#,
        &data,
        100,
        1,
        &y,
    );
    assert_eq!(sbs_model_num_features(model), 1);
    assert_eq!(sbs_model_num_classes(model), -1, "回归模型无类别数");
    assert_eq!(sbs_model_num_trees(model), 60);

    let mut out = vec![0.0f64; 100];
    let n = sbs_predict(model, data.as_ptr(), 100, 1, out.as_mut_ptr(), 100);
    assert_eq!(n, 100);
    // 端点允许偏差，中段应贴近 y = 2x。
    for (i, &p) in out.iter().enumerate().take(80).skip(20) {
        let expected = 2.0 * i as f64;
        assert!((p - expected).abs() < 2.0, "x={i} 预测 {p} 偏离 {expected}");
    }
    sbs_model_free(model);
}

#[test]
fn regression_train_predict_matches_rust_facade_bitwise() {
    // 同 seed 同数据：FFI 与 Rust 门面的预测必须逐位一致（红线 3 经边界层不变）。
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use sooboost_core::api::GradientBoosting;
    use sooboost_core::data::{Dataset, MissingPolicy};
    use std::sync::Arc;

    let (x, y) = linear_data();
    let data: Vec<f64> = x.clone();
    let schema = Schema::new(vec![
        Field::new("f0", DataType::Float64, true),
        Field::new("target", DataType::Float64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(x.clone())),
            Arc::new(Float64Array::from(y.clone())),
        ],
    )
    .expect("batch");
    let ds =
        Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default()).expect("ds");
    let rust_model = GradientBoosting::regressor()
        .n_estimators(30)
        .learning_rate(0.15)
        .seed(42)
        .fit(&ds)
        .expect("Rust 门面训练");
    let rust_preds = rust_model.predict(&ds).expect("Rust 预测");

    let model = train(
        r#"{"task":"regression","n_estimators":30,"learning_rate":0.15,"seed":42}"#,
        &data,
        100,
        1,
        &y,
    );
    let mut out = vec![0.0f64; 100];
    assert_eq!(
        sbs_predict(model, data.as_ptr(), 100, 1, out.as_mut_ptr(), 100),
        100
    );
    assert_eq!(out, rust_preds, "FFI 与 Rust 门面预测必须逐位一致");
    sbs_model_free(model);
}

#[test]
fn binary_outputs_probabilities_in_unit_range() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| if v >= 50.0 { 1.0 } else { 0.0 })
        .collect();
    let model = train(
        r#"{"task":"binary","n_estimators":50,"learning_rate":0.3}"#,
        &x,
        100,
        1,
        &y,
    );
    let mut out = vec![0.0f64; 100];
    assert_eq!(
        sbs_predict(model, x.as_ptr(), 100, 1, out.as_mut_ptr(), 100),
        100
    );
    for &p in &out {
        assert!((0.0..=1.0).contains(&p), "概率越界: {p}");
    }
    assert!(out[90] > 0.9 && out[5] < 0.1, "正负类概率未拉开");
    // predict_proba 与 predict 同口径（二分类均输出正类概率）
    let mut proba = vec![0.0f64; 100];
    assert_eq!(
        sbs_predict_proba(model, x.as_ptr(), 100, 1, proba.as_mut_ptr(), 100),
        100
    );
    assert_eq!(out, proba);
    sbs_model_free(model);
}

#[test]
fn multiclass_predict_and_proba_matrix() {
    let x: Vec<f64> = (0..99).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| (v / 33.0).floor()).collect();
    let model = train(
        r#"{"task":"multiclass","n_classes":3,"n_estimators":30,"learning_rate":0.3}"#,
        &x,
        99,
        1,
        &y,
    );
    assert_eq!(sbs_model_num_classes(model), 3);

    let mut out = vec![0.0f64; 99];
    assert_eq!(
        sbs_predict(model, x.as_ptr(), 99, 1, out.as_mut_ptr(), 99),
        99
    );
    let hits = out
        .iter()
        .zip(y.iter())
        .filter(|&(&c, &t)| c as usize == t as usize)
        .count();
    assert!(hits as f64 / 99.0 > 0.9, "可分数据准确率应 >0.9: {hits}/99");

    // 概率矩阵 n×k 行主序：行和为 1，argmax 与 predict 一致
    let mut proba = vec![0.0f64; 99 * 3];
    assert_eq!(
        sbs_predict_proba(model, x.as_ptr(), 99, 1, proba.as_mut_ptr(), 99 * 3),
        99 * 3
    );
    for r in 0..99 {
        let row = &proba[r * 3..r * 3 + 3];
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "行 {r} 概率和应为 1: {s}");
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("非空");
        assert_eq!(argmax, out[r] as usize, "argmax 应与 predict 一致");
    }
    sbs_model_free(model);
}

#[test]
fn serialize_deserialize_roundtrip_is_bitwise_identical() {
    let (x, y) = linear_data();
    let data: Vec<f64> = x.clone();
    let model = train(
        r#"{"task":"regression","n_estimators":30,"learning_rate":0.1,"max_depth":4,"seed":7}"#,
        &data,
        100,
        1,
        &y,
    );

    // 两段式：先探测
    let needed = sbs_serialize(model, std::ptr::null_mut(), 0);
    assert!(needed > 0, "探测应返回所需长度: {needed}");
    let mut buf = vec![0u8; needed as usize];
    let written = sbs_serialize(model, buf.as_mut_ptr(), needed);
    assert_eq!(written, needed);

    let mut loaded: *mut SbsModel = std::ptr::null_mut();
    assert_eq!(sbs_deserialize(buf.as_ptr(), needed, &mut loaded), 0);
    assert!(!loaded.is_null());
    assert_eq!(sbs_model_num_trees(loaded), 30);

    let mut before = vec![0.0f64; 100];
    let mut after = vec![0.0f64; 100];
    assert_eq!(
        sbs_predict(model, data.as_ptr(), 100, 1, before.as_mut_ptr(), 100),
        100
    );
    assert_eq!(
        sbs_predict(loaded, data.as_ptr(), 100, 1, after.as_mut_ptr(), 100),
        100
    );
    assert_eq!(before, after, "存读后预测必须逐位一致");
    sbs_model_free(loaded);
    sbs_model_free(model);
}

#[test]
fn missing_values_as_nan_are_handled() {
    // NaN = 缺失（默认策略）：含 NaN 的行预测应有限且不崩溃。
    let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let model = train(r#"{"task":"regression","n_estimators":20}"#, &x, 60, 1, &y);
    let mut query = x.clone();
    query[3] = f64::NAN;
    let mut out = vec![0.0f64; 60];
    assert_eq!(
        sbs_predict(model, query.as_ptr(), 60, 1, out.as_mut_ptr(), 60),
        60
    );
    assert!(out.iter().all(|v| v.is_finite()), "含 NaN 行的预测应有限");
    sbs_model_free(model);
}

#[test]
fn invalid_inputs_report_error_via_last_error() {
    let (x, y) = linear_data();
    let data: Vec<f64> = x.clone();
    let mut model: *mut SbsModel = std::ptr::null_mut();

    // 非法 JSON
    let bad = cjson("{not json");
    assert_eq!(
        sbs_train(bad.as_ptr(), data.as_ptr(), 100, 1, y.as_ptr(), &mut model),
        -1
    );
    assert!(!last_error().is_empty(), "失败后 last_error 必须非空");

    // 未知字段（deny_unknown_fields）
    let unknown = cjson(r#"{"task":"regression","unknown_field":1}"#);
    assert_eq!(
        sbs_train(
            unknown.as_ptr(),
            data.as_ptr(),
            100,
            1,
            y.as_ptr(),
            &mut model
        ),
        -1
    );
    assert!(
        last_error().contains("unknown_field"),
        "应点名未知字段: {}",
        last_error()
    );

    // 未知 task
    let wrong_task = cjson(r#"{"task":"ranking"}"#);
    assert_eq!(
        sbs_train(
            wrong_task.as_ptr(),
            data.as_ptr(),
            100,
            1,
            y.as_ptr(),
            &mut model
        ),
        -1
    );

    // multiclass 缺 n_classes
    let no_k = cjson(r#"{"task":"multiclass"}"#);
    assert_eq!(
        sbs_train(no_k.as_ptr(), data.as_ptr(), 100, 1, y.as_ptr(), &mut model),
        -1
    );

    // NULL / 非正维度
    assert_eq!(
        sbs_train(
            std::ptr::null(),
            data.as_ptr(),
            100,
            1,
            y.as_ptr(),
            &mut model
        ),
        -1
    );
    let ok = cjson(r#"{"task":"regression"}"#);
    assert_eq!(
        sbs_train(ok.as_ptr(), data.as_ptr(), 0, 1, y.as_ptr(), &mut model),
        -1
    );
    assert_eq!(
        sbs_train(
            ok.as_ptr(),
            std::ptr::null(),
            100,
            1,
            y.as_ptr(),
            &mut model
        ),
        -1
    );
    assert!(model.is_null(), "失败路径不得产出句柄");

    // 训一个真模型测预测侧错误
    let model = train(
        r#"{"task":"regression","n_estimators":5}"#,
        &data,
        100,
        1,
        &y,
    );
    let mut out = vec![0.0f64; 100];
    // 特征数不匹配
    assert_eq!(
        sbs_predict(model, data.as_ptr(), 100, 2, out.as_mut_ptr(), 100),
        -1
    );
    assert!(
        last_error().contains("特征数"),
        "应报特征数不匹配: {}",
        last_error()
    );
    // out_cap 不足
    assert_eq!(
        sbs_predict(model, data.as_ptr(), 100, 1, out.as_mut_ptr(), 10),
        -1
    );
    // NULL 句柄
    assert_eq!(
        sbs_predict(
            std::ptr::null(),
            data.as_ptr(),
            100,
            1,
            out.as_mut_ptr(),
            100
        ),
        -1
    );
    // 回归无概率
    assert_eq!(
        sbs_predict_proba(model, data.as_ptr(), 100, 1, out.as_mut_ptr(), 100),
        -1
    );
    // 序列化 cap 不足
    let needed = sbs_serialize(model, std::ptr::null_mut(), 0);
    let mut small = vec![0u8; (needed - 1) as usize];
    assert_eq!(sbs_serialize(model, small.as_mut_ptr(), needed - 1), -1);
    // 反序列化非法字节（同长但全 0xFF：magic 不符 → 显式报错）
    let corrupted = vec![0xFFu8; needed as usize];
    let mut m2: *mut SbsModel = std::ptr::null_mut();
    assert_eq!(
        sbs_deserialize(corrupted.as_ptr(), corrupted.len() as i64, &mut m2),
        -1
    );
    sbs_model_free(model);
    // NULL 安全 free
    sbs_model_free(std::ptr::null_mut());
}
