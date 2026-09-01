//! sooboost C ABI（M12：嵌入接口）。
//!
//! 薄边界层：把 [`sooboost_core::api::GradientBoosting`] 门面暴露为稳定的
//! `extern "C"` 接口，供 C/C++/Python(ctypes)/Node 等宿主语言嵌入使用。
//!
//! # ABI 约定
//!
//! - 状态码：`0`（或返回值≥0 的长度语义）= 成功；`-1` = 失败，错误信息经
//!   [`sbs_last_error`] 取**线程局部**缓冲（`Display` of core `Error` 原样透传）。
//! - 内存：模型句柄由 `sbs_train` / `sbs_deserialize` 产出，调用方以
//!   [`sbs_model_free`] 释放；数据与输出缓冲一律由调用方分配，sooboost 不持有。
//! - 数据布局：**行主序** `data[row * n_features + feature]`；缺失 = `NaN`
//!   （core `MissingPolicy::TreatNanAsMissing` 默认语义，红线 2 单点定义不变）。
//! - 序列化两段式：先 `sbs_serialize(m, NULL, 0)` 探测长度，再分配重调。
//! - `deny_unknown_fields`：参数 JSON 出现未知字段显式报错（红线 6，不静默忽略）。
//!
//! # unsafe 纪律
//!
//! core 保持 `#![forbid(unsafe_code)]` 不变；本 crate 是全仓唯一 unsafe 允许区，
//! 仅限三类：raw 指针 → slice/CStr 借用（调用方契约见 sooboost.h）、
//! `Box` 句柄往来、panic 屏障（`catch_unwind`，禁止 unwind 穿过 FFI）。
//! 每处 unsafe 附 SAFETY 注释。
//!
//! 指针参数按 XGBoost C API 惯例设计为**容忍 NULL**：所有指针先做空值/长度
//! 校验再借用，非法输入返回 -1 + last_error 而非 UB；残余契约（调用方保证
//! 缓冲区真实可读/可写范围内存）记录在 sooboost.h，静态分析不可见，故对
//! `not_unsafe_ptr_arg_deref` 全 crate 豁免。

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::Deserialize;
use sooboost_core::api::{Config, GradientBoosting, Objective};
use sooboost_core::data::{Dataset, MissingPolicy};
use std::sync::Arc;

// 线程局部 last-error 缓冲（ABI 约定：错误不跨线程可取，避免锁与竞争）。
thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: impl std::fmt::Display) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

/// panic 屏障：Rust 实现体 panic 一律转为 `-1` + last_error，禁止 unwind 穿过 FFI。
fn ffi_barrier<T>(body: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "internal panic".to_string());
        Err(format!("FFI 内部 panic（已拦截，未跨越边界）: {msg}"))
    })
}

/// 静态版本串（NUL 结尾，'static 生命周期，调用方不得释放）。
static VERSION_C: &[u8] = concat!("sooboost-ffi ", env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// 返回版本字符串（如 `"sooboost-ffi 0.2.0"`）；指针恒定有效，无需释放。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_version() -> *const c_char {
    // SAFETY：VERSION_C 是含 NUL 的 'static 字节串，仅做不可变借用。
    VERSION_C.as_ptr().cast::<c_char>()
}

/// FFI 训练参数（JSON）。所有字段可选；未提供者取 core 门面默认值。
///
/// `task`: `"regression"`（默认）/ `"binary"` / `"multiclass"`（须给 `n_classes` ≥ 2）。
/// 未知字段显式报错（`deny_unknown_fields`，红线 6）。
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FfiParams {
    task: Option<String>,
    n_classes: Option<usize>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    min_split_gain: Option<f64>,
    reg_lambda: Option<f64>,
    max_bins: Option<usize>,
    max_categories: Option<usize>,
    categorical_alpha: Option<f64>,
    seed: Option<u64>,
}

/// 不透明模型句柄（由 `sbs_train` / `sbs_deserialize` 产出，`sbs_model_free` 释放）。
pub struct SbsModel {
    inner: GradientBoosting,
}

/// 由行主序 f64 矩阵 + 标签构造 Dataset（NaN = 缺失，默认策略）。
///
/// dummy target：预测路径同样需要 target 列占位（全 0，不参与预测）。
fn build_dataset(
    data: &[f64],
    labels: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Result<Dataset, String> {
    if n_rows == 0 {
        return Err("n_rows 必须为正".to_string());
    }
    if n_features == 0 {
        return Err("n_features 必须为正".to_string());
    }
    if labels.len() != n_rows {
        return Err(format!(
            "labels 长度不符：期望 {n_rows}，实际 {}",
            labels.len()
        ));
    }
    if data.len() != n_rows.saturating_mul(n_features) {
        return Err(format!(
            "data 长度不符：期望 n_rows*n_features = {}，实际 {}",
            n_rows.saturating_mul(n_features),
            data.len()
        ));
    }
    let mut names: Vec<String> = (0..n_features).map(|f| format!("f{f}")).collect();
    let mut cols: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(n_features + 1);
    for f in 0..n_features {
        let col: Vec<f64> = (0..n_rows).map(|r| data[r * n_features + f]).collect();
        cols.push(Arc::new(Float64Array::from(col)));
    }
    names.push("target".to_string());
    cols.push(Arc::new(Float64Array::from(labels.to_vec())));
    let schema = Schema::new(
        names
            .iter()
            .map(|n| Field::new(n.as_str(), DataType::Float64, true))
            .collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(Arc::new(schema), cols)
        .map_err(|e| format!("构造 RecordBatch 失败: {e}"))?;
    let feature_refs: Vec<&str> = names[..n_features].iter().map(String::as_str).collect();
    Dataset::from_record_batch(batch, &feature_refs, "target", MissingPolicy::default())
        .map_err(|e| format!("构造 Dataset 失败: {e}"))
}

fn params_to_config(p: &FfiParams) -> Config {
    let mut cfg = Config::default();
    if let Some(v) = p.n_estimators {
        cfg.n_estimators = v;
    }
    if let Some(v) = p.learning_rate {
        cfg.learning_rate = v;
    }
    if let Some(v) = p.max_depth {
        cfg.max_depth = v;
    }
    if let Some(v) = p.min_samples_leaf {
        cfg.min_samples_leaf = v;
    }
    if let Some(v) = p.min_split_gain {
        cfg.min_split_gain = v;
    }
    if let Some(v) = p.reg_lambda {
        cfg.reg_lambda = v;
    }
    if let Some(v) = p.max_bins {
        cfg.max_bins = v;
    }
    if let Some(v) = p.max_categories {
        cfg.max_categories = v;
    }
    if let Some(v) = p.categorical_alpha {
        cfg.categorical_alpha = v;
    }
    if let Some(v) = p.seed {
        cfg.seed = v;
    }
    cfg
}

fn train_impl(
    params_json: &CStr,
    data: &[f64],
    n_rows: usize,
    n_features: usize,
    labels: &[f64],
) -> Result<Box<SbsModel>, String> {
    let raw = params_json
        .to_str()
        .map_err(|_| "params_json 不是合法 UTF-8".to_string())?;
    let p: FfiParams =
        serde_json::from_str(raw).map_err(|e| format!("params_json 解析失败: {e}"))?;
    let task = p.task.as_deref().unwrap_or("regression");
    let cfg = params_to_config(&p);

    let ds = build_dataset(data, labels, n_rows, n_features)?;
    let builder = match task {
        "regression" => GradientBoosting::regressor(),
        "binary" => GradientBoosting::classifier(),
        "multiclass" => {
            let k = p
                .n_classes
                .ok_or_else(|| "task=multiclass 必须提供 n_classes（≥2）".to_string())?;
            GradientBoosting::multiclass_classifier(k)
        }
        other => {
            return Err(format!(
                "未知 task: {other}（可选 regression/binary/multiclass）"
            ));
        }
    };
    let fitted = builder
        .n_estimators(cfg.n_estimators)
        .learning_rate(cfg.learning_rate)
        .max_depth(cfg.max_depth)
        .min_samples_leaf(cfg.min_samples_leaf)
        .min_split_gain(cfg.min_split_gain)
        .reg_lambda(cfg.reg_lambda)
        .max_bins(cfg.max_bins)
        .max_categories(cfg.max_categories)
        .categorical_alpha(cfg.categorical_alpha)
        .seed(cfg.seed)
        .fit(&ds)
        .map_err(|e| e.to_string())?;
    Ok(Box::new(SbsModel { inner: fitted }))
}

/// 训练模型。
///
/// `params_json` 为 NUL 结尾的 UTF-8 JSON（见 [`FfiParams`]）；`data` 为行主序
/// `n_rows × n_features`；`labels` 长 `n_rows`；成功时写入 `*out_model` 并返回 0。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_train(
    params_json: *const c_char,
    data: *const f64,
    n_rows: i64,
    n_features: i64,
    labels: *const f64,
    out_model: *mut *mut SbsModel,
) -> i32 {
    let result = ffi_barrier(|| {
        if params_json.is_null() {
            return Err("params_json 不能为 NULL".to_string());
        }
        if data.is_null() || labels.is_null() {
            return Err("data/labels 不能为 NULL".to_string());
        }
        if out_model.is_null() {
            return Err("out_model 不能为 NULL".to_string());
        }
        if n_rows <= 0 || n_features <= 0 {
            return Err(format!(
                "n_rows/n_features 必须为正，实际 {n_rows}/{n_features}"
            ));
        }
        // SAFETY：调用方契约（sooboost.h）保证三个指针各自指向至少 n_rows/
        // n_rows*n_features 个 f64、一个合法 C 字符串；长度已在上一步校验为正。
        let json = unsafe { CStr::from_ptr(params_json) };
        // SAFETY：同上，切片长度来自调用方声明的 n_rows / n_features。
        let data =
            unsafe { slice::from_raw_parts(data, (n_rows as usize) * (n_features as usize)) };
        // SAFETY：同上。
        let labels = unsafe { slice::from_raw_parts(labels, n_rows as usize) };
        train_impl(json, data, n_rows as usize, n_features as usize, labels)
    });
    match result {
        Ok(model) => {
            // SAFETY：out_model 非空已在屏障内校验；写入调用方提供的槽位。
            unsafe { *out_model = Box::into_raw(model) };
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

fn predict_impl(
    model: &GradientBoosting,
    data: &[f64],
    n_rows: usize,
    n_features: usize,
    out: &mut [f64],
) -> Result<usize, String> {
    if n_features != model.num_features() {
        return Err(format!(
            "特征数不匹配：模型期望 {}，实际 {n_features}",
            model.num_features()
        ));
    }
    let dummy = vec![0.0f64; n_rows];
    let ds = build_dataset(data, &dummy, n_rows, n_features)?;
    let preds = model.predict(&ds).map_err(|e| e.to_string())?;
    if out.len() < preds.len() {
        return Err(format!(
            "out 容量不足：需要 {}，实际 {}",
            preds.len(),
            out.len()
        ));
    }
    out[..preds.len()].copy_from_slice(&preds);
    Ok(preds.len())
}

/// 批量预测（回归 → 原值；二分类 → 正类概率；多分类 → argmax 类别）。
///
/// 成功返回写入 `out` 的元素数（= `n_rows`）；`out_cap` 不足或参数非法返回 -1。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_predict(
    model: *const SbsModel,
    data: *const f64,
    n_rows: i64,
    n_features: i64,
    out: *mut f64,
    out_cap: i64,
) -> i64 {
    let result = ffi_barrier(|| {
        let m = model_ref(model)?;
        if data.is_null() {
            return Err("data 不能为 NULL".to_string());
        }
        if out.is_null() {
            return Err("out 不能为 NULL".to_string());
        }
        if n_rows <= 0 || n_features <= 0 {
            return Err(format!(
                "n_rows/n_features 必须为正，实际 {n_rows}/{n_features}"
            ));
        }
        if out_cap < n_rows {
            return Err(format!("out_cap 不足：至少需要 {n_rows}，实际 {out_cap}"));
        }
        // SAFETY：调用方契约保证 data 至少 n_rows*n_features 个 f64、out 至少 out_cap 个。
        let data =
            unsafe { slice::from_raw_parts(data, (n_rows as usize) * (n_features as usize)) };
        // SAFETY：同上。
        let out = unsafe { slice::from_raw_parts_mut(out, out_cap as usize) };
        predict_impl(&m.inner, data, n_rows as usize, n_features as usize, out)
    });
    match result {
        Ok(n) => n as i64,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

fn proba_impl(
    model: &GradientBoosting,
    data: &[f64],
    n_rows: usize,
    n_features: usize,
    out: &mut [f64],
) -> Result<usize, String> {
    if n_features != model.num_features() {
        return Err(format!(
            "特征数不匹配：模型期望 {}，实际 {n_features}",
            model.num_features()
        ));
    }
    let dummy = vec![0.0f64; n_rows];
    let ds = build_dataset(data, &dummy, n_rows, n_features)?;
    match model.objective() {
        Objective::SquaredError => Err("回归目标无概率输出：数值预测请用 sbs_predict".to_string()),
        Objective::BinaryLogLoss => {
            let proba = model.predict(&ds).map_err(|e| e.to_string())?;
            if out.len() < proba.len() {
                return Err(format!(
                    "out 容量不足：需要 {}，实际 {}",
                    proba.len(),
                    out.len()
                ));
            }
            out[..proba.len()].copy_from_slice(&proba);
            Ok(proba.len())
        }
        Objective::MulticlassSoftmax => {
            let proba = model.predict_proba(&ds).map_err(|e| e.to_string())?;
            let flat: Vec<f64> = proba.into_iter().flatten().collect();
            if out.len() < flat.len() {
                return Err(format!(
                    "out 容量不足：需要 n_rows*n_classes = {}，实际 {}",
                    flat.len(),
                    out.len()
                ));
            }
            out[..flat.len()].copy_from_slice(&flat);
            Ok(flat.len())
        }
    }
}

/// 批量概率预测（二分类 → 正类概率，`n` 个；多分类 → 行主序 `n×k` 概率矩阵；
/// 回归显式报错）。成功返回写入元素数，失败返回 -1。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_predict_proba(
    model: *const SbsModel,
    data: *const f64,
    n_rows: i64,
    n_features: i64,
    out: *mut f64,
    out_cap: i64,
) -> i64 {
    let result = ffi_barrier(|| {
        let m = model_ref(model)?;
        if data.is_null() || out.is_null() {
            return Err("data/out 不能为 NULL".to_string());
        }
        if n_rows <= 0 || n_features <= 0 {
            return Err(format!(
                "n_rows/n_features 必须为正，实际 {n_rows}/{n_features}"
            ));
        }
        // SAFETY：调用方契约保证 data 至少 n_rows*n_features 个 f64、out 至少 out_cap 个。
        let data =
            unsafe { slice::from_raw_parts(data, (n_rows as usize) * (n_features as usize)) };
        // SAFETY：同上。
        let out = unsafe { slice::from_raw_parts_mut(out, out_cap as usize) };
        proba_impl(&m.inner, data, n_rows as usize, n_features as usize, out)
    });
    match result {
        Ok(n) => n as i64,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// SAFETY 共用前置：句柄非空（返回借用引用）。
fn model_ref(model: *const SbsModel) -> Result<&'static SbsModel, String> {
    if model.is_null() {
        return Err("model 句柄不能为 NULL".to_string());
    }
    // SAFETY：调用方契约保证指针来自 sbs_train/sbs_deserialize 且未被 free；
    // 'static 是 FFI 借用惯例的标注（实际生命周期由 sbs_model_free 管理）。
    Ok(unsafe { &*model })
}

/// 模型特征数（无效句柄返回 -1）。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_model_num_features(model: *const SbsModel) -> i64 {
    match ffi_barrier(|| model_ref(model).map(|m| m.inner.num_features() as i64)) {
        Ok(v) => v,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// 模型类别数（多分类为 k；回归/二分类为 -1）。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_model_num_classes(model: *const SbsModel) -> i64 {
    match ffi_barrier(|| model_ref(model).map(|m| m.inner.num_classes().map_or(-1, |k| k as i64))) {
        Ok(v) => v,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// 模型树棵数（多分类为每类棵数；无效句柄返回 -1）。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_model_num_trees(model: *const SbsModel) -> i64 {
    match ffi_barrier(|| model_ref(model).map(|m| m.inner.num_trees() as i64)) {
        Ok(v) => v,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// 序列化为字节（两段式）。
///
/// `out == NULL && cap == 0`：探测模式，返回所需字节数（≥0）；
/// `out` 非空且 `cap` 足够：写入并返回实际字节数；
/// `cap` 不足或句柄无效：返回 -1（错误信息含所需长度）。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_serialize(model: *const SbsModel, out: *mut u8, cap: i64) -> i64 {
    let result = ffi_barrier(|| {
        let m = model_ref(model)?;
        let bytes = m.inner.to_bytes();
        if out.is_null() {
            if cap != 0 {
                return Err("out 为 NULL 时 cap 必须为 0（探测模式）".to_string());
            }
            return Ok(bytes.len());
        }
        if cap < 0 {
            return Err(format!("cap 不能为负，实际 {cap}"));
        }
        if (cap as usize) < bytes.len() {
            return Err(format!("缓冲区不足：需要 {} 字节，实际 {cap}", bytes.len()));
        }
        // SAFETY：out 非空且调用方契约保证至少 cap 字节可写，已验证 cap ≥ len。
        let dst = unsafe { slice::from_raw_parts_mut(out, cap as usize) };
        dst[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len())
    });
    match result {
        Ok(needed_or_written) => needed_or_written as i64,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

fn deserialize_impl(bytes: &[u8]) -> Result<Box<SbsModel>, String> {
    let fitted = GradientBoosting::from_bytes(bytes).map_err(|e| e.to_string())?;
    Ok(Box::new(SbsModel { inner: fitted }))
}

/// 由字节恢复模型（目标自动探测：回归 → 二分类 → 多分类）。
/// 成功写入 `*out_model` 并返回 0；字节非法返回 -1。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_deserialize(
    bytes: *const u8,
    len: i64,
    out_model: *mut *mut SbsModel,
) -> i32 {
    let result = ffi_barrier(|| {
        if bytes.is_null() {
            return Err("bytes 不能为 NULL".to_string());
        }
        if len <= 0 {
            return Err(format!("len 必须为正，实际 {len}"));
        }
        if out_model.is_null() {
            return Err("out_model 不能为 NULL".to_string());
        }
        // SAFETY：调用方契约保证 bytes 至少 len 字节可读。
        let bytes = unsafe { slice::from_raw_parts(bytes, len as usize) };
        deserialize_impl(bytes)
    });
    match result {
        Ok(model) => {
            // SAFETY：out_model 非空已在屏障内校验。
            unsafe { *out_model = Box::into_raw(model) };
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// 释放模型句柄（NULL 安全；释放后句柄失效，不得复用）。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_model_free(model: *mut SbsModel) {
    if !model.is_null() {
        // SAFETY：指针由 Box::into_raw 产出，且调用方保证未重复释放（头文件契约）。
        drop(unsafe { Box::from_raw(model) });
    }
}

/// 取线程局部 last-error（UTF-8，NUL 结尾，截断到 `cap`）。
/// `out` 为 NULL 时返回 0；无错误时写入空串。恒返回 0。
#[unsafe(no_mangle)]
pub extern "C" fn sbs_last_error(out: *mut c_char, cap: i64) -> i32 {
    if out.is_null() || cap <= 0 {
        return 0;
    }
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        // SAFETY：out 非空且 cap ≥ 1，由调用方契约保证可写 cap 字节。
        let dst = unsafe { slice::from_raw_parts_mut(out.cast::<u8>(), cap as usize) };
        let bytes = msg.as_bytes();
        let n = bytes.len().min(cap as usize - 1);
        dst[..n].copy_from_slice(&bytes[..n]);
        dst[n] = 0;
    });
    0
}

/// 便于 FFI 测试与调试的 Rust 侧视图（不属于 C ABI）。
impl SbsModel {
    /// 内部门面模型引用。
    #[must_use]
    pub fn inner(&self) -> &GradientBoosting {
        &self.inner
    }
}
