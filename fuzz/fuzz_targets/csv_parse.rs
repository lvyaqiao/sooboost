//! CSV 字节解析 fuzz 目标（M2-0，红线 6）。
//!
//! 任意字节喂给 `Dataset::from_csv_bytes`（arrow CSV 读入路径）：禁止 panic。

#![no_main]

use libfuzzer_sys::fuzz_target;
use sooboost_core::data::{Dataset, MissingPolicy};

fuzz_target!(|data: &[u8]| {
    let _ = Dataset::from_csv_bytes(data, &["f0"], "target", MissingPolicy::default());
});
