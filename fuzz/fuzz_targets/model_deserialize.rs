//! 模型字节反序列化 fuzz 目标（M2-0，红线 6）。
//!
//! 任意字节喂给 `Booster::deserialize`：禁止 panic / UB / OOM。
//! 校验顺序（contracts §1.2）保证 checksum 在分配前先行，多数输入早退。

#![no_main]

use libfuzzer_sys::fuzz_target;
use sooboost_core::boosting::Booster;
use sooboost_core::loss::SquaredError;

fuzz_target!(|data: &[u8]| {
    let _ = Booster::<SquaredError>::deserialize(data, SquaredError);
});
