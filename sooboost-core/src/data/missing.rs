//! 缺失值语义——全库唯一定义点（红线 2）。
//!
//! 规则（doc/baseline/contracts.md §1.1）：
//! - arrow null 位图 = 缺失，任何情况下成立；
//! - NaN 是否视为缺失由 `MissingPolicy` 决定，一旦确定全库一致；
//! - 任何模块不得自行解释缺失/NaN，一律经本模块查询。

/// 缺失值策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    /// arrow null = 缺失，NaN 也视为缺失（默认，与主流 GBDT 一致）。
    #[default]
    TreatNanAsMissing,
    /// arrow null = 缺失，NaN 保留为数值（仅当训练配置显式选择）。
    KeepNan,
}

/// 值级缺失判断（红线 2 唯一来源；数组级调用方须先取 `is_null` 传入）。
pub fn is_missing_value(value: f64, is_null: bool, policy: MissingPolicy) -> bool {
    match policy {
        MissingPolicy::TreatNanAsMissing => is_null || value.is_nan(),
        MissingPolicy::KeepNan => is_null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_treats_nan_as_missing() {
        assert_eq!(MissingPolicy::default(), MissingPolicy::TreatNanAsMissing);
    }
}
