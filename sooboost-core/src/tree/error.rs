//! 树构建错误：显式传播（易踩坑 10）。

/// 树构建错误。
#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    /// 输入长度不一致。
    #[error("矩阵行数({rows})与梯度长度({grad})/海森长度({hess})不一致")]
    LengthMismatch {
        rows: usize,
        grad: usize,
        hess: usize,
    },

    /// 数据为空。
    #[error("数据为空，无法建树")]
    Empty,
}
