//! 模型序列化错误（contracts §1.2：区分「版本不支持 / checksum 失败 / 结构非法」）。

use std::io;

/// 模型加载/保存错误。
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("不是 sooboost 模型文件（magic 不匹配）")]
    InvalidMagic,
    #[error("不支持的模型版本 {version}（当前支持 {supported}）")]
    UnsupportedVersion { version: u32, supported: u32 },
    #[error("模型字节不足或长度越界（截断/结构非法）")]
    Truncated,
    #[error("checksum 校验失败：期望 {expected:#x}，实际 {actual:#x}")]
    ChecksumFailed { expected: u64, actual: u64 },
    #[error("模型结构非法：{0}")]
    InvalidLayout(&'static str),
    #[error("损失函数不匹配：模型为 {found}，当前加载器为 {expected}")]
    LossMismatch { expected: String, found: String },
    #[error("IO 错误: {0}")]
    Io(#[from] io::Error),
}
