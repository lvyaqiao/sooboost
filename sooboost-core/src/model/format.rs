//! 模型字节格式（contracts §1.2 显式布局，不依赖 serde/bincode 默认布局）。
//!
//! 全部字段小端、固定宽度；无隐式 padding（读侧按字节流顺序解析）。
//! checksum 覆盖 magic 起到 checksum 前全部字节。
//!
//! ```text
//! magic            [u8;4] = b"SOOB"
//! version          u32 = 4（v1：标量模型；v2：+ 类别编码段；v3：+ 树节点 gain/cover；
//!                            v4：+ num_classes 头，多分类 softmax 模型）
//! loss_name_len    u16；loss_name  UTF-8
//! num_classes      u32（v4；标量模型恒为 1，多分类 ≥ 2）
//! init_score       f64（标量）｜init_scores [f64;num_classes]（多分类）
//! learning_rate    f64
//! num_trees        u32（多分类 = K 类 × 每类轮数，类主序平铺）
//! trees×{ num_nodes u32
//!         split_features [u32;num_nodes]
//!         thresholds     [f64;num_nodes]
//!         missing_go_left [u8;num_nodes]
//!         left           [i32;num_nodes]
//!         right          [i32;num_nodes]
//!         leaf_values    [f64;num_nodes]
//!         depths         [u32;num_nodes]
//!         split_gains    [f64;num_nodes]（v3；叶子为 0）
//!         node_counts    [f64;num_nodes]（v3；节点覆盖样本数） }
//! max_bins         u32
//! num_features     u32
//! features×{ num_boundaries u32；boundaries [f64;num_boundaries] }
//! has_categorical  u8（v2；0/1；标量与多分类均可携带类别编码段，M8 起多分类不再恒 0）
//! [若 1] num_cat u32；cat_features [u32;num_cat]
//!        每类别特征×{ num_entries u32；entries [u32 key, f64 value]（按 key 升序）
//!                     prior f64；alpha f64 }
//! metadata_len     u32；metadata UTF-8
//! checksum         u64（FNV-1a 64）
//! ```

use std::io::Cursor;

use super::error::ModelError;

/// 模型 magic。
pub const MAGIC: &[u8; 4] = b"SOOB";
/// 当前模型格式版本（v4 起支持多分类 softmax 模型：num_classes 头 + 类主序平铺树）。
pub const VERSION: u32 = 4;
/// 多分类 softmax 模型的损失名（contracts §1.2；标量名由 `Loss::name()` 提供）。
pub const MULTICLASS_LOSS_NAME: &str = "multiclass_softmax";
/// checksum 算法标识（FNV-1a 64）。
pub const CHECKSUM_FNV1A64: u8 = 1;

// -- 写入原语（小端） ------------------------------------------------------

pub fn push_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

pub fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

// -- 读取原语 ---------------------------------------------------------------

fn read_exact<'a>(c: &mut Cursor<&'a [u8]>, n: usize) -> Result<&'a [u8], ModelError> {
    let pos = c.position() as usize;
    let end = pos + n;
    if end > c.get_ref().len() {
        return Err(ModelError::Truncated);
    }
    c.set_position(end as u64);
    Ok(&c.get_ref()[pos..end])
}

pub fn read_u8(c: &mut Cursor<&[u8]>) -> Result<u8, ModelError> {
    Ok(read_exact(c, 1)?[0])
}

pub fn read_u16(c: &mut Cursor<&[u8]>) -> Result<u16, ModelError> {
    Ok(u16::from_le_bytes(
        read_exact(c, 2)?.try_into().expect("长度 2"),
    ))
}

pub fn read_u32(c: &mut Cursor<&[u8]>) -> Result<u32, ModelError> {
    Ok(u32::from_le_bytes(
        read_exact(c, 4)?.try_into().expect("长度 4"),
    ))
}

pub fn read_u64(c: &mut Cursor<&[u8]>) -> Result<u64, ModelError> {
    Ok(u64::from_le_bytes(
        read_exact(c, 8)?.try_into().expect("长度 8"),
    ))
}

pub fn read_f64(c: &mut Cursor<&[u8]>) -> Result<f64, ModelError> {
    Ok(f64::from_le_bytes(
        read_exact(c, 8)?.try_into().expect("长度 8"),
    ))
}

pub fn read_i32(c: &mut Cursor<&[u8]>) -> Result<i32, ModelError> {
    Ok(i32::from_le_bytes(
        read_exact(c, 4)?.try_into().expect("长度 4"),
    ))
}

pub fn read_bytes<'a>(c: &mut Cursor<&'a [u8]>, n: usize) -> Result<&'a [u8], ModelError> {
    read_exact(c, n)
}

/// 读取长度前缀的 UTF-8 字符串（u16 长度）。
pub fn read_len_str16(c: &mut Cursor<&[u8]>) -> Result<String, ModelError> {
    let len = read_u16(c)? as usize;
    let bytes = read_exact(c, len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| ModelError::InvalidLayout("loss 名非 UTF-8"))
}

/// 读取 u32 长度前缀的 UTF-8 字符串。
pub fn read_len_str32(c: &mut Cursor<&[u8]>) -> Result<String, ModelError> {
    let len = read_u32(c)? as usize;
    let bytes = read_exact(c, len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| ModelError::InvalidLayout("元数据非 UTF-8"))
}

/// FNV-1a 64 位 checksum（简单无表实现，覆盖主体字节）。
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
