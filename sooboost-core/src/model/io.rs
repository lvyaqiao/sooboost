//! Booster 序列化/反序列化（显式字节布局，contracts §1.2）。

use std::collections::HashMap;
use std::io::Cursor;

use crate::binning::BinTable;
use crate::boosting::Booster;
use crate::boosting::multiclass::MulticlassBooster;
use crate::data::target_stats::CategoricalEncoding;
use crate::loss::Loss;
use crate::tree::Tree;

use super::error::ModelError;
use super::format::{
    MAGIC, MULTICLASS_LOSS_NAME, VERSION, fnv1a64, push_f64, push_i32, push_u8, push_u16, push_u32,
    push_u64, read_bytes, read_f64, read_i32, read_len_str16, read_len_str32, read_u8, read_u32,
};

/// 序列化 `Booster` 为字节（小端显式布局 + 尾部 FNV-1a checksum；v4 标量路）。
pub fn serialize<L: Loss>(booster: &Booster<L>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, VERSION);
    let loss_name = booster.loss().name();
    push_u16(&mut out, loss_name.len() as u16);
    out.extend_from_slice(loss_name.as_bytes());
    push_u32(&mut out, 1); // v4：标量模型 num_classes 恒为 1
    push_f64(&mut out, booster.init_score());
    push_f64(&mut out, booster.learning_rate());
    write_trees_and_tail(&mut out, booster.trees(), booster.bin_table(), |out| {
        // 类别编码段（v2）
        match booster.categorical_encoding() {
            None => push_u8(out, 0),
            Some(enc) => {
                push_u8(out, 1);
                let cat = booster.cat_features();
                push_u32(out, cat.len() as u32);
                for &f in cat {
                    push_u32(out, f as u32);
                }
                for fi in 0..enc.num_features() {
                    let entries = enc.entries(fi); // 按 key 升序，确定性
                    push_u32(out, entries.len() as u32);
                    for (k, v) in &entries {
                        push_u32(out, *k);
                        push_f64(out, *v);
                    }
                    push_f64(out, enc.prior(fi));
                    push_f64(out, enc.alpha(fi));
                }
            }
        }
    });
    out
}

/// 序列化多分类模型（v4 多分类路；M6-5a）。
///
/// 树按类主序平铺（`trees[0][0..n], trees[1][0..n], …`），每类 init score
/// 依次写入；多分类暂不支持类别特征，`has_categorical` 恒为 0。
pub fn serialize_multiclass(booster: &MulticlassBooster) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, VERSION);
    push_u16(&mut out, MULTICLASS_LOSS_NAME.len() as u16);
    out.extend_from_slice(MULTICLASS_LOSS_NAME.as_bytes());
    push_u32(&mut out, booster.n_classes() as u32);
    for &s in booster.init_scores() {
        push_f64(&mut out, s);
    }
    push_f64(&mut out, booster.learning_rate());
    let trees: Vec<Tree> = booster.trees_flat().cloned().collect();
    let table = booster.bin_table();
    write_trees_and_tail(&mut out, &trees, table, |out| {
        push_u8(out, 0); // 多分类暂不支持类别特征
    });
    out
}

/// 写树序列 + 分箱表 + 尾段 + checksum（标量/多分类共用的尾部布局）。
fn write_trees_and_tail(
    out: &mut Vec<u8>,
    trees: &[Tree],
    table: &BinTable,
    tail: impl FnOnce(&mut Vec<u8>),
) {
    push_u32(out, trees.len() as u32);
    for t in trees {
        let n = t.num_nodes();
        push_u32(out, n as u32);
        for &f in t.split_features() {
            push_u32(out, f as u32);
        }
        for &x in t.thresholds() {
            push_f64(out, x);
        }
        for &b in t.missing_go_left() {
            push_u8(out, u8::from(b));
        }
        for &x in t.left() {
            push_i32(out, x);
        }
        for &x in t.right() {
            push_i32(out, x);
        }
        for &x in t.leaf_values() {
            push_f64(out, x);
        }
        for &d in t.depths() {
            push_u32(out, d as u32);
        }
        // v3：树节点 gain/cover（特征重要度数据源，M6-3）
        for &x in t.split_gains() {
            push_f64(out, x);
        }
        for &x in t.node_counts() {
            push_f64(out, x);
        }
    }

    // bin 表
    push_u32(out, table.max_bins() as u32);
    push_u32(out, table.num_features() as u32);
    for f in 0..table.num_features() {
        let b = table.boundaries(f);
        push_u32(out, b.len() as u32);
        for &x in b {
            push_f64(out, x);
        }
    }

    tail(out);

    // 元数据（M1 先留空）
    push_u32(out, 0);

    // checksum 覆盖 magic 起到此
    let checksum = fnv1a64(out);
    push_u64(out, checksum);
}

/// 反序列化字节为 `Booster<L>`；`loss` 用于校验模型头中的损失名。
///
/// 校验顺序（contracts §1.2）：magic → 版本 → checksum → 损失名/类别数 → 结构。
/// 仅当字节本身合法、仅损失名不同时报 `LossMismatch`；截断/checksum 失败等
/// 一律原样上抛（红线 6）。
pub fn deserialize<L: Loss>(bytes: &[u8], loss: L) -> Result<Booster<L>, ModelError> {
    let p = parse(bytes)?;
    if p.loss_name != loss.name() {
        return Err(ModelError::LossMismatch {
            expected: loss.name().to_string(),
            found: p.loss_name,
        });
    }
    if p.num_classes != 1 {
        return Err(ModelError::InvalidLayout(
            "标量反序列化要求 num_classes = 1",
        ));
    }
    if !p.init_scores.iter().all(|s| s.is_finite()) || !p.learning_rate.is_finite() {
        return Err(ModelError::InvalidLayout("init/lr 非有限"));
    }
    validate_tree_features(&p.trees, p.table.num_features())?;
    Ok(Booster::from_parts(
        loss,
        p.trees,
        p.table,
        p.init_scores[0],
        p.learning_rate,
        p.encoding,
        p.cat_features,
    ))
}

/// 反序列化字节为多分类模型（M6-5a）。
///
/// 损失名必须为 `multiclass_softmax`，否则报 `LossMismatch`（门面探测据此
/// 依次尝试标量 → 多分类）；树按类主序平铺还原，树数必须能被类别数整除。
pub fn deserialize_multiclass(bytes: &[u8]) -> Result<MulticlassBooster, ModelError> {
    let p = parse(bytes)?;
    if p.loss_name != MULTICLASS_LOSS_NAME {
        return Err(ModelError::LossMismatch {
            expected: MULTICLASS_LOSS_NAME.to_string(),
            found: p.loss_name,
        });
    }
    if p.num_classes < 2 {
        return Err(ModelError::InvalidLayout("多分类类别数必须 ≥ 2"));
    }
    let per_class = p.trees.len() / p.num_classes;
    if p.trees.len() % p.num_classes != 0 || per_class == 0 {
        return Err(ModelError::InvalidLayout(
            "树数必须为类别数正整数倍（类主序平铺）",
        ));
    }
    if !p.init_scores.iter().all(|s| s.is_finite()) || !p.learning_rate.is_finite() {
        return Err(ModelError::InvalidLayout("init/lr 非有限"));
    }
    if p.has_categorical {
        return Err(ModelError::InvalidLayout(
            "多分类模型暂不支持类别特征段（has_categorical 必须为 0）",
        ));
    }
    validate_tree_features(&p.trees, p.table.num_features())?;
    let trees: Vec<Vec<Tree>> = p.trees.chunks(per_class).map(<[Tree]>::to_vec).collect();
    debug_assert_eq!(trees.len(), p.num_classes);
    Ok(MulticlassBooster::from_parts(
        p.num_classes,
        trees,
        p.table,
        p.init_scores,
        p.learning_rate,
    ))
}

/// 解析结果（头部 + 树 + 分箱表；类别编码尾段暂存原始游标位置）。
struct Parsed {
    loss_name: String,
    num_classes: usize,
    init_scores: Vec<f64>,
    learning_rate: f64,
    trees: Vec<Tree>,
    table: BinTable,
    /// 类别编码段（仅标量模型可能携带；多分类恒 None）。
    encoding: Option<CategoricalEncoding>,
    cat_features: Vec<usize>,
    has_categorical: bool,
}

/// 共享解析：magic → 版本 → checksum → 头 → 树 → 分箱表 → 类别段 → 元数据。
fn parse(bytes: &[u8]) -> Result<Parsed, ModelError> {
    if bytes.len() < 8 || &bytes[0..4] != MAGIC {
        return Err(ModelError::InvalidMagic);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("长度 4"));
    if version != VERSION {
        return Err(ModelError::UnsupportedVersion {
            version,
            supported: VERSION,
        });
    }
    if bytes.len() < 16 {
        return Err(ModelError::Truncated);
    }
    let checksum = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().expect("长度 8"));
    let actual = fnv1a64(&bytes[..bytes.len() - 8]);
    if checksum != actual {
        return Err(ModelError::ChecksumFailed {
            expected: checksum,
            actual,
        });
    }

    let mut c = Cursor::new(bytes);
    read_bytes(&mut c, 4)?;
    read_u32(&mut c)?; // version 已校验

    let loss_name = read_len_str16(&mut c)?;
    let num_classes = read_u32(&mut c)? as usize;
    if num_classes == 0 {
        return Err(ModelError::InvalidLayout("num_classes 为 0"));
    }
    let mut init_scores = Vec::with_capacity(num_classes);
    for _ in 0..num_classes {
        init_scores.push(read_f64(&mut c)?);
    }
    let learning_rate = read_f64(&mut c)?;

    let num_trees = read_u32(&mut c)? as usize;
    let mut trees = Vec::with_capacity(num_trees);
    for _ in 0..num_trees {
        trees.push(read_tree(&mut c)?);
    }

    let table = read_bin_table(&mut c)?;

    // 类别编码段（v2；标量模型可能携带，多分类序列化恒为 0）
    let mut encoding = None;
    let mut cat_features = Vec::new();
    let has_categorical = match read_u8(&mut c)? {
        0 => false,
        1 => {
            let num_cat = read_u32(&mut c)? as usize;
            for _ in 0..num_cat {
                let f = read_u32(&mut c)? as usize;
                if f >= table.num_features() {
                    return Err(ModelError::InvalidLayout("类别特征索引越界"));
                }
                cat_features.push(f);
            }
            let mut maps = Vec::with_capacity(num_cat);
            let mut priors = Vec::with_capacity(num_cat);
            let mut alphas = Vec::with_capacity(num_cat);
            for _ in 0..num_cat {
                let num_entries = read_u32(&mut c)? as usize;
                let mut map = HashMap::with_capacity(num_entries);
                let mut prev: Option<u32> = None;
                for _ in 0..num_entries {
                    let k = read_u32(&mut c)?;
                    let v = read_f64(&mut c)?;
                    if prev.is_some_and(|p| p >= k) {
                        return Err(ModelError::InvalidLayout("类别键非严格升序"));
                    }
                    prev = Some(k);
                    map.insert(k, v);
                }
                let prior = read_f64(&mut c)?;
                let alpha = read_f64(&mut c)?;
                maps.push(map);
                priors.push(prior);
                alphas.push(alpha);
            }
            encoding = Some(CategoricalEncoding::from_parts(maps, priors, alphas));
            true
        }
        _ => return Err(ModelError::InvalidLayout("has_categorical 非 0/1")),
    };

    let _metadata = read_len_str32(&mut c)?;

    Ok(Parsed {
        loss_name,
        num_classes,
        init_scores,
        learning_rate,
        trees,
        table,
        encoding,
        cat_features,
        has_categorical,
    })
}

/// 结构校验：树节点分裂索引必须在特征数内。
fn validate_tree_features(trees: &[Tree], num_features: usize) -> Result<(), ModelError> {
    for t in trees {
        for &f in t.split_features() {
            if f >= num_features {
                return Err(ModelError::InvalidLayout("split_feature 越界"));
            }
        }
    }
    Ok(())
}

/// 读分箱表并校验边界合法性。
fn read_bin_table(c: &mut Cursor<&[u8]>) -> Result<BinTable, ModelError> {
    let max_bins = read_u32(c)? as usize;
    let num_features = read_u32(c)? as usize;
    let mut boundaries = Vec::with_capacity(num_features);
    for _ in 0..num_features {
        let nb = read_u32(c)? as usize;
        let mut b = Vec::with_capacity(nb);
        for _ in 0..nb {
            b.push(read_f64(c)?);
        }
        boundaries.push(b);
    }
    let table = BinTable::new(boundaries, max_bins)
        .map_err(|_| ModelError::InvalidLayout("bin 表边界非法（数量超限或非严格升序）"))?;
    if table.num_features() != num_features {
        return Err(ModelError::InvalidLayout("bin 表特征数与头部不一致"));
    }
    Ok(table)
}

/// 读一棵树并校验节点结构。
fn read_tree(c: &mut Cursor<&[u8]>) -> Result<Tree, ModelError> {
    let num_nodes = read_u32(c)? as usize;
    if num_nodes == 0 {
        return Err(ModelError::InvalidLayout("树节点数为 0"));
    }
    let mut split_features = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        split_features.push(read_u32(c)? as usize);
    }
    let mut thresholds = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        thresholds.push(read_f64(c)?);
    }
    let mut missing_go_left = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        match read_u8(c)? {
            0 => missing_go_left.push(false),
            1 => missing_go_left.push(true),
            _ => return Err(ModelError::InvalidLayout("布尔字段非 0/1")),
        }
    }
    let mut left = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        left.push(read_i32(c)?);
    }
    let mut right = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        right.push(read_i32(c)?);
    }
    let mut leaf_values = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        leaf_values.push(read_f64(c)?);
    }
    let mut depths = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        depths.push(read_u32(c)? as usize);
    }
    // v3：树节点 gain/cover
    let mut split_gains = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let g = read_f64(c)?;
        if !g.is_finite() || g < 0.0 {
            return Err(ModelError::InvalidLayout("split_gain 非有限或为负"));
        }
        split_gains.push(g);
    }
    let mut node_counts = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let n = read_f64(c)?;
        if !n.is_finite() || n < 0.0 {
            return Err(ModelError::InvalidLayout("node_count 非有限或为负"));
        }
        node_counts.push(n);
    }

    // 节点结构校验：内部节点必须有合法左右子索引；叶子 left==right==-1。
    let nn = num_nodes as i64;
    for i in 0..num_nodes {
        let l = left[i];
        let r = right[i];
        if l < 0 || r < 0 {
            if l != -1 || r != -1 {
                return Err(ModelError::InvalidLayout("分裂子索引部分缺失"));
            }
        } else {
            if (l as i64) >= nn || (r as i64) >= nn {
                return Err(ModelError::InvalidLayout("子节点索引越界"));
            }
        }
    }

    Ok(Tree::from_soa(
        split_features,
        thresholds,
        missing_go_left,
        left,
        right,
        leaf_values,
        depths,
        split_gains,
        node_counts,
    ))
}
