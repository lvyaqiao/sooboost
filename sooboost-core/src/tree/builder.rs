//! TreeBuilder：level-wise 单线程直方图建树（红线 1 单一内核）。
//!
//! 算法：维护 `rows`（按节点区间 [s,e) 排列的行索引）与节点列表；
//! 每层对每个活动节点构建直方图 → 找最佳分裂 → 原地分区生成子节点区间。
//! 分裂公式（牛顿步，对 L2 / binary logloss 通用）：
//!   gain = 0.5·(G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ))
//! 叶子值 = −G/(H+λ)。

use rayon::prelude::*;

use crate::binning::{BinTable, BinnedMatrix, MISSING_BIN};

use super::error::TreeError;
use super::histogram::Histogram;
use super::model::Tree;

/// 树参数（M0 首版固定，m0-spec §4）。
#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    /// 最大深度（根=0；达到即停止分裂）。
    pub max_depth: usize,
    /// 叶子最少样本数。
    pub min_samples_leaf: usize,
    /// 最小分裂增益（小于等于该值不分裂）。
    pub min_split_gain: f64,
    /// L2 正则 λ（叶子/分裂分母平滑）。
    pub reg_lambda: f64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: 6,
            min_samples_leaf: 5,
            min_split_gain: 0.0,
            reg_lambda: 0.0,
        }
    }
}

/// 候选分裂。
#[derive(Debug, Clone)]
struct Split {
    feature: usize,
    bin: u16,
    threshold: f64,
    gain: f64,
    missing_go_left: bool,
}

/// 构建期节点缓冲。
#[derive(Debug)]
pub(crate) struct NodeBuf {
    pub(crate) range: (usize, usize),
    pub(crate) depth: usize,
    pub(crate) split_feature: Option<usize>,
    pub(crate) threshold: f64,
    pub(crate) missing_go_left: bool,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
    pub(crate) leaf_value: Option<f64>,
}

impl NodeBuf {
    fn new(range: (usize, usize), depth: usize) -> Self {
        Self {
            range,
            depth,
            split_feature: None,
            threshold: 0.0,
            missing_go_left: false,
            left: None,
            right: None,
            leaf_value: None,
        }
    }
}

/// 树构建器（level-wise，单线程）。
#[derive(Debug)]
pub struct TreeBuilder {
    params: TreeParams,
}

impl TreeBuilder {
    pub fn new(params: TreeParams) -> Self {
        Self { params }
    }

    /// 依据（预计算的）梯度/海森构建一棵树。
    pub fn build(
        &self,
        matrix: &BinnedMatrix,
        table: &BinTable,
        grad: &[f64],
        hess: &[f64],
    ) -> Result<Tree, TreeError> {
        let n = matrix.num_rows();
        if n == 0 {
            return Err(TreeError::Empty);
        }
        if grad.len() != n || hess.len() != n {
            return Err(TreeError::LengthMismatch {
                rows: n,
                grad: grad.len(),
                hess: hess.len(),
            });
        }

        let mut rows: Vec<usize> = (0..n).collect();
        let mut nodes: Vec<NodeBuf> = vec![NodeBuf::new((0, n), 0)];
        let mut level: Vec<usize> = vec![0];
        let mut depth = 0usize;

        while !level.is_empty() && depth < self.params.max_depth {
            let mut next_level: Vec<usize> = Vec::new();
            for &node_id in &level {
                let (s, e) = nodes[node_id].range;
                let (g_tot, h_tot) = sum_grad_hess(&rows[s..e], grad, hess);

                let best = find_best_split(
                    &ScanCtx {
                        matrix,
                        table,
                        grad,
                        hess,
                        params: &self.params,
                    },
                    &rows[s..e],
                    g_tot,
                    h_tot,
                );

                if let Some(split) = best.filter(|sp| sp.gain > self.params.min_split_gain) {
                    let m = partition_rows(&mut rows, s, e, |r| {
                        let b = matrix.bin(split.feature, r);
                        if b == MISSING_BIN {
                            split.missing_go_left
                        } else {
                            b <= split.bin
                        }
                    });
                    let left_id = nodes.len();
                    let right_id = left_id + 1;
                    nodes.push(NodeBuf::new((s, m), depth + 1));
                    nodes.push(NodeBuf::new((m, e), depth + 1));
                    let node = &mut nodes[node_id];
                    node.split_feature = Some(split.feature);
                    node.threshold = split.threshold;
                    node.missing_go_left = split.missing_go_left;
                    node.left = Some(left_id);
                    node.right = Some(right_id);
                    next_level.push(left_id);
                    next_level.push(right_id);
                } else {
                    nodes[node_id].leaf_value =
                        Some(leaf_value(g_tot, h_tot, self.params.reg_lambda));
                }
            }
            level = next_level;
            depth += 1;
        }

        // 循环退出时若 level 仍非空（depth 达上限），统一标记为叶子
        for &node_id in &level {
            let node = &mut nodes[node_id];
            if node.leaf_value.is_none() {
                let (s, e) = node.range;
                let (g_tot, h_tot) = sum_grad_hess(&rows[s..e], grad, hess);
                node.leaf_value = Some(leaf_value(g_tot, h_tot, self.params.reg_lambda));
            }
        }

        Ok(Tree::from_nodes(nodes))
    }
}

/// 节点区间内梯度/海森合计。
fn sum_grad_hess(node_rows: &[usize], grad: &[f64], hess: &[f64]) -> (f64, f64) {
    let mut g = 0.0;
    let mut h = 0.0;
    for &r in node_rows {
        g += grad[r];
        h += hess[r];
    }
    (g, h)
}

/// 扫描上下文：建树所需只读输入（避免 find_best_split 参数过多）。
struct ScanCtx<'a> {
    matrix: &'a BinnedMatrix,
    table: &'a BinTable,
    grad: &'a [f64],
    hess: &'a [f64],
    params: &'a TreeParams,
}

/// 对节点内所有特征扫描分裂点，返回最优分裂。
///
/// 并行（M1-3，D2）：特征维度 `par_iter` 独立构建直方图，各特征无共享浮点
/// 归约 → **任意线程数逐位一致**（红线 3 层级二）；合并用 (gain, feature) 全序
/// 比较，reduce 顺序无关，结果唯一确定。
fn find_best_split(ctx: &ScanCtx, node_rows: &[usize], g_tot: f64, h_tot: f64) -> Option<Split> {
    (0..ctx.matrix.num_features())
        .into_par_iter()
        .map(|feature| best_split_for_feature(ctx, feature, node_rows, g_tot, h_tot))
        .reduce(|| None, better_split)
}

/// (gain, feature) 全序合并：增益大者胜；平局取特征号小者 → 结果与归约顺序无关。
fn better_split(a: Option<Split>, b: Option<Split>) -> Option<Split> {
    match (a, b) {
        (Some(x), Some(y)) => {
            if x.gain > y.gain || (x.gain == y.gain && x.feature <= y.feature) {
                Some(x)
            } else {
                Some(y)
            }
        }
        (a, b) => a.or(b),
    }
}

/// 单特征最优分裂（直方图构建 + 分位扫描；并行单元，内部顺序执行保证逐位确定）。
fn best_split_for_feature(
    ctx: &ScanCtx,
    feature: usize,
    node_rows: &[usize],
    g_tot: f64,
    h_tot: f64,
) -> Option<Split> {
    let num_bins = ctx.table.num_bins(feature);
    if num_bins < 2 {
        return None;
    }
    let mut hist = Histogram::new(num_bins);
    let (mut mg, mut mh) = (0.0f64, 0.0f64);
    for &r in node_rows {
        let b = ctx.matrix.bin(feature, r);
        if b == MISSING_BIN {
            mg += ctx.grad[r];
            mh += ctx.hess[r];
        } else {
            hist.accumulate(b, ctx.grad[r], ctx.hess[r]);
        }
    }
    let (g_nm, h_nm, c_nm) = hist.total();

    let mut best: Option<Split> = None;
    let (mut g_l, mut h_l, mut c_l) = (0.0f64, 0.0f64, 0u32);
    for k in 0..num_bins - 1 {
        g_l += hist.grad[k];
        h_l += hist.hess[k];
        c_l += hist.count[k];
        if (c_l as usize) < ctx.params.min_samples_leaf {
            continue;
        }
        let c_r = c_nm.saturating_sub(c_l);
        if (c_r as usize) < ctx.params.min_samples_leaf {
            continue;
        }
        // 两种缺失方向：缺失随左 / 缺失随右，取增益更大者
        let candidates = [
            (g_l + mg, h_l + mh, g_nm - g_l, h_nm - h_l, true),
            (g_l, h_l, g_nm - g_l + mg, h_nm - h_l + mh, false),
        ];
        for (gl, hl, gr, hr, miss_left) in candidates {
            let gain = split_gain(gl, hl, gr, hr, g_tot, h_tot, ctx.params.reg_lambda);
            if gain > ctx.params.min_split_gain && best.as_ref().is_none_or(|b| gain > b.gain) {
                let threshold = ctx.table.boundaries(feature)[k];
                best = Some(Split {
                    feature,
                    bin: k as u16,
                    threshold,
                    gain,
                    missing_go_left: miss_left,
                });
            }
        }
    }
    best
}

/// 分裂增益（牛顿步）。
fn split_gain(gl: f64, hl: f64, gr: f64, hr: f64, gt: f64, ht: f64, lambda: f64) -> f64 {
    let dl = gl * gl / (hl + lambda);
    let dr = gr * gr / (hr + lambda);
    let dt = gt * gt / (ht + lambda);
    0.5 * (dl + dr - dt)
}

/// 叶子值 = −G/(H+λ)；分母过小返回 0。
fn leaf_value(g: f64, h: f64, lambda: f64) -> f64 {
    let denom = h + lambda;
    if denom.abs() < 1e-15 { 0.0 } else { -g / denom }
}

/// 原地分区 rows[s..e)：满足 `pred` 的行移到左部，返回边界 `m`（左 s..m，右 m..e）。
fn partition_rows(rows: &mut [usize], s: usize, e: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut i = s;
    let mut j = e;
    while i < j {
        if pred(rows[i]) {
            i += 1;
        } else {
            j -= 1;
            rows.swap(i, j);
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::{BinTable, BinnedMatrix, DEFAULT_MAX_BINS};
    use crate::data::{Dataset, MissingPolicy};
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// 构造单特征数据集，返回 (table, matrix)。
    fn single_feature(values: Vec<Option<f64>>, y: Vec<f64>) -> (BinTable, BinnedMatrix) {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(values)),
                Arc::new(Float64Array::from(y)),
            ],
        )
        .expect("构造 batch");
        let ds = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .expect("构造 Dataset");
        BinTable::build_from_dataset(&ds, DEFAULT_MAX_BINS).expect("分箱")
    }

    #[test]
    fn max_depth_zero_gives_single_leaf() {
        let (table, matrix) =
            single_feature(vec![Some(1.0), Some(2.0), Some(3.0)], vec![1.0, 2.0, 3.0]);
        let builder = TreeBuilder::new(TreeParams {
            max_depth: 0,
            ..TreeParams::default()
        });
        let tree = builder
            .build(&matrix, &table, &[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0])
            .expect("建树");
        assert_eq!(tree.num_nodes(), 1);
        // leaf = -(1+2+3)/3 = -2
        assert!((tree.predict(&[1.0], &[false]) - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn split_separates_left_and_right_gradients() {
        // 前 3 行 grad=1，后 3 行 grad=-1 → 最优分裂在中间
        let (table, matrix) = single_feature(
            vec![
                Some(0.0),
                Some(1.0),
                Some(2.0),
                Some(100.0),
                Some(101.0),
                Some(102.0),
            ],
            vec![0.0; 6],
        );
        let builder = TreeBuilder::new(TreeParams {
            max_depth: 1,
            min_samples_leaf: 1,
            ..TreeParams::default()
        });
        let tree = builder
            .build(
                &matrix,
                &table,
                &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                &[1.0; 6],
            )
            .expect("建树");
        assert_eq!(tree.num_nodes(), 3);
        // 左叶 = -(3)/3 = -1，右叶 = -(-3)/3 = 1
        let left_pred = tree.predict(&[1.0], &[false]);
        let right_pred = tree.predict(&[101.0], &[false]);
        assert!((left_pred - (-1.0)).abs() < 1e-12, "left={left_pred}");
        assert!((right_pred - 1.0).abs() < 1e-12, "right={right_pred}");
        // 缺失默认随某一方向（不崩溃，值在 [-1,1]）
        let miss_pred = tree.predict(&[0.0], &[true]);
        assert!(miss_pred.abs() <= 1.0 + 1e-12);
    }

    #[test]
    fn missing_rows_follow_declared_direction() {
        let (table, matrix) = single_feature(
            vec![
                Some(0.0),
                Some(1.0),
                Some(2.0),
                None,
                Some(100.0),
                Some(101.0),
                Some(102.0),
            ],
            vec![0.0; 7],
        );
        let builder = TreeBuilder::new(TreeParams {
            max_depth: 1,
            min_samples_leaf: 1,
            ..TreeParams::default()
        });
        let tree = builder
            .build(
                &matrix,
                &table,
                &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
                &[1.0; 7],
            )
            .expect("建树");
        // 缺失行（None）预测应等于某一叶子值（缺失方向被记录）
        let miss = tree.predict(&[0.0], &[true]);
        assert!((miss - (-1.0)).abs() < 1e-12 || (miss - 1.0).abs() < 1e-12);
    }
}
