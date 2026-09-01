//! TreeBuilder：level-wise 建树 + 形状自适应并行（M9）。
//!
//! 算法：维护 `rows`（按节点区间 [s,e) 排列的行索引）与节点列表；每层
//! 并行求各节点 (G, H) 合计 → 并行求各节点最优分裂 → 串行 bookkeeping
//! + 原地分区。分裂扫描按**数据形状**（行数/特征数）确定性二选一：
//!
//! - **行主序单趟（两阶段）**：并行单元 = (节点×行块) 填直方图 + (节点×特征)
//!   扫描。每行只读一次 (grad, hess)，消除逐特征路径 F 倍重复读的内存
//!   流量。适合大行数少特征（california 20000×8：A/B 实测快 ~9–30%）。
//! - **直接路径**：并行单元 = (节点×特征)，每单元独立构建单特征直方图并
//!   扫描，累加顺序与 v0.2.0 **逐位一致**。适合多特征小行数
//!   （breast_cancer 569×30）——浅层节点少、特征多，特征维并行吃满核；
//!   行主序路径在此形状实测反而慢 ~79%（浅层填充并行度塌缩）。
//!
//! 确定性（红线 3 层级二）：选路只依赖数据形状（与线程数无关）；
//! 行块边界只依赖节点行数、块间按块下标定序归并；(gain, feature) 全序
//! 合并与归约顺序无关 → **任意线程数逐位一致**。
//!
//! 分裂公式（牛顿步，对 L2 / binary logloss 通用）：
//!   gain = 0.5·(G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ))
//! 叶子值 = −G/(H+λ)。

use rayon::prelude::*;

use crate::binning::{BinTable, BinnedMatrix, MISSING_BIN};

use super::error::TreeError;
use super::histogram::HistSet;
use super::model::Tree;

/// 直方图填充的行块大小：并行单元粒度（M9）。
const FILL_CHUNK_ROWS: usize = 4096;

/// 行主序单趟路径的形状门槛：平均每特征行数 ≥ 该值才启用（见模块文档）。
const ROW_MAJOR_MIN_ROWS_PER_FEATURE: usize = 1024;

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
    /// 分裂增益（分裂节点记录；特征重要度 gain 口径，M6-3）。
    pub(crate) gain: f64,
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
            gain: 0.0,
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

        let num_bins: Vec<usize> = (0..matrix.num_features())
            .map(|f| table.num_bins(f))
            .collect();
        // 形状自适应选路（只依赖数据形状 → 确定性；见模块文档）
        let nf = matrix.num_features();
        let use_row_major = n / nf.max(1) >= ROW_MAJOR_MIN_ROWS_PER_FEATURE;

        while !level.is_empty() && depth < self.params.max_depth {
            let mut next_level: Vec<usize> = Vec::new();
            // 每节点区间 + 梯度合计（并行按节点；各节点行序固定 → 逐位一致）
            let ranges: Vec<(usize, usize)> = level.iter().map(|&id| nodes[id].range).collect();
            let totals: Vec<(f64, f64)> = ranges
                .par_iter()
                .map(|&(s, e)| sum_grad_hess(&rows[s..e], grad, hess))
                .collect();

            let ctx = ScanCtx {
                matrix,
                table,
                grad,
                hess,
                params: &self.params,
            };

            let mut bests: Vec<Option<Split>> = if use_row_major {
                // ── 行主序路径：阶段 A (节点×行块) 填充 + 阶段 B (节点×特征) 扫描 ──
                // 块边界只依赖节点行数 → 与线程数无关（红线 3）
                let mut fill_units: Vec<(usize, usize, usize)> = Vec::new(); // (层内下标, 块起, 块止)
                for (li, &(s, e)) in ranges.iter().enumerate() {
                    let len = e - s;
                    let mut off = 0;
                    while off < len {
                        let end = usize::min(off + FILL_CHUNK_ROWS, len);
                        fill_units.push((li, off, end));
                        off = end;
                    }
                }
                let partials: Vec<HistSet> = fill_units
                    .par_iter()
                    .map(|&(li, off, end)| {
                        let (s, _) = ranges[li];
                        let mut hs = HistSet::new(&num_bins);
                        hs.fill(matrix, &rows[s + off..s + end], grad, hess);
                        hs
                    })
                    .collect();
                // 按节点依块下标定序归并（任意线程数结果一致）
                let mut hists: Vec<Option<HistSet>> = (0..ranges.len()).map(|_| None).collect();
                for (hs, &(li, _, _)) in partials.into_iter().zip(&fill_units) {
                    match &mut hists[li] {
                        None => hists[li] = Some(hs),
                        Some(acc) => acc.merge_in_place(&hs),
                    }
                }
                let hists: Vec<HistSet> = hists
                    .into_iter()
                    .map(|o| o.unwrap_or_else(|| HistSet::new(&num_bins)))
                    .collect();

                let mut scan_units: Vec<(usize, usize)> = Vec::new(); // (层内下标, 特征)
                for li in 0..ranges.len() {
                    for f in 0..nf {
                        scan_units.push((li, f));
                    }
                }
                let scan_results: Vec<Option<Split>> = scan_units
                    .par_iter()
                    .map(|&(li, f)| {
                        best_split_for_feature(&ctx, f, &hists[li], totals[li].0, totals[li].1)
                    })
                    .collect();
                // 每节点按 (gain, feature) 全序合并（最大值唯一 → 与顺序无关）
                let mut bests: Vec<Option<Split>> = vec![None; ranges.len()];
                for (res, &(li, _)) in scan_results.into_iter().zip(&scan_units) {
                    bests[li] = better_split(bests[li].take(), res);
                }
                bests
            } else {
                // ── 直接路径：单元 = (节点×特征)，独立填单特征直方图并扫描 ──
                // 累加顺序与 v0.2.0 逐位一致；多特征小行数下特征维并行吃满核
                let mut units: Vec<(usize, usize, usize, usize)> = Vec::new(); // (层内下标, 特征, 区间起, 区间止)
                for (li, &(s, e)) in ranges.iter().enumerate() {
                    for f in 0..nf {
                        units.push((li, f, s, e));
                    }
                }
                let results: Vec<Option<Split>> = units
                    .par_iter()
                    .map(|&(li, f, s, e)| {
                        best_split_direct(&ctx, f, &rows[s..e], totals[li].0, totals[li].1)
                    })
                    .collect();
                let mut bests: Vec<Option<Split>> = vec![None; ranges.len()];
                for (res, &(li, _, _, _)) in results.into_iter().zip(&units) {
                    bests[li] = better_split(bests[li].take(), res);
                }
                bests
            };

            // 串行 bookkeeping + 原地分区（各节点区间不相交，顺序固定）
            for (level_idx, &node_id) in level.iter().enumerate() {
                let (s, e) = nodes[node_id].range;
                let (g_tot, h_tot) = totals[level_idx];
                if let Some(split) = bests[level_idx]
                    .take()
                    .filter(|sp| sp.gain > self.params.min_split_gain)
                {
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
                    node.gain = split.gain;
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

/// 扫描上下文：分裂扫描所需只读输入。
struct ScanCtx<'a> {
    matrix: &'a BinnedMatrix,
    table: &'a BinTable,
    grad: &'a [f64],
    hess: &'a [f64],
    params: &'a TreeParams,
}

/// 直接路径：单特征最优分裂（独立填单特征直方图 + 扫描；M9 (节点×特征) 并行单元）。
///
/// 逐行顺序累加（`matrix.bin(feature, r)` 特征主序连续读 bin），
/// 与 v0.2.0 的逐特征处理**逐位一致**。
fn best_split_direct(
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
    let mut grad = vec![0.0f64; num_bins];
    let mut hess = vec![0.0f64; num_bins];
    let mut count = vec![0u32; num_bins];
    let (mut mg, mut mh) = (0.0f64, 0.0f64);
    for &r in node_rows {
        let b = ctx.matrix.bin(feature, r);
        let (g, h) = (ctx.grad[r], ctx.hess[r]);
        if b == MISSING_BIN {
            mg += g;
            mh += h;
        } else {
            grad[b as usize] += g;
            hess[b as usize] += h;
            count[b as usize] += 1;
        }
    }
    let (g_nm, h_nm, c_nm): (f64, f64, u32) = {
        let g: f64 = grad.iter().sum();
        let h: f64 = hess.iter().sum();
        let c: u32 = count.iter().sum();
        (g, h, c)
    };

    scan_histogram(
        ctx, feature, num_bins, &grad, &hess, &count, g_nm, h_nm, c_nm, mg, mh, g_tot, h_tot,
    )
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

/// 行主序路径：单特征最优分裂（扫描预建直方图；(节点×特征) 并行单元）。
fn best_split_for_feature(
    ctx: &ScanCtx,
    feature: usize,
    hist: &HistSet,
    g_tot: f64,
    h_tot: f64,
) -> Option<Split> {
    let num_bins = ctx.table.num_bins(feature);
    if num_bins < 2 {
        return None;
    }
    let (g_nm, h_nm, c_nm) = hist.feature_total(feature);
    let (mg, mh) = (hist.miss_g[feature], hist.miss_h[feature]);

    let s = hist.offsets[feature];
    let grad = &hist.grad[s..s + num_bins];
    let hess = &hist.hess[s..s + num_bins];
    let count = &hist.count[s..s + num_bins];

    scan_histogram(
        ctx, feature, num_bins, grad, hess, count, g_nm, h_nm, c_nm, mg, mh, g_tot, h_tot,
    )
}

/// 分位扫描共享内核：对单特征直方图做前缀累加并取最优分裂。
///
/// 两种缺失方向（随左/随右）取增益更大者；阈值取该 bin 上边界。
/// 逐 bin 顺序扫描 → 结果确定。
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_histogram(
    ctx: &ScanCtx,
    feature: usize,
    num_bins: usize,
    grad: &[f64],
    hess: &[f64],
    count: &[u32],
    g_nm: f64,
    h_nm: f64,
    c_nm: u32,
    mg: f64,
    mh: f64,
    g_tot: f64,
    h_tot: f64,
) -> Option<Split> {
    let mut best: Option<Split> = None;
    let (mut g_l, mut h_l, mut c_l) = (0.0f64, 0.0f64, 0u32);
    for k in 0..num_bins - 1 {
        g_l += grad[k];
        h_l += hess[k];
        c_l += count[k];
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
