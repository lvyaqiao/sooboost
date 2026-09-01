//! 直方图：节点全特征分箱上的梯度/海森/计数累加（M9 单趟版）。
//!
//! 确定性（红线 3 层级二）：每个填充单元（行块）内部严格按行顺序累加；
//! 行块边界只依赖节点行数（与线程数无关）；并行归并严格按块下标定序
//! → **任意线程数逐位一致**。填充单元 = 行块（大节点拆多块并行）、
//! 或整节点（小节点单块），两种路径累加语义一致。

/// 单节点全特征直方图集合（扁平布局）。
///
/// 所有特征的 bin 槽位连续存放（`offsets[feature]` 为该特征 bin 0 的下标），
/// 一次分配、缓存友好；每特征另有缺失行 (g, h) 合计槽位。
#[derive(Debug, Clone, Default)]
pub(crate) struct HistSet {
    /// 各特征 bin 槽位起始下标（长度 nf+1，末项为 total_bins）。
    pub offsets: Vec<usize>,
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
    pub count: Vec<u32>,
    /// 每特征缺失行梯度合计（长 nf）。
    pub miss_g: Vec<f64>,
    /// 每特征缺失行海森合计（长 nf）。
    pub miss_h: Vec<f64>,
}

impl HistSet {
    /// 分配一个全零直方图集合。
    pub fn new(num_bins_per_feature: &[usize]) -> Self {
        let nf = num_bins_per_feature.len();
        let mut offsets = Vec::with_capacity(nf + 1);
        let mut total = 0usize;
        for &nb in num_bins_per_feature {
            offsets.push(total);
            total += nb;
        }
        offsets.push(total);
        Self {
            offsets,
            grad: vec![0.0; total],
            hess: vec![0.0; total],
            count: vec![0; total],
            miss_g: vec![0.0; nf],
            miss_h: vec![0.0; nf],
        }
    }

    /// 单趟填充：对 `node_rows` 的每一行只读一次 (g, h)，
    /// 内层按行主序连续读取该行全部特征的 bin id（M9 关键热路径）。
    /// 累加顺序 = 行顺序 → 与 v0.2.0 逐特征行序累加逐位一致。
    pub fn fill(
        &mut self,
        matrix: &crate::binning::BinnedMatrix,
        node_rows: &[usize],
        grad: &[f64],
        hess: &[f64],
    ) {
        let nf = self.miss_g.len();
        debug_assert_eq!(nf, matrix.num_features());
        for &r in node_rows {
            let g = grad[r];
            let h = hess[r];
            let bins = matrix.row_bins(r);
            for (f, &b) in bins.iter().enumerate() {
                if b == crate::binning::MISSING_BIN {
                    self.miss_g[f] += g;
                    self.miss_h[f] += h;
                } else {
                    let i = self.offsets[f] + b as usize;
                    self.grad[i] += g;
                    self.hess[i] += h;
                    self.count[i] += 1;
                }
            }
        }
    }

    /// 按块下标顺序归并另一集合（确定性：只允许按块下标依次调用）。
    pub fn merge_in_place(&mut self, other: &HistSet) {
        for (a, b) in self.grad.iter_mut().zip(&other.grad) {
            *a += b;
        }
        for (a, b) in self.hess.iter_mut().zip(&other.hess) {
            *a += b;
        }
        for (a, b) in self.count.iter_mut().zip(&other.count) {
            *a += b;
        }
        for (a, b) in self.miss_g.iter_mut().zip(&other.miss_g) {
            *a += b;
        }
        for (a, b) in self.miss_h.iter_mut().zip(&other.miss_h) {
            *a += b;
        }
    }

    /// 原地减法：`self = self − other`（M10 直方图减法：父 − seed 兄弟）。
    ///
    /// 行数上子节点 ⊆ 父节点 → count 差恒非负（用 saturating 防御）；
    /// grad/hess 为单步浮点减法 → 结果与线程数无关（红线 3）。
    pub fn subtract_in_place(&mut self, other: &HistSet) {
        for (a, b) in self.grad.iter_mut().zip(&other.grad) {
            *a -= b;
        }
        for (a, b) in self.hess.iter_mut().zip(&other.hess) {
            *a -= b;
        }
        for (a, b) in self.count.iter_mut().zip(&other.count) {
            *a = a.saturating_sub(*b);
        }
        for (a, b) in self.miss_g.iter_mut().zip(&other.miss_g) {
            *a -= b;
        }
        for (a, b) in self.miss_h.iter_mut().zip(&other.miss_h) {
            *a -= b;
        }
    }

    /// 某特征非缺失 (G, H, count) 合计（按 bin 下标顺序求和）。
    pub fn feature_total(&self, feature: usize) -> (f64, f64, u32) {
        let s = self.offsets[feature];
        let e = self.offsets[feature + 1];
        let g: f64 = self.grad[s..e].iter().sum();
        let h: f64 = self.hess[s..e].iter().sum();
        let c: u32 = self.count[s..e].iter().sum();
        (g, h, c)
    }
}
