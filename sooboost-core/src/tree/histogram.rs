//! 直方图：单特征分箱上的梯度/海森/计数累加（单线程版，红线 1 单一内核）。

/// 单特征直方图：按 bin 累加梯度、海森与计数。
#[derive(Debug, Clone)]
pub struct Histogram {
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
    pub count: Vec<u32>,
}

impl Histogram {
    pub fn new(num_bins: usize) -> Self {
        Self {
            grad: vec![0.0; num_bins],
            hess: vec![0.0; num_bins],
            count: vec![0; num_bins],
        }
    }

    /// 累加一行（`bin` 为有效非缺失 bin id）。
    #[inline]
    pub fn accumulate(&mut self, bin: u16, g: f64, h: f64) {
        let b = bin as usize;
        self.grad[b] += g;
        self.hess[b] += h;
        self.count[b] += 1;
    }

    /// 全 bin 合计（G, H, count）——不含缺失，缺失由调用方单独统计。
    pub fn total(&self) -> (f64, f64, u32) {
        let g: f64 = self.grad.iter().sum();
        let h: f64 = self.hess.iter().sum();
        let c: u32 = self.count.iter().sum();
        (g, h, c)
    }
}
