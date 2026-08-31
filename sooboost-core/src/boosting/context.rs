//! 训练上下文：显式传递的运行时状态载体（红线 4，零全局状态）。
//!
//! M0 无子采样（无随机路径），seed 仅作为确定性契约的显式输入（红线 3
//! 层级一：同输入同 seed → 结果逐位一致）。M1 引入 subsample / 特征采样时，
//! 随机源一律经 `rng()` 取自本上下文，禁止模块内自造全局随机状态。

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

/// 一次训练会话的上下文（显式传递，不落全局）。
#[derive(Debug, Clone)]
pub struct TrainingContext {
    rng_seed: u64,
}

impl TrainingContext {
    /// 由固定种子构造确定性上下文。
    pub fn new(rng_seed: u64) -> Self {
        Self { rng_seed }
    }

    pub fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    /// 由固定种子生成确定性随机源（同种子 → 同序列，红线 3 层级一）。
    pub fn rng(&self) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(self.rng_seed)
    }
}
