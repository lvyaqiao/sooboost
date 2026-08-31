//! `Loss` trait：训练内核泛型于该 trait，编译期内联（架构 D3）。

/// 损失函数接口（monomorphized）。
///
/// - `gradient` / `hessian` 供 GBDT 分裂与叶子值计算；
/// - `value` 为损失值，供数值验证与后续监控/早停使用；
/// - `init_score` 为第 0 棵树的全局偏置；
/// - `transform` 把累加原始分数（init + 各树）映射为最终预测
///   （L2 → 恒等；binary logloss → sigmoid）。
pub trait Loss: Send + Sync {
    /// 损失名称（用于元数据与错误消息）。
    fn name(&self) -> &'static str;

    /// 在给定预测 `pred`（原始分数）与真值 `y` 处的损失值。
    fn value(&self, y: f64, pred: f64) -> f64;

    /// 全局初始预测（训练前偏置）。`y` 为空时返回 0.0（调用方应保证非空，见数据契约）。
    fn init_score(&self, y: &[f64]) -> f64;

    /// 一阶导 d(loss)/d(pred)。
    fn gradient(&self, y: f64, pred: f64) -> f64;

    /// 二阶导 d^2(loss)/d(pred)^2。
    fn hessian(&self, y: f64, pred: f64) -> f64;

    /// 原始分数 → 最终预测。
    fn transform(&self, raw: f64) -> f64;
}
