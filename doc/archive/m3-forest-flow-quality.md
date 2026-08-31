# m3-forest-flow-quality.md ForestFlow 质量硬化

> 状态：已归档（M3 gate 全绿，2026-08-19 收口）
> 类型：T1 变体（方案 = 契约候选）｜写者：维护者｜读者：M3 实施人
> 范围：ForestFlow 生成边际、相关结构、条件填补的实验质量；不改变核心 API 与模型格式
> 关联：doc/ledgers/backlog.md [B-03]；doc/archive/m2-spec.md；sooboost-experiments/src/forest_flow.rs；benchmark/run_benchmark.py
> 更新：2026-08-19

## 1. 背景

M2 原型在 synthetic 数据上结构合理，但 California 的 z-score 流路径难以拟合偏态边际；单次随机条件样本也不适合作为 RMSE 点填补预测。M3 只在 D7 实验 crate 与 benchmark 层修正这两个问题，不把研究实现提升为核心稳定 API。

## 2. 动作

- 每列训练值做经验分位数正态化，采样结束用经验分位数插值还原，降低偏态尾部对流模型的压力。
- 每个观测行生成 4 个分层时间 `(t, epsilon)` 训练样本，降低流匹配目标的蒙特卡洛噪声；按特征派生 seed，保持确定性。
- 采样器从伪 Euler 改为二阶中点积分；观测特征继续逐步固定。
- 保留 `impute` 的单次随机填补语义，新增 `impute_mean` 对多个条件样本求均值，用于点填补 RMSE 与 CLI benchmark；CLI 默认 4 次，可由 `--imputation-samples` 覆盖。
- 为经验正态变换和 Monte Carlo 点填补增加 roundtrip/确定性测试。

## 3. 验收

- `cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace` 全绿。
- `scripts/gate.ps1` 八步全部通过，核心 sooboost 质量对齐与性能参考不回归。
- `benchmark --mode gen --gate` 四个数据集全通过；指标记录 `forest_flow_metrics.json`，明确 `imputation_samples=4`。
- 研究结果仅作为原型证据，不承诺生产分布生成质量或跨数据集泛化。

## 4. 收口结果

California 生成均值相对误差从 M2 的 `1.027` 降至 `0.049`，标准差相对误差从 `4.158` 降至 `0.057`，C2ST AUC 从 `0.803` 降至 `0.620`；4 次条件样本均值使 California 点填补相对均值基线增益达到 `+23.3%`。强相关 synthetic binary 填补增益达到 `+72.4%`。弱相关数据仍略差于均值基线，按依赖感知门禁保留该事实，不虚构收益。
