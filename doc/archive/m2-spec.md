# m2-spec.md M2 里程碑规格

> 状态：已归档（M2 五步工作流全绿，2026-08-19 收口）
> 类型：T1 变体（方案 = 契约候选）｜写者：维护者｜读者：M2 实施人（开工前必读）
> 范围：M2 五步工作流（cargo-fuzz 前置 + 条件分布树原型 + Flow Matching 原型 + 反哺核心 + 门禁扩展）、红线/契约承接、验收标准、测试面、明确不做清单
> 关联：doc/baseline/architecture.md §D7（研究先实验 crate 后反哺核心）；doc/baseline/contracts.md；doc/archive/m1-spec.md（M1 已收口）
> 参照：UnmaskingTrees/BaltoBot（arXiv:2407.05593, TMLR 2025）；ForestDiffusion（arXiv:2309.09968）；BUFF（arXiv:2404.18219）
> 更新：2026-08-19（M2 收口：两条研究方向原型、cargo-fuzz、gen 门禁均完成）

## 1. 目标

M1 已完成核心功能面（序列化/目标函数/并行/类别特征/门禁）。M2 转向产品卖点中的研究方向：
把 GBDT 改造成非参数条件概率分布树（BaltoBot/UnmaskingTrees）与用 Boosting 树拟合 Flow Matching
向量场（ForestDiffusion 思路），实现**表格数据生成与缺失值填补**。D7 约束：研究先放独立实验
crate（`sooboost-experiments`）原型验证，验证后仅回灌被证明的最小原语到核心。

## 2. 五步工作流（按 tranche 顺序执行）

| # | 工作流 | 承接 | 风险 | 验收方式 |
|---|---|---|---|---|
| M2-0 | **cargo-fuzz 正式化**（前置） | 红线 6「fuzz：读入外部字节的路径」操作化（M1 以 proptest 占位，本步转正式 fuzz） | 中（nightly + libFuzzer 工具链） | `fuzz/` 独立 workspace crate + 两 target：`model_deserialize`（Booster::deserialize 任意字节不 panic/不 OOM）、`csv_parse`（CSV 字节读入）；各跑 ≥60s 无 crash；失败样本入 corpus 固化；CI 仍以 proptest 作 stable 回归守护 |
| M2-A | **条件分布树原型**（UnmaskingTrees 填补 + BaltoBot 条件生成） | D7 实验 crate；复用核心 Booster（regression/binary） | 中（方法还原度） | 填补：MC 迭代 unmask，合成相关数据 RMSE 优于 mean 基线；条件生成：边际/相关/C2ST 质量可测；sklearn IterativeImputer 对照留后续研究 |
| M2-B | **Flow Matching 向量场原型**（ForestDiffusion per-feature） | D7 实验 crate；每特征独立 GBT（标量叶子即可复用） | 中高（采样器/ODE） | 生成质量（边际/相关/C2ST）+ 填补（采样式 RMSE）；Euler 步进确定性与同 seed 一致 |
| M2-C | **反哺核心**（最小 API 回灌） | D7「原型验证后再反哺核心 trait」 | 中（核心边界） | 仅回灌被 A/B 验证的原语（如 predict_raw / Dataset 便捷构造），不提前铺向量叶子；核心测试零回归 |
| M2-D | **门禁扩展**（生成/填补质量档） | AGENTS §测试纪律「基准门禁」 | 低 | benchmark 新增 `--mode gen`（marginals/correlation/C2ST/imputation RMSE）+ 对齐门槛入 scripts/gate.ps1 |

## 3. 红线与易踩坑承接

- **红线 3**：实验原型采样固定 seed，同 seed 生成逐位一致（层级二）；并行（若有）顺序确定。
- **红线 6 操作化**：M2-0 把模型/CSV 字节路径从 proptest 占位升级为 cargo-fuzz target。
- **红线 7**：实验 crate 同样 `#![forbid(unsafe_code)]`（fuzz target 亦禁 unsafe）。
- **易踩坑 5**：缺失值语义必须显式断言结果（生成/填补质量不能只看"没崩"）。
- **易踩坑 10**：实验 crate 库代码禁 unwrap/expect，错误显式传播。
- **易踩坑 8**：实验 crate 只依赖 workspace 已有依赖（arrow/rayon）；不引新重依赖；sklearn 对比仅在 benchmark（Python）侧。

## 4. 测试面（入 CI）

| 类别 | 内容 |
|---|---|
| fuzz | M2-0：模型字节 + CSV 解析两 target（libFuzzer ≥60s）+ proptest 回归守护 |
| 确定性 | 生成/采样式路径同 seed 逐位一致；Euler 步进确定 |
| 属性测试 | 生成样本边界（有限值、无 NaN 泄漏、形状正确）；填补单调性（信息越多误差越小） |
| 集成 | 训练→生成→质量指标端到端；填补→再训练回路 |
| 基准门禁 | M2-D gen 档（marginals/correlation/C2ST/imputation RMSE）对齐门槛 |

## 5. 明确不做（M2 边界）

- 不引入多输出/向量叶子（D7：核心保持标量叶子；Flow Matching 用 per-feature 独立 GBT 规避）
- 不做 EFB/GOSS（D9）；不做 SIMD/GPU；不做 C ABI/pyo3
- BaltoBot 不做大规模高维原生（原型尺度：合成数据 + 4 个 numeric benchmark 数据集）
- 类别特征生成先用简化策略（连续特征为主；类别转数值复用 M1-4 ordered TS），不做多类采样器
- 不把研究方向压进核心 API 形状（绑定层/研究方向都不反哺架构，易踩坑 7 精神）

## 6. 收口条件

五步工作流全绿 + fuzz 正式化入 target + gen 门禁达标 + workspace 测试零回归 → 状态标 ✅，下一里程碑立项。
若 A/B 某方向验证结论为"收益不成立"，在 records 登记验证证据后可裁剪该方向（不算失败，属 D7 风险隔离的预期产出）。
