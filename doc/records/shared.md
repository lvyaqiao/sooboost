# shared.md 共享面变更登记 + 待手动验收总表

> 状态：现行
> 类型：T3 流水（只追加，公共区段）｜写者：维护者汇总｜读者：全员（变更感知层）
> 范围：跨模块/对外契约面（红线、模型格式、API、数据语义）变更登记；里程碑手动验收总表
> 关联：doc/baseline/architecture.md；doc/baseline/contracts.md；doc/ledgers/roadmap.md
> 更新：2026-08-13

## 共享面变更登记

| 日期 | 变更 | 影响面 | 关联 |
|---|---|---|---|
| 2026-08-31 | 集成测试 `benchmark_path` 由 `CARGO_MANIFEST_DIR` 改为向上查找 workspace 根（解析到 `benchmark/`） | 测试面（无公共 API 变更） | b4e6427 |
| 2026-08-31 | 5 处源码悬空引用 `doc/plans/m0-spec.md` → `doc/archive/m0-spec.md` | 文档面 | b4e6427 |
| 2026-08-31 | 全量源码/基准数据/CI 首次纳入 git | 仓库面 | 2655416 等 |

## 待手动验收总表

| 里程碑 | 验收项 | 状态 | 证据 |
|---|---|---|---|
| M4 地基修复 | `cargo test --workspace` 全绿（~93 测试） | 已通过（本地复验） | b4e6427 |
| M4 地基修复 | `cargo fmt --check` + `cargo clippy -D warnings` 绿 | 已通过（本地复验） | 本地 run |
| M4 地基修复 | 基准门禁 `sooboost --gate` / `gen --gate` 绿（质量对标 HGB 在 0.05 内） | 已通过（本地复验） | 686a458 |
