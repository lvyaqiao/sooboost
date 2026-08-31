# doc/archive/ 冻结区

> 本目录存放已完成/已归档的方案与审计记录。**只读冻结**：正文不再更新、编号不随现行重编（doc/ 现行编号见 `../README.md`）。
> 引用历史文档时路径保留 `doc/archive/...`；现行文档引用本区内容 = 摘要+指针，不复制正文。

| 编号 | 文档 | 归档原因 |
|---|---|---|
| 01~09 | 编号制文档 | 2026-08-13 文档大重构：废除编号制 → 全目录化 + 类型分层（现行见 doc/baseline/、doc/ledgers/、doc/records/、doc/research/） |
| 06 | 架构重构依据 | 空白骨架；重构记录机制并入 doc/baseline/architecture.md §ADR 区段 |
| M0 | m0-spec.md | 2026-08-19 M0 收口：九项承诺全绿 + 质量门达标（4/4 数据集超 RF、贴近 HGB）+ 确定性测试入 CI |
| M1 | m1-spec.md | 2026-08-19 M1 收口：五项工作流全绿（序列化/目标函数/并行/类别特征/基准门禁）；快照回归 + 并行确定性 + proptest fuzz 占位入 CI；基准门禁达标（scripts/gate.ps1） |
| M2 | m2-spec.md | 2026-08-19 M2 收口：cargo-fuzz 双 target、条件分布树/Flow Matching 实验原型、gen 生成/填补门禁全绿；核心无须引入向量叶子（D7） |
| M3 | m3-forest-flow-quality.md | 2026-08-19 M3 收口：经验分位数正态化 + 分层流匹配采样 + 中点积分 + 4 样本点填补；生成/填补 gate 4/4，全量 8 步门禁通过 |
