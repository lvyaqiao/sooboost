# doc/ 文档索引（AI 与人的第一入口）

> 维护：文档目录任何变更（新增/迁移/归档）必须同步本表与本目录导航。
> 结构：目录 = 生命周期类型（见 §类型定义）；引用一律写可解析路径 `doc/<目录>/<文件>.md §章节`；死链必须拦截。
> 更新：2026-08-13（文档大重构：废除编号制 → 全目录化 + 类型分层 + 写入路由）

## 目录导航

```
doc/
├─ README.md        本文件：导航 + 类型 + 路由 + 治理（唯一入口）
├─ baseline/        T1 基线：契约/决策（product/tech/architecture/contracts）
├─ plans/           模块方案（实施完归档）
├─ ledgers/         T2 台账：状态跟踪（roadmap 权威/backlog 需求池/bugs/risks）
├─ records/         T3 流水：交付与验收（shared 公共区段 + master 域文件）
├─ research/        参考调研（GBDT 三巨头功能/工程/架构与 bug 分族）
├─ roles/           工作区（现行单域 master；mailbox 声明投递 + taskbox 任务分派为未来分支协作留位）
└─ archive/         冻结区（历史方案/编年史，只读）
```

## 类型定义（生命周期分层）

| 类型 | 语义 | 更新规则 | 归档触发 |
|---|---|---|---|
| T1 基线 | 契约/决策，低频改 | **ADR 式**：追加不改旧文，改版旧版冻结进 archive | 改版时 |
| T2 台账 | 状态跟踪，单条目 | 行级更新（不改结构） | 不复用 |
| T3 流水 | 事件记录，只追加 | **按写者分区**（域段 owner 写、公共段维护者写） | 里程碑结束移 archive |
| 参考调研 | 研究资料 | 追加补充 | 结论落地后 |
| 工作区 | 私人/投递 | 本人 + mailbox/taskbox 例外 | 离职/终 |
| 冻结区 | 历史 | 只读，正文不再更新 | — |

## 写入路由表（新内容写哪 → 谁写）

| 我要… | 写哪 | 谁 |
|---|---|---|
| 改需求/技术/架构契约 | doc/baseline/ 对应文件（ADR 追加） | 维护者授权后写 |
| 新增模块方案 | doc/plans/ + plans/README 登记 | 方案提出人 |
| 新需求登记 | doc/ledgers/backlog.md 加一行 | 任何人 |
| 里程碑/进度状态变更 | doc/ledgers/roadmap.md（权威单点） | 维护者 |
| 挂 BUG | doc/ledgers/bugs.md 行 | 修者 |
| 挂风险 | doc/ledgers/risks.md 行 | 发现者 |
| 记录交付/验收 | doc/records/<域>.md | 域 owner |
| 共享面变更登记 | doc/records/shared.md 一行 | 维护者汇总 |
| 跨域开工声明（未来多域时） | doc/roles/mailbox/ 对应收件箱 | 投递人 |
| 立项任务分派（未来多域时） | doc/roles/taskbox/ 对应收件箱 | 维护者 |
| 私有日志/草稿 | doc/roles/<域>/ | 本人 |

> **单一事实源**：一个事实只在一处主写，其他文件只放指针。状态以 doc/ledgers/roadmap.md 为权威，个人文件只是执行视图（对齐规则见 doc/records/README.md）。

## 域路由表（开工前先查本表拿必读清单）

| 你的域/角色 | 必读 | 开工前查 |
|---|---|---|
| 任意 | AGENTS.md（契约）+ 本文件（路由） | — |
| master（现行唯一域） | doc/records/master.md + doc/records/shared.md + doc/baseline/architecture.md + doc/baseline/contracts.md | shared.md 最近登记 + roadmap 状态总览 |
| 未来分支域 | 各自域 records/ + 相关 baseline | 同上 |

**开工仪式**：① AGENTS.md（自动加载）→ ② 本表拿必读清单 → ③ 读必读文件头部过滤 → ④ 查 shared.md 最近登记 + roadmap 状态 → ⑤ git log → 开工。

## 需求与重构管线

### 需求管线（新需求 → 人）

```
登记（doc/ledgers/backlog.md 一行，任何人）→ 裁决（维护者：拒绝/小改动/立项）
→ 立项（backlog 行号 + roadmap 里程碑表）→ 执行（分支 <域>-<任务>/ 或单域直接提交）
→ 收口（roadmap 状态 ✅ + backlog 销号）
```

- 行号引用铁律：待办条目必须带 `[B-<序号>]` / `[M<里程碑>]` 引用，无引用 = 未立项
- 异议：归属/拒绝有异议 → 投 doc/roles/mailbox/ 对应收件箱
- 小改动（≤1 天）免 backlog，直接进域 records/ 待办区

### 重构管线（架构/重构变动）

```
① 意向：doc/baseline/architecture.md ADR 区段追加观察项（谁想重构谁写）
② 授权：触碰红线/契约/模型格式 → 维护者授权（mailbox 申请）；域内重构 owner 自决，
         触碰共享面仍必须共享面登记
③ 方案：doc/plans/<重构名>.md（决策/动作/验证骨架）+ plans/README 登记
④ 执行：分支 + records/<域>.md 交付 + mailbox 跨域声明
⑤ 验证：门禁（测试/基准）+ 手动验收（records/shared.md 总表）
⑥ 收口：ADR ✅ + roadmap 相关行更新 + 方案文件归档 archive/
```

## 文档模板（新建文档必须采用头部）

```markdown
# <标题>

> 状态：现行 / 归档
> 类型：<T1 基线｜T2 台账｜T3 流水｜参考调研｜工作区>｜写者：<谁>｜读者：<谁>
> 范围：<本文档覆盖的主题与边界>
> 关联：<关联文档路径 §章节>；<关联代码路径>
> 更新：<YYYY-MM-DD>

<正文，一个职责一个文档，单文件 ≤500 行，超了按主题拆子文件（目录内加文件 + 本目录 README 登记）>
```

## 治理规则（写入时对照）

1. **单一事实源**：新知识先查写入路由表确认位置；已存在 → 引用不复制
2. **引用可解析**：一律 `doc/<目录>/<文件>.md §章节`；提交前检查旧编号/死链引用（旧编号简写已废弃，复用即纠错）
3. **目录即契约**：文件归属目录决定其类型与更新规则；**新增内容 = 加文件 + 本表/目录 README 登记一行**，永不动结构
4. **头部自描述**：状态/类型/范围/关联/更新 五行必填（agent 读头部即可过滤）
5. **归档**：T3 流水按里程碑移 doc/archive/（登记 archive/README）；T1 改版旧版进 archive；归档后正文不再更新
6. **表格纪律**：散文优先——默认 `###` 小节 + bullets + 加粗字段；表格仅限两类：短格查表/台账（单格 ≤40 字符）、机器解析索引表（archive/README 登记表禁改格式）
7. **重大重构纪律**：触碰红线/契约/模型格式的改动，无 ADR 观察项 + 授权不得开工
