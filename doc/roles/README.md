# doc/roles 人员私有文档目录

> 状态：现行
> 范围：人员私有工作文档（日志/TODO/方案草稿/决策备忘）——现行仅 master 单一域；`mailbox/`、`taskbox/` 为未来分支协作留位（共享例外，见规则 5/6）
> 关联：doc/ledgers/roadmap.md；AGENTS.md 「协作机制」
> 更新：2026-08-13

## 目录

| 子目录 | 人员/域 | 分支前缀 | 状态 |
|---|---|---|---|
| master/ | 维护者（现行唯一域） | `master/` | 现行 |
| mailbox/ | 跨域声明信箱（共享可写） | — | 留位（未来多域时启用） |
| taskbox/ | 任务分派信箱（维护者写/收件人改状态） | — | 留位（未来多域时启用） |

## 规则

1. **私有**：本目录是各人的私人工作区（工作日志/草稿/TODO），不登记 doc/README.md 编号索引
2. **跨人员事项**：必须上升登记 doc/ledgers/roadmap.md 或 doc/records/，禁止写进他人目录（跨域声明除外——投 `mailbox/` 对应收件箱，见规则 5）
3. **归属变更**：人员变动只更新 roadmap 分工快照；离职 = 子目录归档至 doc/archive/，编号不回收
4. **公共品**：面向全员的机制说明一律放 doc/ 主干（README/baseline/ledgers），本目录只放个人视角记录
5. **mailbox/ 共享例外**：`mailbox/` 为全员可写的共享区（跨域声明投递处，规则见 mailbox/README.md）；该目录不收私人日志
6. **taskbox/ 任务分派例外**：`taskbox/` 为维护者单向任务分派区（规则见 taskbox/README.md）——维护者独占写、收件人只改自己任务行状态列；与 mailbox 正交
