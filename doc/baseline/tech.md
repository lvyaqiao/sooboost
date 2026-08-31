# tech.md 技术选型

> 状态：现行
> 类型：T1 基线｜写者：维护者｜读者：全员（契约依据）
> 范围：技术栈选型决策——采用/封装后采用/自建/放弃/观察清单（原 02 迁移）
> 关联：doc/baseline/product.md（需求）；doc/baseline/architecture.md（架构红线与决策表）；doc/research/GBDT三巨头bug分族与代码质量.md（维护负担证据）
> 更新：2026-08-13（编号制废除，原 02 迁移）

> 调研日期：2026-08-11。所有生态结论均经 crates.io API 实测核验（下载量/版本/代码规模/维护状态）。

## 一、总体判断

**Rust 生态的成熟度在"工程层"，不在"算法层"**：

- 工程基础设施（arrow/rayon/serde/测试工具链）完全成熟，达到甚至超过 C++ 生态水平；
- GBDT 算法层没有任何生产级纯 Rust 实现（见第四节实测数据），必须自建——这既是本项目的立项理由，也意味着自建面收敛明确。

**"轻实现、重验证"策略成立**：算法核心自建（5k~15k 行量级），全部底座用成熟 crate，不重复造轮子。

## 二、采用清单（生产级，直接依赖）

| crate | 用途 | 采用原因 |
|---|---|---|
| `arrow` / `parquet` | 数据层底座 | Apache 官方实现，Polars/DataFusion 同款；零拷贝读 Parquet/Polars/DuckDB 是核心卖点；缺失值用原生 null 位图（对应 doc/baseline/architecture.md §红线 2） |
| `rayon` | 并行 | 事实标准数据并行库；直方图/分裂搜索的并行由 rayon 承担（对应 doc/baseline/architecture.md §D2） |
| `serde` + `serde_json` | 模型 JSON 格式 | 调试/跨语言可读；option schema 化（CatBoost 长处） |
| `bincode` / `postcard` | 模型紧凑二进制 | mmap 热加载；与 JSON 双格式并存（对应 doc/baseline/architecture.md §D6） |
| `rand` + `rand_xoshiro` | 确定性 RNG | 分箱 sketch/采样带种子，结果逐位可复现（对应 doc/baseline/architecture.md §红线 3） |
| `proptest` | 属性测试 | 分裂单调性、缺失值语义、序列化往返等性质守护 |
| `criterion` | 基准 | 基准门禁入 CI（LightGBM 无性能门禁的教训） |
| `cargo-fuzz` | 模糊测试 | 三巨头全无 fuzz 的差异化机会（对应 doc/baseline/architecture.md §红线 6） |
| `clap` | CLI | 事实标准 |
| `memmap2` | 模型 mmap 热加载 | 零拷贝推理 + `Arc` 原子热替换 |
| `thiserror` / `anyhow` | 错误处理 | 库/应用分层约定 |
| `tracing` | 日志 | 结构化日志 |
| `bytemuck` / `hashbrown` / `smallvec` | 零拷贝位转 / 高性能容器 | arrow 同款依赖链，行为验证充分 |

## 三、封装后采用（半可用）

| crate | 现状 | 处置 |
|---|---|---|
| `wide` | 稳定 SIMD 封装（f32x8/f64x4 等，跨平台） | **采用**，在树内核内封装一层 SIMD 抽象；显式 SIMD 优于 C++ 依赖自动向量化（doc/baseline/product.md 痛点） |
| `std::simd`（portable-simd） | 实测（2026-08）仍为 nightly-only（`#![feature(portable_simd)]`） | **观察**，不依赖；稳定化后评估替换 `wide` |

## 四、自建清单（玩具/空白，必须自建）

对齐 doc/baseline/architecture.md 模块，自建面收敛为：

| 模块 | 自建内容 | 对应架构 |
|---|---|---|
| `data` | arrow RecordBatch 之上的薄封装：缺失值语义唯一定义、类别/权重/group 元数据 | doc/baseline/architecture.md §D1 |
| `binning` | 确定性 quantile sketch、bin 表（随模型序列化） | doc/baseline/architecture.md §D4 |
| `tree` | 直方图构建 + 分裂搜索内核（泛型特化串行/rayon/SIMD）、推理优先 SoA 树表示 | doc/baseline/architecture.md §D2/D7 |
| `loss` | Loss trait + 内置损失集 + 注册表 | doc/baseline/architecture.md §D3 |
| `boosting` | GBDT 循环、采样、早停/CV/回调、TrainingContext | doc/baseline/architecture.md §D5 |
| `model` | 版本化模型格式、JSON/二进制序列化、热替换 | doc/baseline/architecture.md §D6 |
| 类别特征 | ordered target statistics | doc/baseline/architecture.md §D9 |

## 五、评估后放弃的选型

**纯 Rust GBDT 候选库（实测数据，全部不可用）：**

| 候选 | 实测核验 | 放弃原因 |
|---|---|---|
| `gbdt`（mesalock-linux/gbdt-rs） | **2,588 行 Rust / 8 文件**，2024-01 后停更 | 玩具级：无直方图/并行/类别/完整缺失值语义，且无维护 |
| `forust-ml`（jinlow/forust） | **5,434 行 Rust / 17 文件**，单人维护，近 90 天下载仅 2,367 | 半成品：功能子集（无原生类别/排序/生存），无重工业投入，正是"doc/baseline/product.md"排除的科研玩具 |
| `smartcore` | 29,741 行通用 ML 库（活跃） | 教学级：GBDT 泛而不精，无直方图/并行/工程化硬化 |
| `linfa-trees` | **876 行，只有决策树，无 boosting** | 不可用：无 GBDT 实现 |
| `lightgbm` / `xgboost` / `catboost` crates | FFI 包装 C++ 库 | 功能可用，但正是"引入 C++ 编译工具链的摩擦"本身，与本项目目标矛盾 |

**其他放弃项：**

| 选型 | 放弃原因 |
|---|---|
| 自造 DMatrix / 自造数据格式 | 破坏 arrow 零拷贝卖点；复制 XGBoost 数据层 bug 面（doc/baseline/architecture.md §架构原因） |
| EFB / GOSS | 微软专利风险；纯 CPU 下收益有限（doc/baseline/architecture.md §D9） |
| pyo3 Python 绑定 | 绑定层是三巨头最大维护负担源（doc/research/GBDT三巨头bug分族与代码质量.md §1.1）；本项目定位 Rust 生态闭环 |
| C ABI / WASM / 模型 codegen | 当前排期仅 Rust API；序列化格式先行设计，未来可低成本导出 |
| GPU 支持 | 纯 CPU 定位；从根源消灭 CPU/GPU 双轨维护（CatBoost 修复率 28% 的教训） |

## 六、观察清单（暂不采用，持续跟踪）

- `std::simd` 稳定化进度（若稳定，替换 `wide`）
- `forust-ml` 若出现工业级投入（多维护者、arrow 集成），重新评估借鉴价值
- Rust ML 生态新动向（burn 等框架的表格数据方向与本项目无交集，仅作情报）
