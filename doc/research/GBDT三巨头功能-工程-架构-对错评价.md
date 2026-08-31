# 调研：GBDT 三巨头（XGBoost / LightGBM / CatBoost）功能、工程、架构与对错评价

> 调研日期：2026-08-11
> 数据来源：`reference/` 下三个仓库的完整 git 历史与源码（实测快照 2026-08-10/11）
> 目的：为本项目（纯 Rust GBDT）评估功能路线、工程策略与架构选型
> 关联文档：`doc/05-BUG分族.md`（七族结构成因谱系）、`doc/调研/调研-GBDT三巨头bug分族与代码质量.md`（bug 统计）

## 一、功能完成度对比

### 1.1 目标函数（objective）广度

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 回归 | squarederror、squaredlogerror、absoluteerror、quantileerror、pseudohuber、logistic、gamma、tweedie、poisson、huber | regression(L1/L2)、mape、fair、huber、quantile、poisson、gamma、tweedie | RMSE、MAE、Quantile、MultiQuantile、Expectile、LogLinQuantile、MAPE、SMAPE、Huber、LogCosh、Lq、Tweedie、Cox、RMSPE、MSLE、MedianAbsoluteError、RMSEWithUncertainty |
| 分类 | binary:logistic/logitraw/hinge | binary、cross_entropy(_lambda) | Logloss、CrossEntropy、Focal、CtrFactor |
| 多分类 | softmax/softprob（MT 时代新增多套） | multiclass/ova | MultiClass/OneVsAll/MultiLogloss/MultiCrossEntropy |
| 排序 | rank:pairwise/ndcg/map | lambdarank、rank_xendcg | PairLogit、YetiRank(±Pairwise)、LambdaMart、StochasticRank、QueryRMSE、GroupQuantile、QuerySoftMax、StochasticFilter、PFound/NDCG/ERR/MRR 指标族 |
| 生存/多目标 | cox/aft；MT 多输出树 | 无 | SurvivalAft；MultiRMSE(±Missing)、多标签 |
| 自定义 | objective 回调 | custom | PythonUserDefined(PerObject/MultiTarget)、UserPerObj/Querywise |

**排序：CatBoost ≫ XGBoost > LightGBM**。LightGBM 无生存/多目标；CatBoost 排序与多目标面全场最大。

### 1.2 数据特性

| 维度 | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 稀疏表示 | DMatrix 稀疏 | CSR/CSC | 支持 |
| 外部内存（超大数据） | 有（extmem quantile dmatrix） | 无 | 有 |
| 类别特征 | 原生（2025 默认启用） | 原生 | 原生最强（ordered TS，无需编码） |
| 文本特征 | 无（靠外部） | 无 | 有（BoW/NaiveBayes/BM25/LDA/KNN） |
| 嵌入特征 | 无 | 无 | 有 |
| 缺失值 | 支持 | 支持 | 支持 |
| 权重/group | 支持 | 支持 | 支持 |
| 多输出/多目标 | MT 多输出树 | 无 | MultiRMSE/MultiLogloss 族 |

**排序：CatBoost 最广（类别+文本+嵌入+多目标），XGBoost 次之（稀疏+外存+MT），LightGBM 最简。**

### 1.3 训练算法与后端

- XGBoost：exact/approx/hist 三套 CPU + gpu_hist/gpu_approx + SYCL 实验 + 外部内存 + 多线程训练（MT）
- LightGBM：histogram + 独有 GOSS/EFB、4 种并行分裂（串行/特征/数据/投票）、CUDA/ROCm/OpenCL
- CatBoost：独有 ordered boosting + 对称树、MVS 采样、GPU 全算法

### 1.4 分布式与绑定

| 维度 | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 分布式 | 最强（自研 collective 层 + Spark/Flink jvm 包） | 中（network 自研 + dask.py） | 弱（private/libs/distributed，生态薄） |
| 语言绑定 | Python/R/JVM(Java/Scala/Spark/Flink)/C/C++ | Python/R/Java(SWIG)/C#/Go/C/C++ | Python/R/Java/C++/C（最窄） |

### 1.5 可解释性与工程闭环

- CatBoost 最全：SHAP 全家桶 + 内置特征选择 + 内置超参自动调优 + overfitting detector + 模型融合
- XGBoost/LightGBM：SHAP、单调/交互约束，调参选特征依赖外部 Optuna/featurewiz

### 1.6 模型格式

- XGBoost：json/ubj/binary 三格式，跨语言互操作最好
- CatBoost：专有 cbm + json + ONNX 导出
- LightGBM：文本 + json 配置

### 1.7 小结

**算法/数据/工程功能：CatBoost > XGBoost > LightGBM；生态/分布式/互操作：XGBoost > LightGBM > CatBoost；轻量与简单：LightGBM 最优。**

## 二、工程厚度对比

| 维度 | XGBoost | LightGBM | CatBoost（仅 catboost/ 本体） |
|---|---|---|---|
| 产品源码规模 | 76k 行 / 393 文件 | 57k 行 / 172 文件 | 239k 行 / 1,562 文件 |
| C++ 测试 | 197 文件 + 属性测试 | 15 文件 | 30 个 ut 目录 |
| Python 测试 | 66 文件 | 15 文件 | 51 文件 |
| 独特测试机制 | 分布式 GPU 测试 | — | canondata 规范输出快照比对（11 处） |
| 产品代码 fuzz | 无 | 无 | 无（仅 contrib/library 基础库） |
| CI | 17 workflows（ASan/UBSan 变体） | 14 workflows + windows VS 工程 | 复用式 action.yaml 矩阵 + 平台预生成 CMakeLists 15+ 份 |
| 代码质量工具 | .clang-format + .clang-tidy + pre-commit（三件套最全） | 仅 pre-commit | 无（靠内部 style guide + monorepo 同步元数据） |
| 构建系统 | CMake + 独有 amalgamation 单文件构建 | CMake + windows sln | 456 CMakeLists + 平台矩阵 + Arcadia ya.make 遗产 |
| 发布工程 | 无发布 workflow | 1 个 release workflow | 最全（build/check/publish + node 包发布） |
| 文档 | 77 文件（最厚） | 21 | 21 tutorials（主体在外部） |

**排序：CatBoost ≫ XGBoost > LightGBM。**
- CatBoost 厚度在规模与构建/发布体系（最"工业化"），但测试密度、开放质量工具最弱，靠内部规范与内部用户群兜底
- XGBoost 厚度在测试与质量工程（小代码大测试的正向样板）
- LightGBM 工程最轻，厚度体现在跨平台构建与 SWIG 多语言生态

**启示：代码量不该厚（对标 LightGBM 的轻），测试与工具链必须厚（对标 XGBoost 测试面 + CatBoost canondata 快照思路）——即"轻实现、重验证"。**

## 三、架构对比

### 3.1 各家架构形态

**XGBoost：插件化分层最干净**
- 分层：`include/xgboost` 29 个公共头（稳定契约）+ src 11 模块
- 扩展：31 处 `XGBOOST_REGISTER_*` 注册宏（编译期插件注册表，新增实现零侵入）
- 流水线：Learner → GBM → TreeUpdater 链式组合（`MapTreeMethodToUpdaters` 将 tree_method 映射为 `[grow_* → prune → refresh]`）
- 数据路径：DMatrix 抽象（Simple/SparsePage/Proxy）→ GradientIndex 量化索引 → Histogram
- 状态：Context 对象 147 处贯穿，无全局可变状态
- GPU：gpu_hist 作为同类 updater 注册（同构插件）；分布式 collective 独立层（自研替代 rabit）

**LightGBM：模板组合 + 工厂硬编码**
- 分层：include 17 公共头 + src 10 模块
- 分裂器架构：`CreateTreeLearner` 工厂硬编码列举 10+ 条 new 分支；Serial 为基类，Feature/Data/Voting 并行分裂器为模板包装（`FeatureParallelTreeLearner<SerialTreeLearner>`），Linear 再套一层，GPU 复制一套
- 数据路径：Dataset → 8-bit bin 压缩 → FeatureGroup（EFB 独有特征捆绑）→ Histogram
- 配置驱动：功能全由字符串参数经工厂分发

**CatBoost：选项模式 + 特征估计器管线（monorepo 最重）**
- 组织：catboost/libs（public）+ catboost/private/libs（32 模块）+ catboost/cuda + app CLI + bindings，public/private 双层依赖治理
- 选项 schema：private/libs/options 30+ 参数结构体，全库配置类型安全 + JSON 序列化
- 特征估计器管线（最大亮点）：feature_estimator（CTR/文本/嵌入三类估计器接口）→ quantized_pool → algo 训练
- GPU：catboost/cuda 完整双轨（与 CPU 并行实现）

### 3.2 对比总表

| 维度 | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| 分层清晰度 | 高（11 模块+稳定头） | 中（工厂集中） | 高但复杂（双层 32 模块） |
| 扩展机制 | 注册宏插件（31 处） | 工厂硬编码列举 | 选项 schema + 估计器接口 |
| 状态管理 | Context 贯穿，无全局态 | 全局 config（出过事） | 选项结构体传参 |
| 数据管线 | DMatrix→Index→Hist | Dataset→Bin→EFB→Hist | Pool→QuantPool→特征估计→algo |
| GPU 接入 | 同构 updater 插件 | 独立 src/cuda + 工厂分支 | cuda 双轨 |
| 分布式 | collective 独立层（最强） | network 模块 | 最弱 |
| 架构短板 | updater 副本多（族 1 根源） | treelearner 组合爆炸（10+ 分支） | CPU/GPU 双轨 + monorepo 复杂度 |

**一句话总结**：XGBoost 是"插件式"（每层可替换，代价是副本多）；LightGBM 是"模板组合式"（代码最省，代价是工厂爆炸）；CatBoost 是"选项驱动 + 管线式"（参数最安全、特征估计最通用，代价是双轨与复杂度）。

## 四、做对了什么、做错了什么（含对功能与打地鼠式 BUG 的影响）

### 4.1 XGBoost

**做对了**
1. 稳定公共契约：C API + 29 公共头锁定接口，功能扩展不破坏生态
2. 插件注册机制（31 处注册宏）：新 updater/objective 零侵入加入
3. Context 贯穿、无全局态：从机制上消灭一类全局串扰 bug
4. 测试最厚：197 C++ 测试 + 属性测试 + ASan/UBSan + 分布式 GPU 测试
5. 序列化三格式 + 版本字段：模型互操作最好
6. 独立 collective 层（自研替代 rabit）：分布式前瞻

**做错了**
1. updater 副本多：exact/approx/hist/gpu_hist/gpu_approx/SYCL，同一逻辑 N 份实现
2. 前端生态面膨胀：sklearn/dask/pandas/polars/arrow 跟随上游持续吃维护
3. 无 fuzz、无覆盖率守护；cache 归属反复搬迁（prediction cache 数次改挂载位置）

**影响**
- 功能：插件机制使功能面扩得最快最广（生存/多目标/新损失优先落地）
- 打地鼠：副本漂移是最大地鼠洞（族 1）；但厚测试把大多数地鼠拍死在 CI 里，漏到用户侧的相对少

### 4.2 LightGBM

**做对了**
1. 模板组合分裂器：并行变体代码量最小
2. 功能克制（57k 行最小）：面窄 → bug 面窄，维护压力全场最低
3. EFB/GOSS 独门算法，全局 bin 共享高效
4. 及时进入稳定期，打地鼠频率自然下降

**做错了**
1. treelearner 工厂硬编码列举：新增分裂器必须改工厂；GPU 再复制一套 → 组合爆炸，修复密度 60% 全场最高
2. 全局线程/配置状态：#4705/#5102 双 issue 修同一类
3. 测试最薄 + 无性能基准门禁：加权分位数静默错值两年内被抓两次
4. 无生存/多目标/文本功能，天花板低

**影响**
- 功能：克制换来稳定，但功能面自限，高级需求必须换库
- 打地鼠：地鼠集中在一个洞（treelearner），强度全场最高；C++ 测试薄，地鼠常漏到用户侧，靠 Python 测试 + issue 反哺兜底

### 4.3 CatBoost

**做对了**
1. 选项 schema 化：配置类 bug 结构性预防
2. 特征估计器管线：新特征类型有章法扩展，功能面做大不炸
3. canondata 快照比对：静默回归早期暴露的独特机制
4. 发布工程完整：build/check/publish 全流水线；内部 monorepo 海量使用打磨

**做错了**
1. CPU/GPU 全算法双轨复制：修复率 28% 居首（Tweedie 二阶导符号不一致等典型）
2. monorepo 遗产：复杂度负担 + 重导入制造 19,647 条历史噪声
3. 开放质量工具缺失：无 clang-tidy/pre-commit，sanitizer 未入 CI
4. 平台测试事后补齐：非主流平台问题在用户侧爆发

**影响**
- 功能：schema + 估计器管线使功能面全场最大（文本/嵌入/多目标/排序全谱）
- 打地鼠：双轨 GPU 是最大地鼠洞；canondata + 内部使用量把多数地鼠按在洞里；monorepo 复杂度制造大量低价值"文档/同步"类打地鼠，稀释维护注意力

### 4.4 核心洞察：决定打地鼠频率的不是功能多少，而是"新功能以什么姿势加入"

| | 扩展姿势 | 功能面 | 打地鼠特征 |
|---|---|---|---|
| XGBoost | 加实现（新 updater 副本） | 广 | 地鼠分散多洞，但 CI 兜住大部分 |
| LightGBM | 少加（克制） | 窄 | 地鼠集中一洞（treelearner），强度高、漏出多 |
| CatBoost | 加机制（估计器管线/schema） | 最广 | 结构性地鼠少，仅双轨 GPU 一个重灾区 |

- **做对的共性** = 用机制防错（稳定接口、schema、插件注册、快照测试）：机制替代人肉，地鼠无处可打
- **做错的共性** = 复制实现（updater 副本、treelearner 工厂、GPU 双轨）：复制是打地鼠的孵化器
- **对本项目最高优先级启示**：Rust 用 trait 注册表（抄 XGBoost）+ serde 选项 schema（抄 CatBoost）+ 单一内核泛型特化（消灭复制）——取三家之长、避三家之短
