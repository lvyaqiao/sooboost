# 05-BUG分族

本文件对 BUG 进行分族。BUG 中有单点考虑不周的，但更多是由于架构等结构性原因反复出现的。本文件根据导致成因对 BUG 分族，用于评估重构的必要性。内容分两个部分：一是具体的分族与 BUG 谱系，二是已有 BUG 的登记。每次只登记 BUG，不自动分族。

## 一、BUG 分族与谱系

### 分族判据

仅收录**由架构/结构原因导致、反复出现**的 BUG 族；判据为「架构层面是否有防止该类错误的机制」——无防错机制、且同类错误在仓库内或跨仓库（XGBoost / LightGBM / CatBoost）反复发生者入族；偶发考虑不周（单点逻辑遗漏）不入族。

### 七族谱系

#### 族 1：多实现副本漂移（Copy-Paste Multiplier）

- 成因：同一算法逻辑（分裂搜索 / 直方图构建 / 缺失值处理 / objective）因性能特化在 CPU、GPU、多线程变体、exact/approx 等多套实现中复制，无单一事实来源，修复一处后其他副本漏修。
- 亚型：
  - CPU/GPU 行为不一致（Tweedie 二阶导符号、预测结果差异）
  - 并行/串行分裂器结果不一致
  - exact 与 approx 分裂结果漂移
- 典型证据（git 日志）：
  - LightGBM `src/treelearner` 目录 262 条提交中 158 条为修复类（60%，全场最高）；6 套分裂器（串行/3 种并行/GPU/linear）相互复制；"also fix / same issue / consistency" 类同步补丁 19 条
  - XGBoost 39 条 CPU/GPU 一致性相关提交，近期才有 "Share the model view for CPU and GPU"（#11759）、"Merge approx tests"（#10583），此前分离维护
  - CatBoost GPU 路径全史修复率 28% 居首；`catboost/cuda` 出现 use-after-free（返回临时字符串指针）
- 代价：修复放大（1 处 bug → N 处补丁）、漏修不对称、行为漂移。

#### 族 2：手写缓存与状态失效（Stale Cache & Invalidation）

- 成因：为性能手写直方图缓存、prediction cache、RowSet 等，无统一失效机制，失效条件与消费方假设隐式耦合；缓存生命周期跨语言边界（pickle / C API）出错。
- 亚型：
  - 缓存归属/生命周期错误（GC 后悬垂、pickle 后错位）
  - 缓存未随参数/数据失效，命中旧值
- 典型证据：
  - XGBoost 108 条 cache 相关提交中 17 条为修复；prediction cache 归属反复搬迁（"Move prediction cache to Learner" #5220、#5086、#3217 "Fix logic in GPU predictor cache lookup"）
  - CatBoost cache 相关 132 条
- 代价：**静默错值**（用旧数据不报错），最危险的一族。

#### 族 3：近似算法与确定性（Approximation & Determinism）

- 成因：quantile sketch、分箱、加权分位数的精度与可复现性依赖采样种子、并行聚合顺序、实现细节；分布式/多线程下结果不可复现，参数微调精度变化。
- 亚型：
  - 加权分位数计算错误（静默）
  - 分布式 sketch 结果不具确定性 / 内存估计失准
  - 并行聚合顺序导致浮点结果漂移
- 典型证据：
  - XGBoost quantile 108 条、sketch 57 条提交
  - LightGBM 加权分位数同源问题两年内修复 2 次（#5848 → #7224，`ceff20a0` 修 weighted percentiles）
- 代价：结果微偏且静默、难以测试，依赖用户 issue 反哺发现。

#### 族 4：缺失值/NaN 语义链路（Missing Semantics Chain）

- 成因：缺失值方向语义贯穿 分箱 → 直方图 → 分裂 → 预测 全链路，各层独立实现、未显式化契约；与稀疏表示交互后漂移，训练/预测不一致。
- 亚型：
  - 训练与预测缺失值行为不一致
  - NaN 与稀疏零值混淆
  - 缺失值处理中的未定义行为（UB）
- 典型证据：三家合计 700+ 条 missing/nan 相关提交（XGBoost 153、LightGBM 89、CatBoost 505），多为静默错误或 UB 类。
- 代价：线上分数静默错误。

#### 族 5：全局可变状态与并发（Global Mutable State）

- 成因：OpenMP 全局线程数、静态 buffer、全局随机引擎、全局配置在库内共享，多实例/嵌套并行/重复调用时串扰。
- 亚型：
  - 全局线程数/线程绑定被其他库或嵌套调用覆盖
  - 并行区共享静态缓冲区的 data race
  - 修了又回滚（revert 循环）
- 典型证据：
  - LightGBM 同一类全局线程控制问题双 issue（#4705/#5102）
  - CatBoost threadbinding 相关 28 条、thread+fix 100 条，且存在修后回滚
  - XGBoost race 21 条、thread 126 条
- 代价：偶发不可复现，TSan 才能抓到的 data race。

#### 族 6：序列化与模型格式版本化（Serialization & Versioning）

- 成因：模型/数据格式演进缺乏显式版本契约，新旧版本混读、跨语言边界（pickle / json / binary / C API）类型映射各自为政。
- 亚型：
  - 模型加载路径静默损坏（零拷贝加载）
  - 版本字段缺失导致跨版本不兼容
  - 属性/特征名等元数据序列化丢失
- 典型证据：CatBoost 零拷贝加载错误静默传播为模型损坏；LightGBM 141 条 revert 中大量与格式/兼容相关；XGBoost serializ/version/compatib 66 条。
- 代价：训练产物不可恢复、线上加载失败，事故级别最高。

#### 族 7：接口与生态面（Binding & Platform Matrix）

- 成因：每加一个语言绑定（Python / R / Java / SWIG）即新增一套类型映射与资源生命周期面；平台×编译器×CUDA 版本组合爆炸，构建/CI 修复占用大量维护带宽。
- 亚型：
  - 上游生态（sklearn / numpy / pandas / arrow）版本更新引发的兼容修复
  - 跨语言资源泄漏 / 段错误
  - 平台（Windows/MSVC、ARM64、macOS）特有编译与运行问题
- 典型证据：LightGBM API 兼容类修复占比 ~31%；XGBoost 构建/CI ~30%、前端兼容 ~14%；CatBoost sklearn tags→sklearn_tags 两代 API 迁移。LightGBM `60b0155a`（1 棵树模型 predict 输出形状不一致）即绑定层语义漂移实例。
- 代价：消耗最多维护带宽，基本无算法价值。

### 结构判断

- 前 4 族（多副本、手写缓存、近似确定性、缺失值链路）为 **GBDT 领域结构性特征**，共同根源是：**性能特化迫使同一逻辑复制多份 + 语义契约散落各层未显式化**。
- 族 5~7（并发全局态、序列化、绑定/平台）为大型 C++ 库通病，非 GBDT 特有。
- 对本项目（纯 Rust 单一内核多后端）的启示：族 1 靠泛型/特质消除复制、族 2 靠单一失效点、族 3 靠确定性测试前置、族 4 靠 arrow 原生 null 语义一次性定义。

## 二、BUG 登记

（空白）

## 二、BUG 登记

（空白）
