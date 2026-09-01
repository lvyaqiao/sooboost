/*
 * sooboost C ABI 头文件（M12：嵌入接口）。
 *
 * 与 sooboost-ffi crate 的 #[no_mangle] extern "C" 函数一一对应。
 * 约定：
 * - 状态码：0（或 ≥0 的长度语义）= 成功；-1 = 失败，错误信息经
 *   sbs_last_error 取线程局部缓冲（UTF-8，不跨线程）。
 * - 模型句柄 SbsModel* 由 sbs_train / sbs_deserialize 产出，
 *   必须且只能经 sbs_model_free 释放一次；释放后不得复用。
 * - 数据布局：行主序 data[row * n_features + feature]；缺失 = NaN。
 * - 序列化两段式：先以 out=NULL, cap=0 探测所需长度，再分配重调。
 * - 本头文件不承载 ABI 之外的行为承诺；语义以 sooboost-core 文档为准。
 */
#ifndef SOOBOOST_H
#define SOOBOOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 不透明模型句柄。 */
typedef struct SbsModel SbsModel;

/* 版本字符串（如 "sooboost-ffi 0.2.0"）；静态存储期，调用方不得释放。 */
const char* sbs_version(void);

/*
 * 训练模型。
 *   params_json : NUL 结尾 UTF-8 JSON，字段（均可选）：
 *     task            "regression"(默认) | "binary" | "multiclass"
 *     n_classes       multiclass 必填，≥2
 *     n_estimators    树轮数（默认 100）
 *     learning_rate   学习率（默认 0.1）
 *     max_depth       树最大深度（根=0）
 *     min_samples_leaf / min_split_gain / reg_lambda
 *     max_bins        分箱上限
 *     max_categories  类别基数上限
 *     categorical_alpha  ordered TS smoothing α
 *     seed            确定性种子（默认 0）
 *     未知字段 → 报错（不静默忽略）。
 *   data        : 行主序 n_rows × n_features（缺失 = NaN）
 *   labels      : 长 n_rows
 *   out_model   : 成功时写入新句柄
 * 返回 0 = 成功；-1 = 失败（见 sbs_last_error）。
 */
int32_t sbs_train(const char* params_json,
                  const double* data, int64_t n_rows, int64_t n_features,
                  const double* labels,
                  SbsModel** out_model);

/*
 * 批量预测：回归 → 原值；二分类 → 正类概率；多分类 → argmax 类别。
 * out 至少 out_cap 个 double（out_cap ≥ n_rows）。
 * 返回写入元素数（= n_rows）；失败返回 -1。
 */
int64_t sbs_predict(const SbsModel* model, const double* data,
                    int64_t n_rows, int64_t n_features,
                    double* out, int64_t out_cap);

/*
 * 批量概率预测：二分类 → 正类概率（n 个）；多分类 → 行主序 n×k
 * 概率矩阵；回归显式报错。k 用 sbs_model_num_classes 查询。
 * 返回写入元素数；失败返回 -1。
 */
int64_t sbs_predict_proba(const SbsModel* model, const double* data,
                          int64_t n_rows, int64_t n_features,
                          double* out, int64_t out_cap);

/* 模型特征数；无效句柄返回 -1。 */
int64_t sbs_model_num_features(const SbsModel* model);

/* 模型类别数：多分类为 k；回归/二分类为 -1。 */
int64_t sbs_model_num_classes(const SbsModel* model);

/* 模型树棵数（多分类为每类棵数）；无效句柄返回 -1。 */
int64_t sbs_model_num_trees(const SbsModel* model);

/*
 * 序列化为字节（sooboost v4 模型格式，与 Rust save/load 完全互通）。
 *   out=NULL, cap=0     → 探测模式，返回所需字节数（≥0）
 *   out 非空且 cap 足够 → 写入并返回实际字节数
 *   cap 不足 / 句柄无效 → -1
 */
int64_t sbs_serialize(const SbsModel* model, uint8_t* out, int64_t cap);

/* 由字节恢复模型（目标自动探测）。返回 0 = 成功；-1 = 字节非法。 */
int32_t sbs_deserialize(const uint8_t* bytes, int64_t len,
                        SbsModel** out_model);

/* 释放模型句柄（NULL 安全；释放后不得复用）。 */
void sbs_model_free(SbsModel* model);

/*
 * 取线程局部 last-error（UTF-8，NUL 结尾，超长截断到 cap）。
 * 无错误时写入空串。恒返回 0。
 */
int32_t sbs_last_error(char* out, int64_t cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SOOBOOST_H */
