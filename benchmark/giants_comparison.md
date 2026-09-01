# sooboost vs 三巨头：真实数据集对比

- 统一预算：n_estimators=200，learning_rate=0.1，seed=42；深度/叶子取各库默认或等价容量（xgb depth6 / lgbm num_leaves31 / catboost depth6 / sooboost depth6 / hgb leaves31）
- 版本：sooboost 0.1.0(local), xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, sklearn 1.9.0
- 训练时间含数据读入与库初始化（sooboost 为 CLI 进程冷启动，含进程开销）

## R²（回归）

| 数据集 | sooboost | xgboost | lightgbm | catboost | sklearn_hgb |
|---|---|---|---|---|---|
| california_housing | 0.8403 | 0.8410 | **0.8466** | 0.8243 | 0.8427 |
| diabetes | 0.3877 | 0.3594 | 0.3494 | **0.4521** | 0.3408 |

## AUC（二分类）

| 数据集 | sooboost | xgboost | lightgbm | catboost | sklearn_hgb |
|---|---|---|---|---|---|
| breast_cancer | 0.9950 | 0.9937 | 0.9891 | **0.9970** | 0.9904 |

## Accuracy（二分类）

| 数据集 | sooboost | xgboost | lightgbm | catboost | sklearn_hgb |
|---|---|---|---|---|---|
| breast_cancer | 0.9649 | 0.9474 | 0.9561 | **0.9737** | 0.9649 |

## 训练耗时

| 数据集 | sooboost | xgboost | lightgbm | catboost | sklearn_hgb |
|---|---|---|---|---|---|
| california_housing | 1.32s | 0.39s | 0.14s | 0.73s | 2.71s |
| diabetes | 0.21s | 0.09s | 0.03s | 0.22s | 0.20s |
| breast_cancer | 0.35s | 0.12s | 0.09s | 1.00s | 0.25s |
