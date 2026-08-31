"""
sooboost 金标准 Benchmark v1
============================
三种基准，各司其职（对应设计批评的 P0-3 落地）：

- quality（质量基准）   ：RandomForest 作为通用基线，HistGradientBoosting 作为更接近
                          sooboost 的 GBDT（直方图）基线。目标措辞：在相同数据与预算下
                          达到目标指标，而非"必须全面超过 RandomForest"。
- correctness（正确性基准）：固定数据、固定 seed、固定超参 → 固定预测与模型快照，
                          供 sooboost 实现后做对齐验证（指标与预测分布，非逐位）。
- perf（性能基准）     ：训练/预测耗时重复多次取统计值（均值/min/max/std）。
                         单次运行时间不作为稳定结论。

环境信息（sklearn/Python/numpy/pandas/CPU）写入 benchmark/environment.json；
数据文件 sha256 写入各数据集 dataset_meta.json（california_housing 依赖网络下载，
无官方 checksum，以本地生成时间为准）。

用法：
    python benchmark/run_benchmark.py                # 默认 quality
    python benchmark/run_benchmark.py --mode correctness
    python benchmark/run_benchmark.py --mode perf --repeats 10
    python benchmark/run_benchmark.py --mode gen
    python benchmark/run_benchmark.py --all          # 三档全跑

输出目录结构：
    benchmark/
    ├── environment.json
    ├── <dataset>/
    │   ├── train.csv / test.csv
    │   ├── dataset_meta.json        # 来源 + 任务 + checksum
    │   ├── <model>_predictions.csv
    │   ├── <model>_params.json
    │   └── metrics.json             # quality/correctness 档
    └── <dataset>/perf.json          # perf 档
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.datasets import (
    fetch_california_housing,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
PERF_REPEATS = 5
# Point-imputation RMSE uses a deterministic mean of several conditional draws.
IMPUTATION_SAMPLES = 4
OUTPUT_DIR = pathlib.Path(__file__).parent
GATE_FAILURES: list[str] = []

HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "random_state": SEED,
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 5,
    "random_state": SEED,
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path, index=False)
    print(f"  -> {path.relative_to(OUTPUT_DIR)}  ({len(df)} rows)")


def save_json(obj, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  -> {path.relative_to(OUTPUT_DIR)}")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj):
    """递归清洗 numpy 类型，保证 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_binary(y_true, y_prob, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def collect_environment() -> dict:
    import numpy
    import pandas

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


# ---------------------------------------------------------------------------
# 数据集定义（来源 / 任务 / 生成函数）
# ---------------------------------------------------------------------------

def _make_synthetic_regression():
    X, y = make_regression(n_samples=2000, n_features=10, n_informative=7,
                           noise=10.0, random_state=SEED)
    return X, y


def _make_synthetic_regression_nonlinear():
    # Friedman#1：非线性（乘积/幂交互），是树模型的有力测试，弥补线性 make_regression 的不足
    X, y = make_friedman1(n_samples=2000, n_features=10, noise=5.0,
                          random_state=SEED)
    return X, y


def _make_synthetic_binary():
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=7,
                               n_redundant=2, n_clusters_per_class=2,
                               flip_y=0.05, random_state=SEED)
    return X, y


def _make_california_housing():
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy()
    y = data.target.to_numpy() * 100000  # 转为美元
    return X, y


DATASETS = [
    {
        "name": "synthetic_regression",
        "task": "regression",
        "source": "sklearn.datasets.make_regression(n_samples=2000, n_features=10, n_informative=7, noise=10.0, random_state=42)",
        "gen": _make_synthetic_regression,
    },
    {
        "name": "synthetic_regression_nonlinear",
        "task": "regression",
        "source": "sklearn.datasets.make_friedman1(n_samples=2000, n_features=10, noise=5.0, random_state=42)",
        "gen": _make_synthetic_regression_nonlinear,
    },
    {
        "name": "synthetic_binary",
        "task": "binary",
        "source": "sklearn.datasets.make_classification(n_samples=2000, n_features=10, n_informative=7, n_redundant=2, n_clusters_per_class=2, flip_y=0.05, random_state=42)",
        "gen": _make_synthetic_binary,
    },
    {
        "name": "california_housing",
        "task": "regression",
        "source": "sklearn.datasets.fetch_california_housing（依赖网络下载，无官方 checksum；目标 *100000 转美元；版本以 environment.json 生成时间为准）",
        "gen": _make_california_housing,
    },
]


def _feature_cols(n_features: int) -> list[str]:
    return [f"f{i}" for i in range(n_features)]


def prepare_dataset(ds: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成数据 → 固定切分 → 写 train/test CSV + dataset_meta.json（含 checksum）。"""
    X, y = ds["gen"]()
    cols = _feature_cols(X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    out_dir = OUTPUT_DIR / ds["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=cols)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test, columns=cols)
    test_df["target"] = y_test

    train_path, test_path = out_dir / "train.csv", out_dir / "test.csv"
    save_csv(train_df, train_path)
    save_csv(test_df, test_path)

    meta = {
        "name": ds["name"],
        "task": ds["task"],
        "source": ds["source"],
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_csv_sha256": sha256_file(train_path),
        "test_csv_sha256": sha256_file(test_path),
    }
    save_json(meta, out_dir / "dataset_meta.json")
    return ds, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# quality 档：RF（通用基线）+ HistGB（GBDT 基线）
# ---------------------------------------------------------------------------

def run_quality(ds, X_train, X_test, y_train, y_test) -> dict:
    task = ds["task"]
    out_dir = OUTPUT_DIR / ds["name"]
    print(f"\n[quality] {ds['name']}")

    baselines = [
        ("random_forest", RandomForestRegressor if task == "regression" else RandomForestClassifier, RF_PARAMS),
        ("hist_gradient_boosting", HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier, HGB_PARAMS),
    ]

    results = {}
    for model_name, ModelClass, params in baselines:
        print(f"  Training {model_name} ...")
        model = ModelClass(**params)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        if task == "regression":
            y_pred = model.predict(X_test)
            metrics = evaluate_regression(y_test, y_pred)
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = evaluate_binary(y_test, y_prob, y_pred)
        predict_time = time.perf_counter() - t0

        metrics["train_time_sec"] = round(train_time, 4)
        metrics["predict_time_sec"] = round(predict_time, 4)

        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        if task == "binary":
            pred_df["y_prob"] = y_prob
        save_csv(pred_df, out_dir / f"{model_name}_predictions.csv")
        save_json(json_safe(model.get_params()), out_dir / f"{model_name}_params.json")

        results[model_name] = metrics
        print(f"    metrics: {metrics}")

    save_json(json_safe(results), out_dir / "metrics.json")
    return results


# ---------------------------------------------------------------------------
# correctness 档：固定 seed 的 HistGB，输出固定预测 + 模型快照
# ---------------------------------------------------------------------------

def run_correctness(ds, X_train, X_test, y_train, y_test) -> dict:
    task = ds["task"]
    out_dir = OUTPUT_DIR / ds["name"]
    print(f"\n[correctness] {ds['name']}")

    ModelClass = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    model = ModelClass(**HGB_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task == "regression":
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_binary(y_test, y_prob, y_pred)
    predict_time = time.perf_counter() - t0

    metrics["train_time_sec"] = round(train_time, 4)
    metrics["predict_time_sec"] = round(predict_time, 4)

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if task == "binary":
        pred_df["y_prob"] = y_prob
    save_csv(pred_df, out_dir / "correctness_predictions.csv")
    save_json(json_safe(HGB_PARAMS), out_dir / "correctness_params.json")
    save_json(json_safe(metrics), out_dir / "correctness_metrics.json")

    # 模型快照（joblib；仅同环境可复用，跨环境不稳定——不作为长期格式）
    try:
        import joblib
        joblib.dump(model, out_dir / "correctness_model.joblib")
        print(f"  -> {pathlib.Path(out_dir).name}/correctness_model.joblib")
    except ImportError:
        print("  (joblib 不可用，跳过模型快照)")

    print(f"    metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# sooboost 档：调 CLI 训练 + 预测，输出预测并与金标准对比
# ---------------------------------------------------------------------------

SOOBOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "min_split_gain": 0.0,
    "max_bins": 255,
    "seed": 42,
}


def find_sooboost_cli() -> pathlib.Path:
    env = os.environ.get("SOOBOOST_CLI")
    if env:
        return pathlib.Path(env)
    root = OUTPUT_DIR.parent
    for rel in ("target/release/sooboost-cli", "target/debug/sooboost-cli"):
        for suffix in ("", ".exe"):
            p = root / (rel + suffix)
            if p.exists():
                return p
    raise SystemExit("找不到 sooboost-cli 二进制（设置 SOOBOOST_CLI 或先 cargo build --release -p sooboost-cli）")


def run_sooboost(ds, X_train, X_test, y_train, y_test, gate: bool = False) -> dict:
    """调 sooboost-cli train → 读 predictions → 算指标 → 与金标准对比。

    gate=True 时落后金标准 > 0.05 记入失败（供门禁脚本判定退出码）。
    """
    task = ds["task"]
    out_dir = OUTPUT_DIR / ds["name"]
    cli = find_sooboost_cli()
    pred_path = out_dir / "sooboost_predictions.csv"
    print(f"\n[sooboost] {ds['name']}  cli={cli}")

    features = ",".join(_feature_cols(X_train.shape[1]))
    cmd = [
        str(cli), "train",
        "--train", str(out_dir / "train.csv"),
        "--test", str(out_dir / "test.csv"),
        "--features", features,
        "--target", "target",
        "--task", task,
        "--output", str(pred_path),
    ]
    for k, v in SOOBOOST_PARAMS.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    total_time = time.perf_counter() - t0

    pred_df = pd.read_csv(pred_path)
    assert len(pred_df) == len(y_test), f"预测行数 {len(pred_df)} != 测试行数 {len(y_test)}"

    if task == "regression":
        y_pred = pred_df["y_pred"].to_numpy()
        metrics = evaluate_regression(y_test, y_pred)
    else:
        y_prob = pred_df["y_prob"].to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_binary(y_test, y_prob, y_pred)

    metrics["total_time_sec"] = round(total_time, 4)
    save_json(json_safe(metrics), out_dir / "sooboost_metrics.json")
    save_json(json_safe(SOOBOOST_PARAMS), out_dir / "sooboost_params.json")

    # 与金标准对比
    try:
        gold = json.loads((out_dir / "correctness_metrics.json").read_text(encoding="utf-8"))
        print(f"  sooboost: {metrics}")
        print(f"  gold HGB: {gold}")
        key = "r2" if task == "regression" else "auc"
        gap = metrics[key] - gold[key]
        ok = gap >= -0.05
        verdict = "OK" if ok else "注意：落后金标准 >0.05"
        print(f"  对齐 ({key}): sooboost={metrics[key]:.4f}  gold={gold[key]:.4f}  差={gap:+.4f}  {verdict}")
        if gate and not ok:
            GATE_FAILURES.append(f"{ds['name']}: {key} 差 {gap:+.4f} 落后金标准 >0.05")
    except FileNotFoundError:
        print("  (缺 correctness_metrics.json，跳过对比)")
    return metrics


# ---------------------------------------------------------------------------
# perf 档：HistGB 重复多次，统计训练/预测耗时
# ---------------------------------------------------------------------------

def run_perf(ds, X_train, X_test, y_train, y_test, repeats: int) -> dict:
    task = ds["task"]
    out_dir = OUTPUT_DIR / ds["name"]
    print(f"\n[perf] {ds['name']} (repeats={repeats})")

    ModelClass = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    train_times, predict_times = [], []

    for i in range(repeats):
        model = ModelClass(**HGB_PARAMS)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.predict(X_test)
        predict_times.append(time.perf_counter() - t0)

    def stats(vals: list[float]) -> dict:
        arr = np.array(vals)
        return {
            "mean_sec": round(float(arr.mean()), 4),
            "min_sec": round(float(arr.min()), 4),
            "max_sec": round(float(arr.max()), 4),
            "std_sec": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "raw_sec": [round(v, 4) for v in vals],
        }

    result = {
        "model": "hist_gradient_boosting",
        "params": HGB_PARAMS,
        "repeats": repeats,
        "train_time": stats(train_times),
        "predict_time": stats(predict_times),
        "note": "单机单次时间受负载影响；内存占用与模型大小未测量（v1 不含）",
    }
    save_json(json_safe(result), out_dir / "perf.json")
    print(f"    train: {result['train_time']}")
    print(f"    predict: {result['predict_time']}")
    return result


# ---------------------------------------------------------------------------
# gen 档：ForestFlow 生成/填补质量（M2-D）
# ---------------------------------------------------------------------------

def find_experiments_cli() -> pathlib.Path:
    env = os.environ.get("SOOBOOST_EXPERIMENTS")
    if env:
        return pathlib.Path(env)
    root = OUTPUT_DIR.parent
    for rel in ("target/release/sooboost-experiments", "target/debug/sooboost-experiments"):
        for suffix in ("", ".exe"):
            path = root / (rel + suffix)
            if path.exists():
                return path
    raise SystemExit(
        "找不到 sooboost-experiments 二进制（设置 SOOBOOST_EXPERIMENTS 或先 "
        "cargo build --release -p sooboost-experiments）"
    )


def _corr_matrix(values: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _off_diagonal_mean_abs(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def evaluate_generation(real: np.ndarray, generated: np.ndarray) -> dict:
    """生成质量的轻量结构指标；不把单一 C2ST 当作充分统计量。"""
    real_mean = np.mean(real, axis=0)
    generated_mean = np.mean(generated, axis=0)
    real_std = np.std(real, axis=0)
    generated_std = np.std(generated, axis=0)
    scale = np.maximum(real_std, 1e-12)
    real_corr = _corr_matrix(real)
    generated_corr = _corr_matrix(generated)
    corr_mae = float(np.mean(np.abs(real_corr - generated_corr)))

    combined = np.vstack([real, generated])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(generated))])
    x_train, x_test, y_train, y_test = train_test_split(
        combined, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    c2st = HistGradientBoostingClassifier(
        max_iter=50, max_leaf_nodes=15, random_state=SEED
    )
    c2st.fit(x_train, y_train)
    c2st_auc = float(roc_auc_score(y_test, c2st.predict_proba(x_test)[:, 1]))

    return {
        "n_real": int(len(real)),
        "n_generated": int(len(generated)),
        "mean_relative_error": float(np.mean(np.abs(real_mean - generated_mean) / scale)),
        "std_relative_error": float(np.mean(np.abs(real_std - generated_std) / scale)),
        "correlation_mae": corr_mae,
        "real_offdiag_abs_corr": _off_diagonal_mean_abs(real_corr),
        "generated_offdiag_abs_corr": _off_diagonal_mean_abs(generated_corr),
        "c2st_auc": c2st_auc,
    }


def _imputation_feature(X_train: np.ndarray) -> int:
    corr = np.abs(_corr_matrix(X_train))
    np.fill_diagonal(corr, 0.0)
    return int(np.argmax(np.mean(corr, axis=1)))


def run_gen(ds, X_train, X_test, y_train, y_test, gate: bool = False) -> dict:
    """调用 ForestFlow，评估生成边际/相关性/C2ST 与点填补 RMSE。"""
    del y_train, y_test
    out_dir = OUTPUT_DIR / ds["name"]
    cli = find_experiments_cli()
    features = ",".join(_feature_cols(X_train.shape[1]))
    count = min(400, len(X_test))
    common = [
        str(cli),
        "--train", str(out_dir / "train.csv"),
        "--features", features,
        "--target", "target",
        "--seed", str(SEED),
        "--steps", "20",
    ]
    generated_path = out_dir / "forest_flow_generated.csv"
    print(f"\n[gen] {ds['name']}  cli={cli}")
    subprocess.run(
        common
        + [
            "--mode", "generate",
            "--count", str(count),
            "--output", str(generated_path),
        ],
        check=True,
    )
    generated = pd.read_csv(generated_path).to_numpy(dtype=float)
    metrics = evaluate_generation(X_test[:count], generated)

    train_corr = np.abs(_corr_matrix(X_train))
    np.fill_diagonal(train_corr, 0.0)
    feature_index = _imputation_feature(X_train)
    dependence_signal = float(np.max(train_corr)) if train_corr.size else 0.0
    imputation_path = out_dir / "forest_flow_imputation.csv"
    subprocess.run(
        common
        + [
            "--mode", "impute",
            "--test", str(out_dir / "test.csv"),
            "--feature-index", str(feature_index),
            "--imputation-samples", str(IMPUTATION_SAMPLES),
            "--output", str(imputation_path),
        ],
        check=True,
    )
    imputation = pd.read_csv(imputation_path)
    actual = imputation["actual"].to_numpy(dtype=float)
    predicted = imputation["imputed"].to_numpy(dtype=float)
    baseline = float(np.mean(X_train[:, feature_index]))
    flow_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, np.full_like(actual, baseline))))
    metrics.update(
        {
            "imputation_feature": feature_index,
            "max_train_pair_abs_corr": dependence_signal,
            "imputation_rmse": flow_rmse,
            "imputation_baseline_rmse": baseline_rmse,
            "imputation_relative_gain": 1.0 - flow_rmse / max(baseline_rmse, 1e-12),
            "imputation_samples": IMPUTATION_SAMPLES,
        }
    )
    save_json(json_safe(metrics), out_dir / "forest_flow_metrics.json")
    print(f"  generation: {metrics}")

    finite = bool(np.isfinite(generated).all() and np.isfinite(predicted).all())
    imputation_ok = (
        metrics["imputation_relative_gain"] >= 0.05
        if dependence_signal >= 0.25
        else flow_rmse <= baseline_rmse * 1.5
    )
    quality_ok = (
        finite
        and metrics["mean_relative_error"] <= 2.0
        and metrics["std_relative_error"] <= 5.0
        and metrics["correlation_mae"] <= 0.75
        and metrics["c2st_auc"] <= 0.9
        and imputation_ok
    )
    if gate and not quality_ok:
        GATE_FAILURES.append(
            f"{ds['name']}: gen quality failed "
            f"(finite={finite}, mean_err={metrics['mean_relative_error']:.4f}, "
            f"std_err={metrics['std_relative_error']:.4f}, "
            f"corr_mae={metrics['correlation_mae']:.4f}, "
            f"c2st_auc={metrics['c2st_auc']:.4f}, "
            f"imputation_gain={metrics['imputation_relative_gain']:+.4f}, "
            f"signal={dependence_signal:.4f})"
        )
    return metrics


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="sooboost 金标准 Benchmark v1")
    parser.add_argument("--mode", choices=["quality", "correctness", "perf", "sooboost", "gen"],
                        default="quality", help="运行档位（默认 quality）")
    parser.add_argument("--all", action="store_true", help="三档全跑")
    parser.add_argument("--gate", action="store_true",
                        help="sooboost/gen 档严格模式：质量不达标时退出码 1（基准门禁）")
    parser.add_argument("--repeats", type=int, default=PERF_REPEATS,
                        help=f"perf 档重复次数（默认 {PERF_REPEATS}）")
    args = parser.parse_args()
    modes = ["quality", "correctness", "perf"] if args.all else [args.mode]

    print("=" * 64)
    print("sooboost 金标准 Benchmark v1")
    print(f"modes: {', '.join(modes)}")
    print("=" * 64)

    save_json(json_safe(collect_environment()), OUTPUT_DIR / "environment.json")

    prepared = [prepare_dataset(ds) for ds in DATASETS]

    for mode in modes:
        for ds, X_train, X_test, y_train, y_test in prepared:
            if mode == "quality":
                run_quality(ds, X_train, X_test, y_train, y_test)
            elif mode == "correctness":
                run_correctness(ds, X_train, X_test, y_train, y_test)
            elif mode == "sooboost":
                run_sooboost(ds, X_train, X_test, y_train, y_test, gate=args.gate)
            elif mode == "gen":
                run_gen(ds, X_train, X_test, y_train, y_test, gate=args.gate)
            else:
                run_perf(ds, X_train, X_test, y_train, y_test, args.repeats)

    if args.gate and GATE_FAILURES:
        print("\n[gate] 基准门禁失败：")
        for f in GATE_FAILURES:
            print(f"  - {f}")
        sys.exit(1)

    print("\nDone. 输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
