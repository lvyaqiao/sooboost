# -*- coding: utf-8 -*-
"""sooboost vs 三巨头（XGBoost / LightGBM / CatBoost）真实数据集对比。

设计原则（与 run_benchmark.py 同口径）：
- 统一预算：n_estimators=200，learning_rate=0.1，seed=42；深度取各库默认/等价容量。
- 真实数据集：california_housing（回归，缓存在 benchmark/california_housing）、
  diabetes（回归）、breast_cancer（二分类）——均为真实观测数据，非合成。
- sooboost 经 sooboost-cli 调用（与 CI gate 同一路径），其余经各自 Python API。
- 指标：回归 r2；二分类 auc + accuracy。

用法（先 cargo build --release -p sooboost-cli）:
    python benchmark/compare_giants.py            # 全部运行
    python benchmark/compare_giants.py --report   # 仅从 JSON 生成 Markdown
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
N_TREES = 200
LEARNING_RATE = 0.1
OUTPUT_DIR = pathlib.Path(__file__).parent
OUT_JSON = OUTPUT_DIR / "giants_comparison.json"

FEATURES = "target"


def find_sooboost_cli() -> pathlib.Path:
    import os

    env = os.environ.get("SOOBOOST_CLI")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p
    exe = pathlib.Path(__file__).parent.parent / "target" / "release" / "sooboost-cli.exe"
    if exe.exists():
        return exe
    raise SystemExit("找不到 sooboost-cli（先 cargo build --release -p sooboost-cli 或设 SOOBOOST_CLI）")


def _split(X, y, stratify=None):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify)


def _as_frame(X, y, feature_names):
    df = pd.DataFrame(np.asarray(X), columns=list(feature_names))
    df["target"] = np.asarray(y, dtype=float)
    return df


def _write_csv(df: pd.DataFrame, path: pathlib.Path):
    df.to_csv(path, index=False)


def _feature_cols(n: int):
    return [f"feature_{i}" for i in range(n)]


# ---------------------------------------------------------------- datasets
def load_datasets() -> dict:
    """返回 {name: dict(task, df, feature_cols, stratify)}。"""
    ds = {}

    # california_housing（回归，复用已缓存 CSV；缺失则现场划分）
    cal_dir = OUTPUT_DIR / "california_housing"
    if (cal_dir / "train.csv").exists() and (cal_dir / "test.csv").exists():
        train = pd.read_csv(cal_dir / "train.csv")
        test = pd.read_csv(cal_dir / "test.csv")
        feats = [c for c in train.columns if c != "target"]
        ds["california_housing"] = {
            "task": "regression",
            "train": train,
            "test": test,
            "features": feats,
        }

    # diabetes（回归，真实：442 名患者 10 项基线指标 → 疾病进展）
    d = load_diabetes()
    Xtr, Xte, ytr, yte = _split(d.data, d.target)
    feats = _feature_cols(d.data.shape[1])
    ds["diabetes"] = {
        "task": "regression",
        "train": _as_frame(Xtr, ytr, feats),
        "test": _as_frame(Xte, yte, feats),
        "features": feats,
    }

    # breast_cancer（二分类，真实：569 例肿瘤 30 项细胞核特征 → 良/恶性）
    b = load_breast_cancer()
    Xtr, Xte, ytr, yte = _split(b.data, b.target, stratify=b.target)
    feats = _feature_cols(b.data.shape[1])
    ds["breast_cancer"] = {
        "task": "binary",
        "train": _as_frame(Xtr, ytr, feats),
        "test": _as_frame(Xte, yte, feats),
        "features": feats,
    }
    return ds


# ---------------------------------------------------------------- models
def run_sooboost(name, ds, workdir: pathlib.Path | None = None) -> dict:
    cli = find_sooboost_cli()
    task = ds["task"]
    workdir = workdir or (OUTPUT_DIR / "giants_work" / name)
    workdir.mkdir(parents=True, exist_ok=True)
    train_csv, test_csv = workdir / "train.csv", workdir / "test.csv"
    _write_csv(ds["train"], train_csv)
    _write_csv(ds["test"], test_csv)
    pred_csv = workdir / "sooboost_predictions.csv"
    cmd = [
        str(cli), "train",
        "--train", str(train_csv),
        "--test", str(test_csv),
        "--features", ",".join(ds["features"]),
        "--target", "target",
        "--task", task,
        "--output", str(pred_csv),
        "--n-estimators", str(N_TREES),
        "--learning-rate", str(LEARNING_RATE),
        "--max-depth", "6",
        "--min-samples-leaf", "5",
        "--max-bins", "255",
        "--seed", str(SEED),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    pred = pd.read_csv(pred_csv)
    y_true = pred["y_true"].to_numpy()
    out = {"time_s": round(elapsed, 3)}
    if task == "regression":
        out["r2"] = float(r2_score(y_true, pred["y_pred"]))
    else:
        out["auc"] = float(roc_auc_score(y_true, pred["y_prob"]))
        out["accuracy"] = float(accuracy_score(y_true, pred["y_pred"]))
    return out


def run_xgboost(name, ds) -> dict:
    import xgboost as xgb

    Xtr, ytr = ds["train"].drop(columns=["target"]), ds["train"]["target"]
    Xte, yte = ds["test"].drop(columns=["target"]), ds["test"]["target"]
    base = dict(n_estimators=N_TREES, learning_rate=LEARNING_RATE, max_depth=6,
                random_state=SEED, tree_method="hist", n_jobs=4, verbosity=0)
    if ds["task"] == "regression":
        m = xgb.XGBRegressor(**base)
    else:
        m = xgb.XGBClassifier(**base)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    out = {"time_s": round(elapsed, 3)}
    if ds["task"] == "regression":
        out["r2"] = float(r2_score(yte, m.predict(Xte)))
    else:
        p = m.predict_proba(Xte)[:, 1]
        out["auc"] = float(roc_auc_score(yte, p))
        out["accuracy"] = float(accuracy_score(yte, p >= 0.5))
    return out


def run_lightgbm(name, ds) -> dict:
    import lightgbm as lgb

    Xtr, ytr = ds["train"].drop(columns=["target"]), ds["train"]["target"]
    Xte, yte = ds["test"].drop(columns=["target"]), ds["test"]["target"]
    base = dict(n_estimators=N_TREES, learning_rate=LEARNING_RATE,
                random_state=SEED, n_jobs=4, verbosity=-1, deterministic=True, force_row_wise=True)
    if ds["task"] == "regression":
        m = lgb.LGBMRegressor(**base)
    else:
        m = lgb.LGBMClassifier(**base)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    out = {"time_s": round(elapsed, 3)}
    if ds["task"] == "regression":
        out["r2"] = float(r2_score(yte, m.predict(Xte)))
    else:
        p = m.predict_proba(Xte)[:, 1]
        out["auc"] = float(roc_auc_score(yte, p))
        out["accuracy"] = float(accuracy_score(yte, p >= 0.5))
    return out


def run_catboost(name, ds) -> dict:
    from catboost import CatBoostClassifier, CatBoostRegressor

    Xtr, ytr = ds["train"].drop(columns=["target"]), ds["train"]["target"]
    Xte, yte = ds["test"].drop(columns=["target"]), ds["test"]["target"]
    base = dict(iterations=N_TREES, learning_rate=LEARNING_RATE, depth=6,
                random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False)
    if ds["task"] == "regression":
        m = CatBoostRegressor(**base)
    else:
        m = CatBoostClassifier(**base)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    out = {"time_s": round(elapsed, 3)}
    if ds["task"] == "regression":
        out["r2"] = float(r2_score(yte, m.predict(Xte)))
    else:
        p = m.predict_proba(Xte)[:, 1]
        out["auc"] = float(roc_auc_score(yte, p))
        out["accuracy"] = float(accuracy_score(yte, p >= 0.5))
    return out


def run_hgb(name, ds) -> dict:
    """sklearn HistGradientBoosting 作参照（现有 CI gate 的金标准）。"""
    Xtr, ytr = ds["train"].drop(columns=["target"]), ds["train"]["target"]
    Xte, yte = ds["test"].drop(columns=["target"]), ds["test"]["target"]
    if ds["task"] == "regression":
        m = HistGradientBoostingRegressor(max_iter=N_TREES, learning_rate=LEARNING_RATE, random_state=SEED)
    else:
        m = HistGradientBoostingClassifier(max_iter=N_TREES, learning_rate=LEARNING_RATE, random_state=SEED)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    out = {"time_s": round(elapsed, 3)}
    if ds["task"] == "regression":
        out["r2"] = float(r2_score(yte, m.predict(Xte)))
    else:
        p = m.predict_proba(Xte)[:, 1]
        out["auc"] = float(roc_auc_score(yte, p))
        out["accuracy"] = float(accuracy_score(yte, p >= 0.5))
    return out


LIBRARIES = [
    ("sooboost", run_sooboost),
    ("xgboost", run_xgboost),
    ("lightgbm", run_lightgbm),
    ("catboost", run_catboost),
    ("sklearn_hgb", run_hgb),
]


# ---------------------------------------------------------------- report
def to_markdown(results: dict, versions: dict) -> str:
    lines = [
        "# sooboost vs 三巨头：真实数据集对比",
        "",
        f"- 统一预算：n_estimators={N_TREES}，learning_rate={LEARNING_RATE}，seed={SEED}；"
        "深度/叶子取各库默认或等价容量（xgb depth6 / lgbm num_leaves31 / catboost depth6 / sooboost depth6 / hgb leaves31）",
        f"- 版本：{', '.join(f'{k} {v}' for k, v in versions.items())}",
        "- 训练时间含数据读入与库初始化（sooboost 为 CLI 进程冷启动，含进程开销）",
        "",
    ]
    for metric, label in [("r2", "R²（回归）"), ("auc", "AUC（二分类）"), ("accuracy", "Accuracy（二分类）")]:
        rows = []
        for ds_name, libs in results.items():
            vals = {lib: m.get(metric) for lib, m in libs.items()}
            if all(v is None for v in vals.values()):
                continue
            best = max(vals, key=lambda k: vals[k] or -9)
            cells = []
            for lib in [l for l, _ in LIBRARIES]:
                v = vals.get(lib)
                cell = "—" if v is None else f"{v:.4f}"
                if lib == best:
                    cell = f"**{cell}**"
                cells.append(cell)
            rows.append(f"| {ds_name} | " + " | ".join(cells) + " |")
        if rows:
            lines += [f"## {label}", "",
                      "| 数据集 | " + " | ".join(l for l, _ in LIBRARIES) + " |",
                      "|---|" + "---|" * len(LIBRARIES)] + rows + [""]

    time_rows = []
    for ds_name, libs in results.items():
        cells = []
        for lib, _ in LIBRARIES:
            t = libs.get(lib, {}).get("time_s")
            cells.append("—" if t is None else f"{t:.2f}s")
        time_rows.append(f"| {ds_name} | " + " | ".join(cells) + " |")
    lines += ["## 训练耗时", "",
              "| 数据集 | " + " | ".join(l for l, _ in LIBRARIES) + " |",
              "|---|" + "---|" * len(LIBRARIES)] + time_rows + [""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="仅从已有 JSON 重新生成 Markdown")
    args = ap.parse_args()

    if not args.report:
        datasets = load_datasets()
        results = {}
        for ds_name, ds in datasets.items():
            n_train = len(ds["train"])
            print(f"\n=== {ds_name}（task={ds['task']}，train={n_train}，test={len(ds['test'])}，"
                  f"features={len(ds['features'])}）===")
            results[ds_name] = {}
            for lib, fn in LIBRARIES:
                try:
                    m = fn(ds_name, ds)
                    results[ds_name][lib] = m
                    print(f"  [{lib:>12}] {m}")
                except Exception as e:  # noqa: BLE001 —— 单库失败不拖垮整表
                    print(f"  [{lib:>12}] FAILED: {e}")
                    results[ds_name][lib] = {"error": str(e)}

        import sklearn

        versions = {
            "sooboost": "0.1.0(local)",
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "catboost": __import__("catboost").__version__,
            "sklearn": sklearn.__version__,
        }
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "budget": {"n_estimators": N_TREES, "learning_rate": LEARNING_RATE, "seed": SEED},
            "versions": versions,
            "results": results,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写出 {OUT_JSON}")

    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    md = to_markdown(payload["results"], payload["versions"])
    out_md = OUTPUT_DIR / "giants_comparison.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"已写出 {out_md}")
    print()
    print(md)


if __name__ == "__main__":
    sys.exit(main())
