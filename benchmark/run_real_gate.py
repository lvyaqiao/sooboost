# -*- coding: utf-8 -*-
"""真实数据集性能门禁（M6-6）：sooboost 在 3 个真实数据集上的精度下限。

设计：
- 与 compare_giants.py 完全同口径（统一预算 200 轮 / lr 0.1 / seed 42 / depth 6），
  直接复用其数据加载与 sooboost 运行逻辑，避免两套实现漂移。
- 三巨头（XGBoost/LightGBM/CatBoost）不在 CI 安装（体积与稳定性原因），
  以 giants_comparison.json 记录的 sooboost 实测值减去安全余量作为金标准下限：
    california_housing R²   记录 0.8403 → 下限 0.82
    diabetes          R²   记录 0.3877 → 下限 0.35
    breast_cancer     AUC  记录 0.9950 → 下限 0.985
  任何跌破下限的提交都意味着回归，必须在合并前解释。

用法（先 cargo build --release -p sooboost-cli）:
    python benchmark/run_real_gate.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare_giants as cg  # noqa: E402

# (数据集名, 指标, 下限)
GATES = [
    ("california_housing", "r2", 0.82),
    ("diabetes", "r2", 0.35),
    ("breast_cancer", "auc", 0.985),
]


def main() -> int:
    datasets = cg.load_datasets()
    failures = []
    print("=== 真实数据集性能门禁（sooboost，统一预算 200 轮 / lr 0.1 / seed 42）===")
    for name, metric, floor in GATES:
        ds = datasets[name]
        m = cg.run_sooboost(name, ds, workdir=cg.OUTPUT_DIR / "real_gate_work" / name)
        got = m.get(metric)
        ok = got is not None and got >= floor
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<20} {metric} = {got:.4f}（下限 {floor}，实测耗时 {m.get('time_s')}s）")
        if not ok:
            failures.append((name, metric, got, floor))

    if failures:
        print("\n门禁未通过：以下指标跌破下限（存在回归，须在合并前解释）：")
        for name, metric, got, floor in failures:
            print(f"  - {name}: {metric} = {got} < {floor}")
        return 1
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
