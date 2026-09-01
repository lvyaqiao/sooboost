#!/usr/bin/env python3
"""sooboost C ABI 冒烟测试（M12，纯标准库 ctypes，无第三方依赖）。

加载 sooboost-ffi 的动态库，走一遍 训练 → 预测 → 序列化/反序列化 →
错误路径 的完整闭环；任何一步不符即非零退出。CI（ubuntu）与本地
（Windows/macOS）均可运行。

用法：
    cargo build --release -p sooboost-ffi
    python scripts/ffi_smoke.py
"""

import ctypes
import glob
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def locate_lib() -> str:
    """按平台探测 target/ 下新构建的 sooboost_ffi 动态库。"""
    patterns = {
        "win32": ["target/release/sooboost_ffi.dll", "target/debug/sooboost_ffi.dll"],
        "darwin": [
            "target/release/libsooboost_ffi.dylib",
            "target/debug/libsooboost_ffi.dylib",
        ],
    }.get(sys.platform, ["target/release/libsooboost_ffi.so", "target/debug/libsooboost_ffi.so"])
    for pat in patterns:
        hits = sorted(glob.glob((REPO / pat).as_posix()))
        if hits:
            return max(hits, key=os.path.getmtime)
    raise SystemExit("未找到 sooboost_ffi 动态库，请先: cargo build --release -p sooboost-ffi")


lib = ctypes.CDLL(locate_lib())

# --- 签名声明（对应 sooboost.h） ---
lib.sbs_version.restype = ctypes.c_char_p
lib.sbs_train.restype = ctypes.c_int32
lib.sbs_train.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_void_p),
]
for name in ("sbs_predict", "sbs_predict_proba"):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
    ]
for name in ("sbs_model_num_features", "sbs_model_num_classes", "sbs_model_num_trees"):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_void_p]
lib.sbs_serialize.restype = ctypes.c_int64
lib.sbs_serialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64]
lib.sbs_deserialize.restype = ctypes.c_int32
lib.sbs_deserialize.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.sbs_model_free.restype = None
lib.sbs_model_free.argtypes = [ctypes.c_void_p]
lib.sbs_last_error.restype = ctypes.c_int32
lib.sbs_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int64]


def last_error() -> str:
    buf = ctypes.create_string_buffer(1024)
    lib.sbs_last_error(buf, len(buf))
    return buf.value.decode("utf-8", "replace")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}\n  last_error: {last_error()}")


def main() -> int:
    print(f"[1] lib = {locate_lib()}")
    version = lib.sbs_version().decode()
    print(f"[2] version = {version}")
    check(version.startswith("sooboost-ffi "), "版本串格式")

    # 数据：y = 2x，100 点
    n = 100
    x = [float(i) for i in range(n)]
    y = [2.0 * v for v in x]
    params = (
        b'{"task":"regression","n_estimators":30,"learning_rate":0.15,"seed":42}'
    )
    dbl_arr = ctypes.c_double * n
    model = ctypes.c_void_p()
    rc = lib.sbs_train(params, dbl_arr(*x), n, 1, dbl_arr(*y), ctypes.byref(model))
    check(rc == 0 and model, f"训练失败 rc={rc}")
    print("[3] train ok")

    check(lib.sbs_model_num_features(model) == 1, "num_features")
    check(lib.sbs_model_num_classes(model) == -1, "回归无类别数")
    check(lib.sbs_model_num_trees(model) == 30, "num_trees")

    out = (ctypes.c_double * n)()
    got = lib.sbs_predict(model, dbl_arr(*x), n, 1, out, n)
    check(got == n, "predict 返回元素数")
    for i in range(20, 80):
        check(math.isclose(out[i], 2.0 * i, abs_tol=2.0), f"x={i} 预测 {out[i]} 偏离 {2.0 * i}")
    print("[4] predict ok（y=2x 中段偏差 < 2.0）")

    # 序列化两段式 + roundtrip 逐位一致
    needed = lib.sbs_serialize(model, None, 0)
    check(needed > 0, "serialize 探测")
    buf = (ctypes.c_uint8 * needed)()
    check(lib.sbs_serialize(model, buf, needed) == needed, "serialize 写入")
    loaded = ctypes.c_void_p()
    check(lib.sbs_deserialize(buf, needed, ctypes.byref(loaded)) == 0, "deserialize")
    out2 = (ctypes.c_double * n)()
    lib.sbs_predict(loaded, dbl_arr(*x), n, 1, out2, n)
    check(list(out) == list(out2), "roundtrip 预测逐位一致")
    lib.sbs_model_free(loaded)
    print(f"[5] serialize/deserialize ok（{needed} 字节，逐位一致）")

    # 错误路径
    rc = lib.sbs_train(b"{not json", dbl_arr(*x), n, 1, dbl_arr(*y), ctypes.byref(model))
    check(rc == -1 and last_error() != "", "非法 JSON 显式报错")
    rc = lib.sbs_train(b'{"task":"regression","nope":1}', dbl_arr(*x), n, 1,
                       dbl_arr(*y), ctypes.byref(model))
    check(rc == -1, "未知字段显式报错")
    print("[6] error paths ok")

    lib.sbs_model_free(model)
    print("FFI 冒烟测试全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
