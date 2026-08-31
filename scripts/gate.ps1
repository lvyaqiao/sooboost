# 本地基准门禁（M1-5）：测试 + lint + 格式 + 对齐门禁 + 性能参考
#
# 用法（仓库根目录）：
#   pwsh scripts/gate.ps1
#
# 流程：
#   1. cargo fmt --check            格式门禁
#   2. cargo clippy -- -D warnings  lint 门禁
#   3. cargo test --workspace       测试门禁
#   4. cargo build --release        构建 CLI
#   5. benchmark --mode sooboost --gate  质量对齐门禁（落后金标准 >0.05 → 失败）
#   6. benchmark --mode perf        性能参考（生成 perf.json，不设硬门禁——单机噪声大）
#   7. cargo build --release -p sooboost-experiments 研究 benchmark binary
#   8. benchmark --mode gen --gate  生成/填补结构质量门禁
#
# 任何一步失败 → 退出码非 0。CI（GitHub Actions）镜像见 .github/workflows/ci.yml。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Write-Host "=== [gate] 1/8 cargo fmt --check ===" -ForegroundColor Cyan
    cargo fmt --check
    if ($LASTEXITCODE -ne 0) { throw "fmt --check 失败" }

    Write-Host "=== [gate] 2/8 cargo clippy -- -D warnings ===" -ForegroundColor Cyan
    cargo clippy --workspace --all-targets -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw "clippy 失败" }

    Write-Host "=== [gate] 3/8 cargo test --workspace ===" -ForegroundColor Cyan
    cargo test --workspace
    if ($LASTEXITCODE -ne 0) { throw "test 失败" }

    Write-Host "=== [gate] 4/8 cargo build --release ===" -ForegroundColor Cyan
    cargo build --release -p sooboost-cli
    if ($LASTEXITCODE -ne 0) { throw "build --release 失败" }

    Write-Host "=== [gate] 5/8 benchmark --mode sooboost --gate ===" -ForegroundColor Cyan
    python benchmark/run_benchmark.py --mode sooboost --gate
    if ($LASTEXITCODE -ne 0) { throw "质量对齐门禁失败" }

    Write-Host "=== [gate] 6/8 benchmark --mode perf ===" -ForegroundColor Cyan
    python benchmark/run_benchmark.py --mode perf

    Write-Host "=== [gate] 7/8 cargo build --release -p sooboost-experiments ===" -ForegroundColor Cyan
    cargo build --release -p sooboost-experiments --bin sooboost-experiments
    if ($LASTEXITCODE -ne 0) { throw "experiments release build 失败" }

    Write-Host "=== [gate] 8/8 benchmark --mode gen --gate ===" -ForegroundColor Cyan
    python benchmark/run_benchmark.py --mode gen --gate
    if ($LASTEXITCODE -ne 0) { throw "生成/填补质量门禁失败" }

    Write-Host "=== [gate] 全部通过 ===" -ForegroundColor Green
}
finally {
    Pop-Location
}
