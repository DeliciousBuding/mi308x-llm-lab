#!/usr/bin/env bash
# Run the Qwen3.8-27B benchmark suite (assumes a running server at $VLLM_BASE_URL).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"

echo "=== health check ==="
curl -fsS "$BASE_URL/health" >/dev/null || { echo "server not healthy at $BASE_URL"; exit 1; }

echo
echo "=== decode + cache suite ==="
python3 "$ROOT/scripts/bench/bench_full.py" all

echo
echo "=== agent trace (30 turns, 20K prefix) ==="
python3 "$ROOT/scripts/bench/bench_agent_trace.py" 30 20000

echo
echo "=== tool round-trip (auto, 20K prefix) ==="
python3 "$ROOT/scripts/bench/bench_tool_roundtrip.py" --rounds 5 --mode auto --prefix-tokens 20000

echo
echo "=== high concurrency boundary ==="
python3 "$ROOT/scripts/bench/bench_high_concurrency.py" --concurrencies 1 2 4 8 32 64

echo
echo "=== TTFT isolation (200K cold prefill) ==="
python3 "$ROOT/scripts/bench/bench_ttft_isolation.py" 200000 --rounds 3

echo
echo "Benchmark suite complete. Record results in docs/PERFORMANCE.md."
