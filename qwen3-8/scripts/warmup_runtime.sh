#!/usr/bin/env bash
# Warm representative Qwen3.8-27B inference paths after service startup,
# then optionally persist the venv/JIT/compiler state for the next GPU restore.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
WAIT_SECONDS="${WARMUP_WAIT_SECONDS:-300}"
SNAPSHOT_AFTER_WARMUP="${SNAPSHOT_AFTER_WARMUP:-0}"

for ((i=0; i<WAIT_SECONDS; i++)); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "$BASE_URL/health" >/dev/null

echo "[warmup] short decode path"
python3 "$ROOT/scripts/bench/bench_full.py" decode

echo "[warmup] 20K required-tool/prefill/parser path"
python3 "$ROOT/scripts/bench/bench_tool_roundtrip.py" \
  --rounds 1 --mode required --prefix-tokens 20000

if [ "$SNAPSHOT_AFTER_WARMUP" = "1" ]; then
  echo "[warmup] snapshotting validated runtime state"
  bash "$ROOT/scripts/snapshot_runtime_state.sh"
else
  echo "[warmup] snapshot skipped (set SNAPSHOT_AFTER_WARMUP=1 to persist)"
fi
