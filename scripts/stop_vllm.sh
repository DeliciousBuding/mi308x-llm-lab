#!/usr/bin/env bash
# Stop the complete vLLM process group, including EngineCore/resource-tracker
# children that may survive if only the API-server PID is terminated.
set -euo pipefail

WAIT_SECONDS="${STOP_WAIT_SECONDS:-30}"
VRAM_WAIT_SECONDS="${STOP_VRAM_WAIT_SECONDS:-30}"
VRAM_MAX_BYTES="${STOP_VRAM_MAX_BYTES:-8000000000}"
SELF_PGID="$(ps -o pgid= $$ | tr -d ' ')"

collect_pgids() {
  {
    pgrep -x vllm 2>/dev/null || true
    ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^VLLM::EngineCor/ {print $1}'
  } | while read -r pid; do
    [ -n "$pid" ] || continue
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
  done | awk 'NF && !seen[$0]++'
}

mapfile -t PGIDS < <(collect_pgids)
if [ "${#PGIDS[@]}" -eq 0 ]; then
  echo "[stop] no vLLM process group found"
  exit 0
fi

for pgid in "${PGIDS[@]}"; do
  if [ "$pgid" = "$SELF_PGID" ]; then
    echo "[stop] refusing to signal current shell process group $pgid" >&2
    exit 2
  fi
  echo "[stop] TERM process group $pgid"
  kill -TERM -- "-$pgid" 2>/dev/null || true
done

for ((i=0; i<WAIT_SECONDS; i++)); do
  mapfile -t LEFT < <(collect_pgids)
  [ "${#LEFT[@]}" -eq 0 ] && break
  sleep 1
done

mapfile -t LEFT < <(collect_pgids)
if [ "${#LEFT[@]}" -gt 0 ]; then
  for pgid in "${LEFT[@]}"; do
    [ "$pgid" = "$SELF_PGID" ] && continue
    echo "[stop] KILL lingering process group $pgid"
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
  sleep 2
fi

mapfile -t LEFT < <(collect_pgids)
if [ "${#LEFT[@]}" -gt 0 ]; then
  echo "[stop] ERROR: vLLM processes still present: ${LEFT[*]}" >&2
  exit 1
fi

echo "[stop] vLLM process tree stopped cleanly"
if command -v rocm-smi >/dev/null 2>&1; then
  vram_used_bytes() {
    rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Used Memory/{print $NF; exit}'
  }
  for ((i=0; i<VRAM_WAIT_SECONDS; i++)); do
    USED="$(vram_used_bytes || true)"
    if [[ "$USED" =~ ^[0-9]+$ ]] && [ "$USED" -le "$VRAM_MAX_BYTES" ]; then
      echo "[stop] VRAM released: $USED bytes"
      exit 0
    fi
    sleep 1
  done
  USED="$(vram_used_bytes || true)"
  echo "[stop] ERROR: VRAM still above safe restart threshold after ${VRAM_WAIT_SECONDS}s: ${USED:-unknown} bytes" >&2
  exit 1
fi
