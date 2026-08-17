#!/usr/bin/env bash
# Snapshot runtime-generated compiler caches (AITER, Triton, COMGR, torch_extensions).
# These are warm-start accelerators: their absence only costs first-request JIT time.
set -euo pipefail

PERSIST="${PERSIST_DIR:-/mnt/workspace/.venvs}"
HOME_DIR="${HOME:-/root}"
mkdir -p "$PERSIST"

snapshot_dir() {
  local src="$1" label="$2" out="$3"
  if [ -d "$src" ]; then
    tar -cf "$out.tmp.$$" -C "$(dirname "$src")" "$(basename "$src")" 2>/dev/null
    mv -f "$out.tmp.$$" "$out"
    echo "OK   $label -> $out ($(du -h "$out" | cut -f1))"
  else
    echo "SKIP $label (not found: $src)"
  fi
}

snapshot_dir "$HOME_DIR/.aiter" "AITER runtime cache" "$PERSIST/aiter_cache.tar.gz"
snapshot_dir "$HOME_DIR/.cache/torch_extensions" "torch_extensions cache" "$PERSIST/torch_ext_cache.tar.gz"
snapshot_dir "$HOME_DIR/.cache/comgr" "ROCm COMGR cache" "$PERSIST/comgr_cache.tar.gz"
snapshot_dir "$HOME_DIR/.triton" "Triton cache" "$PERSIST/triton_cache.tar.gz"
