#!/usr/bin/env bash
# Stage a complete local-SSD hot copy of the Qwen3.8-27B checkpoint.
#
# The launcher auto-prefers a complete local hot copy; this script prepares it.
# Local copy is ephemeral (lost on instance rebuild); the persistent checkpoint
# remains the source of truth.
set -euo pipefail

QUANT="${QUANT:-bf16}"
case "$QUANT" in
  bf16) MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}" ;;
  fp8)  MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}" ;;
  *) echo "ERROR: QUANT must be 'bf16' or 'fp8'" >&2; exit 1 ;;
esac

MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
HOT_MODEL_BASE="${HOT_MODEL_BASE:-/root/models}"
SRC="$MODEL_BASE/$MODEL_ID"
DST="$HOT_MODEL_BASE/$MODEL_ID"

[ -d "$SRC" ] || { echo "ERROR: source missing: $SRC" >&2; exit 1; }
mkdir -p "$DST"

echo "[stage] $SRC -> $DST"
echo "[stage] this is an ephemeral local-SSD copy; persistent source remains $SRC"

# rsync if available (faster, resumable), else cp.
if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "$SRC/" "$DST/"
else
  cp -r "$SRC/." "$DST/"
fi

shards=$(find "$DST" -maxdepth 1 -type f \( -name 'model-*.safetensors' -o -name 'layers-*.safetensors' \) 2>/dev/null | wc -l | tr -d ' ')
echo "[stage] complete: $shards shards at $DST"
echo "[stage] launcher will auto-prefer this hot copy on next serve"
