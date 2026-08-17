#!/usr/bin/env bash
# Download/resume the official Qwen3.8-27B checkpoint from ModelScope (or Hugging Face).
#
# Usage:
#   bash scripts/01_download_model.sh qwen38-bf16   # 55.6 GB, 18 shards
#   bash scripts/01_download_model.sh qwen38-fp8     # ~28 GB, FP8 variant
set -euo pipefail

SELECTOR="${1:-qwen38-bf16}"
case "$SELECTOR" in
  qwen38-bf16)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
    EXPECTED_SHARDS="${EXPECTED_SHARDS:-18}"
    ;;
  qwen38-fp8)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}"
    EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"
    ;;
  *)
    echo "Usage: $0 qwen38-bf16 | qwen38-fp8" >&2
    exit 2
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-${MODEL_DEST:-/mnt/workspace/models}}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/$MODEL_ID}"
MAX_WORKERS="${MAX_WORKERS:-16}"

mkdir -p "$MODEL_DIR"

echo "Model      : $MODEL_ID"
echo "Destination: $MODEL_DIR"
echo "Workers    : $MAX_WORKERS"

# Prefer ModelScope (the deployment platform), fall back to huggingface-cli.
if command -v modelscope >/dev/null 2>&1; then
  echo "[download] using modelscope CLI"
  modelscope download \
    --model "$MODEL_ID" \
    --local_dir "$MODEL_DIR" \
    --max-workers "$MAX_WORKERS"
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "[download] using huggingface-cli"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" \
    --max-workers "$MAX_WORKERS"
else
  echo "ERROR: neither modelscope nor huggingface-cli found. Install one:" >&2
  echo "  pip install modelscope   # or" >&2
  echo "  pip install huggingface_hub hf_transfer" >&2
  exit 1
fi

actual=$(find "$MODEL_DIR" -maxdepth 1 -type f \( -name 'model-*.safetensors' -o -name 'layers-*.safetensors' \) 2>/dev/null | wc -l | tr -d ' ')
size=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
echo "Downloaded size: ${size:-unknown}"
echo "Safetensors shards: $actual"

if [ "$EXPECTED_SHARDS" -gt 0 ] && [ "$actual" -ne "$EXPECTED_SHARDS" ]; then
  echo "WARNING: expected $EXPECTED_SHARDS shards, got $actual" >&2
  echo "Re-run this script to resume the missing files." >&2
  exit 1
fi

for required in config.json tokenizer_config.json; do
  if [ ! -f "$MODEL_DIR/$required" ]; then
    echo "WARNING: missing metadata file: $required" >&2
  fi
done

echo "Download complete: $MODEL_ID ($actual shards)"
