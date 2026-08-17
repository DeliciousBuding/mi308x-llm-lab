#!/usr/bin/env bash
# Serve Qwen3.8-27B with SGLang on ROCm (gfx942 / MI308X / MI300X).
#
# SGLang is the RECOMMENDED serving engine for Qwen3.8-27B's hybrid attention
# architecture. Its Unified Radix Cache with MAMBA component caches GDN
# recurrent state across requests — vLLM's APC cannot do this.
#
# Key SGLang advantages for this model:
#   - mamba_radix_cache.py: caches GDN state in radix tree (not just KV)
#   - hi_mamba_radix_cache.py: CPU offload for GDN state (HiCache)
#   - triton_gdn_fused_proj.py: Triton GDN kernel (works on ROCm)
#   - cutedsl_gdn_mtp_ring.py: ReplaySSM for MTP (no mixed-batch crash)
#   - Anthropic-compatible endpoint (Claude Code connects natively)
#
# This script is secret-neutral and does not generate/persist API keys.
set -euo pipefail

if [ "$#" -gt 0 ] && [ "$1" != "qwen38" ] && [ "$1" != "qwen3.8-27b" ]; then
  echo "ERROR: this launcher only serves Qwen3.8-27B; remove model selector '$1'." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_BASE_EXPLICIT="${MODEL_BASE+x}"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
QUANT="${QUANT:-bf16}"

# SGLang venv — SEPARATE from both vLLM venvs (DS0731 and Qwen3.8-vLLM)
VENV_DIR="${SGLANG_VENV:-/root/.venvs/sglang-qwen}"
if [ -z "${USE_SYSTEM_SGLANG:-}" ] && [ -x "$VENV_DIR/bin/python" ]; then
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  echo "[sglang] using isolated qwen venv: $VENV_DIR"
else
  echo "[sglang] using system SGLang (unverified control path)"
fi

if ! command -v sglang >/dev/null 2>&1 && ! command -v python3 -c "import sglang" >/dev/null 2>&1; then
  echo "ERROR: sglang not found" >&2
  echo "Run: bash scripts/install_sglang.sh" >&2
  exit 1
fi

# AITER integration (ROCm fused-kernel library)
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
case "$QUANT" in
  bf16) MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}" ;;
  fp8)  MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}" ;;
  *) echo "ERROR: QUANT must be 'bf16' or 'fp8'" >&2; exit 1 ;;
esac

HOT_MODEL_BASE="${HOT_MODEL_BASE:-/root/models}"
if [ -z "$MODEL_BASE_EXPLICIT" ]; then
  hot_path="$HOT_MODEL_BASE/$MODEL_ID"
  if [ -d "$hot_path" ] && [ -f "$hot_path/config.json" ]; then
    MODEL_BASE="$HOT_MODEL_BASE"
    echo "[model] using local hot copy: $hot_path"
  else
    echo "[model] using persistent base: $MODEL_BASE"
  fi
fi
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/$MODEL_ID}"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: model config not found: $MODEL_PATH/config.json" >&2
  echo "Run: bash scripts/01_download_model.sh qwen38-bf16" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# YaRN RoPE scaling for 512K context (factor 2.0 over 262K native)
# ---------------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
if [ "$MAX_MODEL_LEN" -gt 262144 ]; then
  YARN_FACTOR="${YARN_FACTOR:-2.0}"
  echo "[yarn] factor=$YARN_FACTOR max_model_len=$MAX_MODEL_LEN (native=262144)"
else
  echo "[yarn] disabled; serving at native max_model_len=$MAX_MODEL_LEN"
fi

# ---------------------------------------------------------------------------
# Scheduler / memory profile (SGLang-specific, tuned for hybrid GDN)
# ---------------------------------------------------------------------------
# --mamba-full-memory-ratio: the KEY sizing flag for hybrid GDN models.
# Splits post-weight memory between GDN state pool and attention KV pool.
# Formula: ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)
#   S = 4 (extra_buffer_lazy), D = 4 (EAGLE MTP-3+1)
#   state_bytes = 78.4 MB (bf16), kv_bytes_per_token = 32.8 KB (fp8)
#   L = 80000 (avg agentic context)
# → ratio ≈ 0.24 for 80K context; adjust for your workload
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.24}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"

# MTP speculative decoding (native multi-token prediction via EAGLE algorithm)
MTP_ENABLED="${MTP_ENABLED:-1}"
SPEC_ARGS=()
if [ "$MTP_ENABLED" = "1" ]; then
  SPEC_ARGS+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
  echo "[mtp] enabled EAGLE (num_steps=3, topk=1, draft_tokens=4)"
else
  echo "[mtp] disabled (native decode baseline)"
fi

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
AUTH_ARGS=()
API_KEY_VALUE="${VLLM_API_KEY:-${SGLANG_API_KEY:-}}"
if [ -z "$API_KEY_VALUE" ] && [ -n "${VLLM_API_KEY_FILE:-${SGLANG_API_KEY_FILE:-}}" ]; then
  KEY_FILE="${VLLM_API_KEY_FILE:-${SGLANG_API_KEY_FILE:-}}"
  if [ ! -r "$KEY_FILE" ]; then
    echo "[auth] API key file not readable: $KEY_FILE" >&2
    exit 1
  fi
  API_KEY_VALUE="$(<"$KEY_FILE")"
fi
if [ -n "$API_KEY_VALUE" ]; then
  AUTH_ARGS+=(--api-key "$API_KEY_VALUE")
  echo "[auth] API key injected"
else
  echo "[auth] no API key configured"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[model] $MODEL_PATH -> $SERVED_MODEL_NAME (quant=$QUANT)"
echo "[scheduler] max_model_len=$MAX_MODEL_LEN mem_fraction=$MEM_FRACTION_STATIC"
echo "[gdn] mamba_full_memory_ratio=$MAMBA_FULL_MEMORY_RATIO ssm_dtype=$MAMBA_SSM_DTYPE radix_strategy=$MAMBA_RADIX_CACHE_STRATEGY"
echo "[gdn] chunked_prefill=$CHUNKED_PREFILL_SIZE (2048 recommended for hybrid GDN)"
echo "[runtime] host=$HOST port=$PORT"

# Build the serve command
# Note: SGLang uses --attention-backend triton on AMD (not flashinfer which is NVIDIA-only)
# The GDN Triton kernels work out-of-the-box on ROCm.
exec sglang serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --max-model-len "$MAX_MODEL_LEN" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
  --mamba-radix-cache-strategy "$MAMBA_RADIX_CACHE_STRATEGY" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --kv-cache-dtype fp8 \
  --enable-radix-cache \
  "${SPEC_ARGS[@]}" \
  "${AUTH_ARGS[@]}"
