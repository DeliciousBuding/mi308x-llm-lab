#!/usr/bin/env bash
# Serve Qwen3.8-27B with SGLang on ROCm (gfx942 / MI308X / MI300X).
#
# SGLang is an EXPERIMENTAL CANDIDATE for Qwen3.8-27B on AMD.
# vLLM (02_serve_vllm.sh) is the production baseline.
# Real agent-loop A/B on GPU decides the winner.
#
# Key correction (2026-08-17): SGLang on AMD MI GPUs must use `no_buffer`
# Mamba Radix Cache strategy — the `extra_buffer` branching-point GDN state
# caching is NVIDIA-only. This weakens SGLang's cross-request GDN state
# caching advantage on AMD. See RESEARCH_NOTES.md §9 for details.
#
# SGLang still offers: Triton GDN kernels, ReplaySSM for MTP, --mamba-ssm-dtype,
# Anthropic-compatible endpoint. These are worth testing but not assumed superior.
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
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
QUANT="${QUANT:-bf16}"

# SGLang venv — SEPARATE from both vLLM venvs (DS0731 and Qwen3.8-vLLM)
VENV_DIR="${SGLANG_VENV:-/root/.venvs/sglang}"
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
# Scheduler / memory profile (SGLang-specific, tuned for hybrid GDN on AMD)
# ---------------------------------------------------------------------------
# CRITICAL (2026-08-17 correction): SGLang on AMD MI GPUs must use
# `no_buffer` Mamba Radix Cache strategy. The `extra_buffer` branching-point
# Mamba state caching depends on FLA path (NVIDIA-only). `no_buffer` lowers
# memory but does NOT support overlap scheduler or branching-point GDN state
# cache. See: sgl-project/sglang docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx
#
# This weakens SGLang's primary advantage on AMD (cross-request GDN state
# caching). SGLang is now an EXPERIMENTAL CANDIDATE, not the recommended engine.
# vLLM is the production baseline. Real agent-loop A/B decides the winner.
#
# --mamba-full-memory-ratio: SGLang default is 0.9. Do NOT hardcode a lower
# value here — let the user sweep it. 0.9 = balanced; lower = more KV, less
# GDN state pool. The ratio depends on avg context length and concurrency.
#
# --mamba-ssm-dtype: default follows model config (float32). BF16 halves the
# GDN state pool but MAY have cumulative numerical drift at 200K+ context.
# Test FP32 (correctness reference) vs BF16 (production candidate) on
# 128K/256K/384K/512K recall + agent replay.
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.9}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-no_buffer}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"

# MTP speculative decoding — do NOT lock MTP-3 as default.
# Reports indicate MTP may not help on AMD MI GPUs (issue #23123).
# Sweep: native / MTP-1 / MTP-2 / MTP-3 on GPU.
# Default: OFF (native decode) for correctness reference (Gate G1/G2).
MTP_ENABLED="${MTP_ENABLED:-0}"
MTP_STEPS="${MTP_STEPS:-3}"
SPEC_ARGS=()
if [ "$MTP_ENABLED" = "1" ]; then
  SPEC_ARGS+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "$MTP_STEPS"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$((MTP_STEPS + 1))"
  )
  echo "[mtp] enabled EAGLE (steps=$MTP_STEPS, topk=1, draft_tokens=$((MTP_STEPS + 1)))"
  echo "[mtp] WARNING: MTP may not help on AMD MI GPUs (issue #23123); sweep before production"
else
  echo "[mtp] disabled (native decode — correctness reference baseline)"
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
