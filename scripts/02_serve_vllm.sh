#!/usr/bin/env bash
# Serve Qwen3.8-27B with native vLLM on ROCm (gfx942 / MI308X / MI300X).
#
# This public recipe is intentionally single-model and secret-neutral. It does
# not generate/persist API keys or own ingress/SSH/bootstrap configuration.
#
# Unlike the sibling DeepSeek-V4-Flash launcher, this script applies NO fork
# overlays: Qwen3.8 is upstream-native in vLLM (enabled for AMD ROCm by
# vllm-project/vllm#50068). The Gated DeltaNet linear-attention kernels are
# Triton-based and optimized for gfx942 (vllm-project/vllm#41446).
set -euo pipefail

# Accept a no-op model selector for parity with the sibling recipe's CLI habit.
if [ "$#" -gt 0 ] && [ "$1" != "qwen38" ] && [ "$1" != "qwen3.8-27b" ]; then
  echo "ERROR: this launcher only serves Qwen3.8-27B; remove model selector '$1'." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Paths (all overridable via environment; defaults are reference-sandbox layout)
# ---------------------------------------------------------------------------
MODEL_BASE_EXPLICIT="${MODEL_BASE+x}"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"

# Quantization variant: bf16 (default, reference quality) or fp8 (max KV headroom).
QUANT="${QUANT:-bf16}"

# The Qwen3.8-27B checkpoint is multimodal by design (vision_config in config.json).
# Production defaults keep the vision encoder enabled for coding-agent screenshots.
# Set LANGUAGE_MODEL_ONLY=1 only for controlled text-throughput / max-KV A/B runs.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-1}"
MM_VIDEO_LIMIT="${MM_VIDEO_LIMIT:-0}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-2}"
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-lru}"
DEFAULT_ENABLE_THINKING="${DEFAULT_ENABLE_THINKING:-0}"

VENV_DIR="${VLLM_VENV:-/root/.venvs/vllm-qwen}"
if [ -z "${USE_SYSTEM_VLLM:-}" ] && [ -x "$VENV_DIR/bin/vllm" ]; then
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  echo "[vllm] using isolated qwen venv: $VENV_DIR"
else
  echo "[vllm] using system vLLM (unverified control path)"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm executable not found" >&2
  exit 1
fi

# AITER is the ROCm fused-kernel library. Enable by default; the 80-CU MI308X
# may need its own tuning tables, but default tables are the starting point.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

# Attention backend / block size for the full-attention (GQA, head_dim=256)
# layers. G3 validation (2026-08-18): the default ROCm path rejects head_dim=256
# and falls back to Triton. ROCM_AITER_UNIFIED_ATTN + block 64 removes that
# fallback and improves decode by 13-35%, so it is the validated default.
# Override ATTENTION_BACKEND/BLOCK_SIZE only for controlled A/B benchmarks.
# Use `${VAR-default}` (not `:-`) so `ATTENTION_BACKEND=` explicitly selects
# the automatic/control path without passing --attention-backend.
ATTENTION_BACKEND="${ATTENTION_BACKEND-ROCM_AITER_UNIFIED_ATTN}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

# ---------------------------------------------------------------------------
# Model path resolution (prefer ephemeral local hot copy when complete)
# ---------------------------------------------------------------------------
count_safetensors() {
  local path="$1"
  if [ ! -d "$path" ]; then
    printf '0\n'
    return 0
  fi
  # Qwen3.8-27B BF16 uses model-*.safetensors (18 shards).
  # The FP8 variant may use layers-*.safetensors or model-*.safetensors.
  local count
  count=$(find "$path" -maxdepth 1 -type f \( -name 'model-*.safetensors' -o -name 'layers-*.safetensors' \) 2>/dev/null | wc -l)
  printf '%s\n' "$count"
}

case "$QUANT" in
  bf16)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
    EXPECTED_SHARDS="${EXPECTED_SHARDS:-18}"
    ;;
  fp8)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}"
    EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"  # FP8 shard count varies; 0 = skip strict check
    ;;
  *)
    echo "ERROR: QUANT must be 'bf16' or 'fp8', got: $QUANT" >&2
    exit 1
    ;;
esac

HOT_MODEL_BASE="${HOT_MODEL_BASE:-/root/models}"
if [ -z "$MODEL_BASE_EXPLICIT" ]; then
  hot_path="$HOT_MODEL_BASE/$MODEL_ID"
  hot_shards=$(count_safetensors "$hot_path")
  if [ "$hot_shards" -gt 0 ] && [ -f "$hot_path/config.json" ]; then
    MODEL_BASE="$HOT_MODEL_BASE"
    echo "[model] using local hot copy: $hot_path ($hot_shards shards)"
  else
    echo "[model] local hot copy incomplete; using persistent base: $MODEL_BASE"
  fi
fi
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/$MODEL_ID}"

SHARD_COUNT=$(count_safetensors "$MODEL_PATH")
if [ "$EXPECTED_SHARDS" -gt 0 ] && [ "$SHARD_COUNT" -lt "$EXPECTED_SHARDS" ]; then
  echo "ERROR: checkpoint incomplete: $SHARD_COUNT/$EXPECTED_SHARDS safetensors shards" >&2
  echo "Run: bash scripts/01_download_model.sh qwen38-$QUANT" >&2
  exit 1
fi
if [ "$SHARD_COUNT" -eq 0 ] && [ ! -f "$MODEL_PATH/model.safetensors" ]; then
  echo "ERROR: no safetensors found in $MODEL_PATH" >&2
  echo "Run: bash scripts/01_download_model.sh qwen38-$QUANT" >&2
  exit 1
fi

# A dead EngineCore cannot unlink its mmap-backed native CPU-KV files.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Context policy: native 256K for production agent/tool-loop traffic.
# ---------------------------------------------------------------------------
# Qwen3.8-27B natively supports 262,144 tokens. Keep that native ceiling as the
# production default: it avoids unnecessary YaRN complexity and focuses tuning
# on TTFT, prefix-cache reuse, interactive concurrency, and decode. Longer
# contexts remain an explicit A/B override only.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
if [ "$MAX_MODEL_LEN" -gt 262144 ]; then
  YARN_FACTOR="${YARN_FACTOR:-2.0}"
  HF_OVERRIDES="${HF_OVERRIDES:-{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": $YARN_FACTOR, \"original_max_position_embeddings\": 262144}}}}"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  echo "[yarn] factor=$YARN_FACTOR max_model_len=$MAX_MODEL_LEN (native=262144)"
else
  HF_OVERRIDES="${HF_OVERRIDES:-}"
  echo "[yarn] disabled; serving at native max_model_len=$MAX_MODEL_LEN"
fi

# ---------------------------------------------------------------------------
# Scheduler / decode profile (ported from the DeepSeek-V4-Flash 3072 default)
# ---------------------------------------------------------------------------
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-3072}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# G-gate validation: native CPU-KV offload crashes with madvise(EINVAL) in the
# DSW sandbox. GPU-only KV is the safe default; non-zero is an explicit A/B.
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-0}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-16000000000}"

EXTRA_ARGS=()
if [ "${KV_OFFLOAD_GB:-0}" -gt 0 ] 2>/dev/null; then
  EXTRA_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_BYTES")
  EXTRA_ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GB")
  EXTRA_ARGS+=(--kv-offloading-backend native)
  echo "[kv-offload] CPU tier ${KV_OFFLOAD_GB} GB + GPU pool $((KV_CACHE_BYTES / 1000000000)) GB"
else
  echo "[kv-offload] disabled (GPU-only)"
fi

# Multimodal policy. Production keeps vision enabled but bounds accidental media
# fan-out from agent clients to one image and no video per prompt. Text-only mode
# remains available as a controlled benchmark profile.
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
  EXTRA_ARGS+=(--language-model-only)
  echo "[vision] skipped (language-model-only); text decoder path"
else
  EXTRA_ARGS+=(--limit-mm-per-prompt "{\"image\":${MM_IMAGE_LIMIT},\"video\":${MM_VIDEO_LIMIT}}")
  EXTRA_ARGS+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
  EXTRA_ARGS+=(--mm-processor-cache-type "$MM_PROCESSOR_CACHE_TYPE")
  echo "[vision] enabled; image_limit=$MM_IMAGE_LIMIT video_limit=$MM_VIDEO_LIMIT mm_cache=${MM_PROCESSOR_CACHE_GB}GiB/$MM_PROCESSOR_CACHE_TYPE"
fi

# Qwen thinking is enabled by the upstream chat template unless told otherwise.
# Default it off for general/coding-agent traffic so max-token truncation cannot
# consume the whole response inside <think>. Request-level chat_template_kwargs
# or reasoning_effort still override this server default in vLLM.
if [ "$DEFAULT_ENABLE_THINKING" = "1" ]; then
  EXTRA_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":true}')
  echo "[thinking] enabled by default (request-level override still supported)"
else
  EXTRA_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":false}')
  echo "[thinking] disabled by default (request-level override still supported)"
fi

# YaRN overrides (if set)
if [ -n "$HF_OVERRIDES" ]; then
  EXTRA_ARGS+=(--hf-overrides "$HF_OVERRIDES")
fi

# ---------------------------------------------------------------------------
# MTP speculative decoding (native multi-token prediction, NOT DSpark)
# ---------------------------------------------------------------------------
# G5 validation (2026-08-18) promoted MTP-3: ~65% acceptance, 94.2 tok/s at C1,
# and 1094 tok/s aggregate at C32. Set MTP_ENABLED=0 only for a native-decode
# control benchmark.
MTP_ENABLED="${MTP_ENABLED:-1}"
MTP_K="${MTP_K:-3}"
SPEC_ARGS=()
if [ "$MTP_ENABLED" = "1" ]; then
  SPEC_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  echo "[mtp] enabled K=$MTP_K (native multi-token prediction)"
  echo "[mtp] note: MTP-1 regresses at high concurrency; MTP-3 is the recommended start"
else
  echo "[mtp] disabled (native decode baseline)"
fi

# ---------------------------------------------------------------------------
# Authentication (deployment policy, not repository state)
# ---------------------------------------------------------------------------
AUTH_ARGS=()
API_KEY_VALUE="${VLLM_API_KEY:-}"
if [ -z "$API_KEY_VALUE" ] && [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  if [ ! -r "$VLLM_API_KEY_FILE" ]; then
    echo "[auth] VLLM_API_KEY_FILE is not readable: $VLLM_API_KEY_FILE" >&2
    exit 1
  fi
  API_KEY_VALUE="$(<"$VLLM_API_KEY_FILE")"
fi

if [ -n "$API_KEY_VALUE" ]; then
  AUTH_ARGS+=(--api-key "$API_KEY_VALUE")
  echo "[auth] API key injected"
elif [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  echo "[auth] VLLM_API_KEY_FILE is empty: $VLLM_API_KEY_FILE" >&2
  exit 1
else
  echo "[auth] no API key configured"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[model] $MODEL_PATH -> $SERVED_MODEL_NAME (quant=$QUANT, shards=$SHARD_COUNT)"
echo "[scheduler] max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_batched_tokens=$MAX_BATCHED_TOKENS long_prefill_cap=$LONG_PREFILL_TOKEN_THRESHOLD"
echo "[runtime] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION kv_cache_dtype=$KV_CACHE_DTYPE host=$HOST port=$PORT block_size=$BLOCK_SIZE attention_backend=${ATTENTION_BACKEND:-auto}"

ATTN_ARGS=()
if [ -n "$ATTENTION_BACKEND" ]; then
  ATTN_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
  # ROCM_AITER_UNIFIED_ATTN requires the sub-toggle (defaults to False in vLLM).
  if [ "$ATTENTION_BACKEND" = "ROCM_AITER_UNIFIED_ATTN" ]; then
    export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
    echo "[attention] UNIFIED_ATTN: set VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 (required for head_dim=256)"
  fi
fi

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 1 \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --block-size "$BLOCK_SIZE" \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
  --linear-backend auto \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  "${SPEC_ARGS[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${ATTN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${AUTH_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
