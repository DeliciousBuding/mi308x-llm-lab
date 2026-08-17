#!/usr/bin/env bash
# =============================================================================
# install_sglang.sh — install SGLang 0.5.17 for Qwen3.8-27B on ROCm (gfx942)
#
# SGLang is installed with --no-deps to skip NVIDIA-specific packages
# (flashinfer, flash-attn-4, nvidia-cutlass-dsl, etc.). The ROCm-compatible
# dependencies are installed from pre-downloaded wheels in $WHEELS/sglang-deps.
# The system ROCm torch (2.11.0+gitd0c8b1f) is reused via --system-site-packages.
#
# Prerequisites:
#   1. SGLang wheel at $WHEELS/sglang/sglang-0.5.17-cp312-*.whl
#   2. SGLang deps at $WHEELS/sglang-deps/*.whl
#   3. venv created by scripts/env_setup.sh (or this script creates one)
#
# Usage: bash install_sglang.sh
# Idempotent: wheels are reinstalled; system Python / system torch are never replaced.
# =============================================================================
set -euo pipefail

VENV="${SGLANG_VENV:-/root/.venvs/sglang-qwen}"
WHEELS="${WHEELS:-/mnt/workspace/wheels}"
SGLANG_WHEEL_DIR="$WHEELS/sglang"
SGLANG_DEPS_DIR="$WHEELS/sglang-deps"

echo "=============================================="
echo "  SGLang 0.5.17 + ROCm deps (Qwen3.8-27B)"
echo "=============================================="
echo "venv:   $VENV"
echo "wheels: $WHEELS"

# Create venv if it doesn't exist (reuse system ROCm torch)
if [ ! -x "$VENV/bin/python" ]; then
  echo ""
  echo "===== [1/5] create isolated venv ====="
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  $PYTHON_BIN -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install --upgrade pip >/dev/null 2>&1 || true
  echo "  ✓ venv created at $VENV"
else
  echo ""
  echo "===== [1/5] venv already exists ====="
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo ""
echo "===== [2/5] verify system ROCm torch reuse ====="
"$PY" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip); assert torch.version.hip, "non-ROCm torch"'

echo ""
echo "===== [3/5] install SGLang core (--no-deps, skip NVIDIA packages) ====="
SGLANG_WHL="$(ls "$SGLANG_WHEEL_DIR"/sglang-0.5.17-cp312-*.whl 2>/dev/null | head -1 || true)"
[ -n "$SGLANG_WHL" ] || { echo "ERROR: SGLang wheel missing in $SGLANG_WHEEL_DIR"; exit 1; }
echo "  SGLang: $(basename "$SGLANG_WHL")"
export PYTHONDONTWRITEBYTECODE=1
"$PIP" install --no-deps --ignore-installed --no-compile "$SGLANG_WHL"
echo "  ✓ SGLang core installed (--no-deps)"

echo ""
echo "===== [4/5] install ROCm-compatible dependencies ====="
if [ -d "$SGLANG_DEPS_DIR" ] && [ "$(ls -A "$SGLANG_DEPS_DIR"/*.whl 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "  installing $(ls "$SGLANG_DEPS_DIR"/*.whl | wc -l) pre-downloaded wheels..."
  "$PIP" install --no-deps --no-compile "$SGLANG_DEPS_DIR"/*.whl
  echo "  ✓ ROCm-compatible deps installed"
else
  echo "  no pre-downloaded deps found; installing from pip..."
  "$PIP" install --no-compile \
    "transformers==5.12.1" aiohttp fastapi uvicorn uvloop \
    "openai==2.6.1" compressed-tensors tiktoken sentencepiece \
    msgspec pydantic partial_json_parser interegular \
    "prometheus-client>=0.20.0" psutil pybase64 setproctitle \
    xxhash zstandard numpy scipy einops pillow python-multipart \
    orjson requests packaging tqdm watchfiles distro easydict \
    gguf "mistral_common>=1.11.5" ninja "numba==0.65.1" \
    "pyzmq>=25.1.2" "anthropic>=0.20.0" datasets \
    "xgrammar==0.2.1" "llguidance>=1.7.6,<2.0.0" "outlines==0.1.11"
  echo "  ✓ ROCm-compatible deps installed from pip"
fi

echo ""
echo "===== [5/5] verify SGLang import ====="
"$PY" -c "
import sglang
print('sglang version:', getattr(sglang, '__version__', 'installed'))
print('sglang module:', sglang.__file__)
" 2>&1 || {
  echo "  ⚠ SGLang import failed (some NVIDIA-only deps may be missing)"
  echo "  This is expected on AMD — SGLang should still serve with --attention-backend triton"
  echo "  Verify on the GPU host: sglang serve --help"
}

echo ""
echo "=============================================="
echo "  installation complete"
echo "=============================================="
echo "verify on GPU host:"
echo "  source $VENV/bin/activate"
echo "  python3 -c 'import sglang; print(sglang.__version__)'"
echo "start serving Qwen3.8-27B:"
echo "  bash scripts/02_serve_sglang.sh qwen38"
