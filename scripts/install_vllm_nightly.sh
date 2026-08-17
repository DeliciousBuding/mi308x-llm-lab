#!/usr/bin/env bash
# =============================================================================
# install_vllm_nightly.sh — install vLLM dev306 + AITER 0.1.19 for Qwen3.8-27B
#
# Unlike the sibling DeepSeek-V4-Flash installer, this applies NO fork overlays.
# Qwen3.8 is upstream-native in vLLM (enabled for AMD ROCm by
# vllm-project/vllm#50068). The Gated DeltaNet linear-attention kernels are
# Triton-based and optimized for gfx942 (vllm-project/vllm#41446).
#
# Prerequisites (paths overridable via env):
#   1. wheels downloaded to $WHEELS (vLLM dev306 ROCm, AITER 0.1.19)
#   2. venv created by scripts/env_setup.sh
#
# Usage: bash install_vllm_nightly.sh
# Idempotent: wheels are reinstalled; system Python / system torch are never
# replaced.
#
# Version pins:
#   vLLM  0.26.1rc1.dev306+gcb8104839.rocm723
#   AITER 0.1.19
#   torch 2.11.0+gitd0c8b1f -> reused from system
# =============================================================================
set -euo pipefail

VENV="${VENV:-/root/.venvs/vllm-qwen}"
WHEELS="${WHEELS:-/mnt/workspace/wheels}"

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
PYVER="$("$PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$VENV/lib/$PYVER/site-packages"

echo "=============================================="
echo "  vLLM dev306 + AITER 0.1.19 (Qwen3.8-27B)"
echo "=============================================="
echo "venv:   $VENV"
echo "site:   $SITE"
echo "wheels: $WHEELS"

echo ""
echo "===== [1/6] verify system ROCm torch reuse ====="
"$PY" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip); assert torch.version.hip, "non-ROCm torch"'

echo ""
echo "===== [2/6] locate exact wheels ====="
AITER_WHL="$(ls "$WHEELS"/amd_aiter-0.1.19-*.whl 2>/dev/null | head -1 || true)"
VLLM_WHL="$(ls "$WHEELS"/vllm-0.26.1rc1.dev306+*.whl 2>/dev/null | head -1 || true)"
[ -n "$AITER_WHL" ] || { echo "ERROR: missing AITER wheel in $WHEELS"; exit 1; }
[ -n "$VLLM_WHL" ]  || { echo "ERROR: missing vLLM dev306 wheel in $WHEELS"; exit 1; }
echo "AITER:  $AITER_WHL"
echo "vLLM:   $VLLM_WHL"

"$PY" -c "import zipfile,sys; [zipfile.ZipFile(f).testzip() and sys.exit(1) for f in sys.argv[1:]]" \
  "$AITER_WHL" "$VLLM_WHL"
echo "wheel ZIP integrity OK"

echo ""
echo "===== [3/6] install wheels into isolated venv ====="
export PYTHONDONTWRITEBYTECODE=1
"$PIP" install --no-deps --ignore-installed --no-compile "$AITER_WHL" "$VLLM_WHL"

echo ""
echo "===== [4/6] install transformers >= 5.8.0 ====="
# Qwen3.8 config.json was written by transformers 5.8.0. vLLM parses config with
# its own Qwen3_5Config, but the Qwen3-VL processor needs the matching transformers.
"$PIP" install --no-compile "transformers>=5.8.0" 2>/dev/null || \
  echo "WARN: transformers upgrade failed; vLLM may use its own config parser instead"

echo ""
echo "===== [5/6] verify Qwen3.8 architecture registration ====="
"$PY" - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
needles = ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3_5ForConditionalGeneration")
found = [n for n in needles if n in archs]
missing = [n for n in needles if n not in archs]
for n in found:
    print(f"  OK       {n}")
for n in missing:
    print(f"  MISSING  {n}")
if not found:
    raise SystemExit(
        "ERROR: no Qwen3.8 architecture registered. The vLLM dev306 build "
        "predates PR #50068. Rebuild vLLM from a newer source or apply the "
        "model-registry patch manually."
    )
print(f"\n  {len(found)}/{len(needles)} Qwen3.8 architectures registered")
PY

echo ""
echo "===== [6/6] version check + persist snapshot ====="
"$PY" -c 'import vllm; print("vllm", vllm.__version__)'
"$PY" -c 'import importlib.metadata as m, aiter; print("AITER", m.version("amd-aiter"))'

echo ""
echo "=============================================="
echo "  installation complete"
echo "=============================================="
echo "verify runtime:"
echo "  python3 scripts/audit_runtime.py"
echo "start serving Qwen3.8-27B:"
echo "  source $VENV/bin/activate"
echo "  bash scripts/02_serve_vllm.sh qwen38"
