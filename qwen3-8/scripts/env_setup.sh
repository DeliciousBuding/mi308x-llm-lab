#!/usr/bin/env bash
# Create the isolated venv for Qwen3.8-27B serving.
#
# This venv is SEPARATE from the DeepSeek-V4-Flash venv (/root/.venvs/vllm).
# The DeepSeek venv carries MLA/DSpark/MoE overlay patches that must not
# contaminate Qwen3.8's standard GQA + linear-attention path.
set -euo pipefail

VENV="${VLLM_VENV:-/root/.venvs/vllm-qwen}"
PYTHON="${PYTHON:-python3}"

echo "Creating isolated venv: $VENV"
echo "Python: $($PYTHON --version 2>&1)"

# --system-site-packages reuses the platform ROCm Torch build (torch 2.11.0+gitd0c8b1f,
# HIP/ROCm 7.2.x) without reinstalling it. System Python/Torch remain untouched.
$PYTHON -m venv --system-site-packages "$VENV"

# Upgrade pip silently; the venv inherits system torch so no heavy install.
"$VENV/bin/pip" install --upgrade pip >/dev/null 2>&1 || true

echo "Venv created at: $VENV"
echo "torch: $("$VENV/bin/python" -c 'import torch; print(torch.__version__, "hip="+str(torch.version.hip))' 2>&1)"
echo ""
echo "Next: bash scripts/install_vllm_nightly.sh"
