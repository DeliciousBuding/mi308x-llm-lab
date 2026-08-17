#!/usr/bin/env bash
# Environment probe for the Qwen3.8-27B ROCm serving stack.
# Reference control: ROCm 7.2.3 / vLLM dev306 / AITER 0.1.19 / gfx942 192 GB.
set -uo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/models}"
MODEL_ID_BF16="${MODEL_ID_BF16:-Qwen/Qwen3.8-27B}"
MODEL_ID_FP8="${MODEL_ID_FP8:-Qwen/Qwen3.8-27B-FP8}"

echo "==================== 1. Hardware and system ===================="
echo "--- CPU cores ---"; nproc
echo "--- Memory ---"; free -h
echo "--- Disks ---"; df -h

echo
echo "==================== 2. AMD GPU and ROCm ===================="
echo "--- ROCm version (validated: 7.2.3) ---"
cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/version 2>/dev/null || echo "no /opt/rocm"
echo "--- GPU model (validated: MI308X/gfx942, 192 GB class) ---"
rocm-smi --showproductname 2>/dev/null || amd-smi static --asic 2>/dev/null || echo "no rocm-smi/amd-smi"
echo "--- VRAM ---"
rocm-smi --showmeminfo vram 2>/dev/null || echo "n/a"

echo
echo "==================== 3. Python / torch / HIP ===================="
python3 --version
python3 - <<'PY' 2>&1 | head -10
import torch
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", "None"))
print("accelerator available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM: {gb:.1f} GiB")
PY

echo
echo "==================== 4. vLLM / AITER (qwen venv) ===================="
VENV_DIR="${VLLM_VENV:-/root/.venvs/vllm-qwen}"
if [ -x "$VENV_DIR/bin/python" ]; then
  PY="$VENV_DIR/bin/python"
  "$PY" -c "import vllm; print('vllm:', vllm.__version__)" 2>&1 | tail -1
  "$PY" -c "import aiter; print('aiter:', getattr(aiter, '__version__', 'installed'))" 2>&1 | tail -1
  command -v vllm >/dev/null 2>&1 && echo "vllm CLI: $(command -v vllm)" || echo "vllm CLI not in PATH"
else
  echo "qwen venv not found at $VENV_DIR; run scripts/env_setup.sh"
fi

echo
echo "==================== 5. Qwen3.8 architecture support ===================="
if [ -x "$VENV_DIR/bin/python" ]; then
  "$VENV_DIR/bin/python" - <<'PY' 2>&1 || echo ">>> architecture check failed"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    for needle in ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3_5ForConditionalGeneration"):
        status = "OK" if needle in archs else "MISSING"
        print(f"  {status:8s} {needle}")
except Exception as exc:
    print(f"  cannot query registry: {exc}")
PY
else
  echo "  (skipped: venv not created)"
fi

echo
echo "==================== 6. Model checkpoint ===================="
for variant in "$MODEL_ID_BF16" "$MODEL_ID_FP8"; do
  path="$MODEL_ROOT/$variant"
  if [ -d "$path" ]; then
    shards=$(find "$path" -maxdepth 1 -type f \( -name 'model-*.safetensors' -o -name 'layers-*.safetensors' \) 2>/dev/null | wc -l | tr -d ' ')
    echo "$variant: $(du -sh "$path" 2>/dev/null | awk '{print $1}') ($shards shards)"
  else
    echo "$variant: missing"
  fi
done

echo
echo "==================== Conclusion ===================="
echo "This probe is informational. Run scripts/audit_runtime.py before accepting"
echo "performance numbers or changing the serving profile."
