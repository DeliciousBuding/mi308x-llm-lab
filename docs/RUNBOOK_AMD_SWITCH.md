# Runbook — AMD Instance Switch for Qwen3.8-27B

> Copy-paste-ready guide. Run these commands on the AMD GPU instance after the
> user switches from CPU to GPU. First boot = full install; subsequent boots =
> restore from snapshot.

## 0. Network bootstrap (every boot)

```bash
bash /mnt/workspace/bootstrap.sh
```

Confirms sshd + reverse tunnels. The agent then takes over the session over SSH.

## 1. Restore runtime (every boot)

```bash
bash /mnt/workspace/infra/restore_runtime.sh
```

Restores DS0731 venv, JIT caches (Triton/COMGR/AITER), and — if the snapshot
exists — the Qwen3.8 venv (`vllm-qwen.tar.gz`). First boot will warn that the
Qwen venv snapshot is missing; proceed to step 2.

## 2. Create Qwen3.8 venv (first boot only)

Skip this step if `restore_runtime.sh` reported "Qwen3.8 venv already present"
or "Qwen3.8 venv restored".

```bash
cd /mnt/workspace/qwen3-8-27b-mi308x
bash scripts/env_setup.sh
bash scripts/install_vllm_nightly.sh
```

`install_vllm_nightly.sh` verifies that `Qwen3_5ForCausalLM` is registered in
the dev306 wheel before declaring success. No fork overlays are applied.

## 3. Audit the runtime

```bash
python3 /mnt/workspace/qwen3-8-27b-mi308x/scripts/audit_runtime.py
```

Must report `AUDIT PASSED` before proceeding. Checks vLLM version, AITER
import, Qwen3.8 architecture registration, and GDN linear-attention path.

## 4. Stage model to local SSD (recommended)

```bash
bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/stage_model_local.sh
```

Copies the 52 GB BF16 checkpoint from NFS to local SSD. The launcher
auto-prefers the local hot copy. Skippable if NFS read latency is acceptable.

## 5. Serve Qwen3.8-27B (vLLM — the validated path)

vLLM is the validated production path (G-gate campaign, 2026-08-18). SGLang was
evaluated and **abandoned** for Qwen3.8: `sgl_kernel` ships CUDA-only wheels
(`libnvrtc.so.13`), so SGLang cannot install on ROCm without Docker, and DSW
has no Docker. `02_serve_sglang.sh` is retained for reference only.

```bash
# In shell 1:
export VLLM_API_KEY_FILE=/mnt/workspace/.bootstrap/vllm_api_key
ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN BLOCK_SIZE=64 \
MTP_ENABLED=1 MTP_K=3 KV_OFFLOAD_GB=0 MAX_MODEL_LEN=524288 \
  bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/02_serve_vllm.sh qwen38
```

Wait for `/health` to return 200 (check with `curl -s http://localhost:8000/health`).

The two non-obvious knobs (see README "Known ROCm gotchas"):

- `ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN` + `BLOCK_SIZE=64` — required for
  head_dim=256; the default backend falls back to Triton and loses 13-35% decode.
- `KV_OFFLOAD_GB=0` — GPU-only KV; CPU-KV offload hits a `madvise` EINVAL on this
  sandbox (unlike the DeepSeek sibling, which runs a 12 GB CPU tier).

## 6. Warm up + snapshot (first boot only)

```bash
# In shell 2, after /health is ready:
SNAPSHOT_AFTER_WARMUP=1 \
  bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/warmup_runtime.sh
```

Covers real first-use JIT paths (Gated DeltaNet Triton kernels, MTP) and
persists the validated venv + cache snapshots to NFS. Subsequent boots restore
from these snapshots instead of rebuilding.

## 7. Gateway: LiteLLM + Studio (optional)

The gateway normalizes requests and exposes a single external API. To serve
Qwen3.8 through the gateway instead of raw vLLM:

### 7a. Update `.env` (if serving Qwen3.8 instead of DS0731)

```bash
# In /mnt/workspace/infra/.env, change:
MODEL_NAME=qwen3.8-27b
```

The LiteLLM config (`studio_gateway/config.yaml`) already has both
`deepseek-v4-flash` and `qwen3.8-27b` entries. The `MODEL_NAME` env var tells
the gateway which model to health-check and which model name the Studio UI uses.

### 7b. Restore/start the gateway

```bash
bash /mnt/workspace/infra/studio_gateway.sh restore
bash /mnt/workspace/infra/studio_gateway.sh health
```

First boot: `install` instead of `restore` (installs LiteLLM + Studio venvs
from pip, then snapshots them for future restore).

### 7c. External API topology

```text
client → public HTTPS endpoint (Cloudflare Tunnel) → LiteLLM → vLLM   (OpenAI API; LiteLLM master key required, no key → 401)
Studio UI → local SSH forward only, not publicly exposed
```

LiteLLM normalizes: `max_tokens` alias folding, output ceiling clamp
(`MAX_OUTPUT_TOKENS_CEILING=65536`), null-field stripping.

## 8. Validation (follow VALIDATION_PLAN.md gates)

```bash
# Quick smoke test
curl -s http://localhost:8000/health
python3 /mnt/workspace/qwen3-8-27b-mi308x/scripts/bench/bench_full.py decode

# Tool-call round-trip
python3 /mnt/workspace/qwen3-8-27b-mi308x/scripts/bench/bench_tool_roundtrip.py \
  --rounds 5 --mode auto --prefix-tokens 20000

# MTP vs native A/B
# (with MTP-3, note the decode rate and MTP acceptance from logs)
MTP_ENABLED=0 bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/02_serve_vllm.sh qwen38
python3 /mnt/workspace/qwen3-8-27b-mi308x/scripts/bench/bench_full.py decode

# YaRN 512K extension (after 262K native is validated)
MAX_MODEL_LEN=524288 bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/02_serve_vllm.sh qwen38
python3 /mnt/workspace/qwen3-8-27b-mi308x/scripts/bench/bench_full.py all
```

## 9. Full benchmark suite

```bash
bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/03_benchmark.sh
```

Record new real-machine results in `docs/PERFORMANCE.md`, update gate status in `docs/VALIDATION_PLAN.md`, and keep pre-deployment estimates/methodology in `docs/RESEARCH_NOTES.md` as historical research context.

## Quick reference: serve config knobs

```bash
MAX_MODEL_LEN=524288          # YaRN factor 2.0 over 262K native
MAX_NUM_SEQS=32               # conservative (GDN state + 80-CU); 64 for batch
MAX_BATCHED_TOKENS=3072       # coding-agent latency profile
MTP_ENABLED=1                 # 1=MTP-3 (latency); 0=native (throughput)
MTP_K=3                       # MTP depth; 1 regresses at high concurrency
ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN  # required: head_dim=256
BLOCK_SIZE=64
KV_OFFLOAD_GB=0               # GPU-only; CPU offload unstable for Qwen3.8
QUANT=bf16                     # bf16 (reference) or fp8 (max KV headroom)
LANGUAGE_MODEL_ONLY=1          # skip vision encoder for text-only serving
GPU_MEMORY_UTILIZATION=0.95
```

## First boot vs subsequent boots

| Step | First boot | Subsequent boots |
| --- | --- | --- |
| bootstrap.sh | ✅ | ✅ |
| restore_runtime.sh | ✅ (warns on missing Qwen venv) | ✅ (restores Qwen venv from snapshot) |
| env_setup.sh + install | ✅ required | skip (restored) |
| stage_model_local.sh | ✅ recommended | ✅ (ephemeral, lost on rebuild) |
| serve | ✅ | ✅ |
| warmup + snapshot | ✅ required | skip (caches restored) |
| gateway restore | ✅ (or install) | ✅ |
