# Validation Plan — Qwen3.8-27B on MI308X

> Gate sequence designed to isolate variables: each gate changes ONE thing.
> Do NOT skip ahead — every gate validates the previous gate's assumptions.
> Core principle: **correctness → engine A/B → state dtype → attention backend
> → MTP → KV dtype → scheduler → concurrency → 512K → real Agent loop**

## Gate G0: System check (no model)

```bash
ssh <gpu-host> 'python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//1e9, \"GB\")"'
rocm-smi --showproductname
# Check: NUMA balancing, transparent hugepage, GPU power limit
cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "n/a"
```

**Pass**: `+gitd0c8b1f` torch, gfx942, 192GB VRAM, 80 CU.

## Gate G1: vLLM correctness reference

```bash
# vLLM · 262K native · BF16 weights · BF16 KV · MTP OFF · C1
MAX_MODEL_LEN=262144 MTP_ENABLED=0 \
  bash scripts/02_serve_vllm.sh qwen38
# Verify: /health 200, /v1/models, short chat, reasoning_content present
```

**Pass**: coherent output with reasoning_content, tool calls parse correctly.

## Gate G2: SGLang correctness reference

```bash
# SGLang · 262K native · float32 SSM · no_buffer · MTP OFF · C1
MAX_MODEL_LEN=262144 MTP_ENABLED=0 \
  bash scripts/02_serve_sglang.sh qwen38
# Verify same as G1
```

**Pass**: same correctness as G1. If SGLang fails to start → fallback to vLLM.

## Gate G3: Attention backend sweep (SGLang only)

```bash
# Triton (default) vs AITER
ATTENTION_BACKEND=aiter MAX_MODEL_LEN=262144 MTP_ENABLED=0 \
  bash scripts/02_serve_sglang.sh qwen38
```

**Measure**: decode-512 tok/s, TTFT p95. Pick winner for subsequent gates.

## Gate G4: SSM dtype A/B (SGLang only)

```bash
# FP32 (correctness reference) vs BF16 (production candidate)
MAMBA_SSM_DTYPE=bfloat16 MAX_MODEL_LEN=262144 MTP_ENABLED=0 \
  bash scripts/02_serve_sglang.sh qwen38
# Run multi-needle recall at 128K/256K/384K
```

**Measure**: decode tok/s, GDN state pool size (from logs), recall accuracy.
BF16 has cumulative drift risk at 200K+. Keep FP32 if recall drops.

## Gate G5: Speculative decoding sweep

```bash
# native / MTP-1 / MTP-2 / MTP-3 (both engines)
MTP_ENABLED=1 MTP_K=1 bash scripts/02_serve_vllm.sh qwen38
MTP_ENABLED=1 MTP_K=2 bash scripts/02_serve_vllm.sh qwen38
MTP_ENABLED=1 MTP_K=3 bash scripts/02_serve_vllm.sh qwen38
# Repeat for SGLang with MTP_STEPS=1/2/3
```

**Measure**: decode-512 tok/s, MTP acceptance rate (from /metrics).
MTP-1 may regress at high concurrency. Find the knee.

## Gate G6: KV dtype A/B

```bash
# BF16 KV (correctness) vs FP8 KV (capacity)
KV_CACHE_DTYPE=fp8 bash scripts/02_serve_vllm.sh qwen38   # vLLM
# SGLang: --kv-cache-dtype fp8 (already default in serve script)
```

**Measure**: KV pool size, max concurrent sessions, recall at 256K.
FP8 KV ≈ 2× capacity. Likely production profile.

## Gate G7: Concurrency knee

```bash
python3 scripts/bench/bench_high_concurrency.py --concurrencies 1 2 5 10 16 24 32
# Also: agent occupancy benchmark (N agents, mixed decode/prefill/idle)
```

**Measure**: aggregate tok/s, per-session tok/s, p95 TTFT.
Find: concurrency where per-session drops below 180 tok/s (5s/turn).

## Gate G8: Context scaling (32K → 256K)

```bash
MAX_MODEL_LEN=262144 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_full.py prefill  # 8K/32K/100K/200K
python3 scripts/bench/bench_long_context_recall.py --lengths 32000 128000 256000
```

**Pass**: all recall needles found, no OOM.

## Gate G9: 384K / 512K extension

```bash
MAX_MODEL_LEN=524288 bash scripts/02_serve_vllm.sh qwen38  # YaRN factor 2.0
python3 scripts/bench/bench_long_context_recall.py --lengths 384000 475000
```

**Measure**: recall at 384K/475K. If 384K drops → 512K is emergency ceiling only.
Determine soft compaction threshold (256K/320K/384K).

## Gate G10: Real agent replay

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000     # 30-turn agent
python3 scripts/bench/bench_session_concurrency.py --sessions 8 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 16 --rounds 4
```

**Measure**: cache hit %, per-turn TTFT, end-to-end session completion time.
This gate decides the production engine (vLLM vs SGLang).

## Post-validation

Replace all estimates in `docs/PERFORMANCE.md` with real numbers.
Record commit hash + manifest + launch args alongside each result.
