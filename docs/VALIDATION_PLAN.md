# Validation Plan — Qwen3.8-27B on MI308X

> Gate sequence isolates variables: correctness → attention backend → MTP → KV
> dtype → scheduler/concurrency → long context → real agent loop.
>
> **Status as of 2026-08-18:** the production vLLM path is validated through
> G0/G1/G3/G5/G7/G8/G10. G9 has passed 512K YaRN startup/capacity but the
> 256K/512K recall ladder remains pending. G6 is pending. G2/G4 were SGLang-only
> experiments and are blocked on ROCm (`sgl_kernel` is CUDA-only in this
> environment), so they are not production gates.

## Gate G0: System check — PASS

```bash
ssh <gpu-host> 'python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//1e9, \"GB\")"'
rocm-smi --showproductname
cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "n/a"
```

Validated: ROCm platform torch, gfx942, 192 GB-class VRAM, 80 CU, NUMA balancing
disabled. Exact measured environment is recorded in `docs/PERFORMANCE.md`.

## Gate G1: vLLM correctness reference — PASS

```bash
ATTENTION_BACKEND= BLOCK_SIZE=256 MAX_MODEL_LEN=262144 \
MTP_ENABLED=0 KV_OFFLOAD_GB=0 KV_CACHE_DTYPE=fp8 \
  bash scripts/02_serve_vllm.sh qwen38
```

Validated: `/health`, `/v1/models`, short generation, reasoning/tool protocol
fixtures, and the native-decode baseline. See `docs/PERFORMANCE.md` for numbers.

## Gate G2: SGLang correctness reference — BLOCKED / RETIRED

The SGLang path is not executable in this DSW ROCm environment because the
available `sgl_kernel` wheel is CUDA-only (`libnvrtc.so.13`). The reference
launcher remains in the repository for research, but production validation does
not wait on this gate.

## Gate G3: vLLM attention backend A/B — PASS

The actual sweep compared the default/fallback path against
`ROCM_AITER_UNIFIED_ATTN` for Qwen3.8's head_dim=256 layers.

```bash
# Control: automatic backend
ATTENTION_BACKEND= BLOCK_SIZE=256 MTP_ENABLED=0 KV_OFFLOAD_GB=0 \
  bash scripts/02_serve_vllm.sh qwen38

# Promoted backend
ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN BLOCK_SIZE=64 \
MTP_ENABLED=0 KV_OFFLOAD_GB=0 \
  bash scripts/02_serve_vllm.sh qwen38
```

Result: UNIFIED_ATTN removes the head_dim=256 fallback and improves decode,
especially at longer generations. It is now the launcher default.

## Gate G4: SSM dtype A/B — NOT APPLICABLE

This was SGLang-only. Because G2 is blocked, G4 is retired for the current
vLLM production recipe.

## Gate G5: MTP sweep — PASS

```bash
MTP_ENABLED=0 bash scripts/02_serve_vllm.sh qwen38
MTP_ENABLED=1 MTP_K=1 bash scripts/02_serve_vllm.sh qwen38
MTP_ENABLED=1 MTP_K=2 bash scripts/02_serve_vllm.sh qwen38
MTP_ENABLED=1 MTP_K=3 bash scripts/02_serve_vllm.sh qwen38
```

Result: MTP-3 is promoted; mean acceptance is about 65%, C1 reaches 94.2 tok/s,
and C32 aggregate reaches 1094 tok/s. MTP-1 regresses at high concurrency.

## Gate G6: BF16 KV vs FP8 KV — PENDING

```bash
# Current validated serving path uses FP8 KV.
KV_CACHE_DTYPE=fp8 bash scripts/02_serve_vllm.sh qwen38
# Model-dtype control (BF16 when QUANT=bf16).
KV_CACHE_DTYPE=auto bash scripts/02_serve_vllm.sh qwen38
```

Pending: run the model-dtype/BF16 correctness-capacity control and quantify the
quality/capacity tradeoff. Do not infer this result from weight quantization.

## Gate G7: Concurrency knee — PASS

```bash
python3 scripts/bench/bench_high_concurrency.py --concurrencies 1 2 5 10 16 24 32
```

Measured through C32. The interactive knee and aggregate throughput are recorded
in `docs/PERFORMANCE.md`.

## Gate G8: Context scaling — PASS for measured 32K/128K range

```bash
MAX_MODEL_LEN=262144 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_full.py prefill
python3 scripts/bench/bench_long_context_recall.py --lengths 32000 128000
```

Warm 128K prefill is fast; the very large cold latency was dominated by Triton
JIT. Decode slows materially at long context, so warmup coverage matters.

## Gate G9: 512K YaRN extension — PARTIAL

```bash
MAX_MODEL_LEN=524288 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_long_context_recall.py --lengths 256000 475000
```

Validated: 512K YaRN server startup/health and KV capacity. Pending: the
256K/512K-class exact-recall ladder. Until that finishes, 512K is a validated
serving ceiling, not a fully validated quality ceiling; correctness-sensitive
production can pin `MAX_MODEL_LEN=262144`.

## Gate G10: Real agent replay — PASS

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 8 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 16 --rounds 4
python3 scripts/bench/bench_tool_roundtrip.py --rounds 5 --mode auto --prefix-tokens 20000
```

Measured: 30-turn prefix-cache hit 84%, warm TTFT below 2.5 s in the trace, and
5/5 tool-call round trips.

## Remaining validation work

1. G6 BF16-KV vs FP8-KV correctness/capacity A/B.
2. G9 256K/512K-class long-context exact recall.
3. Optional scheduler/isolation sweeps (`MAX_BATCHED_TOKENS` 2048/4096/8192).
4. Record every new real-machine result in `docs/PERFORMANCE.md`; keep estimates
   and methodology in `docs/RESEARCH_NOTES.md`.
