# Validation Plan — Qwen3.8-27B on MI308X

> Gate sequence isolates variables: correctness → attention backend → MTP → KV
> dtype → scheduler/concurrency → long context → real agent loop.
>
> **Status as of 2026-08-18:** the production vLLM path is validated through
> G0/G1/G3/G5/G7/G8/G10. G9 has passed 512K YaRN startup/capacity, but native
> 256K exact recall remains the production-quality gate; 512K quality is optional.
> G6 is pending. G2/G4 were SGLang-only
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

## Gate G6: KV-cache precision / scale quality — PENDING

The production checkpoint weights are BF16 while the **KV cache** is FP8; these
are independent controls. The gate must measure quality as well as capacity.
Current vLLM documentation distinguishes three FP8 scale sources: unit/default
scales, warmup-calculated scales, and representative-dataset calibration. Do not
collapse them into one "FP8" result.

Required matrix:

1. `KV_CACHE_DTYPE=auto` — model-dtype/BF16 KV quality control.
2. Current production `KV_CACHE_DTYPE=fp8` — record the actual scale source used
   by the pinned runtime/model and all startup warnings.
3. If the pinned ROCm runtime supports calculated/calibrated KV scales without
   changing the validated serving stack, add that as a **separate** FP8 row;
   otherwise record it as a future-runtime experiment, not a production change.

For every row record: KV token capacity, C1 decode/TTFT, deterministic short
fixtures, tool-call contract, exact-recall fixtures at representative long
contexts, and any output divergence. Use the same prompts/seeds across rows.
A capacity win does not promote a KV mode if correctness regresses.

Upstream references:

- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [LLM Compressor FP8 KV calibration example](https://docs.vllm.ai/projects/llm-compressor/en/0.7.0/examples/quantization_kv_cache/)

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

## Gate G9: Native-256K recall first; optional 512K YaRN — PARTIAL

Production is native 262,144 tokens. Validate that ceiling **before** spending
more time on YaRN:

```bash
MAX_MODEL_LEN=262144 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_long_context_recall.py --lengths 128000 192000 240000
```

Only if native-256K exact recall is green and there is still a real product need
for >256K should the optional YaRN row be revisited:

```bash
MAX_MODEL_LEN=524288 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_long_context_recall.py --lengths 256000 384000 475000
```

512K startup/health/KV capacity is already validated; **quality is not**. It is
therefore an experiment ceiling, not a production target.

## Gate G10: Real agent replay — PASS

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 8 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 16 --rounds 4
python3 scripts/bench/bench_tool_roundtrip.py --rounds 5 --mode auto --prefix-tokens 20000
```

Measured: the final 16K/1K scheduler keeps exact-64K short-request interference
near +1.35 s, C8 long-history decode around 33.6 tok/s/session, and C10 around
31.3 tok/s/session. The private gateway's OpenAI Chat live contract is 31/31 PASS
(system/developer normalization, tool modes/history, thinking aliases, Vision,
SSRF denial, structured output, streaming lifecycle, and cancellation). See
`docs/PERFORMANCE.md` for public performance data; private protocol details stay
in `infra/docs/AGENT_CHAT_CONTRACT.md`.

## Future vLLM upgrade gate

Do **not** chase a newer wheel just because the version number is higher. The
pinned ROCm build is promoted and upstream-native. Upgrade only when there is a
specific benefit for this workload, then rerun the whole G0-G10 matrix plus the
private Chat contract.

Highest-value upstream items to watch:

- Hybrid Mamba/GDN prefix caching in `align` mode remains experimental and can
  lose all reuse when the retained Mamba checkpoint falls in request-unique
  tokens. This matches the measured 1600-token alignment cliffs in our Agent
  traces: [vLLM issue #45238](https://github.com/vllm-project/vllm/issues/45238)
- Newer scheduler branches expose richer partial-prefill controls. Evaluate them
  only if they improve long-history TTFT **without** regressing the exact-64K
  short-request isolation gate or C8/C10 decode.
- Qwen3 tool-parser whitespace is a release blocker for exact-match edit tools.
  The current pinned runtime passes `audit_runtime.py`; any upgrade must keep that
  assertion green: [vLLM issue #48753](https://github.com/vllm-project/vllm/issues/48753)
- Prefix caching + speculative decoding changes are behavior-sensitive; compare
  cached tokens, recomputed tokens, TTFT, draft acceptance, and deterministic
  outputs before promotion.

## Remaining validation work

1. G6 KV-cache precision/scale quality matrix (FP8 vs model-dtype/BF16, with
   scale provenance recorded).
2. G9 native-256K exact recall. Revisit 512K YaRN quality only if 256K is green
   and a real workload requires it.
3. Real New API OpenAI-Chat canary over long-lived coding/tool loops; treat
   system/developer shape and prefix-cache hit behavior as observability data.
4. Record every new real-machine result in `docs/PERFORMANCE.md`; keep estimates
   and historical methodology in `docs/RESEARCH_NOTES.md`.
