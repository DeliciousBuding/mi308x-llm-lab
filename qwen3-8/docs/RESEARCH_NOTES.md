# Research Notes — Qwen3.8-27B Concurrency & ROCm Support

> Status: **pre-deployment research + estimation record**. Sections that derive
> capacity/throughput from first principles are intentionally preserved as the
> original hypothesis; measured Qwen3.8 results from 2026-08-17/18 supersede
> those estimates wherever they differ. The measurement SSOT is
> [`PERFORMANCE.md`](PERFORMANCE.md), and live gate status is in
> [`VALIDATION_PLAN.md`](VALIDATION_PLAN.md).
>
> Validation delta: MTP-3 acceptance is ~65%; measured C1 is 94.2 tok/s and C32
> aggregate is 1094 tok/s; FP8 KV pool is ~3.92M tokens at the native profile;
> interactive concurrency is ~8-16 rather than the original ~20-30 estimate;
> SGLang is blocked on ROCm in this environment; 512K YaRN startup/capacity is
> validated but the 256K/512K recall ladder and G6 KV-dtype A/B remain pending.

## 1. Model architecture (the decisive fact)

From the official Hugging Face model card (`Qwen/Qwen3.8-27B`, verified 2026-08-15):

```text
Type                 Causal Language Model with Vision Encoder
Parameters           27B (27.8B counting vision tower + padded vocab)
Hidden dimension     5,120
Token embedding      248,320 (padded)
Layers               64
Hidden layout         16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
                       = 48 Gated DeltaNet layers + 16 Gated Attention layers
Gated DeltaNet       48 V heads, 16 QK heads, head_dim 128
Gated Attention       24 Q heads,  4 KV heads, head_dim 256
RoPE dim             64
FFN intermediate     17,408
MTP                  trained with multiple steps
Native context       262,144 → extensible to 1,010,000 via YaRN
License              Apache 2.0
Released             2026-08-13/14
```

**The layer mix is the entire point.** Only 16 of 64 layers (25%) run full
GQA attention with a context-growing KV cache. The other 48 run Gated DeltaNet
linear attention with a **constant-size recurrent state** that does not grow
with sequence length. This is why vLLM's recipe reports 6.6M KV-token capacity
even at the 1M-context extension on a single Blackwell GPU.

## 2. KV cache math

### 2.1 Per-token KV volume

Full-attention layers only (16 of 64). The 48 DeltaNet layers have a constant
recurrent state (~75 MB per session, not per-token) folded into overhead.

Per-token per full-attn layer = `num_kv_heads × head_dim × 2 (K+V) × dtype_bytes`
= `4 × 256 × 2 × dtype_bytes`

| KV dtype | per token per layer | × 16 layers | per 512K session |
| ---: | ---: | ---: | ---: |
| BF16 (2 B) | 4 KiB | **64 KiB** | 32 GB |
| FP8 (1 B) | 2 KiB | **32 KiB** | 16 GB |

### 2.2 192 GB memory budget

Same hardware as the DeepSeek-V4-Flash deployment: MI308X / gfx942 / 80 CU /
~192 GiB HBM. The pre-deployment estimate initially borrowed the DeepSeek
recipe's validated `KV_CACHE_BYTES=16GB` GPU pin + `KV_OFFLOAD_GB=12` CPU tier.
**That assumption was later invalidated for Qwen3.8**: native CPU-KV offload
hits `madvise(EINVAL)`, so measured Qwen capacity uses GPU-only KV. The table
below is retained only to show the original estimate; use `PERFORMANCE.md` for
measured capacity.

| Weight config | Weights | Available for KV + overhead | Original estimated pool | 512K extreme | 80K typical |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 weights + FP8 KV | 54 GB | ~130 GB | ~142 GB (incl 12 GB CPU) | ~8 sessions | ~50 sessions |
| **FP8 weights + FP8 KV** | 27 GB | ~157 GB | ~169 GB | ~10 sessions | ~60 sessions |

(80K column: 80,000 × 32 KiB = 2.5 GB per session; effective pool ÷ 2.5 GB)

### 2.3 Contrast with DeepSeek-V4-Flash (MLA)

DeepSeek-V4-Flash uses MLA (Multi-head Latent Attention), compressing KV to
~2-4 KiB per token (FP8). A 512K single session needs only ~1-2 GB. This is why
the DeepSeek recipe can run `MAX_NUM_SEQS=64` at the 524K ceiling with 16 GB
GPU + 12 GB CPU KV.

Qwen3.8-27B's full-attention GQA (4 KV heads × 256 dim on 16 layers) is
~8-16× larger per token than MLA. The hybrid architecture (only 16/64 layers)
offsets this partially, but 512K extreme concurrency is still KV-bound at
~8-10 sessions. **512K is a ceiling, not a working point.**

## 3. Decode throughput ceiling (DeepSeek-V4-Flash as hardware proxy)

Validated on the same MI308X (80 CU, gfx942), from `SERVE_OPS.md §9`:

```text
                    warm decode-512    C32 aggregate    C64 aggregate
DSpark K7                    141.7          730.2            914.6 tok/s
native                        33.4          570.4            921.6 tok/s
```

- C64 aggregate ~915-922 tok/s is this host's **engine-wide decode ceiling**
  (compute-bound; speculation stops helping at high batch).
- Qwen3.8-27B dense 27B has ~54 GFLOPs/token forward (2N convention), comparable
  to DeepSeek-V4-Flash MoE's per-token active compute. **Aggregate decode
  ceiling: ~900 tok/s** (same order of magnitude on the same hardware).
- Single-stream: DeepSeek-V4-Flash is 141.8 tok/s. Qwen3.8-27B BF16 weights
  (54 GB) at ~5 TB/s HBM bandwidth ≈ bandwidth-bound ~95-100 tok/s. With MTP-3
  (if acceptance is high): ~250-300 tok/s. **Single-stream likely below
  DeepSeek-V4-Flash** (dense weights are heavier to read per token than MoE
  active params), but MTP closes the gap at low batch.

## 4. Agentic coding-loop workload model

From the vLLM × Mooncake blog (Codex/SWE-bench Pro corpus, 610 traces) and the
CacheWise paper (arXiv 2606.16824):

```text
shared prefix (sysprompt + skills + tooldefs + AGENTS.md)   ~20K tokens (shared)
context growth                                                ~2,242 tokens/turn
turn-30 context                                               ~80K (median)
long-tail context                                             >180K
input:output ratio                                            ~131:1
prefix cache hit                                              94.2%
inter-turn delay (tool exec)                                  5.2s median / 81.4s P99
decode per turn                                               ~900 tokens
```

**Agent-coding + websearch correction**: web search adds latency (search API
1-3s + page fetch 1-5s + reading results), pushing the median inter-turn delay
from 5.2s to ~10-20s.

## 5. Concurrency estimate (three constraints, take the minimum)

Engine aggregate decode = ~900 tok/s. Each turn generates ~900 tokens.

Concurrency bound = `R_engine × T_turn / 900`, where `T_turn = T_decode + T_tool`.

| Scenario | T_decode | T_tool | T_turn | decode-bound | KV-bound (80K) | KV-bound (512K) | **min** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Interactive, pure code, 2s/turn | 2s | 5s | 7s | 7 | 50 | 8 | **7-8** |
| Interactive, pure code, 5s/turn | 5s | 5s | 10s | 10 | 50 | 8 | **8-10** |
| **Interactive, websearch-heavy** | 3s | 17s | 20s | 20 | 50 | 8 | **20** |
| Interactive, websearch-heavy | 3s | 27s | 30s | 30 | 50 | 8 | **30** |
| Batch, latency-tolerant | 30s | 30s | 60s | 60 | 50 | 8 | **50-60** |
| Batch, all at 512K extreme | 30s | 30s | 60s | 60 | — | 8 | **8-10** |

`MAX_NUM_SEQS=64` is the admission hard cap; all rows ≤ 64.

## 6. Original pre-deployment conclusion (superseded by measured results)

| Workload | Concurrency | Binding constraint |
| --- | ---: | --- |
| **Interactive, websearch, 2-5s decode/turn** | **~20-30** | Decode throughput |
| Interactive, pure code, fast tools | ~7-10 | Decode throughput |
| Batch / background, latency-tolerant | ~40-60 | KV memory + admission |
| All sessions at 512K extreme | ~8-10 | KV memory |

**One-liner**: on the 192 GB MI308X, Qwen3.8-27B @ 512K for agent-coding loops
with websearch supports ~20-30 concurrent interactive sessions (decode-bound);
~50-60 if latency-tolerant (KV/admission-bound). The 512K ceiling is a
worst-case memory limit — simultaneously filling it caps at ~8-10 sessions.

## 7. Contrast with DeepSeek-V4-Flash (validated)

| Dimension | DeepSeek-V4-Flash (validated) | Qwen3.8-27B (original estimate; see PERFORMANCE for measured) |
| --- | --- | --- |
| MAX_NUM_SEQS | 64 | 64 (same admission) |
| 524K/512K extreme concurrency | ~64 (MLA KV tiny) | ~8-10 (GQA KV 8-16× larger) |
| 80K typical concurrency | ~64 (KV not binding) | ~50-60 |
| decode-512 single stream | 141.8 tok/s | ~95-100 BF16 / ~250-300 MTP-3 |
| C64 aggregate | 914.6 tok/s | ~900 (same hardware ceiling) |
| 500K cold prefill | 77.3s | est. ~20-30s (hybrid: only 16/64 layers O(L²)) |
| Interactive agent concurrency | ~20-30 (same decode bound) | ~20-30 (same decode bound) |

**Key asymmetry**: DeepSeek-V4-Flash wins on 524K extreme concurrency (MLA).
Qwen3.8-27B wins on long-prefill TTFT (hybrid attention). At the typical
agentic working point (80K, 94% prefix-cache hit), both are decode-bound at
~20-30 interactive / ~50-60 batch.

## 8. ROCm / gfx942 support evidence

### 8.1 AMD Day-0 announcement (2026-08-12)

AMD announced Day-0 support for Qwen 3.8 on MI300X/MI325X/MI355X with vLLM and
SGLang. The official vLLM serve command from AMD's article (for the 2.4T MoE):

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --max-model-len 32768 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.8 \
  --reasoning-parser qwen3 \
  --language-model-only --no-enable-prefix-caching --mamba-cache-mode none
```

Note: AMD's 2.4T config uses `--mamba-cache-mode none` and
`--no-enable-prefix-caching` for the MoE multi-node setup. For the single-GPU
27B dense agentic workload, we **want** prefix caching (94% hit rate) and the
DeltaNet recurrent state cache enabled. These are tuning decisions for
validation, not blockers.

### 8.2 Gated DeltaNet kernel optimization for gfx942

[vllm-project/vllm#41446](https://github.com/vllm-project/vllm/pull/41446)
"Optimize GatedDeltaNet FLA prefill kernels on MI300X":

- Fused `fwd_h + fwd_o` kernel keeping recurrent state in registers.
- Shape-aware dispatch (fused wins 1.15× at B=1 T≤1024).
- AMD MFMA hint expansion (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`).
- Single static Triton Config per kernel (eliminates runtime autotune cost).
- **Result: 1.43× kernel-level speedup (geomean over 16 shapes).**
- **E2E on Qwen3.5-122B-A10B-FP8: 27/27 TTFT wins, 0 regressions.**
- BugFix: `act_quant_fusion.py` ROCm `hasattr` guard for `silu_and_mul_per_block_quant`.

This directly addresses the #1 risk from the initial estimate: the Gated
DeltaNet linear-attention path IS optimized for gfx942, not a naive fallback.

### 8.3 Qwen3.8 enablement for AMD ROCm

[vllm-project/vllm#50068](https://github.com/vllm-project/vllm/pull/50068)
"Enable Qwen3.8 for AMD Rocm":

- Registers text-only `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`.
- Advertises hybrid and M-RoPE support on the causal implementation.
- Exposes Gated DeltaNet Mamba cache dtype/shape/copy metadata.
- Validated with a two-node ROCm vLLM deployment (TP=8, PP=2).
- Approved and auto-merge enabled (2026-08-03/07).

### 8.4 MTP (speculative decode)

From the vLLM recipe for Qwen3.8-2.4T-A95B:

```text
MTP-3: ~2.3× per-user output rate (recommended)
MTP-1: 64.8% acceptance, NOT worth it (+3.4% at C1, -9% at C128, -23% at C256)
       "the draft pass displaces real work once the batch is compute-bound"
```

MTP config: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`

The model card confirms Qwen3.8-27B has MTP ("trained with multiple steps").
The vLLM MTP docs list `method: mtp` as the native multi-token path. This was a
pre-deployment uncertainty; G5 later measured **~65% mean MTP-3 acceptance** on
gfx942 and promoted MTP-3 as the default. See `PERFORMANCE.md` for the measured
C1/C32 curve.

### 8.5 Checkpoint inventory

| Variant | Repo | Size | Shards | Notes |
| --- | --- | ---: | ---: | --- |
| BF16 | `Qwen/Qwen3.8-27B` | 55.6 GB | 18 | reference quality |
| FP8 | `Qwen/Qwen3.8-27B-FP8` | ~28 GB | varies | block-size-128 fine-grained FP8 |
| FP8 (unsloth) | `unsloth/Qwen3.8-27B-FP8` | ~28 GB | varies | may include KV-cache calibration |
| NVFP4 | `Inferact/Qwen3.8-27B-NVFP4` | 24.6 GiB | — | NVIDIA-only (not for gfx942) |

**MXFP4 does not load on NVIDIA devices** (vLLM missing linear-method support).
On AMD/gfx942, the FP8 checkpoint is the right quantization path.

## 9. Remaining risks (updated 2026-08-18)

1. **80-CU compute is real**: this host reports 80 CUs, not MI300X's 304. The
   DeepSeek-V4-Flash 914 tok/s is the validated ceiling proxy. Published
   MI300X benchmarks will overestimate by ~3-4×.
2. **Long-context quality ceiling**: 512K YaRN startup and KV capacity are
   validated, but the 256K/512K-class exact-recall ladder is still pending.
   Treat 512K as a serving ceiling rather than a fully validated quality ceiling.
3. **GDN state allocation** (vllm-metal#400): Qwen3.5 hybrid models allocate
   GDN recurrent state for `max_num_seqs` at startup. With 48 GDN layers,
   this can be ~5 GB at `max_num_seqs=64` on top of the KV budget. **Start
   with `max_num_seqs=32`** to avoid startup OOM; raise only after measuring
   actual GDN state size.
4. **Mixed batch spec-decode crash** (vllm#36918): GDN attention backend
   crashes when a batch contains both regular decode and speculative-decode
   requests (happens when concurrent requests approach `max_model_len` at
   different times). Fix merged upstream; verify dev306 includes it. If the
   crash recurs, disable MTP as a workaround.
5. **FP8 KV cache calibration**: the official FP8 checkpoint may lack KV-cache
   calibration scales (`--kv-cache-dtype fp8` may warn). Workaround: use the
   unsloth FP8 variant, or accept BF16 KV (64 KiB/token, halves KV capacity).
6. **YaRN factor 2.0 quality**: 262K → 512K degrades short-context quality.
   Agent-loop early turns (12K context) should be regression-tested against
   the native 262K baseline.
7. **AITER tuning coverage**: the 80-CU MI308X AITER tables in the sibling
   DeepSeek repo are keyed for DeepSeek GEMM shapes. Qwen3.8's dense GEMMs are
   more standard and may work with default AITER tables, but a tuning pass is
   a post-deployment optimization.
8. **CPU-KV offload is unsafe for Qwen3.8 on this sandbox**: native offload
   hits `madvise(MADV_POPULATE_WRITE)` EINVAL. The launcher therefore defaults
   to `KV_OFFLOAD_GB=0`; only re-enable it for an explicit regression test.

## 10. Remaining validation work

The live gate checklist is `VALIDATION_PLAN.md`; do not maintain a second full
sequence here. The unresolved research questions are:

1. **G6 KV dtype A/B**: compare current FP8 KV against model-dtype (`auto` on
   BF16 weights) for capacity, recall, and decode stability.
2. **G9 long-context quality**: complete the 256K/512K-class exact-recall
   ladder before calling 512K a validated quality ceiling.
3. **Scheduler/isolation tuning**: sweep `MAX_BATCHED_TOKENS` and long-prefill
   isolation only if it materially improves the measured agent-loop profile.
4. **AITER tuning coverage**: tune Qwen3.8 dense GEMM shapes only after the
   correctness gates above are closed.
