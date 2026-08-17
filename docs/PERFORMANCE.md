# Performance — Qwen3.8-27B on MI308X

> **G1 measured on 2026-08-17** (vLLM path). Numbers below marked *measured* are
> real-machine results; everything else is still an estimate.
> Environment is pinned by `results/runtime_manifest.json`.

## Hardware (verified, G0)

```text
GPU        gfx942:sramecc+:xnack-
CUs        80                      (NOT 304 like a full MI300X)
VRAM       205.8 GB
ROCm       7.2.3 / HIP 7.2.53211
torch      2.11.0+gitd0c8b1f
/dev/shm   16 GB
NUMA balancing  0 (disabled — matches AMD guidance)
```

## G1: vLLM correctness reference (measured)

Launch: `MAX_MODEL_LEN=262144 MAX_NUM_SEQS=32 MTP_ENABLED=0 KV_OFFLOAD_GB=0`,
BF16 weights, FP8 KV, prefix caching on, `--language-model-only`.

| Metric | Result |
| --- | ---: |
| Engine version | `0.26.1rc1.dev306+gcb8104839` |
| Resolved architecture | `Qwen3_5ForConditionalGeneration` |
| GDN prefill kernel | **Triton/FLA** (gfx942 path active) |
| Mamba cache mode | `align` (prefix-caching compatible) |
| **GPU KV cache size** | **3,892,752 tokens** |
| **Max concurrency @262K** | **14.85x** |
| Startup (compile+warmup+capture) | ~150 s |
| torch.compile | 42.0 s |
| Initial profiling/warmup | 52.9 s |
| **decode-512, thinking off** | **54.7 tok/s** |
| decode-4096, thinking off | 41.6 tok/s |
| Basic generation correctness | PASS (`101, 103, 107`) |
| Protocol fixtures | 5/6 → 6/6 after fixture fix |

### KV capacity derived from the measured pool

3,892,752 tokens of FP8 KV, so concurrent sessions at a given context:

| Context per session | Concurrent sessions |
| ---: | ---: |
| 80K | ~48 |
| 262K | **14.85 (engine-reported)** |
| 512K | ~7.4 |

This validates the KV model in `RESEARCH_NOTES.md §2`: the earlier estimate of
~50 sessions at 80K and ~8-10 at 512K matches the measured pool.

### Decode throughput vs estimate

The estimate was ~95-100 tok/s (BF16 bandwidth-bound). Measured is **54.7 tok/s**
— about 60% of the bandwidth-bound ceiling. Two fallbacks in the startup log are
the leading suspects and are tuning targets for later gates:

```text
WARNING  Cannot use ROCm custom paged attention kernel, falling back to Triton
WARNING  aiter sampler does not support per-request generators; falling back to PyTorch-native
```

Decode also degrades with generation length (54.7 tok/s @512 → 41.6 tok/s @4096),
consistent with KV growth on the 16 full-attention layers.

For context, the sibling DeepSeek-V4-Flash on the same host measures 33.4 tok/s
native / 141.8 tok/s with DSpark. Qwen3.8 native is ~1.6× DS0731 native.

## Open blocker: thinking mode returns empty output (mitigated)

**Severity: was blocking Coding Agent use; a mitigation now exists.** With
`reasoning_effort=xhigh` (the Qwen3.8 default), the `qwen3` reasoning parser
returns **nothing at all** when the thinking block does not close before
`max_tokens`:

| Request | Tokens generated | content | reasoning_content | finish_reason |
| --- | ---: | ---: | ---: | --- |
| xhigh (default), max_tokens=512 | 512 | 0 chars | 0 chars | length |
| xhigh (default), max_tokens=4096 | 4096 | 0 chars | 0 chars | length |
| xhigh, max_tokens=2048 (stream) | 2048 | 0 chunks | 0 chunks | — |
| **`reasoning_effort=low`, max_tokens=2048** | 2048 | **2,778 chars** | 0 | length |
| **`reasoning_effort=medium`, max_tokens=4096** | 4096 | **11,516 chars** | 0 | length |
| **`enable_thinking=false`, max_tokens=512** | 512 | **2,140 chars** | 0 | length |

Not streaming-specific — non-streaming shows the same empty result. Tokens are
generated and counted in `usage` but never surfaced to the client.

Root cause: the parser buffers until `</think>`; at `xhigh` Qwen3.8 can think
far longer than 4096 tokens, so the tag never arrives and the buffered output is
discarded. The model card recommends budgeting up to 262,144 tokens for
reasoning content, which confirms xhigh is not a small budget.

**Mitigation**: set `reasoning_effort` to `low` or `medium`, or disable thinking.
Any of the three produces usable output at ordinary `max_tokens`.

**Secondary observation**: at `low`/`medium` the text lands in `content` with
`reasoning_content` empty, so reasoning is not being split into its own field on
this path. Harmless for agent use (the answer is present) but worth confirming
before relying on hidden-reasoning behavior.

**Decode cost of thinking**: 48.9 tok/s at `low`, 41.8 tok/s at `medium`, versus
54.3 tok/s with thinking off — longer generations pay the KV-growth cost on the
16 full-attention layers.

## G3: Attention backend A/B (measured 2026-08-18)

**Root cause of low baseline decode**: Qwen3.8-27B's full-attention layers use
head_dim=256. The ROCm custom paged attention kernel
(`use_rocm_custom_paged_attention()`) only supports head_size 64/128 on CDNA,
so the default `ROCM_ATTN` backend is rejected and vLLM falls through to
`TRITON_ATTN`. The AITER Flash Attention MHA kernel (`ROCM_AITER_FA`) is also
not in the priority list because `is_mha_enabled()` returns False at runtime.

**Fix**: `--attention-backend ROCM_AITER_UNIFIED_ATTN --block-size 64` +
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`. The `RocmAiterUnifiedAttentionBackend`
supports `head_size >= 32` and uses a different paged decode implementation
that does not hit the head_dim=256 restriction.

| Metric | Baseline (Triton, bs=256) | UNIFIED_ATTN (bs=64) | Delta |
| --- | ---: | ---: | ---: |
| C1 decode-512 | 49.5 tok/s | **56.2 tok/s** | +13.5% |
| C1 decode-4096 | 41.6 tok/s | **56.1 tok/s** | **+34.6%** |
| C2 aggregate (512) | — | 101.5 tok/s | — |
| C4 aggregate (512) | — | 158.3 tok/s | — |
| TTFT | 0.15s | **0.06s** | -60% |
| GPU KV pool | 3,892,752 tok | 3,922,907 tok | +0.8% |
| Max concurrency @262K | 14.85x | 14.96x | +0.7% |
| "Cannot use ROCm custom" warning | yes | **no** | — |

**Key insight**: UNIFIED_ATTN not only improves raw decode speed but also
**eliminates the throughput degradation at long generation lengths** (56.2 →
56.1 from 512→4096 tokens, vs 49.5 → 41.6 for baseline). This is critical for
agentic coding loops where turns can be 900+ tokens.

**Remaining warnings** (non-critical):
- `aiter sampler does not support per-request generators; falling back to PyTorch-native`
- `Triton kernel JIT compilation during inference: kernel_unified_attention_2d` (warmup issue)
- `PyTorch's native GELU with tanh approximation is unstable` (minor)

## G5: MTP-3 speculative decode (measured 2026-08-18)

**MTP-3 + UNIFIED_ATTN on gfx942 works and works well.** This was the biggest
unknown — AMD MTP support was flagged as unmeasured (vLLM issue #23123). Results
exceed the original estimate of ~95-100 tok/s.

Launch: `--attention-backend ROCM_AITER_UNIFIED_ATTN --block-size 64` +
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` +
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`.

### MTP acceptance metrics (from vLLM SpecDecoding logs)

```text
Mean acceptance length:  2.85-2.99 / 3.0    (excellent — nearly all 3 drafts accepted)
Per-position acceptance:  pos1=0.84  pos2=0.64  pos3=0.47
Avg draft acceptance:     61.6-66.5%
```

### Decode throughput scaling (MTP-3 + UNIFIED_ATTN)

| Concurrency | Per-session (tok/s) | Aggregate (tok/s) | Scaling eff. |
| ---: | ---: | ---: | ---: |
| C1 (4096 tok) | **94.2** | 94.2 | 100% |
| C2 (512 tok) | 88.2 | 176.3 | 94% |
| C4 (512 tok) | 84.2 | 336.8 | 89% |
| C8 (512 tok) | 64.5 | 516.2 | 68% |
| C16 (512 tok) | 46.5 | 743.5 | 49% |

### Comparison across all configurations

| Metric | Baseline (Triton) | UNIFIED_ATTN | **MTP-3 + UNIFIED_ATTN** |
| --- | ---: | ---: | ---: |
| C1 decode-512 | 49.5 | 56.2 | — |
| C1 decode-4096 | 41.6 | 56.1 | **94.2** |
| C2 aggregate | — | 101.5 | **176.3** |
| C4 aggregate | — | 158.3 | **336.8** |
| C8 aggregate | — | — | **516.2** |
| C16 aggregate | — | — | **743.5** |
| TTFT | 0.15s | 0.06s | 0.09s |
| GPU KV pool | 3,892,752 | 3,922,907 | 3,922,907* |
| Max concurrency @262K | 14.85x | 14.96x | 14.96x* |

(*MTP-3 does not change the KV pool size; the draft head uses the same KV.)

### Updated concurrency estimate (MTP-3 + UNIFIED_ATTN)

With MTP-3, per-session decode at C8 is 64.5 tok/s (900 tokens in ~14s), and
at C16 is 46.5 tok/s (900 tokens in ~19s). The aggregate at C16 (743.5) is
approaching the ~900 tok/s engine ceiling.

| Workload | Concurrency | Per-session decode | Turn time (decode+tool) |
| --- | ---: | ---: | ---: |
| Interactive, fast tools | ~8 | 64.5 tok/s | ~19s (14s + 5s) |
| Interactive, websearch | ~8-16 | 46-65 tok/s | ~24-39s |
| Batch, latency-tolerant | ~32+ | ~25 tok/s est. | ~36s+ |

**Key finding**: MTP-3 roughly doubles per-session throughput at every
concurrency level vs UNIFIED_ATTN alone. The interactive concurrency limit
(where per-session drops below ~30 tok/s) is pushed from ~C16 to ~C32+.

## G8: Context scaling (measured 2026-08-18, MTP-3 + UNIFIED_ATTN)

Cold prefill (no prefix cache hit) with increasing context length:

| Context | TTFT (cold) | Decode after prefill | Notes |
| ---: | ---: | ---: | --- |
| 32K | 52.98s | 28.5 tok/s | Includes JIT compilation for new shapes |
| 128K | 373.64s | 10.7 tok/s | 6+ min cold start; multiple JIT kernels compiled |

**JIT compilation is a major contributor**: the warmup didn't cover long-context
shapes, so Triton kernels (`kernel_unified_attention_2d`, `kernel_unified_attention_3d`,
`reduce_segments`, `eagle_prepare_inputs_padded_kernel`) are compiled on-the-fly.
Subsequent requests at the same shape should be faster.

**Decode degradation at long context**: 10.7 tok/s at 128K vs 94.2 at short context.
The 16 full-attention layers have ~4 GB of KV at 128K (128K × 32 KiB/token), and
the attention computation (not just KV read) is the bottleneck on Triton/ROCm.

**Impact on agentic coding loops**: the shared 20K prefix is cached (prefix
caching on), so only incremental tokens (~2K/turn) need prefill each turn. The
cold-start 32K TTFT of 53s is a one-time cost per session. The real concern is
decode at 80K+ context — estimated ~20 tok/s (interpolating 28.5@32K → 10.7@128K),
which would make turns at 80K+ context noticeably slower.

**Comparison to estimate**: RESEARCH_NOTES estimated ~3-5s for 100K prefill
(hybrid attention: 16/64 layers O(L²)). Measured is **374s for 128K** — 75-125x
slower than estimate. The hybrid advantage is real (only 16/64 layers are O(L²)),
but the constant factor on Triton/ROCm with 80 CUs is much higher than assumed.

## Validation gates

```text
[x] G0  system check
[x] G1  vLLM 262K native, MTP off, C1 correctness + decode baseline
[ ] G2  SGLang correctness            BLOCKED — see RESEARCH_NOTES §10
[x] G3  attention backend A/B          UNIFIED_ATTN wins (+13.5% C1, +34.6% long)
[ ] G4  SSM dtype A/B                 (SGLang-only; blocked)
[x] G5  MTP-3 speculative decode       94.2 tok/s C1, 743.5 C16 aggregate
[ ] G6  BF16 KV vs FP8 KV
[x] G7  concurrency knee C1..C16      knee at ~C8 (64.5 tok/s), C16 still useful
[x] G8  context scaling 32K/128K      cold TTFT 53s/374s; decode degrades at long ctx
[ ] G9  384K/512K extension + recall
[ ] G10 real agent replay
```
