# Performance — Estimated Targets (pre-deployment)

> **All numbers on this page are estimates, not validated measurements.**
> Real-machine validation has not yet been performed. See
> [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for the methodology and
> [`VALIDATION_PLAN.md`](VALIDATION_PLAN.md) for the measurement sequence.
>
> Validated reference: the sibling DeepSeek-V4-Flash recipe on the same MI308X
> (80 CU, gfx942, 192 GB) achieved decode-512 141.8 tok/s, C64 aggregate 914.6
> tok/s, 500K endpoint 77.3s. See `deepseek-v4-flash-mi308x/docs/PERFORMANCE.md`.

## Hardware

```text
GPU        MI308X / MI300X class, gfx942
CUs        80 (reported by torch/rocminfo/AITER; NOT 304 like full MI300X)
HBM        ~192 GiB
ROCm       7.2.3
```

## Estimated decode throughput

Derived from the DeepSeek-V4-Flash validated baseline (same hardware) scaled
by per-token compute ratio. See `RESEARCH_NOTES.md §3` for derivation.

| Metric | DeepSeek-V4-Flash (validated) | Qwen3.8-27B (estimated) |
| --- | ---: | ---: |
| decode-512 single stream | 141.8 tok/s | ~95-100 (BF16) / ~250-300 (MTP-3) |
| C1 aggregate | 129.2 tok/s | ~90-100 |
| C2 aggregate | 235.8 tok/s | ~180-200 |
| C4 aggregate | 375.0 tok/s | ~350-400 |
| C8 aggregate | 549.6 tok/s | ~500-550 |
| C32 aggregate | 730.2 tok/s | ~700-750 |
| C64 aggregate | 914.6 tok/s | ~850-920 |

## Estimated long-context

| Metric | DeepSeek-V4-Flash (validated) | Qwen3.8-27B (estimated) |
| --- | ---: | ---: |
| 500K cold prefill | 77.3s | ~20-30s (hybrid attn: 16/64 layers O(L²)) |
| 100K prefill | ~10s | ~3-5s |
| hot TTFT p95 | ~2.9s | est. ~1-2s (hybrid prefill cheaper) |

## Estimated concurrency (agentic coding loop)

| Workload | Concurrency | Binding constraint |
| --- | ---: | --- |
| Interactive, websearch-heavy | ~20-30 | Decode throughput |
| Interactive, pure coding | ~7-10 | Decode throughput |
| Batch / background | ~40-60 | KV memory + admission |
| All at 512K extreme | ~8-10 | KV memory |

## KV cache capacity

| Config | KV pool | 512K sessions | 80K sessions |
| --- | ---: | ---: | ---: |
| BF16 weights + FP8 KV | ~142 GB | ~8 | ~50 |
| FP8 weights + FP8 KV | ~169 GB | ~10 | ~60 |

## Validation gates (all TBD)

```text
[ ] 262K native serve + health
[ ] YaRN factor 2.0 → 512K, 50K→500K ladder survival
[ ] MTP-3 acceptance measurement (C1, C8, C32, C64)
[ ] multi-needle exact recall 100K/256K/384K/475K
[ ] decode ladder C1/C2/C4/C8/C32/C64
[ ] agent trace 30-turn prefix-cache hit
[ ] session concurrency sweep (knee finder)
[ ] cold 200K TTFT isolation
[ ] FP8 vs BF16 A/B
```

When real measurements exist, they replace the estimates on this page and the
file header changes from "Estimated Targets" to "Validated Baseline".
