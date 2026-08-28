# qwen3-8-mi308x

<div align="center">

**Qwen3.8 series on a single AMD Instinct MI300X / MI308X (gfx942), served with native vLLM on ROCm.**

Two profiles, one upstream-native stack: Qwen3.8-27B (dense hybrid attention, validated) · Qwen3.8-Flash-Next (125B MoE + Qwen Sparse Attention, FP8, in preparation) · MTP speculative decode · prefix caching · agentic-coding workload target · no Docker required · no fork patches

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)
[![vLLM](https://img.shields.io/badge/vLLM-dev306-4B32C3)](https://github.com/vllm-project/vllm)
![GPU](https://img.shields.io/badge/GPU-gfx942%20%7C%20192GB-ED1C24)

</div>

> **Repository scope (2026-08-29).** Renamed from `qwen3-8-27b-mi308x` (GitHub
> redirects the old name): this repository is the home of the Qwen3.8 series on
> gfx942. Both models share one upstream-native vLLM stack, one benchmark suite,
> and one serving venv; the single 192 GB card serves one model at a time.
>
> **Qwen3.8-27B — validated, frozen (2026-08-18).** Production vLLM path
> validated on the 192 GB gfx942 host: G0/G1/G3/G5/G7/G8/G10 passed. G9
> validated 512K YaRN startup + KV capacity; the native-256K recall ladder and
> the G6 KV precision/scale control were not completed and are no longer
> pursued. The SGLang-only G2/G4 path is blocked on ROCm (`sgl_kernel` is
> CUDA-only) and is not part of the recipe. Headline results: final production
> warmup C1 decode ~100-101.5 tok/s, C32 aggregate 1094 tok/s, C8/C10
> long-history decode ~33.6/~31.3 tok/s per session, exact-64K short-request
> interference ~+1.35 s. Real-machine numbers in
> [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md); the pre-deployment estimates in
> [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) are retained for
> methodology only.
>
> **Qwen3.8-Flash-Next — in preparation (2026-08-29).** Weights staging and
> repo consolidation done; engine support and gfx942 validation pending. See the
> [Flash-Next profile section](#qwen38-flash-next-profile-in-preparation).

## Qwen3.8-Flash-Next profile (in preparation)

Qwen3.8-Flash-Next (released 2026-08-26) is the open-weight preview of the
upcoming Qwen4 architecture: an ultra-sparse multimodal MoE with a 125B main
model (6B activated per token), a 51B N-gram embedding table, a 4B MTP head,
and hybrid Gated DeltaNet + Qwen Sparse Attention (QSA).

```text
layers             48 = 12 x (3 x GDN -> MoE + 1 x QSA -> MoE)
experts            512 (10 routed + 1 shared active per token)
n-gram embedding   51B bigram/trigram lookup at layer 2
QSA                24 Q heads / 2 KV heads, head_dim 256,
                   indexer budget 512 blocks / 2048 tokens
context            262,144 native, YaRN-extensible to ~1M
MTP                1 layer, multi-step (native speculative decode)
```

### Checkpoint choice: FP8-only on one 192 GB card

| checkpoint | size | single gfx942 card (205.8 GB VRAM) |
| --- | ---: | --- |
| BF16 | ~335 GiB | does not fit |
| **FP8** (official) | **~172.8 GiB** | fits, ~20 GB headroom for KV/activations |

```bash
bash scripts/01_download_model.sh flashnext-fp8   # ~185.6 GB, 131 shards
```

QSA KV stays compact (2 KV heads on 12/48 layers ≈ 12 KiB/token), so the
remaining headroom still supports long-context sessions.

### Engine status (2026-08-29, not yet run on this host)

- Model support requires vLLM containing
  [vllm-project/vllm#53896](https://github.com/vllm-project/vllm/pull/53896)
  (optionally #53899 for PLE offload). The dev306 wheel validated for
  Qwen3.8-27B **predates this model**; the serving venv must be upgraded and
  SHA-pinned before first launch.
- The official [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
  validates GB300 / GB200 / H200 and MI355X (BF16 only). **gfx942 is not yet
  validated** — the QSA attention kernels and indexer are the main unknowns.
  The QSA layers also use head_dim 256, the same shape the ROCm custom
  paged-attention kernel rejects on Qwen3.8-27B (see Known ROCm gotchas).
- N-gram embedding offload (PLE, `VLLM_PLE_CPU_OFFLOAD=1`) is NVIDIA-only at
  the moment, so on AMD the full 51B embedding table must live in HBM — that
  is what makes the FP8 memory budget tight.
- Single-card TP1 has no official validation (the recipe's validated minimum is
  TP2 on GB300); expect the first-launch matrix to be exploratory.

Recipe starting point (unvalidated here): `--gpu-memory-utilization 0.90
--max-num-seqs 256 --enable-prefix-caching --tool-call-parser qwen3_xml
--reasoning-parser qwen3`.

## Why Qwen3.8-27B on gfx942

Qwen3.8-27B is the dense 27B member of the Qwen3.8 family, on the same
**hybrid-attention backbone** as the 2.4T MoE flagship. The architecture is the
reason this model matters for long-context agentic serving on a single 192 GB
GPU:

```text
layers               64
full-attention layers 16/64   (GQA: 24 Q heads, 4 KV heads, head_dim 256)
linear-attn layers    48/64   (Gated DeltaNet — constant recurrent state)
native context        262,144 → YaRN factor 2.0 → 524,288
MTP                   trained natively (speculative decode without a draft model)
```

**Only 16 of 64 layers have a context-growing KV cache.** The other 48 run
Gated DeltaNet linear attention with a fixed-size recurrent state. This makes
long-context prefill dramatically cheaper than a conventional 64-layer
full-attention model, and it is why vLLM's recipe reports 6.6M KV-token
capacity even at the 1M-context extension on a single GPU.

### Contrast with the sibling DeepSeek-V4-Flash recipe

The sibling repo `deepseek-v4-flash-mi308x` serves a MoE model with MLA
(Multi-head Latent Attention). This repository serves a **dense** model with
**standard GQA on 16 layers + linear attention on 48 layers**. Key differences:

| Dimension | DeepSeek-V4-Flash-0731 | Qwen3.8-27B (this repo) |
| --- | --- | --- |
| Model type | MoE + MLA | dense + hybrid attention |
| Patch source | pinned fork (ryanzhou) + 18 overlays | **upstream-native** (no fork) |
| Speculation | DSpark (probabilistic draft) | **MTP-3** (native multi-token) |
| KV per token (FP8) | ~2-4 KiB (MLA, very compact) | ~32 KiB (GQA on 16 layers) |
| 512K single-session KV | ~1-2 GB | ~16 GB |
| Parsers | `deepseek_v4` | `qwen3` / `qwen3_coder` |
| Long prefill advantage | full attention on all layers | **only 16/64 layers are O(L²)** |
| Weight footprint (BF16) | MoE (larger) | 54 GB (18 shards) |

The DeepSeek recipe wins on 524K **extreme concurrency** (MLA's compact KV lets
64 sessions coexist at full ceiling). Qwen3.8-27B wins on **long-prefill TTFT**
(hybrid attention is ~4× cheaper at 500K) and on **simplicity** (no fork, no
DeepSeek-specific overlays, upstream vLLM supports it directly).

## Concurrency for agentic coding loops (measured)

Model in [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md), anchored to the
measured MTP-3 decode curve (G5): per-session 64.5 tok/s at C8, 46.5 tok/s at
C16, 34.2 tok/s at C32; aggregate ceiling 1094 tok/s. For an agent-coding loop
with web search (~900 decode tokens/turn, 80K median context):

| Workload profile | Concurrency | Per-session decode | Binding constraint |
| --- | ---: | ---: | --- |
| Interactive, fast tools | ~8-16 | 46-65 tok/s | Decode throughput |
| Interactive, websearch-heavy | ~16 | 46.5 tok/s | Decode throughput |
| Batch / background, latency-tolerant | ~32+ | 34.2 tok/s | KV memory + admission |
| All sessions at 512K extreme | ~6-7 | — | KV memory (6.73× measured) |

The binding constraint for interactive agent loops is **engine decode
throughput**; MTP-3 pushes the interactive limit (per-session > ~30 tok/s)
from ~C16 to ~C32. The measured KV pool is 3.92M FP8 tokens (14.96× at 262K,
6.73× at 512K). The 512K ceiling is a worst-case memory limit, not a typical
operating point — real agentic sessions sit at 12K-80K context.

## Quick start

### 0. Verify the environment

```bash
bash scripts/00_check_env.sh
```

Reports ROCm/torch versions, detected AITER `(gfx, cu_num)` key, wheel and
model-shard presence, and disk headroom.

### 1. Create the isolated venv

```bash
bash scripts/env_setup.sh
```

The venv uses `--system-site-packages` so the platform ROCm Torch build is
reused. System Python/Torch remain untouched. **This is a separate venv from
the DeepSeek serving stack** — the DeepSeek venv carries MLA/DSpark overlays
that must not contaminate Qwen3.8's standard attention path.

### 2. Install the serving runtime

Place the vLLM/AITER wheels under `$WHEELS`, then:

```bash
bash scripts/install_vllm_nightly.sh
```

Unlike the DeepSeek recipe, this installer applies **no fork overlays**.
Qwen3.8 is upstream-native in vLLM (enabled for AMD ROCm by
[vllm-project/vllm#50068](https://github.com/vllm-project/vllm/pull/50068)).
The installer verifies that the `Qwen3_5ForCausalLM` architecture and the
Gated DeltaNet linear-attention path are importable before declaring success.

### 3. Download the model

```bash
# BF16 (54 GB, 18 shards) — reference quality
bash scripts/01_download_model.sh qwen38-bf16

# FP8 (~28 GB) — frees ~27 GB for KV, recommended for max concurrency
bash scripts/01_download_model.sh qwen38-fp8
```

### 4. Audit and serve

```bash
python3 scripts/audit_runtime.py
export VLLM_API_KEY_FILE=/run/secrets/vllm_api_key

bash scripts/02_serve_vllm.sh qwen38

# In a second shell after /health is ready:
SNAPSHOT_AFTER_WARMUP=1 bash scripts/warmup_runtime.sh
```

### 5. Benchmark

```bash
curl -s http://localhost:8000/health
python3 scripts/bench/bench_agent_trace.py 30 20000
```

## Validated serving profile

Promoted 2026-08-18 after the G3/G5 gates (`docs/PERFORMANCE.md`). The launcher
defaults now match the validated runtime: UNIFIED_ATTN + block 64, MTP-3, and
**GPU-only KV**. Production now targets native `MAX_MODEL_LEN=262144` for multi-round coding-agent/tool-loop traffic; 512K YaRN remains an explicit experiment rather than a production default. Vision is enabled and thinking is disabled by default, with request-level overrides still supported:

```text
--max-model-len 262144                    (native; no YaRN in production)
--attention-backend ROCM_AITER_UNIFIED_ATTN   (required: head_dim=256)
--block-size 64
--kv-cache-dtype fp8
--enable-prefix-caching
--max-num-seqs 32                         (interactive) / 64 (batch)
--max-num-batched-tokens 16384            (5-10 Agent production profile)
--long-prefill-token-threshold 1024         (short-request isolation / fairness)
--tensor-parallel-size 1
Vision encoder enabled; `--limit-mm-per-prompt {"image":4,"video":0}`; remote HTTP(S) media deny-by-default
--default-chat-template-kwargs '{"enable_thinking":false}'
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--enable-prompt-tokens-details
MTP: method=mtp, num_speculative_tokens=3 (acceptance ~65%)
--gpu-memory-utilization 0.95
KV_OFFLOAD_GB=0                           (GPU-only KV; CPU offload unstable)
```

### Controlled A/B overrides (maintenance experiments only)

The launcher already contains the production values above. Override **one variable
at a time** only for a recorded validation gate; never copy experiment values into
the normal restore command.

```bash
# Native-decode control for G5/MTP regression checks
MTP_ENABLED=0 bash scripts/02_serve_vllm.sh qwen38

# Model-dtype/BF16 KV control for G6
KV_CACHE_DTYPE=auto bash scripts/02_serve_vllm.sh qwen38

# Optional YaRN quality experiment after native-256K recall is green
MAX_MODEL_LEN=524288 bash scripts/02_serve_vllm.sh qwen38

# Text-only control to quantify Vision overhead; not the production profile
LANGUAGE_MODEL_ONLY=1 bash scripts/02_serve_vllm.sh qwen38
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_BASE` | `<persistent-storage>/models` | checkpoint directory |
| `HOT_MODEL_BASE` | `<local-ssd>/models` | optional local-SSD hot copy |
| `WHEELS` | `<persistent-storage>/wheels` | pinned vLLM/AITER wheels |
| `PERSIST_DIR` | `<persistent-storage>/.venvs` | venv/JIT snapshots |
| `VLLM_VENV` | `<local-disk>/.venvs/vllm-qwen` | active serving venv |
| `QUANT` | `bf16` | checkpoint variant (`bf16` or `fp8`) |
| `KV_CACHE_DTYPE` | `fp8` | KV cache dtype; use `auto` for the pending model-dtype/BF16 G6 control |
| `MAX_MODEL_LEN` | `262144` | native production context ceiling; `524288` is an explicit YaRN experiment |
| `MAX_BATCHED_TOKENS` | `16384` | promoted total scheduler budget for 5-10 Agent traffic |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | per-long-request prefill cap; protects short-request latency |
| `MM_IMAGE_LIMIT` / `MM_VIDEO_LIMIT` | `4` / `0` | bounded Vision input policy |
| `DEFAULT_ENABLE_THINKING` | `0` | Agent/tool traffic defaults to non-thinking; requests may override |
| `VLLM_API_KEY` / `VLLM_API_KEY_FILE` | *(unset)* | optional serving/client auth |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | benchmark client endpoint |

## Coding-agent benchmark suite

```text
scripts/bench/
├── bench_full.py                  decode/prefill/cache/context suite
├── bench_latency.py               TTFT / decode latency fixture
├── bench_agent_trace.py           one growing agent; per-request cache accounting
├── bench_session_concurrency.py   N independent long-lived agent histories
├── bench_high_concurrency.py      clean C32/C64 MTP-vs-native boundary
├── bench_long_context_recall.py   exact multi-needle 100K-475K recall
├── bench_tool_roundtrip.py        streamed tool call -> tool result -> final answer
└── bench_ttft_isolation.py        short request injected during long prefill
```

## Stable runtime provenance

```text
GPU        MI308X / MI300X class, gfx942, 192 GB
ROCm       7.2.3
Python     3.12
Torch      2.11.0+gitd0c8b1f (platform ROCm build, reused)
vLLM       0.26.1rc1.dev306+gcb8104839.rocm723
AITER      0.1.19
patch src  none (upstream-native; Qwen3.8 enabled by vllm-project/vllm#50068)
```

## ROCm support status

- **AMD Day-0 announcement**: 2026-08-12, covering MI300X/MI325X/MI355X.
  MI308X is not listed by name but is the same gfx942 ISA.
- **Gated DeltaNet kernels**: Triton-based (`fused_recurrent_gated_delta_rule`),
  work out-of-the-box on ROCm. Optimized specifically for gfx942 by
  [vllm-project/vllm#41446](https://github.com/vllm-project/vllm/pull/41446)
  (1.43× kernel speedup, 27/27 TTFT wins, 0 regressions).
- **Qwen3.8 enablement for AMD ROCm**: merged via
  [vllm-project/vllm#50068](https://github.com/vllm-project/vllm/pull/50068),
  validated with a two-node TP=8 PP=2 ROCm deployment.
- **MTP on ROCm**: validated 2026-08-18 (G5). MTP-3 + UNIFIED_ATTN on gfx942
  reaches ~65% mean acceptance (per-position 0.84 / 0.64 / 0.47) and lifts C1
  decode from 56.2 to **94.2 tok/s**; C32 aggregate reaches 1094 tok/s. MTP-1
  regresses at high concurrency, so MTP-3 is the promoted default.

## Known ROCm gotchas (validated)

Hard-won during the G-gate campaign; each is a real failure mode, not a style
preference.

1. **head_dim=256 rejects the default attention backend.** The full-attention
   layers use head_dim=256. The ROCm custom paged kernel only supports
   head_size 64/128, so the default backend silently falls back to Triton and
   decode drops ~13-35%. Fix: `--attention-backend ROCM_AITER_UNIFIED_ATTN
   --block-size 64` + `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`.
2. **Thinking mode can return empty output.** With `reasoning_effort=xhigh`
   (the default), the `qwen3` parser waits for a closing `</think>`; if the
   model is still thinking when `max_tokens` is hit, the whole response is
   discarded (0 content, `finish_reason=length`). Fix: `reasoning_effort=low`
   / `medium`, or `enable_thinking=false`.
3. **CPU-KV offload crashes.** Unlike the DeepSeek sibling, Qwen3.8 hits a
   `madvise(MADV_POPULATE_WRITE)` EINVAL on the Kata/tmpfs sandbox when CPU-KV
   offload is enabled. Fix: `KV_OFFLOAD_GB=0` (GPU-only KV).
4. **Cold long-context TTFT is JIT, not prefill.** The first request at a new
   shape pays Triton JIT (128K cold 374s vs warm 5.0s). Extend warmup to cover
   32K/64K/128K shapes so production requests stay on the warm path.

## Risks (see docs/RESEARCH_NOTES.md for detail)

1. **80-CU MI308X compute**: this host reports 80 CUs vs MI300X's 304. The
   DeepSeek-V4-Flash validated decode ceiling (~915 tok/s at C64) is the
   hardware proxy; do not use published MI300X benchmarks.
2. **MTP acceptance on gfx942**: resolved (G5, ~65% acceptance, C1 94.2 tok/s).
   MTP-1 still regresses at high concurrency; MTP-3 is the promoted default.
3. **FP8 KV cache calibration**: the official FP8 checkpoint may lack KV-cache
   calibration scales; `--kv-cache-dtype fp8` may emit warnings. The unsloth
   FP8 variant includes calibration.
4. **YaRN quality at factor 2.0**: extending 262K → 512K degrades short-context
   quality. Agent-loop early turns (12K) should be regression-tested.
5. **AITER tuning coverage**: the 80-CU MI308X AITER tables in the sibling
  DeepSeek repo are keyed for DeepSeek GEMM shapes, not Qwen3.8's. The dense
  27B has more standard GEMM shapes and may work with default AITER tables,
  but a tuning pass is a post-deployment optimization, not a blocker.

## License

Apache 2.0 — see [LICENSE](LICENSE). The Qwen3.8-27B model weights are
Apache 2.0 (Alibaba) and are **not** redistributed here; scripts download them
at runtime.
