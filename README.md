# qwen3-8-27b-mi308x

<div align="center">

**Qwen3.8-27B (dense, hybrid attention) on a single AMD Instinct MI300X / MI308X (gfx942), served with native vLLM on ROCm.**

512K configured context (YaRN factor 2.0) · MTP-3 speculative decode · prefix caching · agentic-coding workload target · no Docker required · upstream-native (no fork patches)

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)
[![vLLM](https://img.shields.io/badge/vLLM-dev306-4B32C3)](https://github.com/vllm-project/vllm)
![GPU](https://img.shields.io/badge/GPU-gfx942%20%7C%20192GB-ED1C24)

</div>

> **Status: preparation / pre-deployment.** This repository is a researched
> serving recipe and benchmark harness for Qwen3.8-27B on the same 192 GB
> gfx942 hardware validated for DeepSeek-V4-Flash-0731. Performance numbers in
> `docs/PERFORMANCE.md` are **estimated targets**, not validated measurements.
> The concurrency analysis in `docs/RESEARCH_NOTES.md` is a structured estimate
> (±40% confidence) pending real-machine validation. AMD announced Day-0
> support for Qwen 3.8 on MI300X/MI325X/MI355X on 2026-08-12; this recipe ports
> that to the MI308X (80-CU gfx942 variant) and adds coding-agent benchmarks.

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

## Estimated concurrency for agentic coding loops

Full analysis in [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md). Headline
estimate for an agent-coding loop with web search (20-30s turn interval, ~900
decode tokens per turn, 80K median context):

| Workload profile | Estimated concurrency | Binding constraint |
| --- | ---: | --- |
| Interactive, websearch-heavy (2-5s decode/turn) | **~20-30** | Decode throughput |
| Interactive, pure coding, fast tools (2-5s/turn) | ~7-10 | Decode throughput |
| Batch / background, latency-tolerant (30-60s/turn) | ~40-60 | KV memory + admission |
| All sessions at 512K extreme | ~8-10 | KV memory |

The binding constraint for interactive agent loops is **engine decode
throughput** (~900 tok/s aggregate on this 80-CU MI308X, proxied from the
DeepSeek-V4-Flash validated baseline). The 512K ceiling is a worst-case memory
limit, not a typical operating point — real agentic sessions sit at 12K-80K
context (turn-30 median).

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

## Stable launch profile (planned)

```text
--max-model-len 524288                    (YaRN factor 2.0 over 262K native)
--hf-overrides '{"text_config":{"rope_parameters":{...,"factor":2.0,...}}}'
--kv-cache-dtype fp8
--enable-prefix-caching
--max-num-seqs 32                         (interactive) / 64 (batch)
--max-num-batched-tokens 3072             (coding-agent latency profile)
--tensor-parallel-size 1
--language-model-only                     (skip vision encoder, text-only)
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--enable-prompt-tokens-details
MTP: method=mtp, num_speculative_tokens=3
--gpu-memory-utilization 0.95
--kv-cache-memory-bytes <hardware-dependent>
--kv-offloading-size 12                   (native CPU-KV, 16GB /dev/shm limit)
```

### A/B without editing the launcher

```bash
MAX_MODEL_LEN=524288
MAX_NUM_SEQS=32
MAX_BATCHED_TOKENS=3072
MTP_ENABLED=1
MTP_K=3
KV_OFFLOAD_GB=12
GPU_MEMORY_UTILIZATION=0.95
QUANT=bf16            # or fp8
```

Examples:

```bash
# Disable MTP for native-decode baseline (throughput-bound high-batch control)
MTP_ENABLED=0 bash scripts/02_serve_vllm.sh qwen38

# FP8 weights for maximum KV headroom
QUANT=fp8 bash scripts/02_serve_vllm.sh qwen38

# Throughput profile (faster 500K endpoint, worse cold-isolation tail)
MAX_BATCHED_TOKENS=4096 bash scripts/02_serve_vllm.sh qwen38
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

## Stable runtime provenance (planned)

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
- **MTP on ROCm**: native multi-token prediction is supported; acceptance and
  throughput characteristics on gfx942 are not yet independently measured for
  the 27B dense variant. This is a validation priority.

## Risks (see docs/RESEARCH_NOTES.md for detail)

1. **80-CU MI308X compute**: this host reports 80 CUs vs MI300X's 304. The
   DeepSeek-V4-Flash validated decode ceiling (~915 tok/s at C64) is the
   hardware proxy; do not use published MI300X benchmarks.
2. **MTP acceptance on gfx942**: unmeasured for Qwen3.8-27B. MTP-1 is known to
   regress at high concurrency; MTP-3 is the recommended starting point.
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
