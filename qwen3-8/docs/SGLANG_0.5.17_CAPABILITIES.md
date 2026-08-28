# SGLang 0.5.17 CLI Capability Snapshot

> **Status: RETIRED / reference-only for Qwen on this DSW ROCm stack.** The
> available `sgl_kernel` dependency is CUDA-only and DSW has no Docker path for
> the AMD image. Production and future validation use vLLM; this snapshot is kept
> only to preserve the research record.
>
> Extracted from the actual `sglang-0.5.17-cp312-cp312-manylinux_2_34_x86_64.whl`
> by parsing `sglang/srt/server_args.py`. Do not rely on SGLang `main` docs —
> they drift. This file is the ground truth for the installed wheel.

## Dependency Audit

```text
Total non-extra deps: 75
Extras (optional):     53

NVIDIA-only (SKIP on AMD, 18):
  cuda-python, flash-attn-4, flashinfer-python[cu13], helion, humming-kernels[cu13],
  nvidia-cutlass-dsl[cu13], nvidia-mathdx, nvidia-ml-py, quack-kernels,
  sgl-deep-gemm, sglang-kernel, tilelang, tokenspeed-mla, torch,
  torch-memory-saver, torchao, torchaudio, torchvision

Pre-downloaded (satisfied, 41):
  aiohttp, anthropic, compressed-tensors, datasets, distro, easydict, einops,
  fastapi, gguf, interegular, llguidance, mistral-common, msgspec, ninja,
  numba, numpy, openai, orjson, outlines, packaging, partial-json-parser,
  pillow, prometheus-client, psutil, pybase64, pydantic, python-multipart,
  pyzmq, requests, scipy, sentencepiece, setproctitle, tiktoken, tqdm,
  uvicorn, uvloop, watchfiles, xxhash, zstandard
  (+ sentencepiece, xgrammar via deps)

Missing (install from pip on AMD, ~16):
  apache-tvm-ffi, blobfile, build, IPython, kernels, openai-harmony,
  py-spy, soundfile, timm, transformers (pinned 5.12.1)
```

Install strategy: `pip install --no-deps sglang-0.5.17-*.whl` + install the
41 pre-downloaded wheels + pip install the missing ones from PyPI.

## Attention Backend Choices (AMD-relevant only)

```python
ATTENTION_BACKEND_CHOICES = [
    # Common (work on AMD)
    "triton",           # ← AMD default for GDN models
    "torch_native",
    "flex_attention",
    "dsa",
    "dsv4",             # DeepSeek V4
    # AMD specific
    "aiter",            # ← AMD AITER-backed, may support MLA
    "wave",             # ← new AMD backend
]
```

**On AMD, sweep: `triton` vs `aiter` for full-attention; `triton` only for GDN.**

## Mamba / GDN Choices

```python
MAMBA_RADIX_CACHE_STRATEGY_CHOICES = ["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"]
MAMBA_BACKEND_CHOICES = ["triton", "flashinfer"]     # AMD: triton only
LINEAR_ATTN_KERNEL_BACKEND_CHOICES = ["triton", "cutedsl", "flashinfer", "flashkda", "nvidia_kda", "ptx_kda"]
```

**CRITICAL**: On AMD, use `no_buffer` (not `extra_buffer`/`extra_buffer_lazy`).
The `extra_buffer` branching-point Mamba state caching depends on FLA path
(NVIDIA-only). See SGLang Qwen3.5 cookbook AMD section.

## Key Server Args (from server_args.py)

```text
--attention-backend <choice>           # triton (AMD default) or aiter
--mamba-backend <choice>               # triton (AMD only)
--mamba-ssm-dtype <float32|bfloat16|float16>  # default: model config (float32)
--mamba-full-memory-ratio <float>      # default: 0.9; split between GDN state and KV
--mamba-radix-cache-strategy <choice>  # AMD: no_buffer
--mamba-max-states-per-path <int>      # -1 = unlimited; controls branching cache depth
--max-mamba-cache-size <int>           # explicit GDN state pool size override
--chunked-prefill-size <int>           # 2048 recommended for hybrid GDN
--kv-cache-dtype <fp8|auto>            # FP8 KV for max capacity
--mem-fraction-static <float>          # 0.90 default
--enable-radix-cache                   # prefix + GDN state cache
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

## Speculative Decoding

```text
# EAGLE (MTP)
--speculative-algorithm EAGLE
--speculative-num-steps <N>            # 1/2/3 — sweep on GPU
--speculative-eagle-topk <N>           # 1 default
--speculative-num-draft-tokens <N>    # steps + 1

# DSPARK (DeepSeek)
--speculative-algorithm DSPARK
--speculative-draft-model-path <path>
--speculative-dspark-block-size <N>    # gamma; auto-inferred from draft checkpoint

# Both
--speculative-rejection-sampling       # requires topk=1
```

**On AMD, MTP may not help (issue #23123). Sweep native/MTP-1/MTP-2/MTP-3.**
