# Architecture

This repository is deliberately narrow: it turns the official
`Qwen/Qwen3.8-27B` checkpoint (BF16 or FP8) into a reproducible,
OpenAI-compatible vLLM service on a single 192 GB gfx942-class AMD Instinct GPU.

```text
OpenAI-compatible client
        |
        |  /v1/chat/completions
        v
      vLLM
        |
        |  Qwen3 tokenizer / qwen3 reasoning parser / qwen3_coder tool parser
        |  MTP-3 speculative decode (native multi-token prediction)
        |  prefix cache + FP8 KV (16 full-attention layers only)
        |  Gated DeltaNet linear attention (48 layers, constant recurrent state)
        |  optional native CPU-KV tier
        v
Qwen3.8-27B (text-only language model path)
        |
        v
AMD Instinct MI308X / MI300X class (gfx942, 192 GB)
```

## Repository boundary

This project owns only the inference/runtime layer:

- environment and runtime validation;
- model download and shard verification;
- vLLM launch configuration and (future) MI308X-specific tuning;
- correctness and performance benchmarks;
- restart/provenance auditing and performance-cache snapshots.

It intentionally does **not** own:

- chat UI or user accounts;
- LiteLLM or another multi-provider gateway;
- Open WebUI deployment;
- public ingress, reverse proxy or tunnels;
- SSH/bootstrap automation;
- platform credentials or API-key persistence;
- ModelScope Studio packaging.

Those concerns belong in a separate application/Studio repository or a private
infrastructure layer. Keeping the boundary strict makes runtime regressions and
benchmark changes attributable to this serving stack.

## Public interface

The stable product boundary is the OpenAI-compatible vLLM HTTP interface. The
launcher publishes the model as `qwen3.8-27b` by default.

Authentication is optional at this layer. When required, inject
`VLLM_API_KEY` from the deployment platform or secret manager. The public
recipe never generates, stores or commits machine credentials.

## Isolation from the sibling DeepSeek serving stack

This repository uses a **separate venv** (`/root/.venvs/vllm-qwen` by default)
from the DeepSeek-V4-Flash serving stack (`/root/.venvs/vllm`). The DeepSeek
venv carries MLA, DSpark, and DeepSeek-MoE-specific overlay patches that must
not contaminate Qwen3.8's standard GQA + linear-attention path. The two
serving stacks can coexist on the same host but must not share a venv.

## Upstream-native (no fork)

Unlike the DeepSeek-V4-Flash recipe (which pins a fork source with 18 overlay
patches), Qwen3.8-27B is **upstream-native** in vLLM. The Qwen3.8 architecture
is enabled for AMD ROCm by
[vllm-project/vllm#50068](https://github.com/vllm-project/vllm/pull/50068),
and the Gated DeltaNet linear-attention kernels are optimized for gfx942 by
[vllm-project/vllm#41446](https://github.com/vllm-project/vllm/pull/41446).
No fork source, no overlay patches, no patched binary artifacts.
