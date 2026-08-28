# mi308x-llm-lab

<div align="center">

**LLM serving lab on a single AMD Instinct MI308X (gfx942 / 80 CU / ~192 GB HBM), vLLM on ROCm.**

One card · one model resident at a time · per-line recipes with measured baselines · coding-agent workload target · no Docker required

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)
![GPU](https://img.shields.io/badge/GPU-gfx942%20%7C%20192GB-ED1C24)

</div>

This repository consolidates all LLM serving lines for one MI308X-class
deployment host. Each model line lives in its own directory with its own
scripts, docs, and benchmark suite; the lines share nothing at runtime
except the machine and an external OpenAI-compatible gateway.

## Model matrix

| Directory | Model | Stack | Status |
| --- | --- | --- | --- |
| [`qwen3-8/`](qwen3-8/README.md) | Qwen3.8-27B (dense hybrid attention) | upstream-native vLLM, MTP-3 | **validated 2026-08-18, frozen** |
| [`qwen3-8/`](qwen3-8/README.md) | Qwen3.8-Flash-Next (125B MoE + QSA, FP8) | upstream-native vLLM (≥ PR #53896) | **active line, in preparation** |
| [`deepseek-v4-flash/`](deepseek-v4-flash/README.md) | DeepSeek-V4-Flash-0731 (MoE + MLA) | fork-overlay vLLM, DSpark K7 | **validated 2026-08-16, frozen** |
| — | GLM-5.3-Flash (320B/18B active) | — | **not servable on one card** (FP8 328 GB > ~195 GB usable VRAM; no vLLM-loadable INT4 quantization as of 2026-08-29). Would land here as `glm-5-3-flash/` if that changes. |

## Why the lines stay separate inside one repo

- **Engine builds differ.** The DeepSeek line runs a fork-overlay vLLM build
  (pinned base + DeepSeek-specific overlays, DSpark). The Qwen line runs
  upstream-native vLLM. The two never share a venv and never mix overlays.
- **Launch profiles differ.** Attention backends, speculative decoding
  (DSpark vs MTP), KV policy (CPU tier vs GPU-only), parsers and scheduler
  budgets are model-specific; "only the weights differ" is not true.
- **What is shared:** the machine, the benchmark-suite shape, the public
  gateway boundary (OpenAI Chat Completions), and this repository's CI.

## Provenance

Merged on 2026-08-29 from the former standalone repositories
`qwen3-8-mi308x` (itself renamed from `qwen3-8-27b-mi308x`) and
`deepseek-v4-flash-mi308x`, with both git histories fully preserved under
`qwen3-8/` and `deepseek-v4-flash/`. GitHub redirects the old names.

## Quick start

Pick a line and follow its README:

```bash
# Qwen3.8 series (active)
cd qwen3-8 && cat README.md

# DeepSeek-V4-Flash-0731 (frozen, maintenance-only)
cd deepseek-v4-flash && cat README.md
```

Both lines assume: ROCm 7.2 platform torch untouched, venv on local disk,
weights on shared storage, ModelScope as the weight channel (Hugging Face is
not reachable from the reference deployment host).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Model weights are downloaded at
runtime from their official sources and are governed by their own licenses.
