# Contributing

Thanks for your interest in contributing to qwen3-8-27b-mi308x!

This repository is a production recipe for serving Qwen3.8-27B on a single AMD
Instinct MI308X / MI300X-class (gfx942) accelerator. Contributions in any of
these areas are especially welcome:

- **Performance**: GEMM tuning tables for the 80-CU MI308X variant, MTP depth
  sweeps, linear-attention (Gated DeltaNet) kernel improvements on gfx942.
- **Stability**: reproductions of long-context crashes, fixes for YaRN-scaled
  512K edge cases, prefix-cache eviction patterns under agentic load.
- **Runtime compatibility**: focused ports and validation for newer pinned
  vLLM, AITER, ROCm, or Qwen3.8 revisions.
- **Docs**: corrections, clearer instructions, benchmark methodology.

## Ground rules

- The model weights are **not** redistributed here — scripts download them at
  runtime from ModelScope or Hugging Face.
- The API key and any instance-specific state must stay out of the repository.
  Use environment variables (`VLLM_API_KEY`, `VLLM_API_KEY_FILE`,
  `VLLM_BASE_URL`, ...) — the scripts already honor them.
- Keep scripts idempotent: re-running them must be safe.
- Performance claims go into `docs/PERFORMANCE.md` with the commit hash,
  environment versions, and the harness command that produced them.
- Do not conflate estimates with validated numbers. Estimates belong in
  `docs/RESEARCH_NOTES.md`; only real-machine measurements belong in
  `docs/PERFORMANCE.md`.

## Development flow

1. Fork the repository and create a feature branch.
2. Make your change; keep commits small and descriptive.
3. Validate scripts with `bash -n scripts/*.sh`.
4. Open a pull request describing the problem and the evidence for your change.

## Reporting issues

Include: GPU model (and reported CU count), ROCm version, vLLM/AITER versions,
the exact serve command, and a minimal reproduction script. Raw numbers are
better than screenshots.
