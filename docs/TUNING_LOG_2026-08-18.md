# Qwen3.8-27B MI308X tuning log — 2026-08-18

Current goal: production serving for multi-round coding-agent / tool-loop traffic. The production context target is **262,144 tokens (native 256K)**; 512K YaRN is no longer the default target.

## 1. Vision production enablement

Production launcher changed from text-only to multimodal by default:

- `LANGUAGE_MODEL_ONLY=0`
- one image per prompt: `MM_IMAGE_LIMIT=1`
- video disabled: `MM_VIDEO_LIMIT=0`
- multimodal processor cache: `2 GiB`, `lru`
- server default thinking disabled with `--default-chat-template-kwargs '{"enable_thinking":false}'`
- request-level thinking remains overridable
- existing `ROCM_AITER_UNIFIED_ATTN + block 64 + FP8 KV + MTP-3 + GPU-only KV` retained

First multimodal cold start exposed a hidden runtime dependency: AITER Vision FlashAttention imports FlyDSL and rejects the platform-visible `flydsl 0.2.0` (`>=0.2.4` required). Fixed by installing the already-persisted `flydsl-0.2.4` wheel into `/root/.venvs/vllm-qwen`; installer and runtime audit now pin/check it.

Vision AITER cold JIT compiled `mha_varlen_fwd_bf16...` once (~81.2 s). Runtime and compiler caches were then re-snapshotted to NFS.

Persistent artifacts after the fix:

- `vllm-qwen.tar.gz`: 6.3G
- `aiter_cache.tar.gz`: 55M
- `torch_ext_cache.tar.gz`: 17M
- `comgr_cache.tar.gz`: 626M
- `triton_cache.tar.gz`: 223M

## 2. Vision-profile runtime measurements (512K ceiling, superseded as production target)

These are retained as an A/B baseline only. Production is moving to native 256K below.

- model load: 52.01 GiB, 22.996 s after page cache warmup
- encoder cache: 16,384 tokens; profiled for one maximum-size image
- available GPU KV cache: 121.85 GiB
- GPU KV cache capacity: 3,589,003 tokens
- graph capture: 17 s / 4.54 GiB
- full engine init: 219.03 s on the first Vision/FlyDSL cold-JIT run
- steady VRAM after init: ~188.7 GB reported by ROCm SMI

Functional regression:

- default streaming: HTTP 200; first SSE assistant role present; content present; no reasoning; `[DONE]` present; `finish_reason=stop`
- explicit thinking override: reasoning + final content both returned
- 1-image request: HTTP 200; synthetic red image correctly answered `Red`
- 2-image request: HTTP 400 as intended by the production one-image guard
- auto tool choice: HTTP 200; `get_weather` tool call parsed correctly

Text performance with Vision enabled:

- 512-output-token C1, 3 rounds: avg TTFT **0.136 s**; warm TTFT ~**0.072 s**; avg decode **102.1 tok/s**; MTP draft acceptance **68–69%**
- longer single request: TTFT **0.077 s**, decode **98.0 tok/s**, 1,702 generated tokens before natural stop; MTP acceptance 67%
- C8 256-token fixture: **398.5 tok/s aggregate**, 49.8 tok/s/session; MTP acceptance 86%
- C16 256-token fixture: **693.7 tok/s aggregate**, 43.4 tok/s/session; MTP acceptance 83%

Conclusion: enabling the Vision encoder does **not** materially regress the text decode path on this 192-GiB-class MI308X. Keep Vision enabled for production agent traffic.

## 3. Production direction: native 256K + agent-loop latency

User decision: stop optimizing for 500K/512K; make **262,144** the production ceiling.

Optimization priorities, in order:

1. multi-round agent/tool-loop TTFT and warm-turn latency
2. prefix-cache hit rate and long-session reuse
3. C5–C10 interactive concurrency; retain safe headroom toward C16+
4. decode / inter-token latency
5. Vision + tool-call + streaming protocol correctness
6. cold-start restoration from persistent venv/JIT snapshots

Controlled scheduler A/B will start from the existing `max_num_batched_tokens=3072`, `max_num_seqs=32`, `long_prefill_token_threshold=1024`, MTP-3 baseline and compare larger batched-token budgets for TTFT/throughput. Promotion must be based on agent-trace/TTFT/concurrency measurements, not warning suppression alone.

## 4. Native-256K scheduler A/B — baseline (`max_num_batched_tokens=3072`)

Runtime after switching the production ceiling to native 262,144:

- YaRN disabled
- Vision enabled, one image / no video
- default thinking off; explicit override supported
- available GPU KV: 121.84 GiB
- GPU KV capacity: 3,438,626 tokens
- engine init for the new 256K scheduler/compile shape: 137.98 s; compilation 51.37 s
- steady VRAM: ~188.72 GB

3072 baseline measurements:

- C1 512-token output: cold-shape TTFT 4.966 s; warm TTFT 0.074 / 0.086 s; avg decode 94.2 tok/s; MTP acceptance 56–65%
- 12-turn agent trace with ~20K initial repository prefix:
  - turn 1 cold TTFT 12.05 s
  - turns 2–10: 16,000 cached tokens (~83–86% hit), TTFT 1.67–2.78 s
  - turns 11–12: 17,600 cached (~91% hit), TTFT 1.09–1.32 s
  - overall prefix-cache hit 78.62%
  - average decode 78.2 tok/s (short natural-stop outputs after the first turn)
- C8 / 256-token fixture: 379.1 tok/s aggregate, 47.4/session
- C16 / 256-token fixture: 708.1 tok/s aggregate, 44.3/session

Assessment: 3072 strongly protects decode/ITL but chunks agent prefills aggressively. It is the control for larger scheduler budgets.

## 5. Scheduler A/B findings: 8192 budget

### 8192 with the old 1024 long-prefill cap

Increasing only `max_num_batched_tokens` from 3072 to 8192 produced almost no agent-latency benefit because `long_prefill_token_threshold=1024` still clamps each long prefill scheduling step to 1024 tokens.

- C1 512-output-token: warm TTFT ~0.086 s; avg decode 93.0 tok/s
- 12-turn ~20K agent trace: effectively identical TTFT/cache behavior to the 3072 baseline
- C8: 379.9 tok/s aggregate
- C16: 717.2 tok/s aggregate
- GPU KV capacity: 3,432,768 tokens

Conclusion: the 1024 long-prefill cap, not the total 3072 token budget, is the main limiter for agent prefill latency.

### 8192 with long-prefill cap disabled (`threshold=0`)

Disabling the per-request long-prefill cap allowed the 8192 budget to be used by a single long prompt, but it is unsafe for interactive multi-agent traffic:

- C1 512-output-token warm TTFT: 0.085–0.132 s; avg decode 94.3 tok/s
- warm ~20K multi-turn trace: most turns ~0.95–1.54 s TTFT, with align-boundary spikes around 2.8 s
- 64K TTFT isolation: short-request added TTFT **+17.692 s / +17.575 s**, average **+17.634 s** → FAIL
- during isolation: 2 requests running, 0 waiting, GPU 100%; the short request was co-scheduled but still suffered from the very large prefill compute chunk
- GPU KV capacity: 3,443,019 tokens

Conclusion: `threshold=0` improves the ability to consume long prefills but destroys short-request isolation. Do not promote.

### Prefix-cache observation

Qwen3.8/Qwen3.5 hybrid caching is in Mamba `align` mode. The measured attention/Mamba aligned block size is **1600 tokens**. Cache hits jump at 1600-token boundaries (for example 16,000 → 17,600 cached tokens), and TTFT varies with those boundaries. This matches the known experimental behavior of vLLM hybrid/Mamba prefix caching; benchmark turn 1 must use a salted first block to avoid false warm results.

The local `bench_agent_trace.py` is being extended with `--cold-prefix` for that purpose.

### Current vLLM version constraint

Latest vLLM exposes `max_num_partial_prefills` and `max_long_partial_prefills`, including the documented pattern `max_long_partial_prefills < max_num_partial_prefills` so short prompts can jump ahead of long prefills. The pinned ROCm dev306 runtime does **not** contain these SchedulerConfig fields or CLI arguments. Do not upgrade the validated ROCm runtime solely for this feature; record it as a future upgrade gate.

Next safe A/B: keep total budget 8192 and test intermediate long-prefill caps.

### Intermediate cap results

`long_prefill_token_threshold=4096`:

- first salted ~20K agent turn: 14.38 s TTFT (included first-use 4096-prefill JIT); later turns mostly 1.36–1.53 s with align-boundary spikes
- long-prefill isolation, two rounds: added TTFT **+10.095 / +9.945 s**, average **+10.020 s** → reject

`long_prefill_token_threshold=2048`:

- long-prefill isolation, two rounds: added TTFT **+3.018 / +2.963 s**, average **+2.990 s** → still above the +2 s production gate
- startup reuses the 8192 compile profile: ~31 s engine init, 122.0 GiB KV, 3,443,019-token KV capacity

The isolation fixture currently labels its generated prefix by an approximate target; `/metrics` shows the `64000` fixture can actually tokenize to roughly 120K. Treat the 0/4096/2048 measurements as controlled relative A/B results, not exact 64K absolute latency. The benchmark must be corrected to report tokenizer-measured input length.

Next candidate: **1600 tokens**, matching the hybrid attention/Mamba aligned cache page size observed at runtime.
