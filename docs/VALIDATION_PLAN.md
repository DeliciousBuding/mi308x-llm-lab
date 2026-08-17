# Validation Plan — Qwen3.8-27B on MI308X

> Pre-deployment checklist. Each gate must pass before proceeding to the next.
> Gates 1-3 are CPU-cheap or quick; gates 4-9 need GPU time.

## Gate 1: Architecture import check (CPU, no GPU needed)

Verify that the dev306 venv can import the Qwen3.8 architecture and that the
Gated DeltaNet linear-attention kernel path is available.

```bash
# In the qwen venv:
python3 -c "
from vllm.model_executor.models.registry import ModelRegistry
names = ModelRegistry.get_supported_archs()
assert 'Qwen3_5ForCausalLM' in names, 'Qwen3.8 text-only architecture not registered'
print('OK: Qwen3_5ForCausalLM registered')
"

python3 -c "
# Verify the Gated DeltaNet FLA kernel imports (Triton, works on ROCm)
from vllm.attention.backends.utils import is_torch_available
print('Triton on ROCm:', is_torch_available())
"
```

**Pass**: both print OK. **Fail**: the vLLM dev306 build predates PR #50068;
rebuild from a newer source or patch the model registry.

## Gate 2: Model download + shard verification

```bash
bash scripts/01_download_model.sh qwen38-bf16
# Verify 18/18 shards + config.json + tokenizer_config.json + index.json
```

## Gate 3: 262K native serve + health

Serve at the model's native 262K context (no YaRN). Confirm the basic
OpenAI-compatible interface works.

```bash
bash scripts/02_serve_vllm.sh qwen38
# In a second shell:
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
# Short chat completion
python3 - <<'PY'
from openai import OpenAI
c = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
r = c.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role":"user","content":"Give me three primes above 100."}],
    max_tokens=256,
)
print(r.choices[0].message.content)
print("usage:", r.usage)
PY
```

**Pass**: health 200, models listed, chat returns coherent output with usage.
**Fail**: kernel import error, OOM, or garbled output — check ROCm/HIP version
and the Gated DeltaNet Triton kernel path.

## Gate 4: Tool-call round trip

```bash
python3 scripts/bench/bench_tool_roundtrip.py --rounds 5 --mode auto --prefix-tokens 20000
```

**Pass**: 5/5 tool calls parsed and executed, no raw chat-template leakage.
This validates the `qwen3_coder` tool parser and the streaming tool protocol.

## Gate 5: MTP-3 acceptance measurement

```bash
# MTP-3 enabled (default)
python3 scripts/bench/bench_full.py decode
# Note the tok/s and MTP acceptance rate from logs.

# Native baseline (MTP disabled)
MTP_ENABLED=0 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_full.py decode
```

**Pass**: MTP-3 single-stream decode > 1.5× native. If MTP-3 < 1.2× native,
acceptance is too low on gfx942 — investigate and consider MTP disabled for
high-batch serving.

## Gate 6: YaRN 512K extension + context ladder

```bash
MAX_MODEL_LEN=524288 bash scripts/02_serve_vllm.sh qwen38
python3 scripts/bench/bench_full.py context
```

**Pass**: 50K / 128K / 256K / 384K / 500K all complete without OOM or crash.
This is survival, not correctness.

## Gate 7: Multi-needle exact recall

```bash
python3 scripts/bench/bench_long_context_recall.py --lengths 100000 256000 384000 475000
```

**Pass**: all needles found at all lengths. If 384K fails (as DSpark did for
DeepSeek-V4-Flash), record it as a known correctness risk, not a blocker.

## Gate 8: Decode throughput ladder

```bash
python3 scripts/bench/bench_high_concurrency.py --concurrencies 1 2 4 8 32 64
```

**Pass**: aggregate scales sublinearly; C64 aggregate within ±15% of the
~900 tok/s estimate. Record the actual numbers in `PERFORMANCE.md`.

## Gate 9: Agentic trace + session concurrency

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 8 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 16 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 24 --rounds 4
python3 scripts/bench/bench_session_concurrency.py --sessions 32 --rounds 4
```

**Pass**: 30-turn cache hit > 90%; find the concurrency knee where per-session
decode drops below 180 tok/s (5s/turn threshold). That knee is the interactive
concurrency limit.

## Gate 10: Cold TTFT isolation

```bash
python3 scripts/bench/bench_ttft_isolation.py 200000 --rounds 3
```

**Pass**: short-request added TTFT < +2.0s (the DeepSeek recipe's validated
gate was +0.5s; +1.3s was reported as still-open). For Qwen3.8, the hybrid
attention should make long prefill cheaper, so the isolation penalty may be
smaller — but this is unmeasured.

## Post-validation

If all gates pass, replace the estimates in `docs/PERFORMANCE.md` with the
real numbers, change the header to "Validated Baseline", and record the commit
hash + environment versions alongside each number.
