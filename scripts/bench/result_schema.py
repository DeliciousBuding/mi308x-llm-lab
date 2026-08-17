#!/usr/bin/env python3
"""Unified result schema for all benchmark runs.

Every benchmark writes one JSONL line per result. This guarantees that
performance numbers are always accompanied by the environment that produced them.

Usage:
    from result_schema import ResultRecord, write_result
    record = ResultRecord(...)
    write_result(record, "results/g1_vllm.jsonl")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ResultRecord:
    # Identity
    gate: str                          # "G1", "G2", "G3", etc.
    engine: str                        # "vllm" or "sglang"
    engine_version: str                # "0.26.1rc1.dev306" or "0.5.17"
    repo_commit: str                    # git HEAD of the serve script's repo
    manifest_hash: str = ""             # SHA256 of runtime_manifest.json (if available)

    # Launch config
    launch_args: dict = field(default_factory=dict)  # all env vars + CLI flags
    model: str = "qwen3.8-27b"
    quant: str = "bf16"
    max_model_len: int = 262144
    kv_cache_dtype: str = "fp8"
    mtp_enabled: bool = False
    mtp_k: int = 0
    attention_backend: str = ""
    mamba_ssm_dtype: str = ""
    mamba_radix_cache_strategy: str = ""
    mamba_full_memory_ratio: float = 0.0
    chunked_prefill_size: int = 0
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    mem_fraction_static: float = 0.0

    # Workload
    context_length: int = 0             # prompt tokens
    concurrency: int = 1
    output_tokens: int = 0              # max_tokens per request

    # Metrics (fill what's available)
    ttft_p50: float = 0.0              # seconds
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    itl_p50: float = 0.0              # inter-token latency, ms
    itl_p95: float = 0.0
    tpot_p50: float = 0.0             # time per output token, ms
    single_stream_tps: float = 0.0     # tokens/s, C1
    aggregate_output_tps: float = 0.0  # tokens/s, all sessions
    aggregate_total_tps: float = 0.0   # input + output tokens/s
    cache_hit_tokens: int = 0
    cache_hit_pct: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    preemptions: int = 0
    mtp_acceptance_rate: float = 0.0
    mtp_mean_accepted_len: float = 0.0
    hbm_peak_gb: float = 0.0
    request_failures: int = 0
    gpu_utilization_pct: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""


def write_result(record: ResultRecord, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


def current_env_summary() -> dict:
    """Capture current env vars for the launch_args field."""
    keys = [
        "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_BATCHED_TOKENS", "MTP_ENABLED",
        "MTP_K", "MTP_STEPS", "KV_CACHE_DTYPE", "QUANT", "LANGUAGE_MODEL_ONLY",
        "KV_OFFLOAD_GB", "GPU_MEMORY_UTILIZATION", "MAMBA_SSM_DTYPE",
        "MAMBA_RADIX_CACHE_STRATEGY", "MAMBA_FULL_MEMORY_RATIO",
        "CHUNKED_PREFILL_SIZE", "MEM_FRACTION_STATIC", "ATTENTION_BACKEND",
        "SGLANG_USE_AITER", "VLLM_ROCM_USE_AITER",
    ]
    return {k: os.environ.get(k, "") for k in keys}
