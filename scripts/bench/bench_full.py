#!/usr/bin/env python3
"""bench_full.py — Qwen3.8-27B comprehensive benchmark.

Covers: decode rate, long-context prefill ladder, prefix cache cold/hot,
concurrency sweep, MTP acceptance.

Usage:
    python3 bench_full.py [decode|prefill|prefix|concurrency|all]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import (
    chat_completion,
    report_mtp_acceptance,
    spec_decode_metrics,
    stream_completion,
    health_check,
)


def bench_decode() -> None:
    """Single-stream decode rate (512-token generation)."""
    print("=== decode-512 (single stream) ===")
    prompt = "Write a detailed technical explanation of how hybrid attention"
    " architectures reduce long-context KV memory. " * 3
    mtp_before = spec_decode_metrics()
    result = stream_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    mtp_after = spec_decode_metrics()
    print(f"  TTFT: {result['ttft_s']:.3f}s")
    print(f"  total: {result['total_s']:.3f}s")
    print(f"  completion_tokens: {result['completion_tokens']}")
    print(f"  decode_rate: {result['decode_rate']:.1f} tok/s")
    print(f"  cached_tokens: {result['cached_tokens']}")
    report_mtp_acceptance("decode-512", mtp_before, mtp_after)


def bench_prefill() -> None:
    """Long-context prefill ladder."""
    print("=== prefill ladder ===")
    for target_tokens in [8_000, 32_000, 100_000, 200_000]:
        filler = "This is a context-padding sentence for measuring prefill latency. " * (target_tokens // 14)
        result = chat_completion(
            [{"role": "user", "content": filler + "\n\nReply with OK."}],
            max_tokens=8,
        )
        print(
            f"  {target_tokens:>7,d} tokens: {result['elapsed_s']:.3f}s "
            f"(prompt={result['prompt_tokens']}, cached={result['cached_tokens']})"
        )


def bench_prefix_cache() -> None:
    """Prefix cache cold vs hot comparison."""
    print("=== prefix cache cold/hot ===")
    prefix = ("Repository context: parser.py validates JSON configs. " * 200
               + "service.py implements LRU cache with TTL. " * 200)
    question = "\n\nWhat does parser.py do?"

    cold = chat_completion([{"role": "user", "content": prefix + question}], max_tokens=16)
    print(f"  cold: {cold['elapsed_s']:.3f}s (cached={cold['cached_tokens']}/{cold['prompt_tokens']})")

    hot = chat_completion([{"role": "user", "content": prefix + question}], max_tokens=16)
    print(f"  hot:  {hot['elapsed_s']:.3f}s (cached={hot['cached_tokens']}/{hot['prompt_tokens']})")

    speedup = cold["elapsed_s"] / hot["elapsed_s"] if hot["elapsed_s"] > 0 else 0
    print(f"  speedup: {speedup:.2f}x")


def bench_concurrency() -> None:
    """Concurrency sweep: C1/C2/C4/C8 aggregate decode."""
    print("=== concurrency sweep (256-token output) ===")
    prompt = "Write a short Python function that reads a JSON config file. " * 5

    for concurrency in [1, 2, 4, 8]:
        mtp_before = spec_decode_metrics()
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = [
                pool.submit(
                    chat_completion,
                    [{"role": "user", "content": prompt}],
                    256,
                )
                for _ in range(concurrency)
            ]
            results = [f.result() for f in futures]
        elapsed = time.time() - start_time
        total_tokens = sum(r["completion_tokens"] for r in results)
        aggregate = total_tokens / elapsed
        mtp_after = spec_decode_metrics()
        print(
            f"  C{concurrency}: aggregate={aggregate:.1f} tok/s, "
            f"wall={elapsed:.2f}s, tokens={total_tokens}"
        )
        report_mtp_acceptance(f"C{concurrency}", mtp_before, mtp_after)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.8-27B comprehensive benchmark")
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["decode", "prefill", "prefix", "concurrency", "all"])
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    modes = {
        "decode": bench_decode,
        "prefill": bench_prefill,
        "prefix": bench_prefix_cache,
        "concurrency": bench_concurrency,
    }
    if args.mode == "all":
        for name, func in modes.items():
            func()
    else:
        modes[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
