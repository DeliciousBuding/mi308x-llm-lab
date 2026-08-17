#!/usr/bin/env python3
"""Comprehensive benchmark runner for Qwen3.8-27B on MI308X.

Runs all G-gate tests with proper thinking mode, timeouts, and result logging.
"""
from __future__ import annotations

import json
import os
import sys
import time
import concurrent.futures
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import (
    stream_completion,
    chat_completion,
    report_mtp_acceptance,
    spec_decode_metrics,
)

API_KEY = os.environ.get("VLLM_API_KEY", "")
BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b"
RESULTS = []


def log(label: str, **kwargs):
    entry = {"test": label, "timestamp": time.time(), **kwargs}
    RESULTS.append(entry)
    parts = [f"  {k}={v}" for k, v in kwargs.items()]
    print(f"[{label}] " + " ".join(parts), flush=True)


def bench_decode():
    """G1: Single-stream decode rate."""
    print("=== G1: decode-512 ===", flush=True)
    prompt = "Write a detailed technical explanation of how hybrid attention architectures reduce long-context KV memory. " * 3
    mtp_before = spec_decode_metrics()
    result = stream_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    mtp_after = spec_decode_metrics()
    log("decode-512",
        ttft=result["ttft_s"],
        total=result["total_s"],
        tokens=result["completion_tokens"],
        decode_rate=round(result["decode_rate"], 1),
        cached=result["cached_tokens"])
    report_mtp_acceptance("decode-512", mtp_before, mtp_after)


def bench_prefill():
    """G8: Prefill ladder at 8K/32K/64K/128K."""
    print("=== G8: prefill ladder ===", flush=True)
    for target_tokens in [8_000, 32_000, 64_000, 128_000]:
        filler = "This is a context-padding sentence for measuring prefill latency. " * (target_tokens // 14)
        start = time.time()
        try:
            result = chat_completion(
                [{"role": "user", "content": filler + "\n\nReply with OK."}],
                max_tokens=8,
            )
            elapsed = time.time() - start
            log("prefill",
                target=target_tokens,
                elapsed=round(elapsed, 3),
                prompt_tokens=result["prompt_tokens"],
                cached=result["cached_tokens"])
        except Exception as exc:
            elapsed = time.time() - start
            log("prefill",
                target=target_tokens,
                elapsed=round(elapsed, 3),
                error=str(exc)[:100])


def bench_prefix_cache():
    """Prefix cache cold vs hot."""
    print("=== prefix cache cold/hot ===", flush=True)
    prefix = ("Repository context: parser.py validates JSON configs. " * 200
              + "service.py implements LRU cache with TTL. " * 200)
    question = "\n\nWhat does parser.py do?"

    cold = chat_completion([{"role": "user", "content": prefix + question}], max_tokens=16)
    log("prefix-cold", elapsed=round(cold["elapsed_s"], 3), cached=cold["cached_tokens"])

    hot = chat_completion([{"role": "user", "content": prefix + question}], max_tokens=16)
    log("prefix-hot", elapsed=round(hot["elapsed_s"], 3), cached=hot["cached_tokens"])

    speedup = cold["elapsed_s"] / hot["elapsed_s"] if hot["elapsed_s"] > 0 else 0
    log("prefix-speedup", speedup=round(speedup, 2))


def bench_concurrency():
    """G7: Concurrency sweep C1/C2/C4/C8/C16/C32."""
    print("=== G7: concurrency sweep ===", flush=True)
    prompt = "Write a short Python function that reads a JSON config file. " * 5

    for concurrency in [1, 2, 4, 8, 16, 32]:
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
        per_session = aggregate / concurrency
        mtp_after = spec_decode_metrics()
        log("concurrency",
            C=concurrency,
            aggregate=round(aggregate, 1),
            per_session=round(per_session, 1),
            wall=round(elapsed, 2),
            tokens=total_tokens)
        report_mtp_acceptance(f"C{concurrency}", mtp_before, mtp_after)


def bench_long_recall():
    """G8b: Long-context recall at 32K/64K/128K."""
    print("=== G8b: long-context recall ===", flush=True)
    # Insert a needle at a known position
    needle = "The secret code is: ORANGE-MANGO-7291."
    for context_tokens in [32_000, 64_000, 128_000]:
        filler = "The quick brown fox jumps over the lazy dog. " * (context_tokens // 12)
        # Insert needle at 50% position
        midpoint = len(filler) // 2
        prompt = filler[:midpoint] + f"\n\n{needle}\n\n" + filler[midpoint:]
        prompt += "\n\nWhat is the secret code? Reply with only the code."

        start = time.time()
        try:
            result = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=32,
            )
            elapsed = time.time() - start
            answer = result.get("content", "") or ""
            hit = "7291" in answer or "ORANGE" in answer.upper()
            log("recall",
                context=context_tokens,
                elapsed=round(elapsed, 1),
                prompt_tokens=result["prompt_tokens"],
                hit=hit,
                answer=answer[:80])
        except Exception as exc:
            elapsed = time.time() - start
            log("recall",
                context=context_tokens,
                elapsed=round(elapsed, 1),
                error=str(exc)[:100])


def bench_agent_trace():
    """G10: Simulated agent trace (5 turns with tool calls)."""
    print("=== G10: agent trace (5 turns) ===", flush=True)
    messages = [{"role": "system", "content": "You are a helpful coding agent. Keep responses concise."}]
    total_tokens = 0
    turn_times = []

    turns = [
        "Create a Python function called parse_config that reads a JSON file with error handling.",
        "Now add type hints and a docstring to that function.",
        "Add a CLI wrapper using argparse that accepts --config path.",
        "Add a --verbose flag that enables logging.",
        "Write a unit test using pytest for parse_config.",
    ]

    for i, turn in enumerate(turns):
        messages.append({"role": "user", "content": turn})
        start = time.time()
        result = stream_completion(messages, max_tokens=256)
        elapsed = time.time() - start
        turn_times.append(elapsed)
        total_tokens += result["completion_tokens"]
        messages.append({"role": "assistant", "content": f"[Response {i+1}]"})
        log("agent-turn",
            turn=i + 1,
            elapsed=round(elapsed, 2),
            tokens=result["completion_tokens"],
            decode_rate=round(result["decode_rate"], 1))

    avg_turn = sum(turn_times) / len(turn_times)
    log("agent-summary",
        turns=len(turns),
        avg_turn=round(avg_turn, 2),
        total_tokens=total_tokens)


def main():
    print(f"Server: {BASE_URL} | Model: {MODEL}", flush=True)
    print(f"Thinking: OFF (bench mode)", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("", flush=True)

    tests = sys.argv[1:] if len(sys.argv) > 1 else ["decode", "prefill", "prefix", "concurrency", "recall", "agent"]
    test_map = {
        "decode": bench_decode,
        "prefill": bench_prefill,
        "prefix": bench_prefix_cache,
        "concurrency": bench_concurrency,
        "recall": bench_long_recall,
        "agent": bench_agent_trace,
    }

    for test_name in tests:
        if test_name in test_map:
            try:
                test_map[test_name]()
            except Exception as exc:
                print(f"  ERROR in {test_name}: {exc}", flush=True)
            print("", flush=True)

    # Write JSON results
    results_path = "/tmp/gate_results.json"
    with open(results_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"Results written to {results_path}", flush=True)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
