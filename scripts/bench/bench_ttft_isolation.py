#!/usr/bin/env python3
"""TTFT isolation benchmark.

Injects a short request during a long cold prefill and measures the added TTFT
penalty. The first cache block is salted with a unique nonce so APC does not
create a false-green from a deterministic prefix.

Usage:
    python3 bench_ttft_isolation.py 200000 --rounds 3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import chat_completion, health_check, repeated_text_for_tokens


def run_isolation_round(prefix_tokens: int) -> dict:
    nonce = f"iso-{prefix_tokens}-{random.randint(0, 999999)}"
    filler_unit = (
        "Repository context block: detailed parser and service descriptions "
        "for measuring cold prefill isolation. "
    )
    long_filler, calibrated_prefix_tokens = repeated_text_for_tokens(
        filler_unit, prefix_tokens, prefix=f"{nonce}\n"
    )
    short_prompt = f"Reply with OK. (nonce {nonce})"

    # Baseline: short request alone (no competing long prefill)
    baseline_start = time.time()
    baseline = chat_completion([{"role": "user", "content": short_prompt}], max_tokens=8)
    baseline_ttft = time.time() - baseline_start

    # Concurrent: long prefill + short request
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        long_future = pool.submit(
            chat_completion,
            [{"role": "user", "content": long_filler + "\n\nReply with OK."}],
            8,
        )
        time.sleep(0.5)  # let the long prefill start
        short_start = time.time()
        short_future = pool.submit(
            chat_completion,
            [{"role": "user", "content": short_prompt}],
            8,
        )
        short_future.result()
        short_isolated_ttft = time.time() - short_start
        long_result = long_future.result()

    added_ttft = short_isolated_ttft - baseline_ttft
    return {
        "baseline_ttft": baseline_ttft,
        "isolated_ttft": short_isolated_ttft,
        "added_ttft": added_ttft,
        "calibrated_prefix_tokens": calibrated_prefix_tokens,
        "actual_prompt_tokens": long_result["prompt_tokens"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TTFT isolation benchmark")
    parser.add_argument("prefix_tokens", type=int, nargs="?", default=200000, help="long prefill prefix tokens")
    parser.add_argument("--rounds", type=int, default=3, help="number of rounds")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print(f"=== TTFT isolation: {args.prefix_tokens} prefix, {args.rounds} rounds ===")
    added_ttf = []
    for round_index in range(1, args.rounds + 1):
        result = run_isolation_round(args.prefix_tokens)
        added_ttf.append(result["added_ttft"])
        print(
            f"  round {round_index}: calibrated_prefix={result['calibrated_prefix_tokens']} "
            f"actual_prompt={result['actual_prompt_tokens']} "
            f"baseline={result['baseline_ttft']:.3f}s, "
            f"isolated={result['isolated_ttft']:.3f}s, "
            f"added={result['added_ttft']:+.3f}s"
        )

    avg_added = sum(added_ttf) / len(added_ttf)
    print(f"\n  avg added TTFT: {avg_added:+.3f}s")
    if avg_added < 0.5:
        print("  PASS (<+0.5s gate)")
    elif avg_added < 2.0:
        print(f"  MARGINAL (<+2.0s gate; target is <+0.5s)")
    else:
        print(f"  FAIL (>{2.0}s added TTFT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
