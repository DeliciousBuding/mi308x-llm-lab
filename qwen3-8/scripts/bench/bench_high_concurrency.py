#!/usr/bin/env python3
"""High-concurrency boundary benchmark.

Clean C1/C2/C4/C8/C32/C64 boundary: indexed 256-token coding fixture,
standalone rerun to avoid interference from other traffic. Measures MTP vs
native decode aggregate at each concurrency level.

Usage:
    python3 bench_high_concurrency.py --concurrencies 1 2 4 8 32 64
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import chat_completion, report_mtp_acceptance, spec_decode_metrics, health_check

FIXTURE_PROMPT = (
    "Write a Python function that parses a JSON config file with error "
    "handling. Include type hints and docstring. " * 5
)


def run_concurrency_level(concurrency: int) -> dict:
    mtp_before = spec_decode_metrics()
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futures = [
            pool.submit(chat_completion, [{"role": "user", "content": FIXTURE_PROMPT}], 256)
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]
    elapsed = time.time() - start_time
    total_tokens = sum(r["completion_tokens"] for r in results)
    aggregate = total_tokens / elapsed
    mtp_after = spec_decode_metrics()
    per_session = aggregate / concurrency

    print(
        f"  C{concurrency:>2d}: aggregate={aggregate:7.1f} tok/s, "
        f"per_session={per_session:6.1f} tok/s, "
        f"wall={elapsed:.2f}s, tokens={total_tokens}"
    )
    report_mtp_acceptance(f"C{concurrency}", mtp_before, mtp_after)
    return {"concurrency": concurrency, "aggregate": aggregate, "per_session": per_session}


def main() -> int:
    parser = argparse.ArgumentParser(description="High-concurrency boundary benchmark")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 32, 64],
                        help="concurrency levels to test")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print("=== high-concurrency boundary (256-token fixture) ===")
    results = []
    for concurrency in args.concurrencies:
        result = run_concurrency_level(concurrency)
        results.append(result)

    print("\n=== summary ===")
    print(f"{'C':>4s} {'aggregate':>12s} {'per_session':>12s}")
    for r in results:
        print(f"{r['concurrency']:>4d} {r['aggregate']:>10.1f}   {r['per_session']:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
