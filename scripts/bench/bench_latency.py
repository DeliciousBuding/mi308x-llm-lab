#!/usr/bin/env python3
"""Latency benchmark — TTFT and decode rate fixture.

Single-stream streaming completion measuring TTFT and tokens-per-second decode
rate. Useful for MTP vs native A/B comparisons.

Usage:
    python3 bench_latency.py [--output-tokens 512]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import report_mtp_acceptance, spec_decode_metrics, stream_completion, health_check


def main() -> int:
    parser = argparse.ArgumentParser(description="TTFT and decode latency benchmark")
    parser.add_argument("--output-tokens", type=int, default=512, help="max output tokens")
    parser.add_argument("--rounds", type=int, default=3, help="number of rounds")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    prompt = "Write a detailed explanation of how hybrid attention reduces KV memory. " * 3
    print(f"=== latency: {args.output_tokens} tokens, {args.rounds} rounds ===")

    ttft_values = []
    decode_rates = []
    for round_index in range(1, args.rounds + 1):
        mtp_before = spec_decode_metrics()
        result = stream_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=args.output_tokens,
        )
        mtp_after = spec_decode_metrics()
        ttft_values.append(result["ttft_s"])
        decode_rates.append(result["decode_rate"])
        print(
            f"  round {round_index}: ttft={result['ttft_s']:.3f}s, "
            f"decode={result['decode_rate']:.1f} tok/s, "
            f"total={result['total_s']:.3f}s, "
            f"tokens={result['completion_tokens']}"
        )
        report_mtp_acceptance(f"latency-r{round_index}", mtp_before, mtp_after)

    avg_ttft = sum(ttft_values) / len(ttft_values)
    avg_decode = sum(decode_rates) / len(decode_rates)
    print(f"\n  avg TTFT:   {avg_ttft:.3f}s")
    print(f"  avg decode: {avg_decode:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
