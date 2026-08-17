#!/usr/bin/env python3
"""Session-concurrency benchmark.

N independent long-lived agent histories running concurrently. Each session
grows its own context over multiple turns. Finds the concurrency knee where
per-session decode drops below a latency threshold.

Usage:
    python3 bench_session_concurrency.py --sessions 8 --rounds 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import stream_completion, health_check

SESSION_PREFIX = (
    "You are a coding agent. Repository context: parser.py validates JSON, "
    "service.py implements LRU cache, handler/auth.go does OIDC verification. " * 10
)


def run_single_session(session_id: int, num_rounds: int) -> dict:
    messages = [
        {"role": "system", "content": "You are an expert coding agent."},
        {"role": "user", "content": SESSION_PREFIX + "\n\nHelp me understand this codebase."},
    ]
    total_decode = 0.0
    total_tokens = 0
    total_ttft = 0.0

    for round_index in range(num_rounds):
        messages.append({"role": "user", "content": f"Round {round_index}: explain module {round_index}."})
        result = stream_completion(messages, max_tokens=256)
        messages.append({"role": "assistant", "content": result["content"]})
        messages.append({"role": "tool", "content": f"module_{round_index} has {20+round_index*3} lines."})
        total_decode += result["total_s"] - result["ttft_s"]
        total_tokens += result["completion_tokens"]
        total_ttft += result["ttft_s"]

    avg_decode_rate = total_tokens / total_decode if total_decode > 0 else 0
    avg_ttft = total_ttft / num_rounds
    return {
        "session_id": session_id,
        "total_tokens": total_tokens,
        "total_decode_s": total_decode,
        "avg_decode_rate": avg_decode_rate,
        "avg_ttft": avg_ttft,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-concurrency benchmark")
    parser.add_argument("--sessions", type=int, default=4, help="number of concurrent sessions")
    parser.add_argument("--rounds", type=int, default=4, help="turns per session")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print(f"=== session concurrency: {args.sessions} sessions x {args.rounds} rounds ===")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.sessions) as pool:
        futures = [
            pool.submit(run_single_session, session_id, args.rounds)
            for session_id in range(args.sessions)
        ]
        results = [f.result() for f in futures]
    wall_time = time.time() - start_time

    total_all_tokens = sum(r["total_tokens"] for r in results)
    avg_per_session_rate = sum(r["avg_decode_rate"] for r in results) / len(results)
    avg_ttft = sum(r["avg_ttft"] for r in results) / len(results)
    aggregate = total_all_tokens / wall_time

    print(f"\n  wall_time:        {wall_time:.2f}s")
    print(f"  total tokens:     {total_all_tokens}")
    print(f"  aggregate:        {aggregate:.1f} tok/s")
    print(f"  avg per-session:  {avg_per_session_rate:.1f} tok/s")
    print(f"  avg TTFT:         {avg_ttft:.3f}s")
    print(f"  per-session detail:")
    for r in sorted(results, key=lambda x: x["session_id"]):
        print(f"    session {r['session_id']}: {r['avg_decode_rate']:.1f} tok/s, ttft={r['avg_ttft']:.3f}s, {r['total_tokens']} tok")

    threshold = 180.0  # 5s/turn = 900 tokens / 5s
    if avg_per_session_rate < threshold:
        print(f"\n  >>> below {threshold:.0f} tok/s per-session threshold (knee found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
