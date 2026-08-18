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
from bench_client import health_check, repeated_text_for_tokens, stream_completion

SESSION_PREFIX_UNIT = (
    "Repository context: parser.py validates JSON, service.py implements an "
    "LRU cache, and handler/auth.go performs OIDC verification. "
)


def initial_messages(session_prefix: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are an expert coding agent."},
        {"role": "user", "content": session_prefix + "\n\nHelp me understand this codebase."},
    ]


def run_session_round(session_id: int, round_index: int, messages: list[dict]) -> tuple[int, dict]:
    result = stream_completion(messages, max_tokens=256)
    return session_id, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-concurrency benchmark")
    parser.add_argument("--sessions", type=int, default=4, help="number of concurrent sessions")
    parser.add_argument("--rounds", type=int, default=4, help="turns per session")
    parser.add_argument("--prefix-tokens", type=int, default=20000, help="tokenizer-calibrated repository prefix per session")
    parser.add_argument("--min-decode-tps", type=float, default=30.0, help="interactive per-session decode floor")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    session_prefixes: list[str] = []
    calibrated_counts: list[int] = []
    salt_base = time.time_ns()
    for session_id in range(args.sessions):
        prefix, count = repeated_text_for_tokens(
            SESSION_PREFIX_UNIT,
            args.prefix_tokens,
            prefix=f"session-{salt_base}-{session_id}\n",
        )
        session_prefixes.append(prefix)
        calibrated_counts.append(count)

    print(
        f"=== session concurrency: {args.sessions} sessions x {args.rounds} rounds, "
        f"target_prefix={args.prefix_tokens}, calibrated_range="
        f"{min(calibrated_counts)}-{max(calibrated_counts)} ==="
    )
    histories = [initial_messages(prefix) for prefix in session_prefixes]
    round_summaries: list[dict] = []
    total_all_tokens = 0
    total_prompt_tokens = 0
    total_cached_tokens = 0
    overall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(args.sessions) as pool:
        for round_index in range(args.rounds):
            for session_id, messages in enumerate(histories):
                messages.append({
                    "role": "user",
                    "content": f"Round {round_index}: explain module {round_index} for session {session_id}.",
                })

            round_start = time.time()
            futures = [
                pool.submit(run_session_round, session_id, round_index, histories[session_id])
                for session_id in range(args.sessions)
            ]
            round_results = dict(f.result() for f in futures)
            round_wall = time.time() - round_start

            for session_id, result in round_results.items():
                histories[session_id].append({"role": "assistant", "content": result["content"]})
                histories[session_id].append({
                    "role": "tool",
                    "content": f"module_{round_index} has {20 + round_index * 3} lines.",
                })

            round_tokens = sum(r["completion_tokens"] for r in round_results.values())
            round_prompt = sum(r["prompt_tokens"] for r in round_results.values())
            round_cached = sum(r["cached_tokens"] for r in round_results.values())
            round_cache_hit = round_cached / round_prompt if round_prompt else 0.0
            round_avg_ttft = sum(r["ttft_s"] for r in round_results.values()) / args.sessions
            round_avg_decode = sum(r["decode_rate"] for r in round_results.values()) / args.sessions
            round_aggregate = round_tokens / round_wall if round_wall > 0 else 0.0
            summary = {
                "round": round_index + 1,
                "wall": round_wall,
                "tokens": round_tokens,
                "aggregate": round_aggregate,
                "avg_decode": round_avg_decode,
                "avg_ttft": round_avg_ttft,
                "cache_hit": round_cache_hit,
            }
            round_summaries.append(summary)
            total_all_tokens += round_tokens
            total_prompt_tokens += round_prompt
            total_cached_tokens += round_cached
            print(
                f"  round {round_index + 1}: wall={round_wall:.2f}s "
                f"agg={round_aggregate:.1f} tok/s per_session={round_avg_decode:.1f} tok/s "
                f"ttft={round_avg_ttft:.3f}s cache={round_cache_hit:.1%}",
                flush=True,
            )

    wall_time = time.time() - overall_start
    aggregate = total_all_tokens / wall_time if wall_time > 0 else 0.0
    overall_cache_hit = total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
    steady = round_summaries[1:] if len(round_summaries) > 1 else round_summaries
    steady_decode = sum(r["avg_decode"] for r in steady) / len(steady)
    steady_ttft = sum(r["avg_ttft"] for r in steady) / len(steady)
    steady_cache = sum(r["cache_hit"] for r in steady) / len(steady)

    print(f"\n  total wall_time:  {wall_time:.2f}s")
    print(f"  total tokens:     {total_all_tokens}")
    print(f"  aggregate:        {aggregate:.1f} tok/s")
    print(f"  overall cache hit:{overall_cache_hit:9.1%}")
    print(f"  steady decode:    {steady_decode:.1f} tok/s/session")
    print(f"  steady TTFT:      {steady_ttft:.3f}s")
    print(f"  steady cache hit: {steady_cache:.1%}")

    if steady_decode < args.min_decode_tps:
        print(f"\n  >>> below {args.min_decode_tps:.0f} tok/s steady per-session interactive floor (knee found)")
    else:
        print(f"\n  PASS: steady >= {args.min_decode_tps:.0f} tok/s per-session interactive floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
