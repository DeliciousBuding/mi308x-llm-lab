#!/usr/bin/env python3
"""Coding-agent multi-turn benchmark.

Simulates a growing agent conversation: stable system prompt + repository
context prefix, then multiple turns of user/assistant/tool exchanges. Measures
per-request prefix-cache hit rate and decode rate across the trace.

Authoritative cache metric: usage.prompt_tokens_details.cached_tokens.

Usage:
    python3 bench_agent_trace.py <num_turns> <prefix_tokens>
    python3 bench_agent_trace.py 30 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import chat_completion, health_check, repeated_text_for_tokens, stream_completion

SYSTEM_PROMPT = (
    "You are an expert coding agent. Follow repository conventions, prefer "
    "explicit error handling, and explain non-obvious choices."
)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
REPO_CONTEXT_UNIT = (
    "File utils/parser.py: parse JSON config with strict field validation, "
    "return typed errors on unknown keys. File service/cache.go: LRU cache "
    "with per-entry TTL and single-flight refresh. "
)


def make_repo_context(target_tokens: int, prefix: str = "") -> tuple[str, int]:
    return repeated_text_for_tokens(REPO_CONTEXT_UNIT, target_tokens, prefix=prefix)


def run_agent_trace(num_turns: int, prefix_tokens: int, cold_prefix: bool = False) -> int:
    if not health_check():
        print("ERROR: server not healthy")
        return 1

    prefix_salt = f"cold-prefix-{time.time_ns()}\n" if cold_prefix else ""
    repo_context, calibrated_prefix_tokens = make_repo_context(prefix_tokens, prefix=prefix_salt)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": repo_context + "\n\nHelp me understand this codebase."},
    ]

    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0
    total_decode_time = 0.0

    print(
        f"=== agent trace: {num_turns} turns, target={prefix_tokens} prefix tokens, "
        f"tokenizer_calibrated={calibrated_prefix_tokens} ==="
    )

    for turn_index in range(1, num_turns + 1):
        user_message = (
            f"Turn {turn_index}: Read file src/module_{turn_index}.py and "
            f"explain what it does. Then suggest one improvement."
        )
        messages.append({"role": "user", "content": user_message})

        result = stream_completion(messages, max_tokens=512, temperature=0.0)
        assistant_content = result["content"]

        messages.append({"role": "assistant", "content": assistant_content})

        # Simulate tool result
        tool_result = (
            f"File src/module_{turn_index}.py contains a data validator "
            f"with {20 + turn_index * 3} lines of code."
        )
        messages.append({"role": "tool", "content": tool_result})

        total_prompt_tokens += result["prompt_tokens"]
        total_cached_tokens += result["cached_tokens"]
        total_completion_tokens += result["completion_tokens"]
        total_decode_time += result["total_s"] - result["ttft_s"]

        cache_hit = 100 * result["cached_tokens"] / result["prompt_tokens"] if result["prompt_tokens"] else 0
        print(
            f"  turn {turn_index:2d}: ctx~{result['prompt_tokens']:>6d} "
            f"cached={result['cached_tokens']:>6d} ({cache_hit:.1f}%) "
            f"decode={result['completion_tokens']:>4d} tok "
            f"ttft={result['ttft_s']:.2f}s"
        )

    overall_hit = 100 * total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0
    avg_decode_rate = total_completion_tokens / total_decode_time if total_decode_time > 0 else 0
    print()
    print(f"  total prompt tokens:   {total_prompt_tokens:>10,}")
    print(f"  total cached tokens:   {total_cached_tokens:>10,}")
    print(f"  overall cache hit:      {overall_hit:.2f}%")
    print(f"  total completion tokens: {total_completion_tokens:>10,}")
    print(f"  avg decode rate:        {avg_decode_rate:.1f} tok/s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Coding-agent multi-turn benchmark")
    parser.add_argument("num_turns", type=int, nargs="?", default=30, help="number of agent turns")
    parser.add_argument("prefix_tokens", type=int, nargs="?", default=20000, help="prefix context tokens")
    parser.add_argument("--cold-prefix", action="store_true", help="salt the first cache block so turn 1 is guaranteed cold")
    args = parser.parse_args()
    return run_agent_trace(args.num_turns, args.prefix_tokens, args.cold_prefix)


if __name__ == "__main__":
    raise SystemExit(main())
