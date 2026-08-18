#!/usr/bin/env python3
"""Deterministic benchmark corpus generator.

Generates fixed, reproducible prompts at various context lengths for benchmarking.
Uses deterministic seeds — no randomness, same output every run.

Generates:
  - Context padding prompts: 1K / 8K / 32K / 128K / 256K / 512K tokens
  - Multi-needle recall prompts (5 needles at varying depths)
  - Agent trace prefix (20K repo context + system prompt)
  - Branching agent prefix (shared 80K → fork 8 children)

Usage:
    python3 generate_corpus.py --output-dir /tmp/bench_corpus
    python3 generate_corpus.py --lengths 1024 8192 32768 131072
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import json
from pathlib import Path

REPO_UNIT = (
    "File src/parser.py: parse JSON config with strict field validation, "
    "return typed errors on unknown keys. "
    "File src/cache.py: LRU cache with per-entry TTL and single-flight refresh. "
    "File src/auth.py: OIDC token verification with clock-skew tolerance. "
)
SYSTEM_PROMPT = (
    "You are an expert coding agent. Follow repository conventions, prefer "
    "explicit error handling, and explain non-obvious choices."
)
NEEDLES = [
    "The magic password for the north gate is s3r3nd1p1ty-42.",
    "The deployment config file is named haiku-deploy-2026.yaml.",
    "The secret API key for staging is staging-7f3a9b2e1c.",
    "The third checkpoint is located in /opt/nexus/archive/delta-7.",
    "The maintainer's favorite color is vermillion.",
]


def deterministic_prompt(target_tokens: int, seed: int = 42) -> str:
    """Generate a deterministic padding prompt of approximately target_tokens tokens."""
    rng = random.Random(seed)
    unit = REPO_UNIT
    repeats = max(1, target_tokens // 14)
    parts = []
    for i in range(repeats):
        # Vary the unit slightly with deterministic noise
        noise = rng.randint(0, 9999)
        parts.append(f"[{noise:04d}] {unit}")
    return "".join(parts)[:target_tokens * 4]  # rough char-to-token ratio


def generate_context_prompt(target_tokens: int) -> dict:
    """Generate a benchmark prompt at approximately target_tokens context."""
    if target_tokens <= 500:
        return {"role": "user", "content": "Write a Python function that reads a JSON file."}
    padding = deterministic_prompt(target_tokens - 100)
    return {
        "role": "user",
        "content": padding + "\n\nBased on the above repository context, write a Python function that reads and validates a JSON config file. Include type hints."
    }


def generate_multi_needle(length: int, seed: int = 42) -> dict:
    """Generate a multi-needle recall prompt at the given length."""
    rng = random.Random(seed)
    filler = deterministic_prompt(length, seed=seed)
    positions = [len(filler) // 6, len(filler) // 3, len(filler) // 2, len(filler) * 2 // 3, len(filler) * 5 // 6]
    needles_shuffled = NEEDLES[:]
    rng.shuffle(needles_shuffled)
    context = filler
    for idx, (needle, pos) in enumerate(zip(needles_shuffled, positions)):
        pos = min(pos, len(context))
        context = context[:pos] + f"\n\n[NOTE {idx}]: {needle}\n\n" + context[pos:]
    return {
        "role": "user",
        "content": context + "\n\nWhat is the exact value stated in [NOTE 0]? Reply with only the value."
    }


def generate_agent_trace_prefix(prefix_tokens: int) -> list[dict]:
    """Generate an agent trace prefix (system + repo context + initial user)."""
    repo_padding = deterministic_prompt(prefix_tokens)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": repo_padding + "\n\nHelp me understand this codebase."},
    ]


def generate_branching_agent(parent_prefix_tokens: int, num_children: int = 8) -> dict:
    """Generate a branching agent scenario: shared parent prefix → fork children."""
    parent_prefix = deterministic_prompt(parent_prefix_tokens)
    children = []
    for i in range(num_children):
        children.append({
            "child_id": i,
            "message": f"Child agent {i}: analyze module_{i}.py and report its purpose.",
        })
    return {
        "parent_prefix_tokens": parent_prefix_tokens,
        "parent_prefix": parent_prefix[:200] + "...",
        "children": children,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark corpus generator")
    parser.add_argument("--output-dir", default="/tmp/bench_corpus",
                        help="output directory for corpus files")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[1024, 8192, 32768, 131072, 262144, 524288],
                        help="context lengths to generate")
    parser.add_argument("--needles", type=int, nargs="+",
                        default=[32000, 128000, 256000, 384000, 475000],
                        help="multi-needle lengths to generate")
    parser.add_argument("--agent-prefix", type=int, default=20000,
                        help="agent trace prefix tokens")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Context prompts
    for length in args.lengths:
        prompt = generate_context_prompt(length)
        out_file = output_dir / f"context_{length}.json"
        out_file.write_text(json.dumps({"target_tokens": length, "messages": [prompt]}))
        print(f"  generated: {out_file.name} (~{length} tokens)")

    # Multi-needle
    for length in args.needles:
        prompt = generate_multi_needle(length)
        out_file = output_dir / f"needle_{length}.json"
        out_file.write_text(json.dumps({"target_tokens": length, "messages": [prompt]}))
        print(f"  generated: {out_file.name} (~{length} tokens, 5 needles)")

    # Agent trace prefix
    agent_prefix = generate_agent_trace_prefix(args.agent_prefix)
    out_file = output_dir / f"agent_prefix_{args.agent_prefix}.json"
    out_file.write_text(json.dumps({"messages": agent_prefix}))
    print(f"  generated: {out_file.name} (~{args.agent_prefix} prefix tokens)")

    # Branching agent
    branching = generate_branching_agent(80000, 8)
    out_file = output_dir / "branching_agent_80K_8children.json"
    out_file.write_text(json.dumps(branching))
    print(f"  generated: {out_file.name} (80K parent → 8 children)")

    print(f"\n  corpus generated at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
