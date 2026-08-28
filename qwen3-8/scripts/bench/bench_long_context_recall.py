#!/usr/bin/env python3
"""Long-context multi-needle exact-recall benchmark.

Inserts multiple "needles" (unique factual statements) at various depths in a
long context, then asks the model to retrieve each. Tests exact recall, not
survival. The first cache block is salted with a nonce to defeat APC false-green.

Usage:
    python3 bench_long_context_recall.py --lengths 100000 256000 384000 475000
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import chat_completion, health_check

NEEDLES = [
    "The magic password for the north gate is s3r3nd1p1ty-42.",
    "The deployment config file is named haiku-deploy-2026.yaml.",
    "The secret API key for staging is staging-7f3a9b2e1c.",
    "The third checkpoint is located in /opt/nexus/archive/ delta-7.",
    "The maintainer's favorite color is vermillion.",
]


def build_context(length: int, nonce: str) -> tuple[str, list[dict]]:
    unit = (
        f"Repository module {nonce}: handles data validation with strict "
        f"field checking and typed error returns on unknown keys. "
    )
    filler = unit * (length // 16)
    positions = [len(filler) // 6, len(filler) // 3, len(filler) // 2, len(filler) * 2 // 3, len(filler) * 5 // 6]
    random.seed(length)
    needles_shuffled = NEEDLES[:]
    random.shuffle(needles_shuffled)

    context = filler
    needle_locations = []
    for idx, (needle, pos) in enumerate(zip(needles_shuffled, positions)):
        pos = min(pos, len(context))
        context = context[:pos] + f"\n\n[NOTE {idx}]: {needle}\n\n" + context[pos:]
        needle_locations.append({"needle": needle, "index": idx})

    return context, needle_locations


def run_recall_test(length: int) -> dict:
    nonce = f"n{length}-{random.randint(0, 999999)}"
    context, needle_locations = build_context(length, nonce)

    results = []
    for item in needle_locations:
        question = (
            f"\n\nWhat is the exact value stated in [NOTE {item['index']}]? "
            f"Reply with only the value, nothing else."
        )
        response = chat_completion(
            [{"role": "user", "content": context + question}],
            max_tokens=64,
        )
        found = item["needle"].split("is ", 1)[-1].rstrip(".") in response["content"]
        results.append({
            "needle_index": item["index"],
            "found": found,
            "response": response["content"][:80],
        })

    passed = sum(1 for r in results if r["found"])
    total = len(results)
    return {"length": length, "passed": passed, "total": total, "details": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-needle exact-recall benchmark")
    parser.add_argument("--lengths", type=int, nargs="+", default=[100000, 256000, 384000, 475000],
                        help="context lengths to test")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print("=== multi-needle exact recall ===")
    all_passed = True
    for length in args.lengths:
        result = run_recall_test(length)
        status = "PASS" if result["passed"] == result["total"] else "PARTIAL"
        if result["passed"] < result["total"]:
            all_passed = False
        print(f"  {length:>7d}: {result['passed']}/{result['total']} {status}")
        for detail in result["details"]:
            if not detail["found"]:
                print(f"    MISSED note {detail['needle_index']}: {detail['response']}")

    print(f"\n  overall: {'ALL PASS' if all_passed else 'SOME MISSED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
