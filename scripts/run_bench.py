#!/usr/bin/env python3
"""Simple gate-aware benchmark runner.

Maps --profile to the right bench commands. Not a full YAML orchestration platform —
just a convenience wrapper so GPU time isn't wasted on manual command lookup.

Usage:
    python3 run_bench.py --profile g1          # vLLM correctness reference
    python3 run_bench.py --profile g2          # SGLang correctness reference
    python3 run_bench.py --profile g5-mtp3      # MTP-3 sweep
    python3 run_bench.py --profile g7-conc     # concurrency knee
    python3 run_bench.py --profile g10-agent   # real agent replay
    python3 run_bench.py --profile corpus      # generate corpus
    python3 run_bench.py --profile manifest    # generate manifest
    python3 run_bench.py --profile canary      # canary sequence
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(RECIPE_ROOT, "scripts", "bench")
RESULTS_DIR = os.path.join(RECIPE_ROOT, "results")

PROFILES = {
    # Environment prep
    "manifest": {
        "desc": "Generate immutable runtime manifest",
        "cmd": f"python3 {RECIPE_ROOT}/scripts/runtime_manifest.py > {RESULTS_DIR}/runtime_manifest.json",
    },
    "corpus": {
        "desc": "Generate benchmark corpus",
        "cmd": f"python3 {BENCH_DIR}/generate_corpus.py --output-dir {RESULTS_DIR}/corpus",
    },
    "canary": {
        "desc": "Canary sequence (health → gen → stream → tool → 32K → fixtures)",
        "cmd": f"bash {RECIPE_ROOT}/scripts/canary_rollback.sh",
    },
    # Correctness
    "g1": {
        "desc": "G1: vLLM 262K native BF16 KV MTP-off C1 correctness",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode && python3 {BENCH_DIR}/protocol_fixtures.py",
    },
    "g2": {
        "desc": "G2: SGLang 262K native float32 SSM no_buffer MTP-off C1 correctness",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode && python3 {BENCH_DIR}/protocol_fixtures.py",
    },
    # Performance
    "g3": {
        "desc": "G3: decode-512 single-stream (engine A/B)",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp0": {
        "desc": "G5: native decode (MTP off)",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp1": {
        "desc": "G5: MTP-1 decode",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp3": {
        "desc": "G5: MTP-3 decode",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g7-conc": {
        "desc": "G7: concurrency sweep C1/C2/C5/C10/C16/C24/C32",
        "cmd": f"python3 {BENCH_DIR}/bench_high_concurrency.py --concurrencies 1 2 5 10 16 24 32",
    },
    "g8-prefill": {
        "desc": "G8: prefill ladder 8K/32K/100K/200K",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py prefill",
    },
    "g8-recall": {
        "desc": "G8: multi-needle recall 32K/128K/256K",
        "cmd": f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 32000 128000 256000",
    },
    "g9-recall": {
        "desc": "G9: 384K/475K recall (YaRN 512K)",
        "cmd": f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 384000 475000",
    },
    "g10-agent": {
        "desc": "G10: 30-turn agent trace (20K prefix)",
        "cmd": f"python3 {BENCH_DIR}/bench_agent_trace.py 30 20000",
    },
    "g10-sessions": {
        "desc": "G10: session concurrency 8/16 sessions",
        "cmd": f"python3 {BENCH_DIR}/bench_session_concurrency.py --sessions 8 --rounds 4 && "
               f"python3 {BENCH_DIR}/bench_session_concurrency.py --sessions 16 --rounds 4",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate-aware benchmark runner")
    parser.add_argument("--profile", required=True, choices=list(PROFILES.keys()),
                        help="benchmark profile to run")
    parser.add_argument("--dry-run", action="store_true", help="print command without executing")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    cmd = profile["cmd"]

    print(f"=== {args.profile}: {profile['desc']} ===")
    print(f"  cmd: {cmd}")

    if args.dry_run:
        return 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = subprocess.run(cmd, shell=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
