#!/usr/bin/env python3
"""Simple gate-aware benchmark runner.

Maps --profile to the right bench commands. Not a full YAML orchestration platform —
just a convenience wrapper so GPU time isn't wasted on manual command lookup.

Usage:
    python3 run_bench.py --profile g1               # current-row correctness
    python3 run_bench.py --profile g6-kv-row        # repeat for each KV A/B row
    python3 run_bench.py --profile g9-native-recall # native-256K production recall
    python3 run_bench.py --profile g10-sessions     # C8/C10 long-lived Agent replay
    python3 run_bench.py --profile corpus           # native-256K corpus defaults
    python3 run_bench.py --profile manifest         # runtime manifest
    python3 run_bench.py --profile canary           # canary sequence
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
        "desc": "Generate native-256K benchmark corpus (512K YaRN is explicit opt-in)",
        "cmd": f"python3 {BENCH_DIR}/generate_corpus.py --output-dir {RESULTS_DIR}/corpus",
    },
    "canary": {
        "desc": "Canary sequence against the currently running vLLM row",
        "cmd": f"bash {RECIPE_ROOT}/scripts/canary_rollback.sh",
    },
    # Correctness/performance. Server profile changes are intentionally external:
    # switch one controlled row in a maintenance window, then run the same client gate.
    "g1": {
        "desc": "G1: client correctness against the current native-256K vLLM row",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode && python3 {BENCH_DIR}/protocol_fixtures.py",
    },
    "g3": {
        "desc": "G3: decode-512 against the current attention-backend row",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp0": {
        "desc": "G5: decode gate for a server row intentionally started with MTP off",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp1": {
        "desc": "G5: decode gate for a server row intentionally started with MTP-1",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g5-mtp3": {
        "desc": "G5: decode gate for a server row intentionally started with MTP-3",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py decode",
    },
    "g6-kv-row": {
        "desc": "G6: quality/capacity client gate for the currently running KV-cache row",
        "cmd": (
            f"python3 {BENCH_DIR}/bench_full.py decode && "
            f"python3 {BENCH_DIR}/protocol_fixtures.py && "
            f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 32000 128000 240000"
        ),
    },
    "g7-conc": {
        "desc": "G7: concurrency sweep C1/C2/C5/C10/C16/C24/C32",
        "cmd": f"python3 {BENCH_DIR}/bench_high_concurrency.py --concurrencies 1 2 5 10 16 24 32",
    },
    "g8-prefill": {
        "desc": "G8: prefill ladder against native production ceiling",
        "cmd": f"python3 {BENCH_DIR}/bench_full.py prefill",
    },
    "g8-recall": {
        "desc": "G8: multi-needle recall 32K/128K",
        "cmd": f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 32000 128000",
    },
    "g9-native-recall": {
        "desc": "G9: native-256K exact recall (production gate)",
        "cmd": f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 128000 192000 240000",
    },
    "g9-yarn-recall": {
        "desc": "G9: optional 512K YaRN recall; server must be intentionally started at 524288",
        "cmd": f"python3 {BENCH_DIR}/bench_long_context_recall.py --lengths 256000 384000 475000",
    },
    "g10-agent": {
        "desc": "G10: 30-turn agent trace (20K prefix)",
        "cmd": f"python3 {BENCH_DIR}/bench_agent_trace.py 30 20000",
    },
    "g10-sessions": {
        "desc": "G10: long-lived Agent session concurrency at C8 and C10",
        "cmd": f"python3 {BENCH_DIR}/bench_session_concurrency.py --sessions 8 --rounds 3 && "
               f"python3 {BENCH_DIR}/bench_session_concurrency.py --sessions 10 --rounds 3",
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
