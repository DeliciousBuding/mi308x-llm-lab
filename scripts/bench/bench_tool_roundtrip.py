#!/usr/bin/env python3
"""Tool-call round-trip benchmark.

Tests the full streamed tool protocol: model emits a tool call, the harness
executes it and feeds back a tool result, then the model produces a final
answer. Validates the qwen3_coder tool parser under auto/forced/required modes.

Usage:
    python3 bench_tool_roundtrip.py --rounds 5 --mode auto --prefix-tokens 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import health_check, repeated_text_for_tokens, stream_completion

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
}


def make_prefix(target_tokens: int) -> tuple[str, int]:
    unit = "Repository context: parser.py validates JSON configs with strict fields. "
    return repeated_text_for_tokens(unit, target_tokens, prefix=f"tool-{time.time_ns()}\n")


def run_tool_roundtrip(mode: str, prefix_tokens: int) -> dict:
    prefix, calibrated_prefix_tokens = make_prefix(prefix_tokens)
    messages = [
        {"role": "user", "content": prefix + "\n\nRead the file src/config.py and tell me what it does."},
    ]
    extra = {
        "tools": [TOOL_DEFINITION],
        "tool_choice": mode if mode != "auto" else "auto",
    }

    first_response = stream_completion(messages, max_tokens=256, temperature=0.0, extra=extra)
    tool_calls = first_response.get("tool_calls", [])

    if not tool_calls:
        return {
            "passed": False,
            "reason": "no tool call emitted",
            "content": first_response["content"][:120],
        }

    # Feed back a tool result
    messages.append({"role": "assistant", "content": first_response["content"], "tool_calls": tool_calls})
    tool_result_content = 'File src/config.py contains:\nDATABASE_URL = "localhost:5432"\nDEBUG = True'
    messages.append({"role": "tool", "content": tool_result_content})

    final_response = stream_completion(messages, max_tokens=256, temperature=0.0)
    return {
        "passed": (
            first_response.get("role_assistant", False)
            and first_response.get("finish_reason") == "tool_calls"
            and bool(final_response["content"])
            and final_response.get("role_assistant", False)
        ),
        "tool_call": json.dumps(tool_calls[0], separators=(",", ":"))[:120] if tool_calls else "",
        "final_content": final_response["content"][:120],
        "calibrated_prefix_tokens": calibrated_prefix_tokens,
        "tool_ttft": first_response["ttft_s"],
        "final_ttft": final_response["ttft_s"],
        "tool_finish": first_response.get("finish_reason", ""),
        "total_s": first_response["total_s"] + final_response["total_s"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-call round-trip benchmark")
    parser.add_argument("--rounds", type=int, default=5, help="number of rounds")
    parser.add_argument("--mode", choices=["auto", "forced", "required"], default="auto", help="tool_choice mode")
    parser.add_argument("--prefix-tokens", type=int, default=20000, help="prefix context tokens")
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print(f"=== tool round-trip: mode={args.mode}, prefix={args.prefix_tokens}, rounds={args.rounds} ===")
    passed = 0
    for round_index in range(1, args.rounds + 1):
        result = run_tool_roundtrip(args.mode, args.prefix_tokens)
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1
        print(
            f"  round {round_index}: {status} | "
            f"prefix={result.get('calibrated_prefix_tokens', 0)} "
            f"tool_ttft={result.get('tool_ttft', 0):.3f}s "
            f"final_ttft={result.get('final_ttft', 0):.3f}s | "
            f"{result.get('tool_call', result.get('reason', ''))}"
        )
        if not result["passed"]:
            print(f"    reason: {result.get('reason', 'unknown')}")

    print(f"\n  {passed}/{args.rounds} passed")
    return 0 if passed == args.rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())
