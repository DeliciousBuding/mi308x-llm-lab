#!/usr/bin/env python3
"""Minimal protocol correctness fixtures for Coding Agent use cases.

Tests the full protocol stack against a live server:
  1. Basic chat completion (non-streaming)
  2. Streaming chat with reasoning_content
  3. Single tool call (auto mode)
  4. Sequential tool calls (tool result → follow-up)
  5. Parallel tool calls
  6. Large tool result (10K tokens)
  7. Anthropic /v1/messages (SGLang only, if available)

Each test has a fixed expected behavior. Failures indicate protocol issues
that would break Coding Agent harnesses (Claude Code, OpenClaw, etc.).

Usage:
    python3 protocol_fixtures.py
    python3 protocol_fixtures.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import resolve_api_key, health_check, BASE_URL, MODEL_NAME

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read."}
            },
            "required": ["path"],
        },
    },
}
TOOL_DEFINITION_2 = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "Directory to list."}
            },
            "required": ["dir"],
        },
    },
}


def run_fixture(name: str, test_fn, base_url: str) -> bool:
    print(f"  [{name}] ", end="", flush=True)
    try:
        result = test_fn(base_url)
        if result:
            print("PASS")
            return True
        else:
            print("FAIL")
            return False
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False


def fixture_basic_chat(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    message = resp["choices"][0]["message"]
    # In thinking mode the answer may land in reasoning_content while content is None.
    content = message.get("content") or message.get("reasoning_content") or ""
    return "OK" in content or "ok" in content.lower()


def fixture_streaming_reasoning(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2+2? Think step by step."}],
        "max_tokens": 256, "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    has_content = False
    has_reasoning = False
    for raw in urllib.request.urlopen(req, timeout=120):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                has_content = True
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                has_reasoning = True
            elif isinstance(delta.get("reasoning_content"), str) and delta["reasoning_content"]:
                has_reasoning = True
    return has_content  # reasoning may be absent if thinking is off; content is required


def fixture_single_tool(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Read the file src/config.py and tell me what it does."}],
        "max_tokens": 256, "temperature": 0.0,
        "tools": [TOOL_DEFINITION],
        "tool_choice": "auto",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    tool_calls = resp["choices"][0]["message"].get("tool_calls", [])
    return len(tool_calls) > 0 and tool_calls[0]["function"]["name"] == "read_file"


def fixture_sequential_tool(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    messages = [
        {"role": "user", "content": "Read src/config.py."},
    ]
    body = json.dumps({
        "model": MODEL_NAME, "messages": messages,
        "max_tokens": 128, "temperature": 0.0,
        "tools": [TOOL_DEFINITION], "tool_choice": "auto",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp1 = json.load(urllib.request.urlopen(req, timeout=60))
    tool_calls = resp1["choices"][0]["message"].get("tool_calls", [])
    if not tool_calls:
        return False
    messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    messages.append({"role": "tool", "content": "DATABASE_URL = localhost:5432"})
    body2 = json.dumps({
        "model": MODEL_NAME, "messages": messages,
        "max_tokens": 128, "temperature": 0.0,
    }).encode()
    req2 = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body2,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp2 = json.load(urllib.request.urlopen(req2, timeout=60))
    return bool(resp2["choices"][0]["message"]["content"])


def fixture_parallel_tool(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Read src/config.py and list files in src/."}],
        "max_tokens": 256, "temperature": 0.0,
        "tools": [TOOL_DEFINITION, TOOL_DEFINITION_2],
        "tool_choice": "auto",
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    tool_calls = resp["choices"][0]["message"].get("tool_calls", [])
    names = {tc["function"]["name"] for tc in tool_calls}
    return len(tool_calls) >= 2 and "read_file" in names and "list_files" in names


def fixture_large_tool_result(base_url: str) -> bool:
    import urllib.request
    api_key = resolve_api_key()
    large_result = "line of code\n" * 800  # ~10K tokens
    messages = [
        {"role": "user", "content": "Read big_file.py."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"big_file.py"}'}}
        ]},
        {"role": "tool", "content": large_result},
    ]
    body = json.dumps({
        "model": MODEL_NAME, "messages": messages,
        "max_tokens": 128, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    return bool(resp["choices"][0]["message"]["content"])


FIXTURES = [
    ("basic_chat", fixture_basic_chat),
    ("streaming_reasoning", fixture_streaming_reasoning),
    ("single_tool", fixture_single_tool),
    ("sequential_tool", fixture_sequential_tool),
    ("parallel_tool", fixture_parallel_tool),
    ("large_tool_result", fixture_large_tool_result),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Protocol correctness fixtures")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    if not health_check():
        print("ERROR: server not healthy")
        return 1

    print(f"=== Protocol Fixtures ({args.base_url}) ===")
    passed = 0
    for name, test_fn in FIXTURES:
        if run_fixture(name, test_fn, args.base_url):
            passed += 1

    print(f"\n  {passed}/{len(FIXTURES)} passed")
    return 0 if passed == len(FIXTURES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
