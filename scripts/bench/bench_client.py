#!/usr/bin/env python3
"""Shared HTTP client and helpers for the Qwen3.8-27B benchmark suite.

All bench scripts import from this module to keep API-key resolution, request
dispatch, and metric parsing in one place. Uses only stdlib (urllib) so the
suite runs without extra pip installs on CI and developer hosts.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
REQUEST_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "3600"))


def resolve_api_key() -> str:
    """Resolve credentials only when a request is actually sent.

    Keeping this lazy makes ``--help`` and CI imports safe on hosts that do not
    have the private DSW key file.
    """
    configured_api_key = os.environ.get("VLLM_API_KEY")
    if configured_api_key:
        return configured_api_key

    api_key_file = os.environ.get("VLLM_API_KEY_FILE")
    if not api_key_file:
        raise RuntimeError(
            "No API key configured; set VLLM_API_KEY or VLLM_API_KEY_FILE"
        )

    try:
        with open(api_key_file, encoding="utf-8") as api_key_stream:
            resolved_api_key = api_key_stream.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"VLLM_API_KEY_FILE does not exist: {api_key_file}"
        ) from exc

    if not resolved_api_key:
        raise RuntimeError(f"VLLM_API_KEY_FILE is empty: {api_key_file}")
    return resolved_api_key


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    try:
        headers["Authorization"] = "Bearer " + resolve_api_key()
    except RuntimeError:
        pass
    return headers


def chat_completion(
    messages: list[dict],
    max_tokens: int = 64,
    temperature: float = 0.0,
    stream: bool = False,
    extra: dict | None = None,
) -> dict:
    """Send a non-streaming chat completion and return parsed response + timing."""
    body_dict: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        body_dict.update(extra)
    body = json.dumps(body_dict).encode()
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=body,
        headers=_headers(),
    )
    start_time = time.time()
    response = json.load(urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT))
    elapsed = time.time() - start_time
    usage = response.get("usage", {})
    choice = response["choices"][0]
    message = choice.get("message", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if usage.get("prompt_tokens_details")
            else 0
        ),
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "tool_calls": message.get("tool_calls", []),
    }


def stream_completion(
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    extra: dict | None = None,
) -> dict:
    """Stream a chat completion, measuring TTFT and decode rate."""
    body_dict: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        body_dict.update(extra)
    body = json.dumps(body_dict).encode()
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=body,
        headers=_headers(),
    )
    start_time = time.time()
    first_token_time = None
    content_chunks: list[str] = []
    usage_data = {}

    response_stream = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
    for raw_line in response_stream:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if first_token_time is None and chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                first_token_time = time.time()
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                content_chunks.append(delta["content"])
        if chunk.get("usage"):
            usage_data = chunk["usage"]

    end_time = time.time()
    ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
    total_elapsed = end_time - start_time
    completion_tokens = usage_data.get("completion_tokens", 0)
    decode_rate = completion_tokens / (end_time - first_token_time) if first_token_time and (end_time > first_token_time) else 0.0

    return {
        "ttft_s": ttft,
        "total_s": total_elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage_data.get("prompt_tokens", 0),
        "cached_tokens": (
            usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if usage_data.get("prompt_tokens_details")
            else 0
        ),
        "content": "".join(content_chunks),
        "decode_rate": decode_rate,
    }


def spec_decode_metrics() -> dict[str, float]:
    """Parse speculative-decode (MTP) counters from the /metrics endpoint."""
    try:
        raw = urllib.request.urlopen(BASE_URL + "/metrics", timeout=10).read().decode()
    except urllib.error.URLError:
        return {"drafts": 0.0, "accepted": 0.0, "draft_tokens": 0.0}

    def extract(name: str) -> float:
        match = re.search(rf"^vllm:{name}_total\S*\s([0-9.e+]+)$", raw, re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    return {
        "drafts": extract("spec_decode_num_drafts"),
        "accepted": extract("spec_decode_num_accepted_tokens"),
        "draft_tokens": extract("spec_decode_num_draft_tokens"),
    }


def report_mtp_acceptance(label: str, before: dict, after: dict) -> None:
    """Print MTP acceptance stats for a labeled measurement window."""
    delta_drafts = after["drafts"] - before["drafts"]
    delta_accepted = after["accepted"] - before["accepted"]
    delta_draft_tokens = after["draft_tokens"] - before["draft_tokens"]
    if delta_drafts > 0:
        mean_accepted = 1 + delta_accepted / delta_drafts
        accept_rate = 100 * delta_accepted / delta_draft_tokens if delta_draft_tokens else 0
        print(
            f"  [MTP] {label}: mean_accepted_len={mean_accepted:.2f} "
            f"(1+{delta_accepted:.0f}/{delta_drafts:.0f}), "
            f"accept_rate={accept_rate:.0f}% ({delta_accepted:.0f}/{delta_draft_tokens:.0f})",
            flush=True,
        )
    else:
        print(f"  [MTP] {label}: no drafts (MTP may not have triggered)", flush=True)


def health_check() -> bool:
    try:
        urllib.request.urlopen(BASE_URL + "/health", timeout=10)
        return True
    except Exception:
        return False
