#!/usr/bin/env python3
"""Generate an immutable runtime manifest (SHA256 + metadata) for reproducibility.

Run on CPU or GPU before any benchmark. The manifest pins:
  - all wheel files (full SHA256)
  - model checkpoint metadata (shard count, total size, index/config/tokenizer hashes)
  - all git repo commits + clean status
  - Python/torch/ROCm versions

Output: JSON to stdout (redirect to a file for persistence).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "/mnt/workspace"))
WHEELS_DIR = WORKSPACE / "wheels"
MODELS_DIR = WORKSPACE / "models"
REPOS = [
    "mi308x-llm-lab",
    "deepseek-v4-flash-mi300x",
    "deepseek-v4-flash-studio",
    "infra",
    "minimax-h3-mi308x",
    "sglang-src",
]
MODEL_PATHS = [
    ("DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ("Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("Qwen3.8-Flash-Next-FP8", "Qwen/Qwen3.8-Flash-Next-FP8"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_dir: Path) -> dict:
    if not (repo_dir / ".git").exists():
        return {"path": str(repo_dir), "commit": "not-a-git-repo", "clean": None}
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    ).stdout.strip())
    return {"path": repo_dir.name, "commit": commit, "clean": not dirty}


def collect_wheels() -> list[dict]:
    wheels = []
    for whl_path in sorted(WHEELS_DIR.rglob("*.whl")):
        wheels.append({
            "name": whl_path.name,
            "size_bytes": whl_path.stat().st_size,
            "sha256": sha256_file(whl_path),
            "rel_path": str(whl_path.relative_to(WHEELS_DIR)),
        })
    return wheels


def collect_model(model_name: str, model_rel: str) -> dict:
    model_dir = MODELS_DIR / model_rel
    if not model_dir.exists():
        return {"name": model_name, "path": str(model_dir), "status": "missing"}
    shards = sorted(model_dir.glob("*.safetensors"))
    metadata_files = ["config.json", "tokenizer_config.json", "generation_config.json",
                      "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja"]
    metadata_hashes = {}
    for mf in metadata_files:
        mf_path = model_dir / mf
        if mf_path.exists():
            metadata_hashes[mf] = sha256_file(mf_path)
    return {
        "name": model_name,
        "path": str(model_dir),
        "status": "present",
        "shard_count": len(shards),
        "total_size_bytes": sum(s.stat().st_size for s in shards),
        "metadata_sha256": metadata_hashes,
    }


def collect_env() -> dict:
    env = {"python": sys.version, "platform": sys.platform}
    try:
        import torch
        env["torch"] = torch.__version__
        env["hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        env["torch"] = "not-installed"
    return env


def main() -> int:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "environment": collect_env(),
        "wheels": collect_wheels(),
        "models": [collect_model(name, rel) for name, rel in MODEL_PATHS],
        "repos": [git_info(WORKSPACE / r) for r in REPOS],
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
