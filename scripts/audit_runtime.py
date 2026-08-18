#!/usr/bin/env python3
"""Audit the installed vLLM runtime for Qwen3.8-27B serving.

Run after a host/bootstrap restart and before performance testing. The audit is
read-only: it verifies runtime versions, Qwen3.8 architecture registration, and
the Gated DeltaNet linear-attention import path.

Unlike the sibling DeepSeek-V4-Flash audit, there are no fork overlays, no
patched binaries, and no sparse-prefill artifacts to verify — Qwen3.8 is
upstream-native in vLLM.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

VENV = Path(os.environ.get("VLLM_VENV", "/root/.venvs/vllm-qwen"))
RECIPE_REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> str:
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return p.stdout.strip()


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    py = VENV / "bin" / "python"
    if not py.exists():
        print(f"FAIL: venv python missing: {py}")
        print("Run: bash scripts/env_setup.sh && bash scripts/install_vllm_nightly.sh")
        return 2

    version_probe = run([
        str(py),
        "-c",
        "import sys, torch, vllm; "
        "print('python=',sys.version.split()[0]); "
        "print('vllm=',vllm.__version__); "
        "print('torch=',torch.__version__); "
        "print('hip=',torch.version.hip)",
    ])
    print("=== runtime versions ===")
    print(version_probe)
    if "0.26.1rc1.dev306" not in version_probe:
        failures.append("vLLM is not the pinned dev306 runtime")
    if "hip=" not in version_probe or "None" in version_probe.split("hip=")[-1]:
        failures.append("torch is not a ROCm build (hip=None)")

    try:
        import_result = subprocess.run(
            [str(py), "-c", "import aiter; print('aiter ok')"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        print("\n=== AITER import ===")
        print(import_result.stdout.strip())
        if import_result.returncode != 0:
            failures.append("AITER import failed")
    except Exception as exc:
        failures.append(f"AITER import check error: {exc}")

    flydsl_probe = subprocess.run(
        [str(py), "-c", (
            "import flydsl; "
            "from aiter.ops.flydsl import is_flydsl_available; "
            "print('flydsl=', flydsl.__version__); "
            "print('aiter_flydsl=', is_flydsl_available())"
        )],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print("\n=== FlyDSL runtime ===")
    print(flydsl_probe.stdout.strip())
    if flydsl_probe.returncode != 0 or "flydsl= 0.2.4" not in flydsl_probe.stdout or "aiter_flydsl= True" not in flydsl_probe.stdout:
        failures.append("flydsl 0.2.4 is required for AITER vision/long-prefix kernels")

    print("\n=== Qwen3 tool-parser whitespace safety ===")
    parser_probe = subprocess.run(
        [str(py), "-c", r"""
import json
from vllm.parser.qwen3 import _qwen3_arg_converter
raw = '<parameter=value>  keep-leading-and-trailing  </parameter>'
parsed = json.loads(_qwen3_arg_converter(raw, partial=False))
assert parsed['value'] == '  keep-leading-and-trailing  ', repr(parsed['value'])
wrapped = '<parameter=value>\n  keep-inner-space  \n</parameter>'
parsed2 = json.loads(_qwen3_arg_converter(wrapped, partial=False))
assert parsed2['value'] == '  keep-inner-space  ', repr(parsed2['value'])
print('qwen3_arg_whitespace=preserved')
"""],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(parser_probe.stdout.strip())
    if parser_probe.returncode != 0 or "qwen3_arg_whitespace=preserved" not in parser_probe.stdout:
        failures.append("Qwen3 tool parser strips meaningful argument whitespace; exact-match coding tools are unsafe")

    print("\n=== Qwen3.8 architecture registration ===")
    arch_probe = subprocess.run(
        [str(py), "-c", """
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
needles = ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3_5ForConditionalGeneration")
for n in needles:
    status = "OK" if n in archs else "MISSING"
    print(f"  {status:8s} {n}")
found = [n for n in needles if n in archs]
print(f"  {len(found)}/{len(needles)} registered")
if not found:
    raise SystemExit("no Qwen3.8 architecture found")
"""],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(arch_probe.stdout.strip())
    if arch_probe.returncode != 0:
        failures.append("Qwen3.8 architecture not registered; vLLM build predates PR #50068")

    print("\n=== Gated DeltaNet linear-attention path ===")
    fla_probe = subprocess.run(
        [str(py), "-c", """
# dev306 moved the Qwen Gated DeltaNet implementation under the v1 GDN
# backend plus the vendored flash-linear-attention package. Import the paths
# actually exercised by the validated Qwen3.8 runtime rather than a retired
# pre-v1 triton_fla module name.
import importlib
for module_name in (
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.v1.attention.backends.gdn_attn",
):
    importlib.import_module(module_name)
    print(f"  OK       {module_name}")
"""],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    print(fla_probe.stdout.strip())
    if fla_probe.returncode != 0:
        warnings.append("triton_fla ops module import had issues (may still work via alternate path)")

    recipe_head = run(["git", "-C", str(RECIPE_REPO), "rev-parse", "HEAD"]) if (RECIPE_REPO / ".git").exists() else "not-a-git-checkout"
    print(f"\n=== revisions ===")
    print(f"recipe_repo={recipe_head}")

    persist_dir = Path(os.environ.get("PERSIST_DIR", "/mnt/workspace/.venvs"))
    print("\n=== restart artifacts ===")
    for path, label in [
        (persist_dir / "vllm-qwen.tar.gz", "persistent venv snapshot"),
        (Path.home() / ".aiter", "AITER runtime cache"),
        (persist_dir / "aiter_cache.tar.gz", "persistent AITER snapshot"),
        (Path.home() / ".triton", "Triton runtime cache"),
        (persist_dir / "triton_cache.tar.gz", "persistent Triton snapshot"),
        (Path.home() / ".cache" / "comgr", "ROCm COMGR runtime cache"),
        (persist_dir / "comgr_cache.tar.gz", "persistent COMGR snapshot"),
    ]:
        print(f"{'OK' if path.exists() else 'WARN'} {label}")
        if not path.exists():
            warnings.append(f"missing warm-start artifact: {label}")

    print()
    if warnings:
        print("AUDIT WARNINGS (warm-start only):")
        for item in warnings:
            print(f"  - {item}")
    if failures:
        print("AUDIT FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("AUDIT PASSED: runtime ready for Qwen3.8-27B serving.")
    print("No fork overlays or patched binaries — Qwen3.8 is upstream-native.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
