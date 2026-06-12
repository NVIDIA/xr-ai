# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config/env/GPU helpers for the vLLM-backed service wrappers.

Stdlib-only by contract (see package docstring). ``yaml`` is imported
function-locally in :func:`load_config` so ``import xr_ai_vllm`` stays
dependency-free for the orchestrator ``--stop`` path, which declares no
pyyaml.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_model_cache(cfg: dict, yaml_dir: Path, *, default: str) -> Path:
    """Resolve ``model_cache`` (relative to the YAML dir) and ensure it exists."""
    raw = cfg.get("model_cache", default)
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(argv: list[str] | None = None) -> tuple[dict, Path, Path | None]:
    """Parse ``--config``/``--ready-file`` and load the YAML config.

    Reconfigures stdout/stderr to line-buffered so logs flush under the
    launcher's piped stdout. Returns ``(cfg, yaml_dir, ready_file)``;
    ``yaml_dir`` is the config's directory (cwd when no config is given),
    used as the base for relative paths like ``model_cache``.
    """
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args(argv)

    cfg: dict = {}
    yaml_dir = Path.cwd()
    if ns.config and ns.config.exists():
        import yaml

        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    return cfg, yaml_dir, ns.ready_file


def setup_hf_env(cfg: dict, model_cache: Path) -> str | None:
    """Apply the shared HuggingFace / CUDA env block.

    Sets ``CUDA_VISIBLE_DEVICES`` (when configured), ``HF_TOKEN`` (when
    provided), ``HF_HUB_ENABLE_HF_TRANSFER``, ``HF_HOME``, and
    ``TRANSFORMERS_CACHE``.

    Both ``HF_HOME`` and ``TRANSFORMERS_CACHE`` are set via ``setdefault``, so
    an externally-set value wins over the YAML default. This intentionally
    differs from the pre-consolidation per-server code where the LLM wrappers
    used an unconditional assignment for ``HF_HOME``; callers that need the
    YAML value to take priority must unset the env var before calling.
    ``TRANSFORMERS_CACHE`` mirrors ``HF_HOME`` for Transformers <4.36 and
    third-party libraries that have not yet adopted the ``HF_HOME`` superset.

    Returns the resolved ``cuda_visible_devices`` string (or ``None``) so
    callers that run GPU detection can confirm the device filter is applied.
    """
    cuda_devices = cfg.get("cuda_visible_devices")
    if cuda_devices is not None:
        cuda_devices = str(cuda_devices)
        # Pip mode reads it from the env; docker mode forwards via --gpus.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache))

    return cuda_devices


def gpu_compute_major() -> int:
    """Return the GPU's compute-capability major version, or 0 if unknown.

    Reads ``CUDA_VISIBLE_DEVICES`` from the env, so set the device filter
    before calling. Logs a warning on failure and falls back to 0.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if out:
            return int(out[0].split(".")[0])
    except Exception as exc:
        log.warning(
            "nvidia-smi compute-cap query failed (%s) — "
            "defaulting to pre-Blackwell model variant", exc,
        )
    return 0
