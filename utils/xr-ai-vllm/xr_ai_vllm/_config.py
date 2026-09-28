# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config, identity, environment, and GPU helpers for model services.

Stdlib-only by contract (see package docstring). ``yaml`` is imported
function-locally in :func:`load_config` so ``import xr_ai_vllm`` stays
dependency-free for the orchestrator ``--stop`` path, which declares no
pyyaml.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from ._docker import credential_digest

log = logging.getLogger(__name__)

_TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
_FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}
_CONFIG_IDENTITY_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "HF_TOKEN",
    "NGC_API_KEY",
    "HF_XET_HIGH_PERFORMANCE",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_ENABLE_HF_TRANSFER",
)
_CREDENTIAL_IDENTITY_ENV = {"HF_TOKEN", "NGC_API_KEY"}
_DESCRIBE_LAUNCH_OPTION = "--describe-launch"
_LOCAL_CONFIG_DIGEST_ENV = "XR_AI_SERVICE_CONFIG_DIGEST"
_LOCAL_SERVICE_IDENTITY_ENV = "XR_AI_SERVICE_IDENTITY"


def parse_config_bool(value: object, key: str) -> bool:
    """Parse a YAML boolean without treating every non-empty string as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_STRINGS:
            return True
        if normalized in _FALSE_BOOL_STRINGS:
            return False
    accepted = sorted(_TRUE_BOOL_STRINGS | _FALSE_BOOL_STRINGS)
    raise ValueError(
        f"{key} must be a boolean or one of {accepted} (got {value!r})"
    )


def resolve_model_cache(cfg: dict, yaml_dir: Path, *, default: str) -> Path:
    """Resolve ``model_cache`` and create it outside launch-description mode."""
    raw = cfg.get("model_cache", default)
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    if not describe_launch_requested():
        p.mkdir(parents=True, exist_ok=True)
    return p


def describe_launch_requested(argv: list[str] | None = None) -> bool:
    """Return whether the wrapper should report its effective launch identity."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(_DESCRIBE_LAUNCH_OPTION, action="store_true")
    ns, _ = parser.parse_known_args(argv)
    return ns.describe_launch


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


def prepare_requested(argv: list[str] | None = None) -> bool:
    """Return whether the service wrapper was invoked in artifact-only mode."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare", action="store_true")
    ns, _ = parser.parse_known_args(argv)
    return ns.prepare


def service_config_digest(
    path: Path | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Digest a service config path, contents, and launch environment identity.

    Credential values use the package's PBKDF2 digest before joining the
    fingerprint. *env* supplies the effective child environment for an
    ownership probe. The canonical path is part of the digest because relative
    artifact paths resolve from the config's directory. Returns ``None`` when no
    readable config path was supplied.
    """
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        resolved = path.resolve()
        digest.update(str(resolved).encode())
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
    except OSError:
        return None
    identity_env = os.environ if env is None else env
    for key in _CONFIG_IDENTITY_ENV:
        value = identity_env.get(key)
        if value is not None:
            digest.update(b"\0")
            digest.update(key.encode())
            digest.update(b"=")
            if key in _CREDENTIAL_IDENTITY_ENV:
                digest.update(credential_digest(value).encode())
            else:
                digest.update(value.encode())
    return digest.hexdigest()[:20]


def local_service_identity_env(
    config_path: Path | None,
    service_identity: str,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment carrying strict local-service identity.

    The config path is resolved while building the digest, so wrappers may
    keep the caller's relative ``--config`` argument without making ownership
    depend on the child's working directory. The source mapping is copied and
    never mutated.
    """
    child_env = dict(os.environ if env is None else env)
    digest = service_config_digest(config_path, env=child_env)
    if digest is not None:
        child_env[_LOCAL_CONFIG_DIGEST_ENV] = digest
    else:
        child_env.pop(_LOCAL_CONFIG_DIGEST_ENV, None)
    child_env[_LOCAL_SERVICE_IDENTITY_ENV] = service_identity
    return child_env


def source_config_digest(argv: list[str] | None = None) -> str | None:
    """Return the path-, content-, and environment-bound config digest.

    Local ownership compares this source identity strictly. Docker exposes it
    as diagnostic metadata and uses ``--describe-launch`` for compatibility.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    ns, _ = parser.parse_known_args(argv)
    if ns.config is None:
        return None
    return service_config_digest(ns.config)


def setup_hf_env(cfg: dict, model_cache: Path) -> str | None:
    """Apply the shared HuggingFace / CUDA env block.

    Sets ``CUDA_VISIBLE_DEVICES`` (when configured), ``HF_TOKEN`` (when
    provided), ``HF_XET_HIGH_PERFORMANCE``, ``HF_HOME``, and
    ``TRANSFORMERS_CACHE``.

    A non-empty env value for ``HF_TOKEN``, ``HF_HOME``, or
    ``TRANSFORMERS_CACHE`` wins over the YAML value, per
    docs/source/getting_started/credentials.md. ``TRANSFORMERS_CACHE`` mirrors
    ``HF_HOME`` for Transformers <4.36 and libraries that have not adopted
    ``HF_HOME``.

    Returns the resolved ``cuda_visible_devices`` string (or ``None``) so
    callers that run GPU detection can confirm the device filter is applied.
    """
    cuda_devices = cfg.get("cuda_visible_devices")
    if cuda_devices is not None:
        cuda_devices = str(cuda_devices)
        # Pip mode reads it from the env; docker mode forwards it through the
        # NVIDIA runtime.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    hf_token = os.environ.get("HF_TOKEN") or cfg.get("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
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


def _gpu_is_dgx_spark() -> bool:
    """Return whether the selected GPU is the GB10 in a DGX Spark."""
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception:
        return False

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    for row in rows:
        index, uuid, name = (part.strip() for part in row.split(",", 2))
        if selected and selected not in {index, uuid}:
            continue
        return "GB10" in name.upper() or "DGX SPARK" in name.upper()
    return False
