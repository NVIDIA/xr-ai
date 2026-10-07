# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and validation for the Clef server."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "Cloudflare/clef-flash"
DEFAULT_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


@dataclass(frozen=True)
class ServerConfig:
    model_name: str
    model_revision: str
    model_path: Path | None
    model_cache: Path
    host: str
    port: int
    device: str
    dtype: str
    max_length: int
    max_body_bytes: int


def identity(config: ServerConfig) -> dict[str, str]:
    """Return the exact, non-secret runtime identity used for safe reuse."""
    configuration = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "model_path": str(config.model_path) if config.model_path is not None else None,
        "model_cache": str(config.model_cache),
        "host": config.host,
        "port": config.port,
        "device": config.device,
        "dtype": config.dtype,
        "max_length": config.max_length,
        "max_body_bytes": config.max_body_bytes,
    }
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "ready",
        "service": "clef-server",
        "model": config.model_name,
        "model_revision": config.model_revision,
        "configuration_fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def load_config(path: Path) -> ServerConfig:
    """Read a YAML config and resolve its filesystem paths against that file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("config must contain a YAML mapping")

    base = path.resolve().parent

    def resolve_path(value: str) -> Path:
        result = Path(value).expanduser()
        return (base / result).resolve() if not result.is_absolute() else result.resolve()

    model_name = _string(raw, "model_name", DEFAULT_MODEL)
    revision = _string(raw, "model_revision", DEFAULT_REVISION)
    model_path_value = raw.get("model_path")
    if model_path_value is not None and (not isinstance(model_path_value, str) or not model_path_value.strip()):
        raise ValueError("model_path must be a non-empty path when supplied")
    model_path = resolve_path(model_path_value) if model_path_value else None

    return ServerConfig(
        model_name=model_name,
        model_revision=revision,
        model_path=model_path,
        model_cache=resolve_path(_string(raw, "model_cache", "../../models")),
        host=_string(raw, "host", "127.0.0.1"),
        port=_integer(raw, "port", 8120, minimum=1, maximum=65535),
        device=_string(raw, "device", "cuda:0"),
        dtype=_enum(raw, "dtype", "bfloat16", {"bfloat16", "float16", "float32"}),
        max_length=_integer(raw, "max_length", 4096, minimum=256, maximum=32768),
        max_body_bytes=_integer(raw, "max_body_bytes", 1_048_576, minimum=1024, maximum=16_777_216),
    )


def _string(values: dict[str, Any], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _enum(values: dict[str, Any], key: str, default: str, allowed: set[str]) -> str:
    value = _string(values, key, default)
    if value not in allowed:
        raise ValueError(f"{key} must be one of {', '.join(sorted(allowed))}")
    return value


def _integer(values: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value
