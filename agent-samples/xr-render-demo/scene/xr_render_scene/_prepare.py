# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and cache the LOVR executable used by the scene service."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import urllib.request
from hashlib import sha256
from pathlib import Path

import yaml
from xr_ai_launcher import report_prepare_status

_VERSION = "0.18.0"
_ASSET = f"lovr-v{_VERSION}-x86_64.AppImage"
_RELEASE = f"https://github.com/bjornbytes/lovr/releases/download/v{_VERSION}"
_SHA256 = "730dd56062eb5efcc8dc29737ec5389ba4372f3bac74a5ca0204299e0cbb63b1"


def _cache_dir(config_path: Path, raw: dict) -> Path:
    configured = raw.get("lovr_cache_dir")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (config_path.parent / path).resolve()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "xr-ai" / "lovr" / _VERSION


def _read_config(config_path: Path) -> dict:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"failed to read LOVR configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"LOVR configuration in {config_path} must be a mapping")
    return raw


def _configured_lovr(raw: dict) -> Path | None:
    configured = os.environ.get("LOVR_BIN") or raw.get("lovr_bin")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"LOVR: detected {path}; required an executable file. Set LOVR_BIN to your executable LOVR build."
        )
    return path.resolve()


def _digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as artifact:
        while block := artifact.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def _cached_lovr(config_path: Path, raw: dict) -> Path | None:
    path = _cache_dir(config_path, raw) / _ASSET
    try:
        if path.is_file() and _digest(path) == _SHA256:
            if not os.access(path, os.X_OK):
                path.chmod(path.stat().st_mode | 0o111)
            return path
    except OSError:
        return None
    return None


def resolve_lovr(config_path: Path) -> Path:
    """Return a configured or prepared LOVR executable without downloading."""
    raw = _read_config(config_path)
    if configured := _configured_lovr(raw):
        return configured
    if cached := _cached_lovr(config_path, raw):
        return cached
    raise RuntimeError(
        f"LOVR v{_VERSION} is not prepared. Run `xr_render_scene --config "
        f"{config_path} --prepare`, or set LOVR_BIN to an executable LOVR build."
    )


def prepare_lovr(config_path: Path, *, report_cached: bool = True) -> Path:
    """Return an executable LOVR path, downloading a supported build if needed."""
    raw = _read_config(config_path)
    if path := _configured_lovr(raw):
        if report_cached:
            report_prepare_status("LOVR", "cached", path.stat().st_size)
        return path
    if (sys.platform, platform.machine().lower()) != ("linux", "x86_64"):
        raise RuntimeError(
            f"LOVR: detected {sys.platform}/{platform.machine()}; automatic "
            f"download requires linux/x86_64. Build LOVR v{_VERSION} and set "
            "LOVR_BIN to its executable."
        )
    if path := _cached_lovr(config_path, raw):
        if report_cached:
            report_prepare_status("LOVR", "cached", path.stat().st_size)
        return path
    cache = _cache_dir(config_path, raw)
    path = cache / _ASSET
    cache.mkdir(parents=True, exist_ok=True)
    report_prepare_status("LOVR", "downloading", None)
    with tempfile.TemporaryDirectory(dir=cache, prefix="lovr-") as staging:
        partial_path = Path(staging) / _ASSET
        with partial_path.open("wb") as partial:
            with urllib.request.urlopen(f"{_RELEASE}/{_ASSET}", timeout=60) as response:
                while block := response.read(1024 * 1024):
                    partial.write(block)
        digest = _digest(partial_path)
        if digest != _SHA256:
            raise RuntimeError(f"LOVR download checksum mismatch: expected {_SHA256}, got {digest}")
        partial_path.chmod(0o755)
        partial_path.replace(path)
    report_prepare_status("LOVR", "cached", path.stat().st_size)
    return path
