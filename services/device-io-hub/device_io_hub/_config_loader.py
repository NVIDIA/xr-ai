# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loads LiveKitConnectorConfig from a YAML file.

Looks for device_io_hub.yaml in the current working directory by default.
Pass --config <path> on the command line to use a different file.

Relative paths in the YAML (e.g. web_client_dir: ../../client-samples/web) are
resolved relative to the YAML file's own directory, so the file is portable
regardless of where the process is started from.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import secrets
from pathlib import Path

import yaml
from loguru import logger

from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from device_io_hub.video import VideoRecorderConfig

from ._deployment import apply_network_overrides

DEFAULT_CONFIG_NAME = "device_io_hub.yaml"
LIVEKIT_API_KEY_ENV = "LIVEKIT_API_KEY"
LIVEKIT_API_SECRET_ENV = "LIVEKIT_API_SECRET"

# Clears web_client_dir when set; /token, /cert, /rtc stay up.
NO_WEB_CLIENT_ENV = "DEVICE_IO_HUB_NO_WEB_CLIENT"


def _web_client_disabled() -> bool:
    return os.environ.get(NO_WEB_CLIENT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str, base: Path) -> str:
    p = Path(value)
    return str((base / p).resolve()) if not p.is_absolute() else value


def _apply_env_credentials(data: dict) -> None:
    if LIVEKIT_API_KEY_ENV in os.environ or LIVEKIT_API_SECRET_ENV in os.environ:
        data["api_key"] = os.environ.get(LIVEKIT_API_KEY_ENV, "").strip()
        data["api_secret"] = os.environ.get(LIVEKIT_API_SECRET_ENV, "").strip()
        return
    data.setdefault("api_key", "")
    data.setdefault("api_secret", "")
    for key in ("api_key", "api_secret"):
        if isinstance(data[key], str):
            data[key] = data[key].strip()
    if data["api_key"] == "" and data["api_secret"] == "":
        data["api_key"] = "xr-ai-" + secrets.token_hex(16)
        data["api_secret"] = "xr-ai-" + secrets.token_hex(32)


def _warn_unknown_keys(data: dict, valid: set[str], path: Path, section: str) -> None:
    unknown = [repr(key) for key in data if key not in valid]
    if unknown:
        logger.warning(
            "{}: section {}: ignoring unknown keys: {}; accepted keys: {}",
            path, section, ", ".join(unknown), ", ".join(sorted(valid)),
        )


def load_config() -> LiveKitConnectorConfig:
    """
    Parse --config from argv, load the YAML file if it exists, and return
    a fully populated LiveKitConnectorConfig.

    If either environment credential is set, require a complete, non-blank
    environment pair instead of YAML credentials. Otherwise, use the YAML pair
    or generate a fresh pair when both YAML values are missing or blank.
    A partial pair remains an error. With no --config flag, a missing
    device_io_hub.yaml in CWD uses configuration defaults. An empty YAML
    document also uses defaults. Unknown top-level and video_recording keys
    are ignored with a warning.

    ``DEVICE_IO_HUB_NETWORK`` supplies a YAML mapping of host network overrides
    after sample YAML, before validation and relative asset-path resolution.
    Accepted fields are the five listener ports, ``lk_use_external_ip``,
    ``lk_skip_external_ip_validation``, and ``web_server_extra_sans``.
    ``cert_file`` and ``key_file`` accept only empty strings to select managed
    development certificates. Other fields and invalid value types are rejected.
    Overriding ``lk_port_ws`` also updates the loopback ``lk_internal_url``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path.cwd() / DEFAULT_CONFIG_NAME

    if not config_path.exists():
        if args.config:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info("No {} found; using defaults. Supply --config <path> to load a configuration", config_path)
        data: dict = {"enable_web_server": False, "web_client_dir": ""}
        apply_network_overrides(data)
        _apply_env_credentials(data)
        return LiveKitConnectorConfig(**data)

    logger.info("Loading config from {}", config_path)
    base = config_path.parent

    try:
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{config_path}: configuration must be UTF-8 encoded") from exc
    if data is None:
        logger.info("{} is empty; using defaults", config_path)
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top-level configuration must be a YAML mapping")
    apply_network_overrides(data)

    valid = {f.name for f in dataclasses.fields(LiveKitConnectorConfig)}
    _warn_unknown_keys(data, valid, config_path, "top-level")
    recording = data.get("video_recording")
    if recording is not None:
        if not isinstance(recording, dict):
            raise ValueError(f"{config_path}: video_recording must be a YAML mapping")
        recording_keys = {f.name for f in dataclasses.fields(VideoRecorderConfig)} | {"enabled"}
        _warn_unknown_keys(recording, recording_keys, config_path, "video_recording")
    _apply_env_credentials(data)

    if _web_client_disabled():
        data["web_client_dir"] = ""

    # Resolve any relative path fields relative to the YAML file's directory.
    for key in ("web_client_dir", "cert_file", "key_file"):
        if data.get(key):
            data[key] = _resolve_path(data[key], base)

    if "web_server_extra_sans" in data:
        sans = data["web_server_extra_sans"]
        if sans is None:
            data["web_server_extra_sans"] = []
        elif isinstance(sans, str):
            data["web_server_extra_sans"] = [sans]
        elif not isinstance(sans, list):
            logger.warning("web_server_extra_sans must be a list of strings; ignoring {!r}", sans)
            data["web_server_extra_sans"] = []

    filtered = {k: v for k, v in data.items() if k in valid}

    return LiveKitConnectorConfig(**filtered)
