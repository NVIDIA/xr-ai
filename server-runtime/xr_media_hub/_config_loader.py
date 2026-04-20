"""
Loads LiveKitConnectorConfig from a YAML file.

Looks for xr_media_hub.yaml in the current working directory by default.
Pass --config <path> on the command line to use a different file.

Relative paths in the YAML (e.g. web_client_dir: ../client-samples/web) are
resolved relative to the YAML file's own directory, so the file is portable
regardless of where the process is started from.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from xr_media_hub.transport.livekit.config import LiveKitConnectorConfig

log = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "xr_media_hub.yaml"


def _resolve_path(value: str, base: Path) -> str:
    p = Path(value)
    return str((base / p).resolve()) if not p.is_absolute() else value


def load_config() -> LiveKitConnectorConfig:
    """
    Parse --config from argv, load the YAML file if it exists, and return
    a fully populated LiveKitConnectorConfig.

    If no --config flag is given and no xr_media_hub.yaml exists in CWD,
    returns default config (web server disabled, no client dir).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path.cwd() / DEFAULT_CONFIG_NAME

    if not config_path.exists():
        if args.config:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        log.debug("No %s found — using defaults", DEFAULT_CONFIG_NAME)
        return LiveKitConnectorConfig(enable_web_server=False, web_client_dir="")

    log.info("Loading config from %s", config_path)
    base = config_path.parent

    with config_path.open() as f:
        data: dict = yaml.safe_load(f) or {}

    # Resolve any relative path fields relative to the YAML file's directory.
    for key in ("web_client_dir", "cert_file", "key_file"):
        if data.get(key):
            data[key] = _resolve_path(data[key], base)

    # Filter to only fields that exist on the dataclass.
    import dataclasses
    valid = {f.name for f in dataclasses.fields(LiveKitConnectorConfig)}
    filtered = {k: v for k, v in data.items() if k in valid}

    return LiveKitConnectorConfig(**filtered)
