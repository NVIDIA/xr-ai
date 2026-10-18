# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own host network overrides; imported only by DeviceIOHub's config loader.

``DEVICE_IO_HUB_NETWORK_CONFIG`` names a YAML file applied after sample YAML and before config
validation. It accepts the five listener ports, LiveKit external-IP booleans,
and ``web_server_extra_sans``. Empty ``cert_file`` and ``key_file`` overrides
select XR AI's managed development certificates. Other fields are rejected.
Relative asset paths continue to resolve against the original YAML directory.
Overriding ``lk_port_ws`` also sets the hub's loopback ``lk_internal_url``.
"""

import os
from pathlib import Path

import yaml

PORTS = {"lk_port_ws", "lk_port_tcp", "lk_port_udp", "web_server_port", "token_server_port"}
BOOLEANS = {"lk_use_external_ip", "lk_skip_external_ip_validation"}


def apply_network_overrides(data: dict) -> None:
    """Apply the optional host network YAML file without changing sample data on failure."""
    filename = os.environ.get("DEVICE_IO_HUB_NETWORK_CONFIG")
    if not filename:
        return
    path = Path(filename)
    with path.open(encoding="utf-8") as stream:
        try:
            overrides = yaml.safe_load(stream)
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise ValueError(f"DEVICE_IO_HUB_NETWORK_CONFIG {path}: must contain valid UTF-8 YAML") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"DEVICE_IO_HUB_NETWORK_CONFIG {path}: must be a YAML mapping")
    for key, value in overrides.items():
        if key in PORTS:
            valid = type(value) is int and 1 <= value <= 65535
        elif key in BOOLEANS:
            valid = type(value) is bool
        elif key == "web_server_extra_sans":
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        elif key in {"cert_file", "key_file"}:
            valid = value == ""
        else:
            valid = False
        if not valid:
            raise ValueError(f"DEVICE_IO_HUB_NETWORK_CONFIG {path}: unsupported field or value: {key}")
    data.update(overrides)
    if "lk_port_ws" in overrides:
        data["lk_internal_url"] = f"ws://127.0.0.1:{overrides['lk_port_ws']}"
