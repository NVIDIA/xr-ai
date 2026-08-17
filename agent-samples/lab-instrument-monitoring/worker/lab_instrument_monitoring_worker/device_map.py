# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve visual marker identities to configured lab-device names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from xr_ai_tools.marker_tracking import MarkerType


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    marker_type: MarkerType
    marker_id: str
    device_name: str


class DeviceMap:
    """Immutable marker-family and identifier lookup."""

    def __init__(self, names: dict[tuple[MarkerType, str], str]) -> None:
        self._names = dict(names)

    def resolve(self, marker_type: MarkerType, marker_id: str) -> DeviceIdentity:
        fallback = marker_id if marker_type is MarkerType.QR_CODE else f"ArUco {marker_id}"
        return DeviceIdentity(
            marker_type=marker_type,
            marker_id=marker_id,
            device_name=self._names.get((marker_type, marker_id), fallback),
        )


def load_device_map(path: Path) -> DeviceMap:
    """Load marker names from a YAML mapping grouped by marker family."""

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("devices", {}), dict):
        raise ValueError(f"device map must contain a 'devices' mapping: {path}")

    names: dict[tuple[MarkerType, str], str] = {}
    for raw_type, raw_devices in raw.get("devices", {}).items():
        marker_type = MarkerType(str(raw_type))
        if not isinstance(raw_devices, dict):
            raise ValueError(f"device map family {raw_type!r} must be a mapping: {path}")
        for raw_id, raw_name in raw_devices.items():
            marker_id = str(raw_id).strip()
            device_name = str(raw_name).strip()
            if not marker_id or not device_name:
                raise ValueError(f"device map IDs and names must be non-empty: {path}")
            names[(marker_type, marker_id)] = device_name
    return DeviceMap(names)


__all__ = ["DeviceIdentity", "DeviceMap", "load_device_map"]
