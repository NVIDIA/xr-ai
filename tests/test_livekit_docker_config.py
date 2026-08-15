# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generated LiveKit server configuration."""
from __future__ import annotations

import stat
from pathlib import Path

import yaml
from xr_media_hub.transport.livekit._docker import (
    _render_livekit_config,
    _write_livekit_config,
)
from xr_media_hub.transport.livekit.config import LiveKitConnectorConfig


def _render_config(cfg: LiveKitConnectorConfig) -> dict:
    return yaml.safe_load(_render_livekit_config(cfg))


def test_livekit_nat_options_are_disabled_by_default() -> None:
    rtc = _render_config(LiveKitConnectorConfig())["rtc"]

    assert rtc["use_external_ip"] is False
    assert rtc["skip_external_ip_validation"] is False


def test_livekit_nat_options_can_be_enabled() -> None:
    cfg = LiveKitConnectorConfig(
        lk_use_external_ip=True,
        lk_skip_external_ip_validation=True,
    )
    rtc = _render_config(cfg)["rtc"]

    assert rtc["use_external_ip"] is True
    assert rtc["skip_external_ip_validation"] is True


def test_livekit_config_file_is_owner_only(tmp_path: Path) -> None:
    cfg_path = tmp_path / "livekit.yaml"

    _write_livekit_config(cfg_path, LiveKitConnectorConfig())

    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600
