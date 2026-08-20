# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LiveKit server configuration and docker run command."""
from __future__ import annotations

import asyncio
import re
import stat
from pathlib import Path

import pytest
import yaml
from device_io_hub._errors import StartupError
from device_io_hub.transport.livekit import _docker as _docker_mod
from device_io_hub.transport.livekit._docker import (
    LiveKitDocker,
    _render_livekit_config,
    _write_livekit_config,
)
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig


def _render_config(cfg: LiveKitConnectorConfig) -> dict:
    return yaml.safe_load(_render_livekit_config(cfg))


def test_livekit_nat_options_are_disabled_by_default() -> None:
    cfg = LiveKitConnectorConfig(api_key="test-key", api_secret="test-secret")
    rtc = _render_config(cfg)["rtc"]

    assert rtc["use_external_ip"] is False
    assert rtc["skip_external_ip_validation"] is False


def test_livekit_nat_options_can_be_enabled() -> None:
    cfg = LiveKitConnectorConfig(
        api_key="test-key",
        api_secret="test-secret",
        lk_use_external_ip=True,
        lk_skip_external_ip_validation=True,
    )
    rtc = _render_config(cfg)["rtc"]

    assert rtc["use_external_ip"] is True
    assert rtc["skip_external_ip_validation"] is True


def test_livekit_config_file_is_owner_only(tmp_path: Path) -> None:
    cfg_path = tmp_path / "livekit.yaml"

    cfg = LiveKitConnectorConfig(api_key="test-key", api_secret="test-secret")
    _write_livekit_config(cfg_path, cfg)

    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600


def test_livekit_image_is_pinned() -> None:
    assert re.fullmatch(
        r"livekit/livekit-server:v\d+\.\d+\.\d+", _docker_mod._LIVEKIT_IMAGE
    )


def test_docker_run_uses_pinned_image(monkeypatch: pytest.MonkeyPatch) -> None:
    argv: list[str] = []

    async def fake_exec(*cmd: str, **_kwargs: object) -> None:
        argv.extend(cmd)
        raise FileNotFoundError

    monkeypatch.setattr(_docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(_docker_mod.subprocess, "run", lambda *a, **k: None)

    cfg = LiveKitConnectorConfig(api_key="test-key", api_secret="test-secret")
    with pytest.raises(StartupError):
        asyncio.run(LiveKitDocker(cfg).start())

    image_at = argv.index(_docker_mod._LIVEKIT_IMAGE)
    # Everything after the image is passed to the container, not to docker.
    assert argv[image_at + 1] == "--config"
    assert not any(arg.endswith(":latest") for arg in argv)
