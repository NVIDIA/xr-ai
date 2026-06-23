# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from xr_media_hub._config_loader import load_config


def _reset_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["xr_media_hub"])


def test_load_config_requires_livekit_credentials(tmp_path, monkeypatch) -> None:
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with pytest.raises(ValueError, match="LiveKit credentials are required"):
        load_config()


def test_load_config_accepts_livekit_credentials_from_environment(tmp_path, monkeypatch) -> None:
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    cfg = load_config()

    assert cfg.api_key == "env-key"
    assert cfg.api_secret == "env-secret"
    assert cfg.enable_web_server is False


def test_load_config_environment_overrides_yaml_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "xr_media_hub.yaml"
    config_path.write_text(
        "api_key: yaml-key\n"
        "api_secret: yaml-secret\n"
        "room_name: yaml-room\n",
        encoding="utf-8",
    )
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    cfg = load_config()

    assert cfg.api_key == "env-key"
    assert cfg.api_secret == "env-secret"
    assert cfg.room_name == "yaml-room"
