# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import jwt
import pytest
import yaml
from device_io_hub._config_loader import load_config
from device_io_hub.transport.livekit._docker import _render_livekit_config
from device_io_hub.transport.livekit._token import make_client_token


def _reset_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["device_io_hub"])


@pytest.mark.parametrize(
    ("api_key", "api_secret"),
    [
        ("env-key", None),
        (None, "env-secret"),
        ("   ", "env-secret"),
        ("env-key", "   "),
        ("", None),
        (None, ""),
        ("   ", None),
        (None, "   "),
        ("", ""),
        ("   ", "   "),
    ],
)
@pytest.mark.parametrize(
    "yaml_text",
    [
        None,
        'api_key: ""\napi_secret: ""\n',
        "api_key: yaml-key\napi_secret: yaml-secret\n",
    ],
)
def test_load_config_rejects_partial_livekit_credentials(
    tmp_path,
    monkeypatch,
    api_key,
    api_secret,
    yaml_text,
) -> None:
    if yaml_text is not None:
        (tmp_path / "device_io_hub.yaml").write_text(yaml_text, encoding="utf-8")
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    if api_key is not None:
        monkeypatch.setenv("LIVEKIT_API_KEY", api_key)
    if api_secret is not None:
        monkeypatch.setenv("LIVEKIT_API_SECRET", api_secret)

    with pytest.raises(ValueError, match="LiveKit credentials are required") as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "device_io_hub.yaml" in message
    assert "LIVEKIT_API_KEY" in message
    assert "LIVEKIT_API_SECRET" in message


def test_load_config_accepts_livekit_credentials_from_environment(
    tmp_path, monkeypatch
) -> None:
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    cfg = load_config()

    assert cfg.api_key == "env-key"
    assert cfg.api_secret == "env-secret"
    assert cfg.enable_web_server is False


def test_load_config_strips_livekit_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "device_io_hub.yaml"
    config_path.write_text(
        "api_key: ' yaml-key  '\napi_secret: ' yaml-secret  '\n",
        encoding="utf-8",
    )
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", " env-key\n")
    monkeypatch.setenv("LIVEKIT_API_SECRET", " env-secret\n")

    cfg = load_config()

    assert cfg.api_key == "env-key"
    assert cfg.api_secret == "env-secret"


def test_load_config_environment_overrides_yaml_credentials(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "device_io_hub.yaml"
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


def test_load_config_accepts_yaml_credentials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "device_io_hub.yaml"
    config_path.write_text(
        "api_key: yaml-key\napi_secret: yaml-secret\n",
        encoding="utf-8",
    )
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    cfg = load_config()

    assert cfg.api_key == "yaml-key"
    assert cfg.api_secret == "yaml-secret"


@pytest.mark.parametrize(
    "yaml_text",
    [
        None,
        "room_name: yaml-room\n",
        'api_key: ""\napi_secret: ""\n',
        'api_key: "  "\napi_secret: "  "\n',
    ],
)
def test_load_config_generates_fresh_credentials(tmp_path, monkeypatch, yaml_text) -> None:
    if yaml_text is not None:
        (tmp_path / "device_io_hub.yaml").write_text(yaml_text, encoding="utf-8")
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    first = load_config()
    second = load_config()

    assert re.fullmatch(r"xr-ai-[0-9a-f]{32}", first.api_key)
    assert re.fullmatch(r"xr-ai-[0-9a-f]{64}", first.api_secret)
    assert first.api_key != second.api_key
    assert first.api_secret != second.api_secret
    assert "LIVEKIT_API_KEY" not in os.environ
    assert "LIVEKIT_API_SECRET" not in os.environ

    # The server and token endpoint use the same generated pair.
    server = yaml.safe_load(_render_livekit_config(first))
    claims = jwt.decode(
        make_client_token(first, identity="alice"),
        server["keys"][first.api_key],
        algorithms=["HS256"],
        issuer=first.api_key,
    )
    assert claims["sub"] == "alice"
    assert claims["video"]["room"] == first.room_name


@pytest.mark.parametrize(
    "yaml_text",
    [
        "api_key: yaml-key\n",
        "api_secret: yaml-secret\n",
        'api_key: null\napi_secret: ""\n',
    ],
)
def test_load_config_rejects_partial_or_invalid_yaml_credentials(
    tmp_path, monkeypatch, yaml_text
) -> None:
    (tmp_path / "device_io_hub.yaml").write_text(yaml_text, encoding="utf-8")
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with pytest.raises(ValueError, match="LiveKit credentials are required"):
        load_config()


@pytest.mark.parametrize(
    "config_path",
    [Path("services/device-io-hub/device_io_hub.yaml")]
    + sorted(Path(__file__).resolve().parents[1].glob("agent-samples/*/yaml/device_io_hub.yaml")),
)
def test_shipped_hub_config_runs_without_credential_setup(monkeypatch, config_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["api_key"] == data["api_secret"] == ""
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    config = load_config()

    assert config.api_key
    assert config.api_secret
