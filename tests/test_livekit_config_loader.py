# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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
from loguru import logger


def _reset_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["device_io_hub"])


@pytest.mark.parametrize("with_yaml", [False, True])
def test_deployment_network_overrides_preserve_sample_yaml(tmp_path, monkeypatch, with_yaml):
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "device_io_hub.yaml"
    original = "web_server_port: 8080\nweb_client_dir: web\ncert_file: old.crt\nkey_file: old.key\n"
    if with_yaml:
        path.write_text(original)
    monkeypatch.setenv("DEVICE_IO_HUB_NETWORK", json.dumps({
        "web_server_port": 9443,
        "lk_port_ws": 9880,
        "lk_use_external_ip": True,
        "web_server_extra_sans": ["203.0.113.1"],
        "cert_file": "",
        "key_file": "",
    }))
    cfg = load_config()
    assert cfg.web_server_port == 9443
    assert cfg.lk_port_ws == 9880
    assert cfg.lk_use_external_ip is True
    assert cfg.web_server_extra_sans == ["203.0.113.1"]
    assert cfg.cert_file == cfg.key_file == ""
    if with_yaml:
        assert cfg.web_client_dir == str(tmp_path / "web")
        assert path.read_text() == original
    else:
        assert not path.exists()


@pytest.mark.parametrize("overrides", [
    [], {"web_server_port": 0}, {"web_server_port": True},
    {"lk_use_external_ip": "true"}, {"web_server_extra_sans": "host"},
    {"cert_file": "/private/key"}, {"api_secret": "secret"}, {"unknown": 1},
])
def test_deployment_rejects_invalid_network_overrides(tmp_path, monkeypatch, overrides):
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEVICE_IO_HUB_NETWORK", json.dumps(overrides))
    with pytest.raises(ValueError, match="DEVICE_IO_HUB_NETWORK"):
        load_config()


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


@pytest.fixture
def config_logs():
    messages = []
    sink = logger.add(messages.append, level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.mark.parametrize(
    ("contents", "section", "unknown", "accepted", "secret_value"),
    [
        ("web_server_prt: port-secret\n", "top-level", "web_server_prt", "web_server_port", "port-secret"),
        ("api_secert: credential-secret\n", "top-level", "api_secert", "api_secret", "credential-secret"),
        ("video_recording:\n  enabld: enabled-secret\n", "video_recording", "enabld", "enabled", "enabled-secret"),
        ("video_recording:\n  bit_rate: bitrate-secret\n", "video_recording", "bit_rate", "bitrate", "bitrate-secret"),
    ],
)
def test_unknown_config_keys_warn_with_source_and_accepted_keys(
    tmp_path, monkeypatch, config_logs, contents, section, unknown, accepted, secret_value
) -> None:
    config_path = tmp_path / "custom_hub.yaml"
    config_path.write_text(contents + "web_server_port: 8123\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    cfg = load_config()

    warnings = [message.record["message"] for message in config_logs if message.record["level"].name == "WARNING"]
    assert len(warnings) == 1
    assert str(config_path) in warnings[0]
    assert f"section {section}" in warnings[0]
    assert unknown in warnings[0]
    assert accepted in warnings[0]
    assert secret_value not in warnings[0]
    assert cfg.web_server_port == 8123


def test_unknown_keys_are_grouped_by_section(tmp_path, monkeypatch, config_logs) -> None:
    config_path = tmp_path / "custom_hub.yaml"
    config_path.write_text(
        "web_server_prt: port-secret\napi_secert: credential-secret\n"
        "video_recording:\n  room_name: nested-secret\n  enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    load_config()

    warnings = [message.record["message"] for message in config_logs if message.record["level"].name == "WARNING"]
    assert len(warnings) == 2
    top_level, recording = warnings
    assert "section top-level" in top_level
    assert "web_server_prt" in top_level and "api_secert" in top_level
    assert "section video_recording" in recording
    assert "room_name" in recording and "accepted keys: " in recording and "enabled" in recording
    for warning in warnings:
        assert str(config_path) in warning
        assert all(secret not in warning for secret in ("port-secret", "credential-secret", "nested-secret"))


def test_known_config_keys_do_not_warn(tmp_path, monkeypatch, config_logs) -> None:
    config_path = tmp_path / "device_io_hub.yaml"
    config_path.write_text(
        "room_name: configured\nvideo_recording:\n  enabled: false\n  bitrate: 1000\n",
        encoding="utf-8",
    )
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    assert load_config().room_name == "configured"
    assert not any(message.record["level"].name == "WARNING" for message in config_logs)


def test_credential_typo_warns_when_hub_generates_credentials(tmp_path, monkeypatch, config_logs) -> None:
    config_path = (tmp_path / "device_io_hub.yaml").resolve()
    config_path.write_text("api_secert: intended-secret\n", encoding="utf-8")
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    config = load_config()

    assert re.fullmatch(r"xr-ai-[0-9a-f]{32}", config.api_key)
    assert re.fullmatch(r"xr-ai-[0-9a-f]{64}", config.api_secret)
    messages = [message.record["message"] for message in config_logs]
    assert any(str(config_path) in message and "api_secert" in message for message in messages)
    assert all("intended-secret" not in message and config.api_secret not in message for message in messages)


@pytest.mark.parametrize("contents", [None, "", "# Empty configuration\n"])
def test_config_default_fallback_is_visible(tmp_path, monkeypatch, config_logs, contents) -> None:
    config_path = (tmp_path / "device_io_hub.yaml").resolve()
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")
    _reset_argv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    load_config()

    messages = [message.record["message"] for message in config_logs]
    assert any(str(config_path) in message and "using defaults" in message for message in messages)
    if contents is None:
        assert any("--config <path>" in message for message in messages)


@pytest.mark.parametrize("contents", ["false\n", "[]\n", "room_name\n"])
def test_hub_config_requires_a_mapping(tmp_path, monkeypatch, contents) -> None:
    config_path = tmp_path / "device_io_hub.yaml"
    config_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])

    with pytest.raises(ValueError, match="must be a YAML mapping") as error:
        load_config()

    assert str(config_path) in str(error.value)


@pytest.mark.parametrize("contents", ["true", "false", "[a]", "[]", "0", "''", "recording"])
def test_recording_config_requires_a_mapping(tmp_path, monkeypatch, contents) -> None:
    config_path = tmp_path / "custom_hub.yaml"
    config_path.write_text(f"video_recording: {contents}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])

    with pytest.raises(ValueError, match="video_recording must be a YAML mapping") as error:
        load_config()

    assert str(config_path) in str(error.value)


@pytest.mark.parametrize("contents", ["null", "{}"])
def test_null_or_empty_recording_config_is_disabled(tmp_path, monkeypatch, contents) -> None:
    config_path = tmp_path / "custom_hub.yaml"
    config_path.write_text(f"video_recording: {contents}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")

    assert load_config().video_recording in (None, {})


def test_hub_config_utf8_error_names_the_file(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom_hub.yaml"
    config_path.write_bytes(b"room_name: \xff\n")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config_path)])

    with pytest.raises(ValueError, match="must be UTF-8 encoded") as error:
        load_config()

    assert str(config_path) in str(error.value)
    assert isinstance(error.value.__cause__, UnicodeDecodeError)
