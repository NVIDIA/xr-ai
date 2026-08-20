# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for self-signed cert SAN generation and regeneration."""
from __future__ import annotations

from pathlib import Path

import pytest
from cryptography import x509
from xr_media_hub.transport.livekit import _tls


@pytest.fixture()
def cert_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(_tls, "_CERT_DIR", tmp_path)
    monkeypatch.setattr(_tls, "_CERT_FILE", tmp_path / "web-server.crt")
    monkeypatch.setattr(_tls, "_KEY_FILE", tmp_path / "web-server.key")
    monkeypatch.setattr(_tls, "_local_ipv4_addrs", lambda: {"10.0.0.5"})
    monkeypatch.setattr(_tls.socket, "gethostname", lambda: "testhost")
    return tmp_path


def _cert(path: str) -> x509.Certificate:
    return x509.load_pem_x509_certificate(Path(path).read_bytes())


def test_normalize_san_canonicalizes_ips() -> None:
    assert _tls._normalize_san("203.0.113.7") == "203.0.113.7"
    assert _tls._normalize_san("2001:DB8::0:1") == "2001:db8::1"
    assert _tls._normalize_san("hub.example.com") == "hub.example.com"


def test_san_general_name_types() -> None:
    assert isinstance(_tls._san_general_name("203.0.113.7"), x509.IPAddress)
    assert isinstance(_tls._san_general_name("2001:db8::1"), x509.IPAddress)
    assert isinstance(_tls._san_general_name("hub.example.com"), x509.DNSName)
    assert _tls._san_general_name("hüb.example.com") is None


def test_fresh_cert_covers_locals_and_extras(cert_env: Path) -> None:
    cert_path, key_path = _tls.ensure_self_signed_cert(["203.0.113.7", "proxy.example.com"])

    entries = _tls._cert_san_entries(_cert(cert_path))
    assert {"localhost", "testhost", "127.0.0.1", "10.0.0.5",
            "203.0.113.7", "proxy.example.com"} <= entries
    assert Path(key_path).exists()


def test_same_inputs_reuse_cached_cert(cert_env: Path) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert(["203.0.113.7"])
    serial = _cert(cert_path).serial_number

    cert_path, _ = _tls.ensure_self_signed_cert(["203.0.113.7"])

    assert _cert(cert_path).serial_number == serial


def test_removed_extra_does_not_regenerate(cert_env: Path) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert(["203.0.113.7"])
    serial = _cert(cert_path).serial_number

    cert_path, _ = _tls.ensure_self_signed_cert([])

    assert _cert(cert_path).serial_number == serial


def test_new_extra_regenerates_and_keeps_old_entries(cert_env: Path) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert(["203.0.113.7"])
    serial = _cert(cert_path).serial_number

    cert_path, _ = _tls.ensure_self_signed_cert(["198.51.100.9"])

    cert = _cert(cert_path)
    assert cert.serial_number != serial
    assert {"198.51.100.9", "203.0.113.7"} <= _tls._cert_san_entries(cert)


def test_missing_local_ip_regenerates(cert_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert()
    serial = _cert(cert_path).serial_number

    monkeypatch.setattr(_tls, "_local_ipv4_addrs", lambda: {"10.0.0.5", "192.0.2.33"})
    cert_path, _ = _tls.ensure_self_signed_cert()

    cert = _cert(cert_path)
    assert cert.serial_number != serial
    assert {"192.0.2.33", "10.0.0.5"} <= _tls._cert_san_entries(cert)


def test_invalid_extras_are_skipped_not_fatal(cert_env: Path) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert(
        [None, 10.0, "  ", "hüb.example.com", " 203.0.113.7 "]  # type: ignore[list-item]
    )

    entries = _tls._cert_san_entries(_cert(cert_path))
    assert "203.0.113.7" in entries
    assert "hüb.example.com" not in entries
    # A prior bug class: a scalar iterated per character.
    assert "2" not in entries


def test_unencodable_entries_do_not_regen_loop(cert_env: Path) -> None:
    cert_path, _ = _tls.ensure_self_signed_cert(["hüb.example.com"])
    serial = _cert(cert_path).serial_number

    cert_path, _ = _tls.ensure_self_signed_cert(["hüb.example.com"])

    assert _cert(cert_path).serial_number == serial


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        ("web_server_extra_sans: hub.example.com", ["hub.example.com"]),
        ("web_server_extra_sans:", []),
        ("web_server_extra_sans: true", []),
        ("web_server_extra_sans:\n  - 203.0.113.7", ["203.0.113.7"]),
    ],
)
def test_loader_coerces_extra_sans(
    yaml_value: str, expected: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xr_media_hub._config_loader import load_config

    cfg_file = tmp_path / "xr_media_hub.yaml"
    cfg_file.write_text(yaml_value + "\n")
    monkeypatch.setattr("sys.argv", ["prog", "--config", str(cfg_file)])

    assert load_config().web_server_extra_sans == expected
