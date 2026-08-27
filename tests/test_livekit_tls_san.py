# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the development root CA and signed server leaf."""
from __future__ import annotations

import datetime
import shutil
import ssl
import stat
import subprocess
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from device_io_hub.transport.livekit import _tls, _web_server
from device_io_hub.transport.livekit.config import LiveKitConnectorConfig
from loguru import logger

from _helpers_subprocess import pick_free_port


@pytest.fixture()
def cert_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(_tls, "_CERT_DIR", tmp_path)
    monkeypatch.setattr(_tls, "_ROOT_CERT_FILE", tmp_path / "root-ca.crt")
    monkeypatch.setattr(_tls, "_ROOT_KEY_FILE", tmp_path / "root-ca.key")
    monkeypatch.setattr(_tls, "_CERT_FILE", tmp_path / "web-server.crt")
    monkeypatch.setattr(_tls, "_KEY_FILE", tmp_path / "web-server.key")
    monkeypatch.setattr(_tls, "_local_ipv4_addrs", lambda: {"10.0.0.5"})
    monkeypatch.setattr(_tls.socket, "gethostname", lambda: "testhost")
    return tmp_path


def _cert(path: str | Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(Path(path).read_bytes())


def _mode(path: str | Path) -> int:
    return stat.S_IMODE(Path(path).stat().st_mode)


def test_normalize_san_canonicalizes_ips() -> None:
    assert _tls._normalize_san("203.0.113.7") == "203.0.113.7"
    assert _tls._normalize_san("2001:DB8::0:1") == "2001:db8::1"
    assert _tls._normalize_san("hub.example.com") == "hub.example.com"


def test_san_general_name_types() -> None:
    assert isinstance(_tls._san_general_name("203.0.113.7"), x509.IPAddress)
    assert isinstance(_tls._san_general_name("2001:db8::1"), x509.IPAddress)
    assert isinstance(_tls._san_general_name("hub.example.com"), x509.DNSName)
    assert _tls._san_general_name("hüb.example.com") is None


def test_fresh_chain_has_required_extensions_and_sans(cert_env: Path) -> None:
    leaf_path, leaf_key_path, root_path = _tls.ensure_development_certificates(
        ["203.0.113.7", "proxy.example.com"]
    )

    root = _cert(root_path)
    leaf = _cert(leaf_path)
    root_constraints = root.extensions.get_extension_for_class(x509.BasicConstraints).value
    root_usage = root.extensions.get_extension_for_class(x509.KeyUsage).value
    leaf_constraints = leaf.extensions.get_extension_for_class(x509.BasicConstraints).value
    leaf_usage = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
    leaf_eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value

    assert root_constraints.ca is True
    assert root_usage.key_cert_sign is True
    assert root_usage.crl_sign is True
    assert root.subject == root.issuer
    assert leaf_constraints.ca is False
    assert leaf_usage.key_cert_sign is False
    assert ExtendedKeyUsageOID.SERVER_AUTH in leaf_eku
    assert leaf.issuer == root.subject
    assert _tls._is_signed_by(leaf, root)
    assert {
        "localhost",
        "testhost",
        "127.0.0.1",
        "10.0.0.5",
        "203.0.113.7",
        "proxy.example.com",
    } <= _tls._cert_san_entries(leaf)
    assert Path(leaf_key_path).exists()


def test_openssl_verifies_leaf_against_root(cert_env: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    leaf_path, _, root_path = _tls.ensure_development_certificates()

    result = subprocess.run(
        ["openssl", "verify", "-CAfile", root_path, leaf_path],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{leaf_path}: OK"


def test_same_inputs_reuse_cached_root_and_leaf(cert_env: Path) -> None:
    leaf_path, _, root_path = _tls.ensure_development_certificates(["203.0.113.7"])
    serials = (_cert(leaf_path).serial_number, _cert(root_path).serial_number)

    leaf_path, _, root_path = _tls.ensure_development_certificates(["203.0.113.7"])

    assert (_cert(leaf_path).serial_number, _cert(root_path).serial_number) == serials


def test_removed_extra_does_not_regenerate_leaf(cert_env: Path) -> None:
    leaf_path, _, root_path = _tls.ensure_development_certificates(["203.0.113.7"])
    serials = (_cert(leaf_path).serial_number, _cert(root_path).serial_number)

    leaf_path, _, root_path = _tls.ensure_development_certificates([])

    assert (_cert(leaf_path).serial_number, _cert(root_path).serial_number) == serials


def test_new_san_rotates_only_leaf_and_keeps_old_entries(cert_env: Path) -> None:
    leaf_path, _, root_path = _tls.ensure_development_certificates(["203.0.113.7"])
    leaf_serial = _cert(leaf_path).serial_number
    root_serial = _cert(root_path).serial_number

    leaf_path, _, root_path = _tls.ensure_development_certificates(["198.51.100.9"])

    leaf = _cert(leaf_path)
    assert leaf.serial_number != leaf_serial
    assert _cert(root_path).serial_number == root_serial
    assert {"198.51.100.9", "203.0.113.7"} <= _tls._cert_san_entries(leaf)


def test_missing_local_ip_rotates_only_leaf(
    cert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf_path, _, root_path = _tls.ensure_development_certificates()
    leaf_serial = _cert(leaf_path).serial_number
    root_serial = _cert(root_path).serial_number

    monkeypatch.setattr(_tls, "_local_ipv4_addrs", lambda: {"10.0.0.5", "192.0.2.33"})
    leaf_path, _, root_path = _tls.ensure_development_certificates()

    leaf = _cert(leaf_path)
    assert leaf.serial_number != leaf_serial
    assert _cert(root_path).serial_number == root_serial
    assert {"192.0.2.33", "10.0.0.5"} <= _tls._cert_san_entries(leaf)


def test_legacy_ca_as_leaf_is_migrated_with_clear_log(cert_env: Path) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    legacy_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    legacy_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "XR AI Experiences")])
    legacy_cert = (
        x509.CertificateBuilder()
        .subject_name(legacy_name)
        .issuer_name(legacy_name)
        .public_key(legacy_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("legacy.example.com")]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(legacy_key, hashes.SHA256())
    )
    _tls._write_cert(_tls._CERT_FILE, legacy_cert)
    _tls._write_private_key(_tls._KEY_FILE, legacy_key)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        leaf_path, _, root_path = _tls.ensure_development_certificates()
    finally:
        logger.remove(sink)

    assert _tls._is_server_leaf(_cert(leaf_path))
    assert _tls._is_development_root(_cert(root_path))
    assert _tls._is_signed_by(_cert(leaf_path), _cert(root_path))
    assert {"legacy.example.com"} <= _tls._cert_san_entries(_cert(leaf_path))
    assert any("migrating legacy CA-as-server certificate" in message for message in messages)
    assert any("Install root-ca.crt" in message for message in messages)


def test_generated_file_permissions_are_restrictive(cert_env: Path) -> None:
    leaf_path, leaf_key_path, root_path = _tls.ensure_development_certificates()

    assert _mode(cert_env) == 0o700
    assert _mode(leaf_key_path) == 0o600
    assert _mode(_tls._ROOT_KEY_FILE) == 0o600
    assert _mode(leaf_path) == 0o644
    assert _mode(root_path) == 0o644


def test_public_root_reader_never_returns_private_key(cert_env: Path) -> None:
    _, _, root_path = _tls.ensure_development_certificates()
    root_pem = Path(root_path).read_bytes()
    root_key_pem = Path(_tls._ROOT_KEY_FILE).read_bytes()
    Path(root_path).write_bytes(root_pem + root_key_pem)
    exposed = _tls._read_public_root_ca(root_path)

    assert exposed == root_pem
    assert b"PRIVATE KEY" not in exposed


async def test_external_certificates_terminate_https_without_exposing_leaf(
    cert_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf_path, leaf_key_path, root_path = _tls.ensure_development_certificates()
    external_cert = cert_env / "external-server.crt"
    external_key = cert_env / "external-server.key"
    shutil.copyfile(leaf_path, external_cert)
    shutil.copyfile(leaf_key_path, external_key)
    monkeypatch.setattr(
        _web_server,
        "ensure_development_certificates",
        lambda *_: pytest.fail("external TLS must not auto-generate certificates"),
    )
    port = pick_free_port(8080)
    cfg = LiveKitConnectorConfig(
        api_key="devkey",
        api_secret="test-secret-at-least-32-bytes-long",
        web_server_host="127.0.0.1",
        web_server_port=port,
        cert_file=str(external_cert),
        key_file=str(external_key),
    )
    server = _web_server.WebServer(cfg)
    await server.start()
    try:
        context = ssl.create_default_context(cafile=root_path)
        async with httpx.AsyncClient(verify=context) as client:
            token_response = await client.get(f"https://localhost:{port}/token")
            cert_response = await client.get(f"https://localhost:{port}/cert")
    finally:
        await server.stop()

    assert token_response.status_code == 200
    assert cert_response.status_code == 404


def test_invalid_extras_are_skipped_not_fatal(cert_env: Path) -> None:
    leaf_path, _, _ = _tls.ensure_development_certificates(
        [None, 10.0, "  ", "hüb.example.com", " 203.0.113.7 "]  # type: ignore[list-item]
    )

    entries = _tls._cert_san_entries(_cert(leaf_path))
    assert "203.0.113.7" in entries
    assert "hüb.example.com" not in entries
    assert "2" not in entries


def test_unencodable_entries_do_not_regen_loop(cert_env: Path) -> None:
    leaf_path, _, _ = _tls.ensure_development_certificates(["hüb.example.com"])
    serial = _cert(leaf_path).serial_number

    leaf_path, _, _ = _tls.ensure_development_certificates(["hüb.example.com"])

    assert _cert(leaf_path).serial_number == serial


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
    from device_io_hub._config_loader import load_config

    cfg_file = tmp_path / "device_io_hub.yaml"
    cfg_file.write_text("api_key: devkey\napi_secret: secret\n" + yaml_value + "\n")
    monkeypatch.setattr("sys.argv", ["prog", "--config", str(cfg_file)])

    assert load_config().web_server_extra_sans == expected
