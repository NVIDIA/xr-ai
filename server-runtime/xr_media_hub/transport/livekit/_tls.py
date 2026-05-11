# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-generate a self-signed CA-marked TLS certificate for the web server."""
from __future__ import annotations

import datetime
import ipaddress
import pathlib
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from loguru import logger

_CERT_DIR  = pathlib.Path.home() / ".local" / "share" / "xr-ai"
_CERT_FILE = _CERT_DIR / "web-server.crt"
_KEY_FILE  = _CERT_DIR / "web-server.key"

# iOS only exposes the Full Trust toggle for certs with BasicConstraints
# CA:TRUE; mobile clients that installed an earlier non-CA / IP-stale
# profile have to re-import the new cert before they will trust it.
_REINSTALL_PROFILE_MSG = (
    "Devices that installed the previous profile must remove it from "
    "Settings → General → VPN & Device Management and reinstall."
)


def _load_cert(cert_path: pathlib.Path) -> x509.Certificate | None:
    try:
        return x509.load_pem_x509_certificate(cert_path.read_bytes())
    except (OSError, ValueError):
        return None


def _is_ca_cert(cert: x509.Certificate) -> bool:
    try:
        return bool(cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)
    except x509.ExtensionNotFound:
        return False


def _cert_ipv4_san(cert: x509.Certificate) -> set[str]:
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set()
    return {str(ip) for ip in san.get_values_for_type(x509.IPAddress)
            if isinstance(ip, ipaddress.IPv4Address)}


def _local_ipv4_addrs() -> set[str]:
    """Best-effort enumeration of routable IPv4 addresses on this host."""
    ips: set[str] = set()
    # UDP connect() resolves the egress interface without sending packets,
    # which yields the LAN IP even when gethostbyname() returns the
    # /etc/hosts loopback alias (127.0.1.1 on Ubuntu).
    for dest in (("8.8.8.8", 80), ("1.1.1.1", 80), ("169.254.169.254", 80)):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.settimeout(0.2)
                s.connect(dest)
                ip = s.getsockname()[0]
                if not ip.startswith("127."):
                    ips.add(ip)
            finally:
                s.close()
        except OSError as exc:
            logger.debug("local-IP probe to {} failed: {}", dest, exc)
    try:
        _, _, host_ips = socket.gethostbyname_ex(socket.gethostname())
        ips.update(ip for ip in host_ips if not ip.startswith("127."))
    except (socket.gaierror, socket.herror, OSError) as exc:
        logger.debug("gethostbyname_ex({}) failed: {}", socket.gethostname(), exc)
    return ips


def ensure_self_signed_cert() -> tuple[str, str]:
    """Return (cert_path, key_path), generating them once and reusing thereafter."""
    _CERT_DIR.mkdir(parents=True, exist_ok=True)

    detected_ips = _local_ipv4_addrs()
    reasons: list[str] = []

    if _CERT_FILE.exists() and _KEY_FILE.exists():
        cached = _load_cert(_CERT_FILE)
        if cached is not None:
            if not _is_ca_cert(cached):
                reasons.append(
                    "cached cert is not a CA cert — regenerating so iOS Full "
                    "Trust toggle appears"
                )
            if missing := detected_ips - _cert_ipv4_san(cached):
                reasons.append(
                    f"cached cert SAN is missing local IP(s) {sorted(missing)} "
                    "— regenerating so clients connecting via those addresses "
                    "can validate the host"
                )
            if not reasons:
                return str(_CERT_FILE), str(_KEY_FILE)

    for reason in reasons:
        logger.info("TLS: {}. {}", reason, _REINSTALL_PROFILE_MSG)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    hostname = socket.gethostname()
    san: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    for ip in sorted(detected_ips):
        try:
            san.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
        except (ipaddress.AddressValueError, ValueError):
            continue

    public_key = key.public_key()
    now  = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
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
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .sign(key, hashes.SHA256())
    )

    _KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    _KEY_FILE.chmod(0o600)
    _CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return str(_CERT_FILE), str(_KEY_FILE)
