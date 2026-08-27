# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a stable development root CA and a signed TLS server leaf."""
from __future__ import annotations

import datetime
import ipaddress
import pathlib
import socket
from collections.abc import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from loguru import logger

_CERT_DIR = pathlib.Path.home() / ".local" / "share" / "xr-ai"
_ROOT_CERT_FILE = _CERT_DIR / "root-ca.crt"
_ROOT_KEY_FILE = _CERT_DIR / "root-ca.key"
_CERT_FILE = _CERT_DIR / "web-server.crt"
_KEY_FILE = _CERT_DIR / "web-server.key"

_ROOT_COMMON_NAME = "XR AI Development Root CA"
_REINSTALL_PROFILE_MSG = (
    "Install root-ca.crt on each client once; the root remains stable when "
    "server addresses change."
)


def _load_cert(cert_path: pathlib.Path) -> x509.Certificate | None:
    try:
        return x509.load_pem_x509_certificate(cert_path.read_bytes())
    except (OSError, ValueError):
        return None


def _load_private_key(key_path: pathlib.Path) -> rsa.RSAPrivateKey | None:
    try:
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError):
        return None
    return key if isinstance(key, rsa.RSAPrivateKey) else None


def _is_ca_cert(cert: x509.Certificate) -> bool:
    try:
        return bool(cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)
    except x509.ExtensionNotFound:
        return False


def _is_development_root(cert: x509.Certificate) -> bool:
    if not _is_ca_cert(cert) or cert.subject != cert.issuer:
        return False
    try:
        usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        return False
    return usage.key_cert_sign and usage.crl_sign


def _is_server_leaf(cert: x509.Certificate) -> bool:
    if _is_ca_cert(cert):
        return False
    try:
        usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
        extended_usage = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        return False
    return (
        usage.digital_signature
        and not usage.key_cert_sign
        and ExtendedKeyUsageOID.SERVER_AUTH in extended_usage
    )


def _cert_san_entries(cert: x509.Certificate) -> set[str]:
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set()
    return (
        {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        | set(san.get_values_for_type(x509.DNSName))
    )


def _normalize_san(entry: str) -> str:
    try:
        return str(ipaddress.ip_address(entry))
    except ValueError:
        return entry


def _san_general_name(entry: str) -> x509.GeneralName | None:
    try:
        return x509.IPAddress(ipaddress.ip_address(entry))
    except ValueError:
        pass
    try:
        return x509.DNSName(entry)
    except (ValueError, TypeError):
        return None


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


def _wanted_san_entries(extra_sans: Sequence[str]) -> set[str]:
    """Return the normalized, encodable SAN entries the leaf must cover."""
    candidates = {"localhost", socket.gethostname(), "127.0.0.1"} | _local_ipv4_addrs()
    for entry in extra_sans:
        if isinstance(entry, str) and entry.strip():
            candidates.add(_normalize_san(entry.strip()))
        else:
            logger.warning("TLS: ignoring invalid web_server_extra_sans entry {!r}", entry)
    wanted: set[str] = set()
    for entry in candidates:
        if _san_general_name(entry) is None:
            logger.warning("TLS: cannot encode SAN entry {!r}, skipping", entry)
        else:
            wanted.add(entry)
    return wanted


def _key_matches_cert(key: rsa.RSAPrivateKey, cert: x509.Certificate) -> bool:
    return key.public_key().public_numbers() == cert.public_key().public_numbers()


def _is_current(
    cert: x509.Certificate,
    now: datetime.datetime,
    renew_before: datetime.timedelta = datetime.timedelta(),
) -> bool:
    return cert.not_valid_before_utc <= now < cert.not_valid_after_utc - renew_before


def _is_signed_by(cert: x509.Certificate, issuer: x509.Certificate) -> bool:
    try:
        cert.verify_directly_issued_by(issuer)
    except ValueError:
        return False
    return True


def _write_private_key(path: pathlib.Path, key: rsa.RSAPrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)


def _write_cert(path: pathlib.Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    path.chmod(0o644)


def _read_public_root_ca(cert_path: str) -> bytes:
    """Return one public root certificate, never other PEM material."""
    cert = _load_cert(pathlib.Path(cert_path))
    if (
        cert is None
        or not _is_ca_cert(cert)
        or cert.subject != cert.issuer
        or not _is_signed_by(cert, cert)
    ):
        raise ValueError(f"{cert_path} is not a self-signed root CA certificate")
    return cert.public_bytes(serialization.Encoding.PEM)


def _generate_root(now: datetime.datetime) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _ROOT_COMMON_NAME)])
    public_key = key.public_key()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(public_key), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    return cert, key


def _generate_leaf(
    root: x509.Certificate,
    root_key: rsa.RSAPrivateKey,
    san_entries: set[str],
    now: datetime.datetime,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key()
    san = [_san_general_name(entry) for entry in sorted(san_entries)]
    cert = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())])
        )
        .issuer_name(root.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=397))
        .add_extension(
            x509.SubjectAlternativeName([name for name in san if name is not None]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )
    return cert, key


def ensure_development_certificates(
    extra_sans: Sequence[str] = (),
) -> tuple[str, str, str]:
    """Return ``(leaf_cert, leaf_key, root_ca)`` and create them as needed.

    The root CA is independent of hostnames and IP addresses. Adding a SAN
    therefore rotates only the server leaf, so clients do not need to reinstall
    the root.
    """
    _CERT_DIR.mkdir(parents=True, exist_ok=True)
    _CERT_DIR.chmod(0o700)
    now = datetime.datetime.now(datetime.timezone.utc)

    root = _load_cert(_ROOT_CERT_FILE)
    root_key = _load_private_key(_ROOT_KEY_FILE)
    root_valid = (
        root is not None
        and root_key is not None
        and _is_development_root(root)
        and _key_matches_cert(root_key, root)
        and _is_signed_by(root, root)
        and _is_current(root, now, datetime.timedelta(days=30))
    )
    if not root_valid:
        if _ROOT_CERT_FILE.exists() or _ROOT_KEY_FILE.exists():
            logger.warning("TLS: cached development root is incomplete or invalid; replacing it")
        root, root_key = _generate_root(now)
        _write_private_key(_ROOT_KEY_FILE, root_key)
        _write_cert(_ROOT_CERT_FILE, root)
        logger.info("TLS: generated stable development root CA at {}", _ROOT_CERT_FILE)

    wanted = _wanted_san_entries(extra_sans)
    cached = _load_cert(_CERT_FILE)
    cached_key = _load_private_key(_KEY_FILE)
    cached_entries = _cert_san_entries(cached) if cached is not None else set()
    reasons: list[str] = []

    if cached is not None and _is_ca_cert(cached):
        logger.info(
            "TLS: migrating legacy CA-as-server certificate at {} to a signed CA:FALSE "
            "server leaf. {}",
            _CERT_FILE,
            _REINSTALL_PROFILE_MSG,
        )
        reasons.append("legacy CA-as-server certificate")
    elif cached is None or cached_key is None:
        reasons.append("server certificate or private key is missing or unreadable")
    elif not _is_server_leaf(cached):
        reasons.append("cached server certificate has invalid TLS extensions")
    elif not _is_current(cached, now, datetime.timedelta(days=30)):
        reasons.append("cached server certificate is expired or nearing expiration")
    elif not _key_matches_cert(cached_key, cached):
        reasons.append("cached server certificate and private key do not match")
    elif not _is_signed_by(cached, root):
        reasons.append("cached server certificate is not signed by the development root")

    if missing := wanted - cached_entries:
        reasons.append(f"cached server certificate SAN is missing {sorted(missing)}")

    if reasons:
        # Keep old SANs when a local interface temporarily disappears. This
        # prevents churn while still allowing explicitly added addresses.
        leaf, leaf_key = _generate_leaf(root, root_key, wanted | cached_entries, now)
        _write_private_key(_KEY_FILE, leaf_key)
        _write_cert(_CERT_FILE, leaf)
        logger.info("TLS: generated server leaf at {} ({})", _CERT_FILE, "; ".join(reasons))

    # Repair permissive modes on cached files as well as setting them on new
    # files. Certificates are public; private keys are owner-readable only.
    _ROOT_KEY_FILE.chmod(0o600)
    _KEY_FILE.chmod(0o600)
    _ROOT_CERT_FILE.chmod(0o644)
    _CERT_FILE.chmod(0o644)

    return str(_CERT_FILE), str(_KEY_FILE), str(_ROOT_CERT_FILE)
