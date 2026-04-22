"""Auto-generate a self-signed TLS certificate for the web server."""
from __future__ import annotations

import datetime
import ipaddress
import pathlib
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_CERT_DIR  = pathlib.Path.home() / ".local" / "share" / "xr-ai"
_CERT_FILE = _CERT_DIR / "web-server.crt"
_KEY_FILE  = _CERT_DIR / "web-server.key"


def ensure_self_signed_cert() -> tuple[str, str]:
    """Return (cert_path, key_path), generating them once and reusing thereafter."""
    _CERT_DIR.mkdir(parents=True, exist_ok=True)

    if _CERT_FILE.exists() and _KEY_FILE.exists():
        return str(_CERT_FILE), str(_KEY_FILE)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    hostname = socket.gethostname()
    san: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        local_ip = socket.gethostbyname(hostname)
        if local_ip != "127.0.0.1":
            san.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
    except OSError:
        pass

    now  = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
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
