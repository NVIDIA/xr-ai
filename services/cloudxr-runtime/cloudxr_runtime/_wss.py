# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TLS WebSocket proxy in front of the CloudXR service, for WebRTC clients.

Browsers only open WebXR streaming sessions to a ``wss://`` origin, while the
native service listens on plain ``ws://localhost:49100``. The proxy terminates
TLS with a self-signed certificate and pipes each WebSocket to the service.
Plain HTTPS requests get a page confirming the certificate was accepted, which
is how a user trusts the certificate before connecting.

Ported from ``isaacteleop.cloudxr.wss`` 1.3.131 (Apache-2.0) without its
out-of-band teleop control hub, USB tethering, and static client hosting.
"""
from __future__ import annotations

import asyncio
import errno
import functools
import logging
import shutil
import ssl
import subprocess
from pathlib import Path

import websockets
from loguru import logger
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response, parse_headers

PROXY_PORT = 48322
BACKEND_URI = "ws://localhost:49100"

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Expose-Headers": "*",
}
# Hop-by-hop and handshake headers the backend connection sets itself.
_UNFORWARDED_HEADERS = {
    "host",
    "upgrade",
    "connection",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
}
_CERT_ACCEPTED_PAGE = (
    b"<!doctype html><html><head><meta charset=utf-8>"
    b"<style>body{font-family:system-ui,sans-serif;display:flex;"
    b"align-items:center;justify-content:center;height:100vh;margin:0;"
    b"background:#f5f5f5;color:#222}div{text-align:center}"
    b"h1{font-weight:600;font-size:1.5rem;margin-bottom:.5rem}"
    b"p{color:#555;font-size:1rem}</style></head>"
    b"<body><div><h1>Certificate Accepted</h1>"
    b"<p>You can close this tab and return to the web client.</p>"
    b"</div></body></html>"
)
# A WebSocket keeps its connection open indefinitely and frames may be large;
# pings and size limits are left to the endpoints.
_CONNECTION_OPTIONS = {
    "compression": None,
    "max_size": None,
    "ping_interval": None,
    "ping_timeout": None,
    "close_timeout": 10,
}


def _allow_cors_preflight() -> None:
    """Let HTTP OPTIONS requests reach ``process_request``.

    websockets >= 14 rejects any method but GET while parsing the request. A
    browser sends an OPTIONS preflight before a cross-origin request, so the
    parser is wrapped to surface it as an ordinary request instead.
    """
    original = Request.parse.__func__

    @classmethod
    def parse(cls, read_line):
        try:
            return (yield from original(cls, read_line))
        except ValueError as exc:
            if "got OPTIONS" not in str(exc):
                raise
            headers = yield from parse_headers(read_line)
            return cls("/__cors_preflight__", headers)

    Request.parse = parse


_allow_cors_preflight()


def ensure_certificate(cert_dir: Path) -> tuple[Path, Path]:
    """Return ``(certificate, key)``, generating a self-signed pair if absent."""
    cert, key = cert_dir / "server.crt", cert_dir / "server.key"
    if cert.exists() != key.exists():
        missing = key if cert.exists() else cert
        raise RuntimeError(
            f"{cert_dir} has half a TLS certificate pair (missing {missing.name}); "
            "restore it or remove the other file"
        )
    if cert.exists():
        return cert, key
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl is not on PATH; it is needed to create the proxy's TLS certificate")
    cert_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "365", "-nodes", "-subj", "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    logger.info("Generated a self-signed TLS certificate in {}", cert_dir)
    return cert, key


async def _answer_http(_connection: ServerConnection, request: Request) -> Response | None:
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    content_type = "text/plain" if request.headers.get("Access-Control-Request-Method") else "text/html; charset=utf-8"
    body = b"OK" if content_type == "text/plain" else _CERT_ACCEPTED_PAGE
    return Response(200, "OK", Headers({"Content-Type": content_type, **_CORS_HEADERS}), body)


def _add_cors_headers(_connection: ServerConnection, _request: Request, response: Response) -> None:
    response.headers.update(_CORS_HEADERS)


async def _pipe(source, destination) -> None:
    try:
        async for message in source:
            await destination.send(message)
    except websockets.ConnectionClosed as exc:
        # Pass the peer's close code on so each side sees why the stream ended.
        try:
            if exc.rcvd is not None:
                await destination.close(exc.rcvd.code, exc.rcvd.reason)
            else:
                await destination.close()
        except websockets.ConnectionClosed:
            pass


async def _proxy(client: ServerConnection, backend_uri: str) -> None:
    path = client.request.path or "/"
    headers = {
        name: value
        for name, value in client.request.headers.raw_items()
        if name.lower() not in _UNFORWARDED_HEADERS
    }
    subprotocols = client.request.headers.get_all("Sec-WebSocket-Protocol")
    try:
        backend = await ws_connect(
            f"{backend_uri}{path}",
            additional_headers=headers,
            subprotocols=subprotocols or None,
            **_CONNECTION_OPTIONS,
        )
    except (OSError, websockets.InvalidHandshake) as exc:
        # The service is down or refused this path; the client sees its
        # socket close.
        logger.warning("CloudXR service did not accept {}{}: {}", backend_uri, path, exc)
        return
    logger.info("Proxying {} -> {}{}", client.remote_address, backend_uri, path)
    try:
        await _first_to_finish(_pipe(client, backend), _pipe(backend, client))
    finally:
        await backend.close()
        logger.info("Connection closed: {}", path)


async def _first_to_finish(*coroutines) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run(
    cert_dir: Path,
    stop: asyncio.Future[None],
    *,
    port: int = PROXY_PORT,
    backend_uri: str = BACKEND_URI,
) -> None:
    """Serve the proxy on *port*, forwarding to *backend_uri*, until *stop* completes."""
    # websockets logs every handshake at INFO; keep only its problems.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    cert, key = ensure_certificate(cert_dir)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert, keyfile=key)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        server = ws_serve(
            functools.partial(_proxy, backend_uri=backend_uri),
            host="",
            port=port,
            ssl=context,
            process_request=_answer_http,
            process_response=_add_cors_headers,
            **_CONNECTION_OPTIONS,
        )
        async with server:
            logger.info("WSS proxy listening on port {}", port)
            await stop
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise RuntimeError(f"WSS proxy port {port} is already in use") from exc
        raise
