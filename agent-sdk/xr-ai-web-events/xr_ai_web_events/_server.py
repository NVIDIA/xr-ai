# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP server for the live web-events page."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from ._store import _EventStore

_STATIC = Path(__file__).with_name("static")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_CSP = (
    "default-src 'self'; base-uri 'none'; connect-src 'self'; "
    "frame-ancestors 'none'; img-src 'self' data:; object-src 'none'; "
    "script-src 'self'; style-src 'self'"
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})


def _normalized_host(value: str) -> str:
    return value.strip().lower().rstrip(".").strip("[]")


def _is_ip_literal(value: str) -> bool:
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def _is_loopback(value: str) -> bool:
    if value in _LOOPBACK_HOSTS:
        return True
    try:
        return ip_address(value).is_loopback
    except ValueError:
        return False


class _WebEventsServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        store: _EventStore,
        title: str,
    ) -> None:
        self.store = store
        self.viewer_title = title
        super().__init__(address, _WebEventsHandler)
        self.listener_host = _normalized_host(address[0])
        self.bound_host = _normalized_host(str(self.server_address[0]))

    def _host_is_allowed(self, requested_host: str) -> bool:
        """Reject DNS names that were not selected as listener identities."""

        host = _normalized_host(requested_host)
        if not host:
            return False
        if self.listener_host in _WILDCARD_HOSTS:
            # A wildcard listener has no single external identity. Permit
            # literal addresses for direct private-network access, but reject
            # arbitrary DNS names that can be rebound to this listener.
            return host == "localhost" or _is_ip_literal(host)
        if _is_loopback(self.listener_host):
            return _is_loopback(host)
        return host in {self.listener_host, self.bound_host}


class _WebEventsHandler(BaseHTTPRequestHandler):
    server_version = "XR-AI-Web-Events"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
        if not self._valid_host():
            self.send_error(HTTPStatus.BAD_REQUEST, "unrecognized Host header")
            return
        request = urlparse(self.path)
        if request.path == "/healthz":
            self._json({"status": "ok"})
            return
        if request.path == "/api/events":
            self._events(request.query)
            return
        asset = _ASSETS.get(request.path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name, content_type = asset
        payload = (_STATIC / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _valid_host(self) -> bool:
        values = self.headers.get_all("Host", failobj=[])
        if len(values) != 1:
            return False
        raw_host = values[0].strip()
        try:
            parsed = urlparse(f"//{raw_host}")
            # Reading .port validates a numeric, in-range port when supplied.
            _ = parsed.port
        except ValueError:
            return False
        if (
            parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return False
        server = cast(_WebEventsServer, self.server)
        return server._host_is_allowed(parsed.hostname)

    def _events(self, query: str) -> None:
        values = parse_qs(query)
        raw_after = values.get("after", ["0"])[0]
        try:
            after = int(raw_after)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "after must be an integer")
            return
        if after < 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "after must not be negative")
            return
        server = cast(_WebEventsServer, self.server)
        response = server.store.after(after)
        response["title"] = server.viewer_title
        self._json(response)

    def _json(self, value: dict[str, Any]) -> None:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
