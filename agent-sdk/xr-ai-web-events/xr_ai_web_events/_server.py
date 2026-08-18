# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP server for the live web-events page."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class _WebEventsHandler(BaseHTTPRequestHandler):
    server_version = "XR-AI-Web-Events"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
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
