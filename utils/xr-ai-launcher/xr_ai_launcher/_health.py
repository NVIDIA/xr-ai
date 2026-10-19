# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the optional loopback health listener; imported only inside xr_ai_launcher."""

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from subprocess import Popen
from urllib.parse import urlsplit

log = logging.getLogger(__name__)


@contextmanager
def stack_health(
    processes: Mapping[str, Popen], port: int | None, unchecked_reuse: tuple[str, ...] = (),
) -> Iterator[threading.Event]:
    """Serve live process health during one foreground stack's lifetime."""
    initialized = threading.Event()
    if port is None:
        yield initialized
        return

    class Handler(BaseHTTPRequestHandler):
        timeout = 2.0

        def handle(self) -> None:
            try:
                super().handle()
            except ConnectionError:
                # ignored: a polling client can close its socket before receiving a response.
                return

        def do_GET(self) -> None:
            hosts = self.headers.get_all("Host", [])
            try:
                host = urlsplit(f"//{hosts[0]}") if len(hosts) == 1 else None
                valid_host = (
                    host is not None and host.hostname in {"localhost", "127.0.0.1", "::1"}
                    and host.username is None and host.password is None
                    and not (host.path or host.query or host.fragment)
                )
                if host is not None:
                    _ = host.port
            except ValueError:
                valid_host = False
            if not valid_host:
                self.send_error(400, "unrecognized Host header")
                return
            if urlsplit(self.path).path.rstrip("/") != "/healthz":
                self.send_error(404)
                return
            # Startup adds children while requests run on other threads.
            started = initialized.is_set()
            states = {name: "running" if proc.poll() is None else "exited"
                      for name, proc in tuple(processes.items())}
            healthy = started and initialized.is_set() and all(state == "running" for state in states.values())
            body = json.dumps({"status": "ready" if healthy else "not_ready", "processes": states,
                               "unchecked_reuse": unchecked_reuse}).encode()
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self) -> None:
            self.do_GET()

        def log_message(self, _format: str, *args: object) -> None:
            # Health polling is intentionally quiet; no request payload is logged.
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise OSError(f"health_port={port}: {exc}") from exc
    with server:
        log.info("Stack process-health endpoint: http://127.0.0.1:%s/healthz", server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        try:
            yield initialized
        finally:
            initialized.clear()
            server.shutdown()
            thread.join()
