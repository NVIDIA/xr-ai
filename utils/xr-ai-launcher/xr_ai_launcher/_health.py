# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the optional loopback health listener; imported only inside xr_ai_launcher."""

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from subprocess import Popen


@contextmanager
def stack_health(processes: Mapping[str, Popen], port: int | None) -> Iterator[threading.Event]:
    """Serve live process health during one foreground stack's lifetime."""
    initialized = threading.Event()
    if port is None:
        yield initialized
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            # The process map is complete before initialized is set. Every
            # request checks the actual child handles, not a stored PID or log.
            healthy = initialized.is_set() and all(proc.poll() is None for proc in processes.values())
            body = b"ready\n" if healthy else b"not ready\n"
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            # Health polling is intentionally quiet; no request payload is logged.
            return

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        try:
            yield initialized
        finally:
            initialized.clear()
            server.shutdown()
            thread.join()
