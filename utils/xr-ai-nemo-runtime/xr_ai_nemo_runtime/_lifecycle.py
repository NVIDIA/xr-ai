# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
/health probe + idle loop for the NeMo docker backend.

The NeMo servers expose `/health` via uvicorn; it returns 503 until weights are
loaded and 200 once ready. The wrapper waits on 200 (model load can take tens
of seconds), then sits idle until the container goes away or a signal arrives.
"""
from __future__ import annotations

import logging
import signal
import time
import urllib.request

log = logging.getLogger(__name__)


def health_url(host: str, port: int) -> str:
    """Build the /health URL.

    The servers bind 0.0.0.0 but the wrapper always probes 127.0.0.1 so it
    works regardless of which interface the host listens on.
    """
    del host  # 127.0.0.1 is always reachable from the wrapper
    return f"http://127.0.0.1:{port}/health"


def health_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_until_healthy(url: str, *, is_alive, poll_s: float = 2.0) -> None:
    """Block until *url* responds 200 or *is_alive()* returns False.

    *is_alive* returns True while the container process is still running. If it
    returns False before /health is up, this raises SystemExit so the wrapper
    exits the same way the in-venv path would on a failed model load.
    """
    while True:
        if not is_alive():
            log.error("NeMo container exited before /health became reachable")
            raise SystemExit(1)
        if health_ok(url, timeout=2.0):
            return
        time.sleep(poll_s)


def idle_until_stopped(url: str, log_prefix: str, poll_s: float = 5.0) -> None:
    """Block until /health stops responding or SIGTERM/SIGINT arrives.

    The container is owned by dockerd, not the wrapper, so the wrapper sits
    idle here until either the container goes away or the launcher signals us.
    """
    stopped = [False]
    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int = signal.getsignal(signal.SIGINT)

    def _on_signal(_sig, _frame):
        stopped[0] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        while not stopped[0]:
            if not health_ok(url, timeout=2.0):
                print(f"[{log_prefix}] NeMo /health unreachable — exiting", flush=True)
                return
            time.sleep(poll_s)
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)
