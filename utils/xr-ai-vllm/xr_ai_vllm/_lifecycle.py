# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared lifecycle helpers for both pip and docker vLLM backends.

Both backends need the same /health probe semantics (vLLM exposes /health on
its serving port once weight load + warmup is complete) and the same
"idle until vLLM goes away or a signal arrives" loop used by persistent
wrappers.
"""
from __future__ import annotations

import logging
import signal
import time
import urllib.request

log = logging.getLogger(__name__)


def health_url(host: str, port: int) -> str:
    """Build the vLLM /health URL.

    vLLM binds 0.0.0.0 in our configs but the wrapper always probes 127.0.0.1
    so it works regardless of which interface the host listens on.
    """
    del host  # 127.0.0.1 is always reachable from the wrapper
    return f"http://127.0.0.1:{port}/health"


def health_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_until_healthy(
    url: str,
    *,
    is_alive,
    poll_s: float = 2.0,
) -> None:
    """Block until *url* responds 200 or *is_alive()* returns False.

    *is_alive* is a callable returning True while the underlying vLLM process /
    container is still running. If it returns False before /health is up, this
    function raises SystemExit so the wrapper exits with the same semantics
    as the existing inline polling loops.
    """
    while True:
        if not is_alive():
            log.error("vLLM exited before /health became reachable")
            raise SystemExit(1)
        if health_ok(url, timeout=2.0):
            return
        time.sleep(poll_s)


def idle_until_stopped(url: str, log_prefix: str, poll_s: float = 5.0) -> None:
    """Block until /health stops responding or SIGTERM/SIGINT arrives.

    Used by persistent wrappers: vLLM is owned by something other than the
    wrapper (a new session group or the docker daemon), so the wrapper sits
    idle here until either vLLM goes away or the launcher SIGTERMs us.
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
                print(f"[{log_prefix}] vLLM /health unreachable — exiting", flush=True)
                return
            time.sleep(poll_s)
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)
