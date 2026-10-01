# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare models and run vLLM from the pip-installed vLLM environment.

Persistent serving puts vLLM in a new session group so the launcher's killpg()
does not reach it. Non-persistent serving shares the wrapper's session so
SIGTERM propagates and vLLM exits with the wrapper.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import _docker, _lifecycle

log = logging.getLogger(__name__)
_PROCESS_STOP_TIMEOUT_S = 5.0


def prepare(model: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model)


def _reusable_listener(port: int, health_url: str) -> int | None:
    """Return the stable managed vLLM PID on *port*, or fail closed."""
    pid = _docker.owned_listener_pid(port, "vllm")
    if pid is None:
        return None
    if not _lifecycle.health_ok(health_url):
        raise RuntimeError(
            f"managed vLLM pid {pid} owns port {port}, but its health check failed; "
            "wait for it to finish starting or run `model_servers --stop`"
        )
    if _docker.pid_on_port_checked(port) != (pid, True, True):
        raise RuntimeError(
            f"listener on port {port} changed while its identity was being validated"
        )
    return pid


def _terminate_process(process: subprocess.Popen, persistent: bool) -> None:
    """Stop and reap a vLLM process within a bounded interval."""
    def send(sig: signal.Signals) -> None:
        if persistent:
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)

    def alive() -> bool:
        return _docker.process_group_alive(process.pid) if persistent else process.poll() is None

    try:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not alive():
                return
            send(sig)
            deadline = time.monotonic() + _PROCESS_STOP_TIMEOUT_S
            while time.monotonic() < deadline and alive():
                time.sleep(0.05)
    except ProcessLookupError:
        return
    finally:
        process.poll()


def run(
    *,
    persistent: bool,
    log_prefix: str,
    vllm_argv: list[str],
    host: str,
    port: int,
    ready_file: Path | None,
) -> None:
    health_url = _lifecycle.health_url(host, port)

    # A profile switch can leave an xr-ai container (vLLM docker or NIM) on
    # this port; it can answer the health probe and be silently mistaken for
    # a reusable pip server. Evict it before the reuse check.
    holder, checked = _docker.container_on_port_checked(port)
    if checked and holder:
        print(
            f"[{log_prefix}] port {port} is held by container {holder}; "
            f"stopping it to make way",
            flush=True,
        )
        _docker.stop_container(holder)
        if not _docker.remove_container(holder) and _docker.container_running(holder):
            log.error("could not evict container %s from port %d", holder, port)
            sys.exit(1)

    try:
        reusable_pid = _reusable_listener(port, health_url) if persistent else None
    except RuntimeError as exc:
        raise SystemExit(f"[{log_prefix}] {exc}") from exc

    if reusable_pid is not None:
        print(
            f"[{log_prefix}] vLLM pid {reusable_pid} already running on "
            f"port {port}: reusing",
            flush=True,
        )
        if ready_file:
            ready_file.touch()
        _lifecycle.idle_until_stopped(health_url, log_prefix)
        return

    print(
        f"[{log_prefix}] Launching vLLM (pip)  http://{host}:{port}/v1",
        flush=True,
    )
    # start_new_session=True is what makes persistence work — vLLM is in its
    # own process group so the launcher's killpg() on the wrapper does not
    # reach it. Non-persistent wrappers stay in the wrapper's group so SIGTERM
    # propagates and vLLM exits with the wrapper.
    env = os.environ | {
        "XR_AI_VLLM_MANAGED": "1",
        "XR_AI_VLLM_PORT": str(port),
    }
    proc: subprocess.Popen | None = None
    pending_signal: int | None = None

    def abort_startup(signum, _frame) -> None:
        nonlocal pending_signal
        if pending_signal is not None:
            return
        pending_signal = signum
        if proc is not None:
            raise SystemExit(128 + signum)

    previous_handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, abort_startup)
        proc = subprocess.Popen(vllm_argv, env=env, start_new_session=persistent)
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        _lifecycle.wait_until_healthy(
            health_url,
            is_alive=lambda: proc.poll() is None,
        )
        listener_pid = _docker.owned_listener_pid(port, "vllm")
        try:
            listener_matches = (
                os.getpgid(listener_pid) == proc.pid
                if persistent and listener_pid is not None
                else listener_pid == proc.pid
            )
        except OSError:
            listener_matches = False
        if not listener_matches:
            raise RuntimeError(
                f"health endpoint on port {port} is not owned by spawned vLLM "
                f"session {proc.pid}; wait for another server to finish starting or "
                "run `model_servers --stop`"
            )
    except (RuntimeError, KeyboardInterrupt, SystemExit) as exc:
        pending_signal = pending_signal or 0
        if proc is not None:
            _terminate_process(proc, persistent)
        if isinstance(exc, RuntimeError):
            raise SystemExit(f"[{log_prefix}] {exc}") from exc
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    log.info("Ready  →  http://localhost:%d/v1", port)
    if ready_file:
        ready_file.touch()

    if persistent:
        _lifecycle.idle_until_stopped(health_url, log_prefix)
    else:
        rc = proc.wait()
        if rc != 0:
            sys.exit(rc)
