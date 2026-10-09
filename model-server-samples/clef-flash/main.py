# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the persistent local Clef-Flash 9B SystemOne service."""

from __future__ import annotations

import argparse
import os
import select
import signal
from pathlib import Path

from xr_ai_launcher import Process, read_config_scalar, run_stack
from xr_ai_logging import setup_logging
from xr_ai_vllm._docker import pid_on_port_checked

_BASE = Path(__file__).resolve().parent


_CONFIG = _BASE / "yaml" / "clef_server.yaml"
_MANAGED_ENV = "XR_AI_CLEF_MANAGED"
_PORT_ENV = "XR_AI_CLEF_PORT"
_STOP_TIMEOUT_S = 20.0
_KILL_TIMEOUT_S = 5.0


def _port() -> int:
    try:
        port = int(read_config_scalar(_CONFIG, "port"))
    except ValueError as exc:
        raise ValueError(f"{_CONFIG}: port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{_CONFIG}: port must be between 1 and 65535")
    return port


def _has_clef_ownership(pid: int, port: int) -> bool:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return (
        f"{_MANAGED_ENV}=1".encode() in entries
        and f"{_PORT_ENV}={port}".encode() in entries
    )


def _wait_for_process_exit(pidfd: int, timeout_s: float) -> bool:
    # Uvicorn closes the listener before draining requests and native work.
    # A pidfd becomes readable only when that exact process exits.
    return bool(select.select([pidfd], [], [], timeout_s)[0])


def _stop_clef(port: int) -> bool:
    pid, checked, listening = pid_on_port_checked(port)
    if not checked or (listening and pid is None):
        print(f"clef-flash: cannot inspect ownership of port {port}; not stopping")
        return False
    if not listening:
        print("clef-flash: no persistent Clef server is running")
        return True
    assert pid is not None
    pidfd = None
    try:
        # Capture identity before verification; all later signals address this
        # process handle even if Linux recycles its numeric PID during shutdown.
        pidfd = os.pidfd_open(pid)
        if not _has_clef_ownership(pid, port):
            print(f"clef-flash: listener on port {port} is not an owned Clef server; not stopping")
            return False
        if pid_on_port_checked(port) != (pid, True, True):
            print(f"clef-flash: listener on port {port} changed during inspection; not stopping")
            return False
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        if _wait_for_process_exit(pidfd, _STOP_TIMEOUT_S):
            print(f"clef-flash: stopped Clef server on port {port}")
            return True
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        if _wait_for_process_exit(pidfd, _KILL_TIMEOUT_S):
            print(f"clef-flash: force-stopped Clef server on port {port}")
            return True
    except ProcessLookupError:
        return True
    except OSError as exc:
        print(f"clef-flash: cannot track the configured Clef process: {exc}; not stopping")
        return False
    finally:
        if pidfd is not None:
            os.close(pidfd)
    print(f"clef-flash: Clef server on port {port} is still running")
    return False


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop only the Clef service configured by yaml/clef_server.yaml.",
    )
    args = parser.parse_args()
    setup_logging("orchestrator", namespace="clef-flash")
    root_uv_config = _BASE.parents[1] / "uv.toml"
    if root_uv_config.is_file():
        os.environ.setdefault("UV_CONFIG_FILE", str(root_uv_config))
    port = _port()
    if args.stop:
        if not _stop_clef(port):
            raise SystemExit("clef-flash: failed to stop the configured Clef server")
        return
    process = Process(
        "clef-flash",
        "../../services/clef-server",
        "clef_server",
        config=_CONFIG,
        launch_mode="persist",
        port=port,
    )
    run_stack([process], _BASE, exit_after_ready=True)


if __name__ == "__main__":
    run()
