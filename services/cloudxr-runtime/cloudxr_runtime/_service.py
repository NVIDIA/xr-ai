# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Host the native CloudXR service in its own process.

    python -m cloudxr_runtime._service <native_dir> <logs_dir>

``libcloudxr.so`` links the system ``libssl.so.3``. A Python process that has
imported :mod:`ssl` carries its own OpenSSL, and the two copies conflict and
crash the streaming stack. This module therefore imports only the standard
library modules below and runs as a separate interpreter, so no Python
OpenSSL is ever loaded beside the service. Keep it that way: no third-party
or package-relative imports here.

Ported from ``isaacteleop.cloudxr.runtime.run`` 1.3.131 (Apache-2.0).
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_FILE_LOGGING_OFF = ("false", "off", "no", "n", "f", "0")


def serve(native_dir: Path, logs_dir: Path) -> None:
    """Run the service until SIGINT or SIGTERM."""
    if os.environ.get("NV_CXR_FILE_LOGGING", "true").lower() not in _FILE_LOGGING_OFF:
        # The native library logs to NV_CXR_OUTPUT_DIR; keep its console
        # banner off the terminal but preserve stderr (Vulkan loader and GPU
        # errors) for diagnosis.
        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr_log = os.open(
            logs_dir / "runtime_stderr.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(stderr_log, sys.stderr.fileno())
        os.close(devnull)
        os.close(stderr_log)

    lib = ctypes.CDLL(str(native_dir / "libcloudxr.so"), mode=os.RTLD_DEEPBIND)
    service = ctypes.c_void_p()
    _check_result("nv_cxr_service_create", lib.nv_cxr_service_create(ctypes.byref(service)))
    started = False
    stop_requested = False
    stopping = False
    stop_result = 0
    joined = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested, stopping, stop_result
        stop_requested = True
        if started and not stopping:
            stopping = True
            stop_result = lib.nv_cxr_service_stop(service)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _check_result("nv_cxr_service_start", lib.nv_cxr_service_start(service))
        started = True
        if stop_requested:
            stop(0, None)

        # Only stop may run concurrently with native join. The owner destroys
        # the handle after join completes and signal admission is closed.
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(lib.nv_cxr_service_join, service).result()
        _check_result("nv_cxr_service_join", result, allow_stopped=True)
        joined = True
        _check_result("nv_cxr_service_stop", stop_result, allow_stopped=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if not started or joined:
            lib.nv_cxr_service_destroy(service)


def _check_result(operation: str, result: int, *, allow_stopped: bool = False) -> None:
    # The SDK permits NOT_STARTED when the runtime stops by external factors.
    if result != 0 and not (allow_stopped and result == -6):
        raise RuntimeError(f"{operation} failed with CloudXR result {result}; see the native runtime log")


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]))
