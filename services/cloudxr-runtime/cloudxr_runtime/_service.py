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
import threading
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
    lib.nv_cxr_service_create(ctypes.byref(service))

    def stop(_signum: int, _frame: object) -> None:
        lib.nv_cxr_service_stop(service)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    lib.nv_cxr_service_start(service)

    # join() blocks in native code; run it on a worker so the main thread
    # stays in Python and can handle the stop signals.
    def join_then_destroy() -> None:
        lib.nv_cxr_service_join(service)
        lib.nv_cxr_service_destroy(service)

    worker = threading.Thread(target=join_then_destroy)
    worker.start()
    worker.join()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]))
