# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare and supervise the native CloudXR service and its OpenXR environment.

Ported from ``isaacteleop.cloudxr`` 1.3.131 (``env_config.py`` and
``runtime.py``, Apache-2.0), keeping its file layout under the install
directory so existing ``cloudxr.env`` consumers are unaffected::

    <install_dir>/openxr_cloudxr.json, libopenxr_cloudxr.so   OpenXR runtime
    <install_dir>/run/cloudxr.env                             exported env
    <install_dir>/run/runtime_started                         readiness sentinel
    <install_dir>/run/eula_accepted                           EULA marker
    <install_dir>/logs/                                       native logs
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ._sdk import SDK

# Service defaults; the process environment and the env file override them.
_DEFAULT_ENV = {
    "NV_CXR_ENABLE_PUSH_DEVICES": "true",
    "NV_CXR_ENABLE_TENSOR_DATA": "true",
    "NV_CXR_FILE_LOGGING": "true",
    "NV_DEVICE_PROFILE": "auto-webrtc",
}
# Derived from the install directory; an env file may not override them.
_DERIVED_KEYS = frozenset(
    {"XR_RUNTIME_JSON", "XRT_NO_STDIN", "NV_CXR_RUNTIME_DIR", "NV_CXR_OUTPUT_DIR"}
)
_STALE_RUN_FILES = ("ipc_cloudxr", "runtime_started", "monado.pid", "cloudxr.pid")
_TERMINATE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Layout:
    """Paths the runtime and its OpenXR clients share."""

    install_dir: Path

    @property
    def run_dir(self) -> Path:
        return self.install_dir / "run"

    @property
    def logs_dir(self) -> Path:
        return self.install_dir / "logs"

    @property
    def env_file(self) -> Path:
        return self.run_dir / "cloudxr.env"

    @property
    def runtime_json(self) -> Path:
        return self.install_dir / "openxr_cloudxr.json"

    @property
    def ready_sentinel(self) -> Path:
        return self.run_dir / "runtime_started"

    @property
    def eula_marker(self) -> Path:
        return self.run_dir / "eula_accepted"


def publish_env(layout: Layout, env_file: Path | None = None) -> dict[str, str]:
    """Resolve the runtime environment, apply it, and write ``cloudxr.env``.

    Precedence: *env_file*, then the process environment, then the defaults.
    """
    overrides = _read_env_file(env_file) if env_file is not None else {}
    for key in sorted(_DERIVED_KEYS & overrides.keys()):
        logger.warning("{} sets {}, which the runtime derives; ignoring it", env_file, key)
        del overrides[key]
    env ={key: os.environ.get(key, value) for key, value in _DEFAULT_ENV.items()}
    env.update(overrides)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    env |= {
        "CXR_INSTALL_DIR": str(layout.install_dir),
        "CXR_HOST_VOLUME_PATH": str(layout.install_dir),
        "XR_RUNTIME_JSON": str(layout.runtime_json),
        "NV_CXR_RUNTIME_DIR": str(layout.run_dir),
        "NV_CXR_OUTPUT_DIR": str(layout.logs_dir),
        "XRT_NO_STDIN": "true",
    }
    os.environ.update(env)

    layout.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    layout.run_dir.chmod(0o700)
    fd = os.open(layout.env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(f"export {key}={shlex.quote(env[key])}\n" for key in sorted(env))
    return env


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines, expanding ``$VARS`` and ``~`` in values."""
    result: dict[str, str] = {}
    for line in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            result[key.strip()] = os.path.expanduser(os.path.expandvars(value.strip()))
    return result


def check_eula(layout: Layout, sdk: SDK, *, accept: bool) -> None:
    """Exit unless the CloudXR license agreement has been accepted.

    *accept* records acceptance without prompting; otherwise the operator is
    asked once and the answer is remembered in the marker file.
    """
    if layout.eula_marker.is_file():
        return
    if not accept:
        print(
            f"\nThe NVIDIA CloudXR license agreement must be accepted to run: {sdk.license_file}",
            file=sys.stderr,
        )
        try:
            reply = input("\nAccept the NVIDIA CloudXR license agreement? [y/N]: ")
        except EOFError:
            reply = ""
        if reply.strip().lower() not in ("y", "yes"):
            sys.exit("CloudXR license agreement not accepted.")
    layout.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    layout.eula_marker.write_text("accepted\n")


def latest_runtime_log(layout: Layout) -> Path | None:
    """Return the newest native server log, if any."""
    logs = sorted(layout.logs_dir.glob("cxr_server.*.log"))
    return logs[-1] if logs else None


def prepare_run_dir(layout: Layout, sdk: SDK) -> None:
    """Install the OpenXR runtime manifest and clear the previous run's state."""
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("libopenxr_cloudxr.so", "openxr_cloudxr.json"):
        shutil.copy2(sdk.native_dir / name, layout.install_dir / name)
    for name in _STALE_RUN_FILES:
        (layout.run_dir / name).unlink(missing_ok=True)


def start_service(layout: Layout, sdk: SDK) -> subprocess.Popen[bytes]:
    """Start the native service in its own interpreter (see ``_service``)."""
    return subprocess.Popen(
        [sys.executable, "-m", "cloudxr_runtime._service", str(sdk.native_dir), str(layout.logs_dir)]
    )


def stop_service(process: subprocess.Popen[bytes]) -> None:
    """Stop the service process, escalating from SIGTERM to SIGKILL."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_TERMINATE_TIMEOUT_S)
