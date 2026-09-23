# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare DeviceIOHub container and browser artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from xr_ai_launcher import docker_image_size, path_size, report_prepare_status

from device_io_hub._config_loader import DeviceIOHubArtifactConfig
from device_io_hub.transport.livekit._docker import _LIVEKIT_IMAGE


def prepare_livekit_image() -> None:
    size = docker_image_size(_LIVEKIT_IMAGE)
    if size is not None:
        report_prepare_status(f"container image {_LIVEKIT_IMAGE}", "cached", size)
        return
    report_prepare_status(f"container image {_LIVEKIT_IMAGE}", "downloading", None)
    try:
        subprocess.run(["docker", "pull", _LIVEKIT_IMAGE], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("DeviceIOHub artifact preparation requires docker on PATH and a running daemon") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to pull container image {_LIVEKIT_IMAGE}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _valid_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def prepare_web_xr_vendor(web_client_dir: str, build_script: str) -> None:
    """Build the CloudXR and LiveKit browser modules for the WebXR client."""
    if not build_script:
        return
    if not web_client_dir:
        raise RuntimeError("web_xr_vendor_build_script requires web_client_dir")
    vendor_dir = Path(web_client_dir).resolve() / "vendor"

    cloudxr_out = vendor_dir / "cloudxr-sdk.esm.mjs"
    livekit_out = vendor_dir / "livekit-client.esm.mjs"
    build_sh = Path(build_script).resolve()
    build_dir = build_sh.parent
    if not build_sh.is_file():
        raise RuntimeError(f"WebXR vendor bundle requires the build script at {build_sh}")
    if not os.access(build_sh, os.X_OK):
        raise RuntimeError(f"WebXR vendor build script is not executable: {build_sh}")

    sdk_version_path = build_dir / ".sdk-version"
    package_json_path = build_dir / "package.json"
    version_marker = vendor_dir / ".cloudxr-sdk-version"
    livekit_version_marker = vendor_dir / ".livekit-client-version"
    try:
        sdk_version = sdk_version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"failed to read {sdk_version_path}: {exc}") from exc
    if not sdk_version:
        raise RuntimeError(f"{sdk_version_path} is empty")
    try:
        package = json.loads(package_json_path.read_text(encoding="utf-8"))
        livekit_version = package["dependencies"]["livekit-client"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read {package_json_path}: {exc}") from exc
    if not isinstance(livekit_version, str) or not livekit_version.strip():
        raise RuntimeError(f"{package_json_path} must select a livekit-client dependency")

    built_version = _read_text(version_marker)
    built_livekit_version = _read_text(livekit_version_marker)
    outputs = (cloudxr_out, livekit_out)
    if (
        all(_valid_output(path) for path in outputs)
        and built_version == sdk_version
        and built_livekit_version == livekit_version
    ):
        report_prepare_status(
            f"WebXR vendor bundle {sdk_version}/{livekit_version}",
            "cached",
            sum(path_size(path) for path in outputs),
        )
        return

    if not shutil.which("npm"):
        raise RuntimeError(
            "WebXR vendor bundle requires npm on PATH. Install Node.js from "
            "https://nodejs.org, then retry, or run "
            f"`cd {build_dir} && ./build.sh`"
        )

    report_prepare_status(
        f"WebXR vendor bundle {sdk_version}/{livekit_version}",
        "downloading",
        None,
    )
    result = subprocess.run([str(build_sh)], cwd=str(build_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"WebXR vendor build failed with exit {result.returncode}; check the build output, then retry"
        )
    invalid = [path.name for path in outputs if not _valid_output(path)]
    if invalid:
        raise RuntimeError("WebXR vendor build completed with missing or empty outputs: " + ", ".join(invalid))
    built_version = _read_text(version_marker)
    built_livekit_version = _read_text(livekit_version_marker)
    if built_version != sdk_version or built_livekit_version != livekit_version:
        raise RuntimeError("WebXR vendor build did not record the requested CloudXR and LiveKit versions")


def prepare_artifacts(cfg: DeviceIOHubArtifactConfig) -> None:
    prepare_livekit_image()
    prepare_web_xr_vendor(
        cfg.web_client_dir,
        cfg.web_xr_vendor_build_script,
    )
