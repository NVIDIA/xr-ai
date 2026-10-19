# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fetch the pinned CloudXR Runtime SDK and prepare its native libraries.

The SDK is the tarball NVIDIA publishes on public NGC (no API key needed),
the same artifact IsaacCapture (formerly IsaacTeleop) bundles into its
``isaacteleop`` wheel. It is downloaded on first use, checked against a
pinned SHA-256, and unpacked once into ``<install_dir>/sdk/<version>-<arch>``.
To install offline, place the tarball at the download path printed on
failure; a file already there is verified and used instead of downloading.

The SDK's binaries are used exactly as published: its license agreement
(section 2.4) does not permit modifying them.
"""
from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

SDK_VERSION = "6.2.0"
"""CloudXR runtime version. Keep equal to the web client SDK pin in
``client-samples/web-xr-build/.sdk-version``."""

_NGC_URL = (
    "https://api.ngc.nvidia.com/v2/resources/org/nvidia/"
    "cloudxr-runtime-for-isaac-teleop/{version}/files?redirect=true&path={file}"
)

# SHA-256 of each published tarball, by CloudXR architecture name.
_SHA256 = {
    "amd64": "92ef5d784d30cd63ae162582cf3e8709e04a39f8d94f1e41ff814b925e30111e",
    "arm64": "3aa25e7c052aab4c2e6b1cb188272ae6d691335c586e1667e95e73c492584e88",
}
_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}

# Headers and optional plugins are not needed to run the service.
_SKIPPED_DIRS = ("include/", "plugins/")


class SDKError(RuntimeError):
    """The CloudXR SDK could not be obtained or prepared."""


@dataclass(frozen=True)
class SDK:
    """An unpacked CloudXR Runtime SDK."""

    native_dir: Path
    """Directory holding ``libcloudxr.so`` and its companion libraries."""

    @property
    def license_file(self) -> Path:
        """The CloudXR license agreement shipped with this SDK."""
        return self.native_dir / "LICENSE.txt"


def ensure_sdk(install_dir: Path) -> SDK:
    """Return the pinned SDK under *install_dir*, downloading it if needed."""
    machine = platform.machine()
    arch = _ARCH.get(machine)
    if arch is None:
        raise SDKError(f"CloudXR runtime is not available for {machine}")

    native_dir = install_dir / "sdk" / f"{SDK_VERSION}-{arch}"
    if (native_dir / "libcloudxr.so").is_file():
        return SDK(native_dir)

    file_name = f"CloudXR-{SDK_VERSION}-Linux-{arch}-sdk.tar.gz"
    tarball = install_dir / "sdk" / file_name
    if not tarball.is_file():
        _download(_NGC_URL.format(version=SDK_VERSION, file=file_name), tarball)
    _verify(tarball, _SHA256[arch])

    # Unpack beside the final directory and rename it into place, so an
    # interrupted run never leaves a half-populated SDK that looks complete.
    with tempfile.TemporaryDirectory(dir=tarball.parent) as staging:
        unpacked = Path(staging) / "native"
        _unpack(tarball, unpacked)
        unpacked.rename(native_dir)
    tarball.unlink()
    logger.info("CloudXR runtime SDK {} installed in {}", SDK_VERSION, native_dir)
    return SDK(native_dir)


def _download(url: str, target: Path) -> None:
    logger.info("Downloading CloudXR runtime SDK {} from NGC…", SDK_VERSION)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise SDKError(
            f"could not download the CloudXR runtime SDK: {exc}\n"
            f"  URL: {url}\n"
            f"  To install offline, place the file at {target}"
        ) from exc
    partial.rename(target)


def _verify(tarball: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with tarball.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        tarball.unlink()
        raise SDKError(
            f"{tarball.name} has SHA-256 {digest.hexdigest()}, expected {expected}; "
            "the file was removed, retry to download it again"
        )


def _unpack(tarball: Path, target: Path) -> None:
    with tarfile.open(tarball) as archive:
        members = [
            member
            for member in archive.getmembers()
            if (member.isfile() or member.issym())
            and not member.name.removeprefix("./").startswith(_SKIPPED_DIRS)
        ]
        archive.extractall(target, members=members, filter="data")

