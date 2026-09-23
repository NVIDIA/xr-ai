# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIOHub artifact configuration and WebXR vendor preparation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from device_io_hub import _prepare
from device_io_hub._config_loader import load_artifact_config


def _vendor_paths(tmp_path: Path) -> tuple[Path, Path, tuple[Path, Path]]:
    web_client_dir = tmp_path / "client-samples" / "web-xr"
    vendor_dir = web_client_dir / "vendor"
    vendor_dir.mkdir(parents=True)
    build_script = tmp_path / "client-samples" / "web-xr-build" / "build.sh"
    build_script.parent.mkdir(parents=True)
    build_script.write_text("#!/bin/sh\n", encoding="utf-8")
    build_script.chmod(0o755)
    (build_script.parent / ".sdk-version").write_text("6.2.0\n", encoding="utf-8")
    (build_script.parent / "package.json").write_text(
        json.dumps({"dependencies": {"livekit-client": "^2.21.0"}}),
        encoding="utf-8",
    )
    outputs = (
        vendor_dir / "cloudxr-sdk.esm.mjs",
        vendor_dir / "livekit-client.esm.mjs",
    )
    return web_client_dir, build_script, outputs


def _record_versions(vendor_dir: Path) -> None:
    (vendor_dir / ".cloudxr-sdk-version").write_text("6.2.0\n", encoding="utf-8")
    (vendor_dir / ".livekit-client-version").write_text(
        "^2.21.0\n",
        encoding="utf-8",
    )


def test_web_xr_vendor_repairs_missing_and_empty_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_client_dir, build_script, outputs = _vendor_paths(tmp_path)
    outputs[0].write_text("cloudxr", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: str) -> subprocess.CompletedProcess:
        calls.append(command)
        for output in outputs:
            output.write_text(output.name, encoding="utf-8")
        _record_versions(outputs[0].parent)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_prepare.shutil, "which", lambda _command: "/usr/bin/npm")
    monkeypatch.setattr(_prepare.subprocess, "run", fake_run)

    _prepare.prepare_web_xr_vendor(str(web_client_dir), str(build_script))
    _prepare.prepare_web_xr_vendor(str(web_client_dir), str(build_script))
    outputs[1].write_bytes(b"")
    _prepare.prepare_web_xr_vendor(str(web_client_dir), str(build_script))

    assert calls == [[str(build_script)], [str(build_script)]]


def test_web_xr_vendor_rejects_empty_output_after_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_client_dir, build_script, outputs = _vendor_paths(tmp_path)

    def fake_run(command: list[str], *, cwd: str) -> subprocess.CompletedProcess:
        outputs[0].write_text("cloudxr", encoding="utf-8")
        outputs[1].write_bytes(b"")
        _record_versions(outputs[0].parent)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_prepare.shutil, "which", lambda _command: "/usr/bin/npm")
    monkeypatch.setattr(_prepare.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="missing or empty outputs"):
        _prepare.prepare_web_xr_vendor(str(web_client_dir), str(build_script))


def test_web_xr_vendor_requires_an_executable_script(tmp_path: Path) -> None:
    web_client_dir, build_script, _ = _vendor_paths(tmp_path)
    build_script.chmod(0o644)

    with pytest.raises(RuntimeError, match="not executable"):
        _prepare.prepare_web_xr_vendor(str(web_client_dir), str(build_script))


def test_artifact_config_resolves_paths_without_livekit_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "hub.yaml"
    config.write_text(
        "web_client_dir: ./web-xr\nweb_xr_vendor_build_script: ./web-xr-build/build.sh\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config)])

    loaded = load_artifact_config()

    assert loaded.web_client_dir == str(tmp_path / "web-xr")
    assert loaded.web_xr_vendor_build_script == str(tmp_path / "web-xr-build" / "build.sh")


def test_no_web_client_override_clears_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "hub.yaml"
    config.write_text(
        "web_client_dir: ./web-xr\nweb_xr_vendor_build_script: ./web-xr-build/build.sh\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEVICE_IO_HUB_NO_WEB_CLIENT", "1")
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--config", str(config)])

    loaded = load_artifact_config()

    assert loaded.web_client_dir == ""
    assert loaded.web_xr_vendor_build_script == ""
