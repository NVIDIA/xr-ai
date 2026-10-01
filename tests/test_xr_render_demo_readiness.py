# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_MAIN = Path(__file__).parents[1] / "agent-samples/xr-render-demo/main.py"


def _load_sample():
    spec = importlib.util.spec_from_file_location("xr_render_readiness", _MAIN)
    assert spec and spec.loader
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


@pytest.mark.parametrize(
    ("case", "node", "npm", "expected"),
    [
        ("current", None, None, None),
        ("native", None, None, None),
        ("stale", "v20.19.0", "/usr/bin/npm", ("web_vendor_node", "passed")),
        ("old-node", "v18.20.0", "/usr/bin/npm", ("web_vendor_node", "failed")),
        ("missing-npm", "v20.19.0", None, ("web_vendor_node", "failed")),
        ("no-metadata", None, None, ("web_vendor_bundle", "failed")),
        ("stale-no-build", None, None, None),
    ],
)
def test_node_is_required_only_when_web_vendor_needs_build(
    tmp_path, monkeypatch, case, node, npm, expected,
):
    sample = _load_sample()
    root = tmp_path / "agent-samples/xr-render-demo"
    build_dir = (root / "../../client-samples/web-xr-build").resolve()
    vendor_dir = (root / "../../client-samples/web-xr/vendor").resolve()
    build_dir.mkdir(parents=True)
    vendor_dir.mkdir(parents=True)
    if case != "no-metadata":
        (build_dir / ".sdk-version").write_text("6.2.0\n")
    (build_dir / "package.json").write_text('{"dependencies":{"livekit-client":"2.5"}}')
    if not case.endswith("no-build"):
        (build_dir / "build.sh").touch()
    if case in ("current", "stale-no-build"):
        for name in ("cloudxr-sdk.esm.mjs", "livekit-client.esm.mjs"):
            (vendor_dir / name).touch()
        sdk_marker = "old" if case == "stale-no-build" else "6.2.0"
        (vendor_dir / ".cloudxr-sdk-version").write_text(f"{sdk_marker}\n")
        (vendor_dir / ".livekit-client-version").write_text("2.5\n")
    vulkan = sample._row("vulkan_device", True, "NVIDIA RTX", "NVIDIA GPU", "")
    monkeypatch.setattr(sample, "_BASE", root)
    monkeypatch.setattr(sample, "_vulkan_check", lambda: vulkan)
    monkeypatch.setenv(sample._NO_WEB_CLIENT_ENV, "true" if case == "native" else "0")
    monkeypatch.setattr(sample.shutil, "which", lambda _name: npm)
    monkeypatch.setattr(sample.subprocess, "run", lambda command, **_kwargs:
                        subprocess.CompletedProcess(command, 0, f"{node}\n")
                        if node else pytest.fail("Node must not be probed"))

    rows = sample._check_sample()
    assert rows[0] == vulkan
    assert [(row["name"], row["status"]) for row in rows[1:] if row["name"] != "lovr"] == (
        [] if expected is None else [expected]
    )
    if case == "current":
        seen = {}
        monkeypatch.setattr(sample, "_BASE", _MAIN.parent)
        monkeypatch.setattr(sample, "setup_logging", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sample, "run_stack", lambda *_args, **kwargs: seen.update(kwargs))
        sample.run(["--check"])
        assert (seen["check_sample"], seen["model_profile"]) == (
            sample._check_sample, _MAIN.parent / "yaml/models.json",
        )


@pytest.mark.parametrize(("tool", "result", "expected"), [
    (None, None, ("skipped", "vulkaninfo not installed; device enumeration not checked")),
    ("vulkaninfo", (True, "vendorID = 0x10de"), ("passed", "NVIDIA Vulkan device found")),
    ("vulkaninfo", (True, "vendorID = 0x10005\ndeviceName = llvmpipe"),
     ("failed", "no NVIDIA Vulkan device in vulkaninfo --summary")),
    ("vulkaninfo", (False, "vkCreateInstance failed"), ("failed", "vkCreateInstance failed")),
])
def test_vulkan_summary_requires_an_nvidia_device(monkeypatch, tool, result, expected):
    sample = _load_sample()
    calls = []
    monkeypatch.setattr(sample.shutil, "which", lambda _name: tool)
    monkeypatch.setattr(sample, "_run", lambda command, timeout=5:
                        calls.append((command, timeout)) or result)
    row = sample._vulkan_check()
    assert (row["status"], row["detected"]) == expected
    assert calls == ([(["vulkaninfo", "--summary"], 15)] if tool else [])


@pytest.mark.parametrize(("arch", "env", "yaml", "status"), [
    ("x86_64", "", "", "skipped"), ("aarch64", "", "", "failed"),
    ("aarch64", "/opt/lovr", "", "passed"), ("aarch64", "", "/opt/lovr", "passed"),
])
def test_lovr_check_is_owned_by_render_sample(tmp_path, monkeypatch, arch, env, yaml, status):
    sample = _load_sample()
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/scene_service.yaml").write_text(f"lovr_bin: {yaml}\n")
    monkeypatch.setattr(sample, "_BASE", tmp_path)
    monkeypatch.setattr(sample.platform, "machine", lambda: arch)
    monkeypatch.setenv("LOVR_BIN", env)
    monkeypatch.setattr(sample, "_vulkan_check", lambda: {})
    monkeypatch.setattr(sample, "_web_client_enabled", lambda: False)
    assert sample._check_sample()[1]["status"] == status


def test_vendor_build_cancellation_kills_descendant_after_shell_exits(
    tmp_path, monkeypatch,
) -> None:
    sample = _load_sample()
    child_pid_path = tmp_path / "child.pid"
    build_sh = tmp_path / "build.sh"
    build_sh.touch()
    code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        "open(sys.argv[1], 'w').write(str(child.pid)); time.sleep(30)"
    )
    build = subprocess.Popen(
        [sys.executable, "-c", code, str(child_pid_path)],
        start_new_session=True,
    )

    class InterruptedBuild:
        pid = build.pid

        def wait(self, timeout=None):
            if timeout is not None:
                assert timeout == 5
                return build.wait(timeout=timeout)
            deadline = time.monotonic() + 3
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child_pid_path.exists()
            os.kill(build.pid, signal.SIGKILL)
            build.wait(timeout=3)
            raise KeyboardInterrupt

    def start_build(command, **kwargs):
        assert command == [str(build_sh)]
        assert kwargs == {"cwd": str(tmp_path), "start_new_session": True}
        return InterruptedBuild()

    monkeypatch.setattr(sample, "_vendor_stale", lambda: (True, build_sh))
    monkeypatch.setattr(sample, "_node_npm_status", lambda: (True, "available"))
    monkeypatch.setattr(sample.subprocess, "Popen", start_build)

    try:
        with pytest.raises(KeyboardInterrupt):
            sample._ensure_web_vendor()

        child_pid = int(child_pid_path.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("vendor build descendant survived cancellation")
    finally:
        try:
            os.killpg(build.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        build.wait(timeout=3)
