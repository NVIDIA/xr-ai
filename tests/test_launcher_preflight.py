# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import xr_ai_launcher._preflight as _preflight
from xr_ai_launcher import Process

_ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path, value):
    (tmp_path / "requirements.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("command_failure", [False, True])
def test_known_failures_report_detected_required_and_fix_without_secrets(
    tmp_path, monkeypatch, command_failure,
):
    _contract(tmp_path, {
        "nvidia_driver": 580,
        "docker": 24,
        "nvidia_container_toolkit": True,
        "vulkan": True,
        "lovr_config": "scene.yaml",
        "disk_gb_free": 60,
    })
    (tmp_path / "scene.yaml").write_text("other: value\n", encoding="utf-8")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model_cache: cache\n", encoding="utf-8")
    monkeypatch.setenv("PRIVATE_TOKEN", "super-secret-value")
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(_preflight.ctypes.util, "find_library", lambda _name: None)
    monkeypatch.setattr(_preflight.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=5_000_000_000))
    monkeypatch.setattr(_preflight.shutil, "which", lambda name:
                        None if command_failure and name == "docker" else f"/usr/bin/{name}")
    def run(command, **_kwargs):
        if command_failure and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 9, "driver failure on stdout\n", "")
        output = {"nvidia-smi": "570.1\n"}.get(command[0], "{}\n" if "info" in command else "23.0\n")
        return subprocess.CompletedProcess(command, 0, output, "")
    monkeypatch.setattr(_preflight.subprocess, "run", run)
    rows = _preflight.preflight(
        (Process("model", ".", "model", config=model_config, prepare=True),),
        tmp_path,
        credentials=("PRIVATE_TOKEN",),
    )
    by_name = {row["name"]: row for row in rows}
    failed = {
        "nvidia_driver", "docker", "nvidia_container_toolkit", "vulkan",
        "lovr", "disk",
    }
    assert failed <= by_name.keys()
    assert all(
        not by_name[name]["ok"]
        and by_name[name]["detected"]
        and by_name[name]["required"]
        and by_name[name]["remediation"]
        for name in failed
    )
    assert by_name["credential:PRIVATE_TOKEN"]["ok"]
    diagnostic_names = ("nvidia_driver", "docker", "nvidia_container_toolkit")
    expected = (("driver failure on stdout", "docker not found", "docker not found")
                if command_failure else ("570.1", "23.0", "{}"))
    assert tuple(by_name[name]["detected"] for name in diagnostic_names) == expected
    assert by_name["nvidia_driver"]["remediation"].startswith(
        "Run `nvidia-smi " if command_failure else "Upgrade the driver:")
    assert by_name["docker"]["remediation"].startswith("Install Docker 24 or newer")
    assert ("Install the NVIDIA" in by_name["nvidia_container_toolkit"]["remediation"]) != command_failure
    assert "super-secret-value" not in json.dumps(rows)


def test_ports_use_passive_tcp_udp_inspection_and_respect_launch_mode(
    tmp_path, monkeypatch,
):
    _contract(tmp_path, {
        "vulkan": True,
        "ports": [
            {"name": "hub", "port": 7880, "proto": "tcp", "config_key": "ws"},
            {"name": "hub", "port": 7882, "proto": "udp", "config_key": "udp"},
        ],
    })
    config = tmp_path / "hub.yaml"
    config.write_text("ws: 7880\nudp: 7882\n", encoding="utf-8")
    processes = (
        Process("hub", ".", "hub", config=config),
        Process("persisted", ".", "model", port=8100, launch_mode="persist"),
        Process("reused", ".", "model", port=8101, launch_mode="reuse"),
    )
    commands: list[list[str]] = []
    vulkan: list[str] = []
    monkeypatch.setattr(_preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        _preflight.ctypes.util, "find_library",
        lambda name: vulkan.append(name) or "libvulkan.so.1",
    )
    def run(command, **_kwargs):
        commands.append(command)
        if "-lnup" in command:
            raise subprocess.TimeoutExpired(command, 5)
        if command[-1] == "sport = :7880":
            return subprocess.CompletedProcess(command, 0, "", "harmless warning\n")
        return subprocess.CompletedProcess(command, 0, "listener\n", "")
    monkeypatch.setattr(_preflight.subprocess, "run", run)
    assert _preflight.preflight(processes, tmp_path, runtime=False) == []
    assert commands == vulkan == []
    rows = _preflight.preflight(processes, tmp_path)
    by_name = {row["name"]: row for row in rows}
    assert by_name["port:hub:tcp:7880"]["detected"] == "available"
    udp = by_name["port:hub:udp:7882"]
    assert not udp["ok"]
    assert udp["detected"] == "TimeoutExpired"
    assert "Run `ss -H -lnup 'sport = :7882'` manually" in udp["remediation"]
    assert by_name["port:persisted:tcp:8100"]["ok"]
    assert "port:reused:tcp:8101" not in by_name
    assert commands == [
        ["ss", "-H", "-lntp", "sport = :7880"],
        ["ss", "-H", "-lnup", "sport = :7882"],
        ["ss", "-H", "-lntp", "sport = :8100"],
    ]


def test_disk_is_omitted_without_a_declared_prepare_cache(tmp_path, monkeypatch):
    _contract(tmp_path, {"disk_gb_free": 60})
    config = tmp_path / "worker.yaml"
    config.write_text("port: 8000\n", encoding="utf-8")
    reused = tmp_path / "reused.yaml"
    reused.write_text("model_cache: missing\n", encoding="utf-8")
    for processes in (
        (Process("worker", ".", "worker", config=config, prepare=True),),
        (Process("owned", ".", "worker", config=reused),
         Process("reused", ".", "worker", config=reused, prepare=True,
                 launch_mode="reuse")),
    ):
        assert _preflight.preflight(processes, tmp_path) == []


@pytest.mark.parametrize(("relative", "arguments"), [
    ("agent-samples/simple-vlm-example/main.py", ()),
    ("agent-samples/xr-render-demo/main.py", ()),
    ("agent-samples/tea-making-sample/main.py", ()),
    ("agent-samples/lab-instrument-monitoring/main.py", ()),
    ("model-server-samples/model-servers/main.py", ("default", "spark")),
    ("model-server-samples/model-servers-nim/main.py", ("spark",)),
])
def test_shipped_contracts_pass_representative_host_shapes(
    relative, arguments, monkeypatch,
):
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(
        f"preflight_{path.parent.name.replace('-', '_')}", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "tea-making" in relative:
        arguments = (module._WORKER_CONFIG,)
    built = module._build_processes(*arguments)
    processes = built[0] if isinstance(built, tuple) else built
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_preflight.ctypes.util, "find_library", lambda _name: "libvulkan.so.1")
    monkeypatch.setattr(_preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "iterdir", lambda path: iter((path / "cached",)))
    outputs = {"info": '{"nvidia": {}}\n', "nvidia-smi": "580.1\n", "docker": "24.0\n", "ss": ""}
    monkeypatch.setattr(_preflight.subprocess, "run", lambda command, **_kwargs:
                        subprocess.CompletedProcess(
                            command, 0, outputs["info" if "info" in command else command[0]], ""))

    rows = _preflight.preflight(processes, module._BASE)

    assert rows and all(row["ok"] for row in rows)
    assert "disk" not in {row["name"] for row in rows}


def test_missing_contract_has_no_checks_or_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(_preflight.subprocess, "run", lambda *_args, **_kwargs: None)
    assert _preflight.preflight((), tmp_path) == []
