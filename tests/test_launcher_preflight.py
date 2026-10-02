# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import xr_ai_launcher._endpoints as _endpoints
import xr_ai_launcher._preflight as _preflight
from xr_ai_launcher import Process

_ROOT = Path(__file__).resolve().parents[1]


def _contract(tmp_path: Path, value: dict) -> None:
    (tmp_path / "requirements.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(("value", "detected"), [
    ({"nvidia_driver": 580}, "contract contains unsupported fields: nvidia_driver"),
    ({"docker": 24}, "contract contains unsupported fields: docker"),
    ({"disk_gb_free": "60"}, "disk_gb_free must be a positive number"),
])
def test_invalid_contract_values_return_failed_contract_row(
    tmp_path, value, detected,
) -> None:
    _contract(tmp_path, value)

    expected = _preflight._row(
        "contract", False, detected, "valid requirements.json",
        f"Fix {tmp_path / 'requirements.json'}.",
    )
    assert _preflight.preflight((), tmp_path) == [expected]


@pytest.mark.parametrize("command_failure", [False, True])
def test_known_failures_report_detected_required_and_fix_without_secrets(
    tmp_path, monkeypatch, command_failure,
) -> None:
    _contract(tmp_path, {"disk_gb_free": 60})
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "vllm_backend: docker\nmodel_cache: cache\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRIVATE_TOKEN", "super-secret-value")
    monkeypatch.setattr(
        _preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=5_000_000_000),
    )
    monkeypatch.setattr(
        _preflight.shutil,
        "which",
        lambda name: None if command_failure and name == "docker" else f"/usr/bin/{name}",
    )

    def run(command, **_kwargs):
        if command_failure and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 9, "driver failure on stdout\n", "")
        output = {
            "nvidia-smi": "570.1\n",
            "docker": "{}\n" if "info" in command else "23.0\n",
        }[command[0]]
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    rows = _preflight.preflight(
        (Process("model", ".", "model", config=model_config, prepare=True),),
        tmp_path,
        credentials=("PRIVATE_TOKEN",),
        model_profile=tmp_path / "missing.json",
    )
    by_name = {row["name"]: row for row in rows}
    failed = {
        "nvidia_driver",
        "docker",
        "nvidia_container_toolkit",
        "disk",
        "endpoint-profile:missing.json",
    }

    assert failed <= by_name.keys()
    assert all(
        by_name[name]["status"] == "failed"
        and by_name[name]["detected"]
        and by_name[name]["required"]
        and by_name[name]["remediation"]
        for name in failed
    )
    assert by_name["credential:PRIVATE_TOKEN"]["status"] == "passed"
    diagnostic_names = ("nvidia_driver", "docker", "nvidia_container_toolkit")
    expected = (
        ("driver failure on stdout", "docker not found", "docker not found")
        if command_failure else ("570.1", "23.0", "{}")
    )
    assert tuple(by_name[name]["detected"] for name in diagnostic_names) == expected
    assert by_name["nvidia_driver"]["remediation"].startswith(
        "Run `nvidia-smi " if command_failure else "Upgrade the driver:",
    )
    assert by_name["docker"]["remediation"].startswith("Install Docker 24 or newer")
    assert (
        "Install the NVIDIA" in by_name["nvidia_container_toolkit"]["remediation"]
    ) != command_failure
    assert by_name["credential:PRIVATE_TOKEN"]["remediation"] == ""
    assert "super-secret-value" not in json.dumps(rows)
    assert all("ok" not in row for row in rows)


def test_ports_use_passive_tcp_udp_inspection_and_persist_is_unverified(
    tmp_path, monkeypatch,
) -> None:
    _contract(tmp_path, {})
    config = tmp_path / "hub.yaml"
    config.write_text("lk_port_ws: 7890\nlk_port_udp: 7892\n", encoding="utf-8")
    processes = (
        Process("hub", ".", "device_io_hub", config=config),
        Process("persisted", ".", "model", port=8100, launch_mode="persist"),
        Process("reused", ".", "model", port=8101, launch_mode="reuse"),
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(_preflight.shutil, "which", lambda _name: None)
    assert _preflight._run(["missing-tool"]) == (False, "missing-tool not found")
    monkeypatch.setattr(_preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "580.1\n", "")
        if command[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(command, 0, "24.0\n", "")
        if "-lnup" in command:
            raise subprocess.TimeoutExpired(command, 5)
        if command[-1] == "sport = :7890":
            return subprocess.CompletedProcess(command, 0, "", "harmless warning\n")
        if command[-1] == "sport = :7881":
            return subprocess.CompletedProcess(command, 2, "", "diagnostic failure")
        return subprocess.CompletedProcess(command, 0, "listener\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert [row["status"] for row in _preflight.preflight(
        processes, tmp_path, runtime=False,
    )] == ["passed", "passed"]
    assert [command[0] for command in commands] == ["nvidia-smi", "docker"]
    commands.clear()

    rows = _preflight.preflight(processes, tmp_path)
    by_name = {row["name"]: row for row in rows}
    assert by_name["port:hub:tcp:7890"]["status"] == "passed"
    tcp = by_name["port:hub:tcp:7881"]
    assert tcp["status"] == "failed" and tcp["detected"] == "diagnostic failure"
    udp = by_name["port:hub:udp:7892"]
    assert udp["status"] == "failed"
    assert udp["detected"] == "timed out after 5s"
    assert "Run `ss -H -lnup 'sport = :7892'` manually" in udp["remediation"]
    persisted = by_name["port:persisted:tcp:8100"]
    assert persisted["status"] == "skipped"
    assert "ownership and readiness unverified" in persisted["detected"]
    assert "port:reused:tcp:8101" not in by_name
    assert commands[2:] == [
        ["ss", "-H", "-lntp", "sport = :7890"],
        ["ss", "-H", "-lntp", "sport = :7881"],
        ["ss", "-H", "-lnup", "sport = :7892"],
        ["ss", "-H", "-lntp", "sport = :8100"],
    ]


def test_contractless_owned_process_port_conflict_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        _preflight,
        "_run",
        lambda command: (True, "LISTEN users:((service,pid=42))")
        if command[0] == "ss" else pytest.fail(f"unexpected command: {command}"),
    )

    rows = _preflight.preflight((Process("service", ".", "service", port=9000),), tmp_path)

    assert rows == [_preflight._row(
        "port:service:tcp:9000",
        False,
        "listening (LISTEN users:((service,pid=42)))",
        "available tcp port",
        "Stop the listener on this port, or change the port in this service's config.",
    )]


def test_component_selection_controls_docker_checks(tmp_path, monkeypatch) -> None:
    pip_config = tmp_path / "pip.yaml"
    pip_config.write_text("vllm_backend: pip\n", encoding="utf-8")
    profile = tmp_path / "models.json"
    profile.write_text(json.dumps({"vlm": {
        "base_url": "https://models.example",
        "readiness": "health",
        "deployment": {"ownership": "managed", "service": "vlm"},
    }}), encoding="utf-8")
    commands: list[list[str]] = []

    def run(command, _timeout=5):
        commands.append(command)
        if command[0] == "nvidia-smi":
            return True, "580.1"
        if command[:2] == ["docker", "version"]:
            return True, "24.0"
        if command[0] == "ss":
            return True, ""
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(_preflight, "_run", run)
    monkeypatch.setattr(
        _endpoints._OPENER, "open",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(status=200)),
    )
    pip_rows = _preflight.preflight(
        (Process("vlm", ".", "vlm_server", config=pip_config),),
        tmp_path,
    )
    assert [row["name"] for row in pip_rows] == ["nvidia_driver"]
    assert commands == [[
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ]]

    commands.clear()
    hosted_rows = _preflight.preflight(
        (
            Process("hub", ".", "device_io_hub"),
            Process("worker", ".", "worker"),
            Process("vlm", ".", "external", launch_mode="reuse"),
        ),
        tmp_path,
        model_profile=profile,
    )
    by_name = {row["name"]: row for row in hosted_rows}
    assert set(by_name) == {
        "nvidia_driver", "docker", "video_codecs", "endpoint:vlm",
        "port:hub:tcp:7880", "port:hub:tcp:7881", "port:hub:udp:7882",
    }
    assert by_name["video_codecs"]["status"] == "skipped"
    assert by_name["endpoint:vlm"]["status"] == "passed"
    assert [command[:2] for command in commands[:2]] == [
        ["nvidia-smi", "--query-gpu=driver_version"],
        ["docker", "version"],
    ]
    assert [command[-1] for command in commands[2:]] == [
        "sport = :7880", "sport = :7881", "sport = :7882",
    ]


@pytest.mark.parametrize("command", [
    "vlm_server",
    "embedding_server",
    "nemotron_omni_llm_server",
    "nemotron3_nano_llm_server",
    "llama_nemotron_llm_server",
])
def test_vllm_wrapper_default_backend_requires_only_driver(
    tmp_path, monkeypatch, command,
) -> None:
    config = tmp_path / "model.yaml"
    config.write_text("port: 8100\n", encoding="utf-8")
    commands = []

    def run(argv, _timeout=5):
        commands.append(argv)
        return True, "580.1"

    monkeypatch.setattr(_preflight, "_run", run)

    rows = _preflight.preflight(
        (Process("model", ".", command, config=config),),
        tmp_path,
        runtime=False,
    )

    assert [(row["name"], row["status"]) for row in rows] == [
        ("nvidia_driver", "passed"),
    ]
    assert [argv[0] for argv in commands] == ["nvidia-smi"]


@pytest.mark.parametrize(("driver", "docker", "status"), [
    ("581.0", "25.0", "passed"),
    ("579.99", "23.99", "failed"),
])
def test_launcher_host_minimums_apply_without_contract(
    tmp_path, monkeypatch, driver, docker, status,
) -> None:
    def run(command, _timeout=5):
        return True, driver if command[0] == "nvidia-smi" else docker

    monkeypatch.setattr(_preflight, "_run", run)

    rows = _preflight.preflight(
        (Process("hub", ".", "device_io_hub"),), tmp_path, runtime=False,
    )

    assert [(row["name"], row["status"]) for row in rows] == [
        ("nvidia_driver", status),
        ("docker", status),
    ]


def test_reused_and_cpu_only_processes_omit_host_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        _preflight,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("no host check selected"),
    )
    processes = (
        Process("worker", ".", "worker"),
        Process("reused-hub", ".", "device_io_hub", launch_mode="reuse"),
        Process("reused-model", ".", "nim_server", launch_mode="reuse"),
    )

    assert _preflight.preflight(processes, tmp_path, runtime=False) == []


def test_disk_is_omitted_without_a_declared_prepare_cache(tmp_path) -> None:
    _contract(tmp_path, {"disk_gb_free": 60})
    config = tmp_path / "worker.yaml"
    config.write_text("port: 8000\n", encoding="utf-8")
    reused = tmp_path / "reused.yaml"
    reused.write_text("model_cache: missing\n", encoding="utf-8")
    for processes in (
        (Process("worker", ".", "worker", config=config, prepare=True),),
        (
            Process("owned", ".", "worker", config=reused),
            Process(
                "reused", ".", "worker", config=reused,
                prepare=True, launch_mode="reuse",
            ),
        ),
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
) -> None:
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
    monkeypatch.setattr(
        _endpoints._OPENER,
        "open",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(status=200)),
    )
    monkeypatch.setattr(
        _endpoints.socket,
        "create_connection",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "iterdir", lambda path: iter((path / "cached",)))

    def run(command, _timeout=5):
        if command[0] == "nvidia-smi":
            return True, "580.1"
        if command[:2] == ["docker", "version"]:
            return True, "24.0"
        if command[:2] == ["docker", "info"]:
            return True, '{"nvidia": {}}'
        if command[0] == "ss":
            return True, ""
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(_preflight, "_run", run)
    rows = _preflight.preflight(
        processes,
        module._BASE,
        model_profile=(
            module._resolve_worker_path("models_config")
            if "tea-making" in relative else module._BASE / "yaml/models.json"
        ) if relative.startswith("agent-samples/") else None,
    )

    statuses = {row["name"]: row["status"] for row in rows}
    expected_skipped = {
        name for name in statuses if name.startswith("endpoint:")
    }
    if relative.startswith("agent-samples/"):
        expected_skipped.add("video_codecs")
    assert expected_skipped <= statuses.keys()
    assert statuses == {
        name: "skipped" if name in expected_skipped else "passed"
        for name in statuses
    }
    assert "disk" not in statuses
    if relative.startswith("agent-samples/"):
        assert any(name.startswith("endpoint:") for name in statuses)
        assert {"nvidia_driver", "docker", "video_codecs"} <= statuses.keys()


def test_missing_contract_has_only_selected_component_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        _preflight,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("no component selected; command must not run"),
    )
    assert _preflight.preflight((), tmp_path) == []
    profile = tmp_path / "missing.json"
    assert _preflight.preflight((), tmp_path, runtime=False, model_profile=profile) == []
    row = _preflight.preflight((), tmp_path, model_profile=profile)[0]
    assert row["name"] == "endpoint-profile:missing.json"
    assert row["status"] == "failed"
