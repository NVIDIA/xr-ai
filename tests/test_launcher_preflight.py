# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest
import xr_ai_launcher._preflight as _preflight
from jsonschema import Draft202012Validator
from xr_ai_launcher._models import EndpointProbe
from xr_ai_launcher._stack import Process

_REAL_PORT_IS_FREE = _preflight._port_is_free
_REAL_EPHEMERAL_PORT_CHECKS = _preflight._ephemeral_port_checks


def _inspection(
    state: _preflight._PortInspectionState,
    evidence: str = "",
    *,
    remediation: str = "",
    error_kind: _preflight._PortInspectionErrorKind | None = None,
) -> _preflight._PortInspection:
    return _preflight._PortInspection(
        state=state,
        evidence=evidence,
        remediation=remediation,
        error_kind=error_kind,
    )


@pytest.fixture(autouse=True)
def _isolate_host_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_preflight, "load_credentials", lambda: None)
    monkeypatch.setattr(
        _preflight, "_port_is_free", lambda *_args: _inspection("clear")
    )
    monkeypatch.setattr(_preflight, "_ephemeral_port_checks", lambda _services: [])
    for name in (
        "HF_TOKEN",
        "NGC_API_KEY",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def no_port_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _preflight.socket,
        "socket",
        lambda *_args, **_kwargs: pytest.fail("port inspection must not create sockets"),
    )


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _result(result: _preflight.PreflightResult, name: str) -> _preflight.CheckResult:
    return next(check for check in result.checks if check.name == name)


def _load_sample_main(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_overlay_merges_objects_and_replaces_lists(tmp_path: Path) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "python": ">=3.11,<3.13",
            "disk_gb_free": {
                "config_keys": ["model_cache", "nim_cache"],
                "runtime_gb": 50,
                "preparation": True,
            },
            "commands": ["uv", "npm"],
        },
    )
    _write(
        tmp_path / "requirements.hosted.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache", "nim_cache"],
                "runtime_gb": 5,
            },
            "commands": ["uv"],
        },
    )

    loaded, files = _preflight.load_contract(contract, profile="hosted")

    assert loaded == {
        "python": ">=3.11,<3.13",
        "disk_gb_free": {
            "config_keys": ["model_cache", "nim_cache"],
            "runtime_gb": 5,
            "preparation": True,
        },
        "commands": ["uv"],
    }
    assert files == (contract, tmp_path / "requirements.hosted.json")


def test_contract_validation_reports_precise_unknown_field(tmp_path: Path) -> None:
    contract = _write(tmp_path / "requirements.json", {"dockre": "24"})

    with pytest.raises(_preflight.ContractError, match=r"unknown field\(s\): dockre"):
        _preflight.load_contract(contract)


def test_shipped_contracts_load_and_schema_fields_match_validator() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "utils/xr-ai-launcher/requirements.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)

    assert set(schema["properties"]) == _preflight._TOP_LEVEL_FIELDS
    contracts = [
        *sorted((root / "agent-samples").glob("*/requirements*.json")),
        root / "model-server-samples/model-servers/requirements.json",
    ]
    for contract in contracts:
        loaded, files = _preflight.load_contract(contract)
        assert loaded
        assert files == (contract,)
        assert list(validator.iter_errors(loaded)) == []


def test_process_config_directory_does_not_select_an_overlay(tmp_path: Path) -> None:
    contract = _write(tmp_path / "requirements.json", {"commands": ["uv"]})
    _write(tmp_path / "requirements.hosted.json", {"commands": ["missing-command"]})
    config = tmp_path / "hosted" / "worker.yaml"
    config.parent.mkdir()
    config.write_text("value: true\n", encoding="utf-8")
    process = Process("worker", ".", "worker", config=config)

    result = _preflight.preflight(contract, processes=(process,), base=tmp_path)

    assert result.profiles == ()
    assert all(check.name != "command:missing-command" for check in result.checks)


def test_arch_override_uses_the_declared_config_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"arch": {"allowed": ["x86_64"], "unless_config": "lovr_bin"}},
    )
    config = tmp_path / "worker.yaml"
    config.write_text("lovr_bin: /opt/lovr/lovr\n", encoding="utf-8")
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "aarch64")

    result = _preflight.preflight(
        contract,
        processes=(Process("worker", ".", "worker", config=config),),
        base=tmp_path,
    )

    check = _result(result, "arch")
    assert check.ok
    assert check.skipped
    assert "lovr_bin configured" in check.detected


def test_version_and_command_requirements_honor_declared_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "node": {
                "version": ">=99",
                "unless_env_truthy": "CUSTOM_NODE",
            },
            "commands": [
                {
                    "name": "optional-tool",
                    "unless_env_truthy": "OPTIONAL_TOOL_BIN",
                }
            ],
        },
    )
    monkeypatch.setenv("CUSTOM_NODE", "1")
    monkeypatch.setenv("OPTIONAL_TOOL_BIN", "1")
    monkeypatch.setattr(
        _preflight,
        "_node_version",
        lambda: pytest.fail("overridden Node must not execute"),
    )
    monkeypatch.setattr(
        _preflight.shutil,
        "which",
        lambda _name: pytest.fail("overridden command must not be resolved"),
    )

    result = _preflight.preflight(contract)

    assert result.ok
    assert _result(result, "node").skipped
    assert _result(result, "command:optional-tool").skipped


def test_missing_contract_still_checks_resolved_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        _preflight._HEALTH_OPENER, "open", lambda *_args, **_kwargs: Response(),
    )
    probe = EndpointProbe(
        name="vlm",
        health_url="https://models.example.test/health",
        ownership="external",
    )

    result = _preflight.preflight(tmp_path / "requirements.json", endpoints=(probe,))

    assert result.contract_files == ()
    assert result.ok
    assert _result(result, "endpoint:vlm").detected == (
        "https://models.example.test/health returned HTTP 204"
    )


def test_adjacent_models_json_is_discovered_and_values_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    worker = yaml_dir / "worker.yaml"
    worker.write_text("other: value\n", encoding="utf-8")
    _write(
        yaml_dir / "models.json",
        {
            "models": {
                "vlm": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": {
                        "base_url": "https://user:secret@models.test/v1",
                        "api_key_env": "MODEL_TOKEN",
                    },
                    "deployment": {"ownership": "external"},
                }
            }
        },
    )
    monkeypatch.setenv("MODEL_TOKEN", "super-secret-value")

    def unavailable(_request, *, timeout):
        assert timeout == 3.0
        raise OSError("request containing super-secret-value failed")

    monkeypatch.setattr(_preflight._HEALTH_OPENER, "open", unavailable)
    process = Process("worker", "worker", "worker", config="yaml/worker.yaml")

    result = _preflight.preflight(
        tmp_path / "requirements.json",
        processes=(process,),
        base=tmp_path,
    )

    detected = _result(result, "endpoint:vlm").detected
    assert "secret" not in detected
    assert detected == "https://models.test/v1/health unavailable (OSError)"
    assert _result(result, "env:MODEL_TOKEN").detected == "set"


@pytest.mark.parametrize(
    ("sample_name", "expected_roles"),
    [
        ("simple-vlm-example", {"stt", "vlm", "tts"}),
        ("xr-render-demo", {"llm", "agent_llm", "stt", "vlm", "tts"}),
    ],
)
def test_shipped_consumer_samples_return_preflight_reports(
    sample_name: str,
    expected_roles: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    sample_root = root / "agent-samples" / sample_name
    sample = _load_sample_main(
        f"launcher_preflight_{sample_name.replace('-', '_')}",
        sample_root / "main.py",
    )
    processes = (
        sample.PROCESSES
        if sample_name == "simple-vlm-example"
        else sample._build_processes()
    )
    monkeypatch.setattr(_preflight, "_driver_version", lambda: ("580.1", None))
    monkeypatch.setattr(_preflight, "_node_version", lambda: "20.19.0")
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        _preflight.ctypes.util, "find_library", lambda name: f"lib{name}.so",
    )
    monkeypatch.setattr(
        _preflight.shutil, "which", lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        _preflight,
        "_probe_endpoint",
        lambda probe: _preflight.CheckResult(
            f"endpoint:{probe.role or probe.name}",
            True,
            "reachable",
            "reachable",
            "",
        ),
    )
    monkeypatch.setattr(
        _preflight,
        "_expensive_checks",
        lambda names, **_kwargs: [
            _preflight.CheckResult(
                name, True, "available", "available", "", tier="expensive",
            )
            for name in names
        ],
    )

    result = _preflight.preflight(
        sample_root / "requirements.json",
        processes=processes,
        base=sample_root,
    )

    assert isinstance(result, _preflight.PreflightResult)
    assert result.ok
    assert {
        service.name
        for service in result.services
        if service.ownership == "external"
    } == expected_roles


def test_hosted_profile_skips_local_machine_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "nvidia_driver": "580",
            "docker": "24",
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            },
        },
    )
    profile = _write(
        tmp_path / "models.hosted.json",
        {
            "models": {
                "vlm": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": {
                        "base_url": "https://models.test",
                        "readiness": "none",
                    },
                    "deployment": {"ownership": "external"},
                }
            }
        },
    )
    deployment = _preflight.ModelDeployment(
        profile,
        {},
        (),
        (EndpointProbe(
            name="vlm",
            health_url="https://models.test/health",
            ownership="external",
            readiness="none",
        ),),
    )
    monkeypatch.setattr(
        _preflight,
        "_driver_version",
        lambda: pytest.fail("host driver must not be checked for an external-only profile"),
    )

    result = _preflight.preflight(contract, deployment=deployment)

    assert result.ok
    assert {check.name for check in result.checks} == {"endpoint:vlm"}
    assert result.services[0].ownership == "external"


def test_owned_healthy_responder_is_conflict_without_identity_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "vlm", "port": 8100, "health": "/health"}]},
    )
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible tcp socket"),
    )
    process = Process("vlm", ".", "vlm_server", port=8100)

    result = _preflight.preflight(contract, processes=(process,), base=tmp_path)

    check = _result(result, "port:vlm:tcp:8100")
    assert not check.ok
    assert check.detected.endswith("; ownership unverified")


def test_owned_port_accepts_explicit_compatible_identity_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "vlm", "port": 8100}]},
    )
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible tcp socket"),
    )
    cudnn = tmp_path / "host-cudnn"
    keep = tmp_path / "other-libs"
    cudnn.mkdir()
    keep.mkdir()
    (cudnn / "libcudnn.so.9").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{cudnn}:{keep}")
    probe_envs: list[dict[str, str]] = []
    process = Process(
        "vlm", ".", "vlm_server", gpu="3", port=8100,
        ownership_probe=lambda env: probe_envs.append(dict(env)) or True,
    )

    result = _preflight.preflight(contract, processes=(process,), base=tmp_path)

    check = _result(result, "port:vlm:tcp:8100")
    assert check.ok
    assert check.detected == "occupied by the expected managed service"
    assert probe_envs[0]["CUDA_VISIBLE_DEVICES"] == "3"
    assert probe_envs[0]["LD_LIBRARY_PATH"] == str(keep)


def test_legacy_managed_container_reports_targeted_restart_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "vlm", "port": 8100}]},
    )
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible tcp socket"),
    )

    def reject_legacy(_env: object) -> bool:
        raise _preflight.OwnershipProbeMismatch(
            "legacy managed container 'xr-ai-vllm-vlm' lacks launch identity labels",
            "Stop and restart managed container 'xr-ai-vllm-vlm', then rerun.",
        )

    result = _preflight.preflight(
        contract,
        processes=(
            Process(
                "vlm", ".", "vlm_server", port=8100,
                ownership_probe=reject_legacy,
            ),
        ),
        base=tmp_path,
    )

    check = _result(result, "port:vlm:tcp:8100")
    assert not check.ok
    assert "legacy managed container" in check.detected
    assert check.remediation == (
        "Stop and restart managed container 'xr-ai-vllm-vlm', then rerun."
    )


def test_process_port_and_configured_endpoint_override_contract_port(tmp_path: Path) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "ports": [
                {"name": "vlm", "port": 8100},
                {"name": "scene", "port": 8320},
            ]
        },
    )
    scene_config = tmp_path / "scene.yaml"
    scene_config.write_text("endpoint: 'tcp://0.0.0.0:9320'\n", encoding="utf-8")
    processes = (
        Process("vlm", ".", "vlm", port=9100),
        Process("scene", ".", "scene", config=scene_config),
    )

    loaded, _ = _preflight.load_contract(contract)
    services = _preflight._resolve_services(
        loaded, processes, (), tmp_path, suppress_unprofiled_reuse=False,
    )

    assert {service.name: service.port for service in services} == {
        "vlm": 9100,
        "scene": 9320,
    }


@pytest.mark.parametrize(
    ("first_host", "second_host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
    ],
)
def test_resolved_owned_bind_conflicts_fail_before_host_probes(
    first_host: str,
    second_host: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "ports": [
                {
                    "name": "first",
                    "port": 8100,
                    "config_key": "port",
                    "bind_config_key": "host",
                },
                {
                    "name": "second",
                    "port": 8200,
                    "config_key": "port",
                    "bind_config_key": "host",
                },
            ]
        },
    )
    first_config = tmp_path / "first.yaml"
    first_config.write_text(
        f"port: 9100\nhost: '{first_host}'\n", encoding="utf-8",
    )
    second_config = tmp_path / "second.yaml"
    second_config.write_text(
        f"port: 9100\nhost: '{second_host}'\n", encoding="utf-8",
    )
    probe_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *args: probe_calls.append(args) or _inspection("clear"),
    )

    result = _preflight.preflight(
        contract,
        processes=(
            Process("first", ".", "first", config=first_config),
            Process("second", ".", "second", config=second_config),
        ),
        base=tmp_path,
    )

    conflicts = [
        check for check in result.checks if check.name.startswith("port_conflict:")
    ]
    assert not result.ok
    assert len(conflicts) == 1
    assert "first binds" in conflicts[0].detected
    assert "second binds" in conflicts[0].detected
    assert probe_calls == [
        (9100, "tcp", first_host),
        (9100, "tcp", second_host),
    ]


def test_native_profile_keeps_hub_server_and_skips_cloudxr_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "ports": [
                {
                    "name": "hub",
                    "port": 8080,
                    "config_key": "web_server_port",
                    "enabled_config_key": "enable_web_server",
                },
                {
                    "name": "cloudxr",
                    "port": 48322,
                    "unless_env_truthy": "DEVICE_IO_HUB_NO_WEB_CLIENT",
                },
                {"name": "cloudxr", "port": 49100},
            ]
        },
    )
    hub_config = tmp_path / "hub.yaml"
    hub_config.write_text(
        "enable_web_server: true\nweb_server_port: '9080'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEVICE_IO_HUB_NO_WEB_CLIENT", "yes")
    processes = (
        Process("hub", ".", "hub", config=hub_config),
        Process("cloudxr", ".", "cloudxr"),
    )

    loaded, _ = _preflight.load_contract(contract)
    services = _preflight._resolve_services(
        loaded, processes, (), tmp_path, suppress_unprofiled_reuse=False,
    )

    assert {(service.name, service.port) for service in services} == {
        ("hub", 9080),
        ("cloudxr", 49100),
    }


def test_disabled_optional_hub_servers_do_not_require_ports(tmp_path: Path) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "ports": [
                {
                    "name": "hub",
                    "port": 8000,
                    "enabled_config_key": "enable_token_server",
                },
                {
                    "name": "hub",
                    "port": 8080,
                    "enabled_config_key": "enable_web_server",
                },
            ]
        },
    )
    hub_config = tmp_path / "hub.yaml"
    hub_config.write_text(
        "enable_token_server: false\nenable_web_server: 'off'\n",
        encoding="utf-8",
    )
    process = Process("hub", ".", "hub", config=hub_config)

    loaded, _ = _preflight.load_contract(contract)
    services = _preflight._resolve_services(
        loaded, (process,), (), tmp_path, suppress_unprofiled_reuse=False,
    )

    assert services == ()
    assert _REAL_EPHEMERAL_PORT_CHECKS(
        services,
        port_range_path=tmp_path / "missing-range",
        reserved_ports_path=tmp_path / "missing-reservations",
    ) == []


@pytest.mark.parametrize(
    ("port", "reserved", "warns"),
    [
        (40000, "", True),
        (50000, "", True),
        (39999, "", False),
        (50001, "", False),
        (45000, "45000", False),
        (45000, "44999-45001", False),
        (45000, "41000,43000-45000,47000", False),
    ],
    ids=(
        "lower-bound", "upper-bound", "below", "above",
        "reserved-single", "reserved-range", "reserved-mixed-range-boundary",
    ),
)
def test_ephemeral_port_warnings_follow_range_and_reservations(
    port: int,
    reserved: str,
    warns: bool,
    tmp_path: Path,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    reservations = tmp_path / "ip_local_reserved_ports"
    port_range.write_text("40000 50000\n", encoding="utf-8")
    reservations.write_text(reserved, encoding="utf-8")

    checks = _REAL_EPHEMERAL_PORT_CHECKS(
        (_preflight.ResolvedService("service", "own", port=port),),
        port_range_path=port_range,
        reserved_ports_path=reservations,
    )

    assert bool(checks) is warns
    if warns:
        assert checks[0].name == f"ephemeral_port:service:tcp:{port}"
        assert checks[0].ok
        assert checks[0].status == "warning"
        assert "preserve every existing" in checks[0].remediation.lower()
        assert "replaces the complete list" in checks[0].remediation
        assert "does not release an existing connection" in checks[0].remediation


def test_ephemeral_port_warning_applies_only_to_resolved_owned_tcp_and_udp(
    tmp_path: Path,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    reservations = tmp_path / "ip_local_reserved_ports"
    port_range.write_text("40000 50000\n", encoding="utf-8")
    reservations.write_text("", encoding="utf-8")
    services = (
        _preflight.ResolvedService("tcp", "own", port=45000),
        _preflight.ResolvedService("udp", "own", port=45001, proto="udp"),
        _preflight.ResolvedService("reused", "reuse", port=45002),
        _preflight.ResolvedService("external", "external", port=45003),
    )

    checks = _REAL_EPHEMERAL_PORT_CHECKS(
        services,
        port_range_path=port_range,
        reserved_ports_path=reservations,
    )

    assert {check.name for check in checks} == {
        "ephemeral_port:tcp:tcp:45000",
        "ephemeral_port:udp:udp:45001",
    }


def test_ephemeral_port_policy_read_failure_is_one_nonblocking_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    reservations = tmp_path / "ip_local_reserved_ports"
    reservations.write_text("", encoding="utf-8")
    original_read_text = Path.read_text

    def read_text(path: Path, *args, **kwargs) -> str:
        if path == port_range:
            raise PermissionError(13, "Permission denied", str(path))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    checks = _REAL_EPHEMERAL_PORT_CHECKS(
        (
            _preflight.ResolvedService("first", "own", port=45000),
            _preflight.ResolvedService("second", "own", port=45001),
        ),
        port_range_path=port_range,
        reserved_ports_path=reservations,
    )

    assert len(checks) == 1
    assert checks[0].name == "ephemeral_ports"
    assert checks[0].ok
    assert checks[0].status == "warning"
    assert "Permission denied" in checks[0].detected
    assert "sysctl net.ipv4.ip_local_port_range" in checks[0].remediation


@pytest.mark.parametrize(
    ("port_range_text", "reserved_text"),
    [
        ("40000\n", ""),
        ("40000 50000\n", "45000-bad"),
        ("40000 50000\n", "45001-45000"),
    ],
)
def test_malformed_ephemeral_port_policy_is_one_nonblocking_warning(
    port_range_text: str,
    reserved_text: str,
    tmp_path: Path,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    reservations = tmp_path / "ip_local_reserved_ports"
    port_range.write_text(port_range_text, encoding="utf-8")
    reservations.write_text(reserved_text, encoding="utf-8")

    checks = _REAL_EPHEMERAL_PORT_CHECKS(
        (_preflight.ResolvedService("service", "own", port=45000),),
        port_range_path=port_range,
        reserved_ports_path=reservations,
    )

    assert len(checks) == 1
    assert checks[0].name == "ephemeral_ports"
    assert checks[0].status == "warning"
    assert "could not inspect host ephemeral port policy" in checks[0].detected


def test_ephemeral_port_warning_does_not_gate_expensive_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    reservations = tmp_path / "ip_local_reserved_ports"
    port_range.write_text("40000 50000\n", encoding="utf-8")
    reservations.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        _preflight,
        "_ephemeral_port_checks",
        lambda services: _REAL_EPHEMERAL_PORT_CHECKS(
            services,
            port_range_path=port_range,
            reserved_ports_path=reservations,
        ),
    )
    monkeypatch.setattr(_preflight.ctypes.util, "find_library", lambda _name: "vulkan")
    expensive_calls: list[tuple[tuple[str, ...], bool]] = []

    def expensive(names, *, force):
        expensive_calls.append((tuple(names), force))
        return [
            _preflight.CheckResult(
                "vulkan_device", True, "usable", "usable", "", tier="expensive",
            )
        ]

    monkeypatch.setattr(_preflight, "_expensive_checks", expensive)
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "service", "port": 45000}], "vulkan": True},
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("service", ".", "service", port=45000),),
        base=tmp_path,
    )

    warning = _result(result, "ephemeral_port:service:tcp:45000")
    assert warning.ok
    assert warning.status == "warning"
    assert result.ok
    assert expensive_calls == [(('vulkan_device',), False)]


def test_reused_endpoint_failure_has_shared_stack_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _preflight._LOCAL_HEALTH_OPENER,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("refused")),
    )
    probe = EndpointProbe(
        name="stt",
        health_url="http://127.0.0.1:8103/health",
        ownership="reuse",
    )

    result = _preflight.preflight(tmp_path / "requirements.json", endpoints=(probe,))

    check = _result(result, "endpoint:stt")
    assert not check.ok
    assert "shared model service 'stt'" in check.remediation
    assert "#preflight-reused-service" in check.remediation


def test_health_redirects_are_not_followed_with_bearer_credentials() -> None:
    handler = _preflight._NoRedirect()

    redirected = handler.redirect_request(
        object(), object(), 302, "Found", {}, "https://other.example/health",
    )

    assert redirected is None


def test_http_probe_sends_bearer_token_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def open_request(request, *, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = str(timeout)
        return Response()

    monkeypatch.setenv("MODEL_TOKEN", "test-secret")
    monkeypatch.setattr(_preflight._HEALTH_OPENER, "open", open_request)

    result = _preflight._probe_endpoint(
        EndpointProbe(
            name="hosted",
            health_url="https://models.example.test/health?token=also-secret",
            ownership="external",
            api_key_env="MODEL_TOKEN",
        )
    )

    assert result.ok
    assert captured == {"authorization": "Bearer test-secret", "timeout": "3.0"}
    assert "test-secret" not in result.detected
    assert "also-secret" not in result.detected


def test_tcp_probe_uses_configured_host_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[tuple[tuple[str, int], float]] = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def connect(address, *, timeout):
        connections.append((address, timeout))
        return Connection()

    monkeypatch.setattr(_preflight.socket, "create_connection", connect)

    result = _preflight._probe_endpoint(
        EndpointProbe(
            name="scene",
            health_url="tcp://[::1]:8320",
            ownership="external",
            timeout=1.5,
        )
    )

    assert result.ok
    assert connections == [(('::1', 8320), 1.5)]
    assert result.detected == "tcp://[::1]:8320 accepted a TCP connection"


def test_malformed_health_endpoint_fails_unless_readiness_is_disabled() -> None:
    enabled = _preflight._probe_endpoint(EndpointProbe(
        name="vlm",
        health_url="localhost:8100",
        ownership="external",
    ))
    disabled = _preflight._probe_endpoint(EndpointProbe(
        name="vlm",
        health_url="localhost:8100",
        ownership="external",
        readiness="none",
    ))

    assert not enabled.ok
    assert not enabled.skipped
    assert enabled.detected == "localhost:8100 is not a usable health endpoint"
    assert disabled.ok
    assert disabled.skipped


def test_ipv6_url_redaction_preserves_brackets_and_removes_secrets() -> None:
    redacted = _preflight._redact_url(
        "http://user:password@[2001:db8::7]:8100/health?token=secret#fragment"
    )

    assert redacted == "http://[2001:db8::7]:8100/health"


@pytest.mark.parametrize(
    ("detected", "constraint", "minimum", "expected"),
    [
        ("Python 3.12.10", ">=3.11,<3.13", False, True),
        ("Docker version 24.0.0", 24, True, True),
        ("v20.18.1", ">=20.19.0", False, False),
        ("580.2", "!=580.2", False, False),
        ("not a version", ">=1", False, False),
    ],
)
def test_version_constraints(
    detected: str, constraint: object, minimum: bool, expected: bool,
) -> None:
    assert _preflight._version_satisfies(
        detected, constraint, minimum=minimum
    ) is expected


@pytest.mark.parametrize(
    (
        "proto", "host", "ipv4_output", "ipv6_output", "expected_state",
        "expected_families",
    ),
    [
        (
            "tcp", "0.0.0.0",
            "LISTEN 0 4096 192.0.2.30:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "192.0.2.10",
            "LISTEN 0 4096 192.0.2.11:8100 0.0.0.0:*\n", "",
            "clear", ("-4", "-6"),
        ),
        (
            "tcp", "192.0.2.10",
            "LISTEN 0 4096 192.0.2.10:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "192.0.2.10",
            "LISTEN 0 4096 0.0.0.0:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "2001:db8::10",
            "LISTEN 0 4096 0.0.0.0:8100 0.0.0.0:*\n", "",
            "clear", ("-6",),
        ),
        (
            "tcp", "0.0.0.0", "",
            "LISTEN 0 4096 [::]:8100 [::]:*\n",
            "clear", ("-4", "-6"),
        ),
        (
            "tcp", "0.0.0.0", "",
            "LISTEN 0 4096 [::1]:8100 [::]:*\n",
            "clear", ("-4", "-6"),
        ),
        (
            "tcp", "192.0.2.10", "",
            "LISTEN 0 4096 [::ffff:192.0.2.10]:8100 [::]:*\n",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "127.0.0.1",
            "LISTEN 0 4096 127.0.0.1%lo:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "2001:db8::10", "",
            "LISTEN 0 4096 [2001:db8::10]:8100 [::]:*\n",
            "occupied", ("-6",),
        ),
        (
            "tcp", "2001:db8::10", "",
            "LISTEN 0 4096 [2001:db8::11]:8100 [::]:*\n",
            "clear", ("-6",),
        ),
        (
            "tcp", "2001:db8::10", "",
            "LISTEN 0 4096 *:8100 *:*\n",
            "occupied", ("-6",),
        ),
        (
            "tcp", "0.0.0.0", "",
            "LISTEN 0 4096 *:8100 *:*\n",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "0.0.0.0", "",
            "LISTEN 0 4096 :::8100 :::*\n",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "0.0.0.0", "",
            "LISTEN 0 4096 [::ffff:192.0.2.10]:8100 [::]:*\n",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "192.0.2.10",
            "LISTEN 0 4096 *:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "tcp", "fe80::1%eth0", "",
            "LISTEN 0 4096 [fe80::1]%eth0:8100 [::]:*\n",
            "occupied", ("-6",),
        ),
        (
            "tcp", "2001:db8::10", "",
            "LISTEN 0 4096 *%eth0:8100 *:*\n",
            "occupied", ("-6",),
        ),
        (
            "tcp", "192.0.2.10",
            (
                "LISTEN 0 4096 192.0.2.11:8100 0.0.0.0:*\n"
                "LISTEN 0 4096 192.0.2.10:8100 0.0.0.0:*\n\n"
            ),
            "",
            "occupied", ("-4", "-6"),
        ),
        (
            "udp", "192.0.2.10",
            "ESTAB 0 0 192.0.2.10:8100 192.0.2.20:53000\n", "",
            "occupied", ("-4", "-6"),
        ),
        (
            "udp", "0.0.0.0",
            "UNCONN 0 0 192.0.2.30:8100 0.0.0.0:*\n", "",
            "occupied", ("-4", "-6"),
        ),
    ],
    ids=(
        "wildcard-sees-nonloopback", "disjoint-ipv4-specific",
        "same-ipv4-specific", "ipv4-wildcard-listener",
        "ipv6-target-ignores-ipv4", "ipv6-only-wildcard-allows-ipv4",
        "ipv6-loopback-allows-ipv4", "matching-ipv4-mapped", "scoped-ipv4",
        "same-ipv6-specific",
        "disjoint-ipv6-specific", "ipv6-star-against-ipv6",
        "ipv6-star-against-ipv4", "bracketless-ipv6-wildcard",
        "wildcard-ipv4-against-mapped", "ipv4-star", "scoped-ipv6",
        "scoped-ipv6-star", "two-rows-with-trailing-blank",
        "connected-udp", "unconnected-udp",
    ),
)
def test_port_inspection_detects_visible_socket_conflicts(
    proto: str,
    host: str,
    ipv4_output: str,
    ipv6_output: str,
    expected_state: str,
    expected_families: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    no_port_sockets: None,
) -> None:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        command = list(command)
        commands.append(command)
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": 2.0,
        }
        output = ipv4_output if "-4" in command else ipv6_output
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(_preflight.subprocess, "run", run)
    monkeypatch.setattr(
        _preflight.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("numeric hosts must not be resolved"),
    )

    inspection = _REAL_PORT_IS_FREE(8100, proto, host)

    assert inspection.state == expected_state
    if expected_state == "clear":
        assert inspection.detected == "no visible socket conflict"
    else:
        assert inspection.detected.startswith("occupied (")
    expected_commands = [
        [
            "ss", "-H", "-n", family,
            "-l" if proto == "tcp" else "-a",
            "-t" if proto == "tcp" else "-u",
            "sport = :8100",
        ]
        for family in expected_families
    ]
    assert commands == expected_commands


def test_port_inspection_resolves_hostname_to_numeric_bind_address(
    monkeypatch: pytest.MonkeyPatch,
    no_port_sockets: None,
) -> None:
    commands: list[list[str]] = []
    resolutions: list[tuple[str, object, int]] = []

    def run(command, **_kwargs):
        command = list(command)
        commands.append(command)
        output = (
            "LISTEN 0 4096 192.0.2.10:8100 0.0.0.0:*\n"
            if "-4" in command else ""
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    def resolve(host, port, family):
        resolutions.append((host, port, family))
        return [
            (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.10", 0))
        ]

    monkeypatch.setattr(_preflight.subprocess, "run", run)
    monkeypatch.setattr(_preflight.socket, "getaddrinfo", resolve)

    inspection = _REAL_PORT_IS_FREE(8100, "tcp", "service.internal")

    assert inspection.state == "occupied"
    assert inspection.detected.startswith("occupied (")
    assert resolutions == [("service.internal", None, socket.AF_INET)]
    assert commands == [
        ["ss", "-H", "-n", "-4", "-l", "-t", "sport = :8100"],
        ["ss", "-H", "-n", "-6", "-l", "-t", "sport = :8100"],
    ]


@pytest.mark.parametrize(
    ("failure", "proto", "error_kind", "expected", "remediation"),
    [
        ("missing", "tcp", "missing", "ss not found", "iproute2"),
        (
            "timeout", "tcp", "timeout", "ss timed out after 2 seconds",
            "diagnose the timeout",
        ),
        ("nonzero", "tcp", "command", "permission denied", "same login session"),
        (
            "permission", "tcp", "command", "ss failed: permission denied",
            "same login session",
        ),
        ("malformed", "tcp", "malformed", "malformed ss output", "`ss -V`"),
        ("malformed-ipv6", "tcp", "malformed", "not-an-ip", "`ss -V`"),
        ("wrong-family", "tcp", "malformed", "[::1]:8100", "`ss -V`"),
        ("wrong-port", "tcp", "malformed", "127.0.0.1:8101", "`ss -V`"),
        ("invalid-udp-state", "udp", "malformed", "close", "`ss -V`"),
        ("tcp-non-listen", "tcp", "malformed", "estab", "`ss -V`"),
        ("stderr", "tcp", "command", "unexpected warning", "same login session"),
        ("stderr-long", "tcp", "command", "ss reported", "same login session"),
    ],
)
def test_port_inspection_fails_closed_when_ss_is_unreliable(
    failure: str,
    proto: str,
    error_kind: str,
    expected: str,
    remediation: str,
    monkeypatch: pytest.MonkeyPatch,
    no_port_sockets: None,
) -> None:
    def run(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("ss")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "nonzero":
            return subprocess.CompletedProcess(command, 1, "", "permission denied")
        if failure == "permission":
            raise PermissionError(13, "Permission denied")
        if failure == "malformed":
            return subprocess.CompletedProcess(command, 0, "LISTEN 0 broken\n", "")
        if failure == "malformed-ipv6":
            output = "LISTEN 0 4096 [not-an-ip]:8100 [::]:*\n" if "-6" in command else ""
            return subprocess.CompletedProcess(command, 0, output, "")
        if failure == "wrong-family":
            output = "LISTEN 0 4096 [::1]:8100 [::]:*\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if failure == "wrong-port":
            output = "LISTEN 0 4096 127.0.0.1:8101 0.0.0.0:*\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if failure == "invalid-udp-state":
            output = "CLOSE 0 0 127.0.0.1:8100 0.0.0.0:*\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if failure == "tcp-non-listen":
            output = "ESTAB 0 0 127.0.0.1:8100 192.0.2.10:50000\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if failure == "stderr-long":
            return subprocess.CompletedProcess(command, 0, "", "reason " + "x" * 500)
        return subprocess.CompletedProcess(command, 0, "", "unexpected warning")

    monkeypatch.setattr(_preflight.subprocess, "run", run)

    inspection = _REAL_PORT_IS_FREE(8100, proto, "127.0.0.1")

    assert inspection.state == "uninspectable"
    assert inspection.error_kind == error_kind
    assert inspection.detected.startswith("could not inspect (")
    assert expected in inspection.detected.lower()
    assert remediation in inspection.remediation
    if failure == "stderr-long":
        assert inspection.evidence.endswith("...")
        assert len(inspection.evidence) <= 220


def test_ipv6_inspection_failure_overrides_visible_ipv4_conflict(
    monkeypatch: pytest.MonkeyPatch,
    no_port_sockets: None,
) -> None:
    def run(command, **_kwargs):
        if "-4" in command:
            return subprocess.CompletedProcess(
                command, 0, "LISTEN 0 4096 127.0.0.1:8100 0.0.0.0:*\n", "",
            )
        return subprocess.CompletedProcess(command, 1, "", "IPv6 table unavailable")

    monkeypatch.setattr(_preflight.subprocess, "run", run)

    inspection = _REAL_PORT_IS_FREE(8100, "tcp", "127.0.0.1")

    assert inspection.state == "uninspectable"
    assert inspection.error_kind == "command"
    assert "ipv6 table unavailable" in inspection.detected.lower()


def test_port_inspection_reports_bind_host_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
    no_port_sockets: None,
) -> None:
    monkeypatch.setattr(
        _preflight.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            socket.gaierror(-2, "Name or service not known")
        ),
    )
    monkeypatch.setattr(
        _preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("ss must not run for an unknown host"),
    )

    inspection = _REAL_PORT_IS_FREE(8100, "tcp", "not-a-host.invalid")

    assert inspection.state == "uninspectable"
    assert inspection.error_kind == "bind_host"
    assert "could not resolve bind host 'not-a-host.invalid'" in inspection.detected
    assert "Name or service not known" in inspection.detected


@pytest.mark.parametrize(
    ("error_kind", "evidence", "outcome"),
    [
        ("missing", "ss not found", "verified"),
        ("malformed", "malformed ss output: 'bad row'", "verified"),
        ("command", "ss reported: unexpected warning", "verified"),
        ("command", "ss exited with status 1: permission denied", "verified"),
        ("missing", "ss not found", "false"),
        ("malformed", "malformed ss output: 'bad row'", "raises"),
        ("missing", "ss not found", "mismatch"),
    ],
    ids=(
        "missing-verified", "malformed-verified", "stderr-verified",
        "nonzero-verified", "false", "raises", "mismatch",
    ),
)
def test_non_timeout_inspection_failure_uses_managed_ownership_probe(
    error_kind: str,
    evidence: str,
    outcome: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ownership_calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection(
            "uninspectable",
            evidence,
            remediation=f"remediate: {evidence}",
            error_kind=error_kind,
        ),
    )

    def ownership(env: dict[str, str]) -> bool:
        ownership_calls.append(dict(env))
        if outcome == "raises":
            raise RuntimeError("probe unavailable")
        if outcome == "mismatch":
            raise _preflight.OwnershipProbeMismatch(
                "managed identity mismatch",
                "stop the mismatched managed service",
            )
        return outcome == "verified"

    result = _preflight.preflight(
        tmp_path / "requirements.json",
        processes=(
            Process(
                "vlm",
                ".",
                "vlm_server",
                port=8100,
                ownership_probe=ownership,
            ),
        ),
        base=tmp_path,
    )

    check = _result(result, "port:vlm:tcp:8100")
    assert len(ownership_calls) == 1
    assert check.required == "no conflicting socket on tcp port 8100"
    if outcome == "verified":
        assert check.remediation == f"remediate: {evidence}"
        assert check.ok
        assert check.status == "warning"
        assert check.detected == (
            "verified as the expected managed service; "
            f"socket inspection failed ({evidence})"
        )
        assert result.services[0].verified_running
    elif outcome == "mismatch":
        assert not check.ok
        assert check.status == "failed"
        assert check.detected == (
            f"could not inspect ({evidence}); managed identity mismatch"
        )
        assert "unverified" not in check.detected
        assert check.remediation == "stop the mismatched managed service"
        assert not result.services[0].verified_running
    else:
        assert check.remediation == f"remediate: {evidence}"
        assert not check.ok
        assert check.status == "failed"
        assert check.detected.startswith(f"could not inspect ({evidence})")
        assert check.detected.endswith("; ownership unverified")
        if outcome == "raises":
            assert "ownership probe failed (RuntimeError)" in check.detected
        assert not result.services[0].verified_running


@pytest.mark.parametrize(
    ("error_kind", "evidence"),
    [
        ("timeout", "ss timed out after 2 seconds"),
        ("bind_host", "could not resolve bind host 'bad.invalid'"),
    ],
)
def test_timeout_and_invalid_bind_host_skip_managed_ownership_probe(
    error_kind: str,
    evidence: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection(
            "uninspectable",
            evidence,
            remediation="correct inspection failure",
            error_kind=error_kind,
        ),
    )
    result = _preflight.preflight(
        tmp_path / "requirements.json",
        processes=(
            Process(
                "vlm",
                ".",
                "vlm_server",
                port=8100,
                ownership_probe=lambda _env: pytest.fail(
                    f"{error_kind} must not invoke ownership"
                ),
            ),
        ),
        base=tmp_path,
    )

    check = _result(result, "port:vlm:tcp:8100")
    assert not check.ok
    assert check.detected == f"could not inspect ({evidence}); ownership not probed"
    assert check.remediation == "correct inspection failure"
    assert not result.services[0].verified_running


def test_cheap_failure_gates_expensive_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"nvidia_driver": "580", "nvidia_container_toolkit": True},
    )
    monkeypatch.setattr(_preflight, "_driver_version", lambda: ("570.1", None))
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda command, **_kwargs: (True, '{"nvidia": {}}')
        if command[:3] == ["docker", "info", "--format"]
        else (False, "unavailable"),
    )
    monkeypatch.setattr(
        _preflight,
        "_expensive_checks",
        lambda *_args, **_kwargs: pytest.fail("expensive checks must be gated"),
    )

    result = _preflight.preflight(
        contract,
        processes=(
            Process(
                "service", ".", "service", port=9000, needs_docker=True,
            ),
        ),
        base=tmp_path,
        force_expensive=True,
    )

    assert not result.ok
    assert _result(result, "nvidia_driver").detected == "570.1"
    assert _result(result, "nvidia_container_toolkit").ok
    blocked = _result(result, "container_gpu")
    assert blocked.ok is None
    assert blocked.status == "blocked"
    assert "nvidia_driver" in blocked.detected


def test_docker_service_runs_toolkit_and_container_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"docker": "24", "nvidia_container_toolkit": True},
    )
    monkeypatch.setattr(_preflight, "_docker_version", lambda: ("28.0", None))
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda command, **_kwargs: (
            (True, '{"nvidia": {}}')
            if command[:3] == ["docker", "info", "--format"]
            else (False, "unexpected command")
        ),
    )
    expensive_calls: list[tuple[tuple[str, ...], bool]] = []

    def expensive(names, *, force):
        expensive_calls.append((tuple(names), force))
        return [
            _preflight.CheckResult(
                "container_gpu", True, "visible", "visible", "",
                tier="expensive",
            )
        ]

    monkeypatch.setattr(_preflight, "_expensive_checks", expensive)

    result = _preflight.preflight(
        contract,
        processes=(Process("service", ".", "service", needs_docker=True),),
        base=tmp_path,
    )

    assert result.ok
    assert _result(result, "docker").ok
    assert _result(result, "nvidia_container_toolkit").ok
    assert _result(result, "container_gpu").ok
    assert expensive_calls == [(('container_gpu',), False)]


def test_pip_only_model_profile_skips_docker_and_container_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"docker": "24", "nvidia_container_toolkit": True},
    )
    config = tmp_path / "vlm.yaml"
    config.write_text("vllm_backend: pip\n", encoding="utf-8")
    process = Process(
        "vlm", ".", "vlm_server", config=config, port=8100, needs_docker=False
    )
    monkeypatch.setattr(
        _preflight,
        "_docker_version",
        lambda: pytest.fail("pip-only services must not inspect Docker"),
    )
    monkeypatch.setattr(
        _preflight, "_port_is_free", lambda *_args: _inspection("clear")
    )

    result = _preflight.preflight(contract, processes=(process,), base=tmp_path)

    assert result.ok
    assert {check.name for check in result.checks} == {"port:vlm:tcp:8100"}


def test_expensive_checks_do_not_persist_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def check(*, strict: bool) -> _preflight.CheckResult:
        calls.append(strict)
        return _preflight.CheckResult(
            "container_gpu", True, "visible", "visible", "", tier="expensive",
        )

    monkeypatch.setattr(_preflight, "_container_gpu_check", check)
    first = _preflight._expensive_checks(("container_gpu",), force=False)
    second = _preflight._expensive_checks(("container_gpu",), force=False)

    assert calls == [False, False]
    assert first[0].to_dict() == second[0].to_dict()
    assert "cached" not in first[0].to_dict()
    assert not hasattr(_preflight, "_cached_expensive_checks")


def test_expensive_check_force_applies_only_to_container_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def container(*, strict: bool) -> _preflight.CheckResult:
        calls.append(f"container:{strict}")
        return _preflight.CheckResult(
            "container_gpu", True, "visible",
            "visible", "fix container runtime", tier="expensive",
        )

    def vulkan() -> _preflight.CheckResult:
        calls.append("vulkan")
        return _preflight.CheckResult(
            "vulkan_device", True, "hardware", "hardware", "", tier="expensive",
        )

    monkeypatch.setattr(_preflight, "_container_gpu_check", container)
    monkeypatch.setattr(_preflight, "_vulkan_device_check", vulkan)

    first = _preflight._expensive_checks(
        ("container_gpu", "vulkan_device"), force=False,
    )
    forced = _preflight._expensive_checks(("container_gpu",), force=True)

    assert [check.ok for check in first] == [True, True]
    assert forced[0].ok
    assert calls == ["container:False", "vulkan", "container:True"]


def test_deferred_rerun_replaces_only_the_container_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_python = _preflight.CheckResult(
        "python", True, "3.12", ">=3.11", "",
    )
    deferred = _preflight.CheckResult(
        "container_gpu", None, "no image", "GPU visible", "",
        tier="expensive", skipped=True, status="deferred",
    )
    result = _preflight.PreflightResult(
        tmp_path / "requirements.json", (), (),
        (original_python, deferred), (),
    )
    calls: list[tuple[tuple[str, ...], bool]] = []

    def rerun(names, *, force):
        calls.append((tuple(names), force))
        return [_preflight.CheckResult(
            "container_gpu", True, "visible", "GPU visible", "",
            tier="expensive",
        )]

    monkeypatch.setattr(_preflight, "_expensive_checks", rerun)

    updated = _preflight.rerun_deferred_checks(result)

    assert updated.checks[0] is original_python
    assert updated.checks[1].detected == "visible"
    assert calls == [(('container_gpu',), True)]


def test_deferred_rerun_leaves_blocked_container_probe_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = _preflight.CheckResult(
        "container_gpu",
        None,
        "blocked by failed cheap checks: docker",
        "GPU visible",
        "fix Docker",
        tier="expensive",
        skipped=True,
        status="blocked",
    )
    result = _preflight.PreflightResult(
        tmp_path / "requirements.json", (), (), (blocked,), ()
    )
    monkeypatch.setattr(
        _preflight,
        "_expensive_checks",
        lambda *_args, **_kwargs: pytest.fail("blocked checks must not rerun"),
    )

    updated = _preflight.rerun_deferred_checks(result)

    assert updated is result
    assert updated.checks == (blocked,)


def test_container_gpu_probe_uses_the_stack_runtime_and_local_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], float]] = []

    def output(command, *, timeout=10.0):
        assert command[:3] == ["docker", "image", "ls"]
        return True, "nvcr.io/nim/nvidia/model:1"

    def run(command, *, timeout=10.0):
        calls.append((list(command), timeout))
        return subprocess.CompletedProcess(
            command, 0, stdout="GPU 0: NVIDIA", stderr="",
        )

    monkeypatch.setattr(_preflight, "_command_output", output)
    monkeypatch.setattr(_preflight, "_run", run)

    result = _preflight._container_gpu_check(strict=True)

    assert result.ok
    probe = calls[-1][0]
    assert probe[:4] == ["docker", "run", "--rm", "--name"]
    assert probe[4].startswith("xr-ai-preflight-gpu-")
    assert probe[5:8] == ["--pull=never", "--runtime", "nvidia"]
    assert "--device" not in probe
    assert "nvcr.io/nim/nvidia/model:1" in probe


def test_strict_container_gpu_probe_fails_when_no_local_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (True, ""),
    )

    result = _preflight._container_gpu_check(strict=True)

    assert not result.ok
    assert not result.skipped
    assert result.status == "failed"
    assert result.detected == "not run: no local container image"


def test_timed_out_gpu_probe_removes_a_late_created_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []
    cleanup_codes = iter((1, 0))
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (True, "nvidia/cuda:latest"),
    )

    def timeout(command, *, timeout):
        probe_commands.append(list(command))
        raise subprocess.TimeoutExpired(command, timeout)

    def cleanup(command, **kwargs):
        cleanup_commands.append(list(command))
        assert kwargs["timeout"] <= _preflight._GPU_PROBE_CLEANUP_COMMAND_TIMEOUT
        return subprocess.CompletedProcess(command, next(cleanup_codes))

    monkeypatch.setattr(_preflight, "_run", timeout)
    monkeypatch.setattr(_preflight.subprocess, "run", cleanup)
    monkeypatch.setattr(_preflight.time, "sleep", lambda _seconds: None)

    result = _preflight._container_gpu_check(strict=True)

    container_name = probe_commands[0][4]
    assert not result.ok
    assert result.detected.startswith("TimeoutExpired:")
    assert cleanup_commands == [
        ["docker", "rm", "-f", container_name],
        ["docker", "rm", "-f", container_name],
    ]


def test_interrupted_gpu_probe_cleans_up_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (True, "nvidia/cuda:latest"),
    )

    def interrupt(command, *, timeout):
        probe_commands.append(list(command))
        raise KeyboardInterrupt

    def cleanup(command, **_kwargs):
        cleanup_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_preflight, "_run", interrupt)
    monkeypatch.setattr(_preflight.subprocess, "run", cleanup)

    with pytest.raises(KeyboardInterrupt):
        _preflight._container_gpu_check(strict=True)

    container_name = probe_commands[0][4]
    assert cleanup_commands == [["docker", "rm", "-f", container_name]]


def test_sigterm_gpu_probe_cleans_up_and_restores_signal_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_commands: list[list[str]] = []
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (True, "nvidia/cuda:latest"),
    )

    def interrupt(_command, *, timeout):
        handler = _preflight.signal.getsignal(_preflight.signal.SIGTERM)
        assert callable(handler)
        handler(_preflight.signal.SIGTERM, None)

    def cleanup(command, **_kwargs):
        cleanup_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_preflight, "_run", interrupt)
    monkeypatch.setattr(_preflight.subprocess, "run", cleanup)
    monkeypatch.setattr(
        _preflight.os,
        "kill",
        lambda pid, signum: forwarded.append((pid, signum)),
    )

    result = _preflight._container_gpu_check(strict=True)

    assert not result.ok
    assert result.detected == f"interrupted by signal {_preflight.signal.SIGTERM}"
    assert len(cleanup_commands) == 1
    assert cleanup_commands[0][:3] == ["docker", "rm", "-f"]
    assert forwarded == [(_preflight.os.getpid(), _preflight.signal.SIGTERM)]


def test_vulkan_expensive_check_rejects_software_only_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_preflight.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (
            True,
            "GPU0:\n\tvendorID = 0x10005\n"
            "\tdeviceType = PHYSICAL_DEVICE_TYPE_CPU\n"
            "\tdeviceName = llvmpipe (LLVM 19.1.7)",
        ),
    )

    result = _preflight._vulkan_device_check()

    assert not result.ok
    assert result.detected == "software-only Vulkan device found"


def test_vulkan_expensive_check_accepts_hardware_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_preflight.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda *_args, **_kwargs: (
            True,
            "GPU0:\n\tvendorID = 0x10de\n"
            "\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
            "\tdeviceName = NVIDIA RTX PRO 6000",
        ),
    )

    result = _preflight._vulkan_device_check()

    assert result.ok
    assert result.detected == "hardware Vulkan device found"


def test_managed_profile_cannot_use_reuse_only_process_definition(tmp_path: Path) -> None:
    profile = _write(
        tmp_path / "models.managed.json",
        {
            "models": {
                "vlm": {
                    "adapter": {"preset": "cosmos_vlm"},
                    "endpoint": {"base_url": "http://127.0.0.1:8100"},
                    "deployment": {"ownership": "managed", "service": "vlm"},
                }
            }
        },
    )
    deployment = _preflight.ModelDeployment(
        profile,
        {"vlm": "own"},
        (),
        (EndpointProbe(
            name="vlm",
            health_url="http://127.0.0.1:8100/health",
            ownership="own",
        ),),
    )
    process = Process("vlm", ".", "vlm", launch_mode="reuse", port=8100)

    result = _preflight.preflight(
        tmp_path / "requirements.json",
        processes=(process,),
        deployment=deployment,
    )

    assert not _result(result, "ownership:vlm").ok


def test_managed_profile_requires_a_declared_process(tmp_path: Path) -> None:
    profile = tmp_path / "models.managed.json"
    deployment = _preflight.ModelDeployment(
        profile,
        {"vlm": "own"},
        (),
        (EndpointProbe(
            name="vlm",
            health_url="http://127.0.0.1:8100/health",
            ownership="own",
        ),),
    )

    result = _preflight.preflight(
        tmp_path / "requirements.json",
        deployment=deployment,
    )

    check = _result(result, "ownership:vlm")
    assert not check.ok
    assert check.detected == "selected profile declares a managed service with no process"


def test_credentials_load_before_owned_process_identity_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def load() -> None:
        events.append("credentials")
        monkeypatch.setenv("PROBE_TOKEN", "loaded")

    def ownership(env: dict[str, str]) -> bool:
        events.append("ownership")
        assert env["PROBE_TOKEN"] == "loaded"
        return True

    monkeypatch.setattr(_preflight, "load_credentials", load)
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible tcp socket"),
    )
    result = _preflight.preflight(
        tmp_path / "requirements.json",
        processes=(
            Process(
                "vlm",
                ".",
                "vlm_server",
                port=8100,
                ownership_probe=ownership,
            ),
        ),
        base=tmp_path,
    )

    assert result.ok
    assert events == ["credentials", "ownership"]


def test_named_port_does_not_match_an_unrelated_process_by_number(
    tmp_path: Path,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "vlm", "port": 8100}]},
    )

    with pytest.raises(
        _preflight.ContractError,
        match="port requirement 'vlm' has no matching process or endpoint",
    ):
        _preflight.preflight(
            contract,
            processes=(Process("other", ".", "other", port=8100),),
            base=tmp_path,
        )


def test_numeric_versions_are_rejected_by_schema_runtime_and_shipped_contract(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "utils/xr-ai-launcher/requirements.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors({"nvidia_driver": "580"})) == []
    assert list(validator.iter_errors({"nvidia_driver": 580}))
    numeric = _write(tmp_path / "requirements.json", {"nvidia_driver": 580})
    with pytest.raises(_preflight.ContractError, match="version string"):
        _preflight.load_contract(numeric)

    shipped, _ = _preflight.load_contract(
        root / "model-server-samples/model-servers/requirements.json"
    )
    assert list(validator.iter_errors(shipped)) == []
    assert shipped["nvidia_driver"] == "580"
    assert shipped["disk_gb_free"] == {
        "config_keys": ["model_cache", "nim_cache"],
        "runtime_gb": 5,
        "preparation": True,
    }


@pytest.mark.parametrize(
    ("instance", "accepted"),
    [
        ({"nvidia_driver": "580"}, True),
        ({"nvidia_driver": {"version": ">=580", "unless_env_truthy": "SKIP"}}, True),
        ({"nvidia_driver": 580}, False),
        ({"nvidia_driver": {"version": 580}}, False),
        ({"env": [{"name": "TOKEN", "required": False, "docs": "help"}]}, True),
        ({"env": [{"name": "TOKEN", "required": "false"}]}, False),
        ({
            "disk_gb_free": {
                "config_keys": ["model_cache", "nim_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        }, True),
        ({"disk_gb_free": {"config_keys": ["model_cache"]}}, False),
        ({
            "ports": [{
                "name": "hub",
                "port": 8000,
                "proto": "tcp",
                "unless_env_truthy": "NO_WEB_CLIENT",
            }]
        }, True),
        ({"ports": [{"name": "hub", "port": "8000"}]}, False),
    ],
)
def test_schema_and_runtime_accept_the_same_contract_shapes(
    tmp_path: Path,
    instance: dict[str, object],
    accepted: bool,
) -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "utils/xr-ai-launcher/requirements.schema.json").read_text(
            encoding="utf-8"
        )
    )
    schema_accepts = not list(Draft202012Validator(schema).iter_errors(instance))
    contract = _write(tmp_path / "requirements.json", instance)
    try:
        _preflight.load_contract(contract)
    except _preflight.ContractError:
        runtime_accepts = False
    else:
        runtime_accepts = True

    assert schema_accepts is accepted
    assert runtime_accepts is accepted


def test_shipped_model_server_contract_runs_with_real_gpu_profile_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    sample_root = root / "model-server-samples/model-servers"
    sample = _load_sample_main(
        "launcher_preflight_model_servers",
        sample_root / "main.py",
    )
    processes, credentials = sample._build_processes("default", "spark")
    deployment = sample.load_deployment_profile(
        sample_root / "yaml/models.default.json"
    )
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(_preflight, "_driver_version", lambda: ("580.1", None))
    monkeypatch.setattr(_preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        _preflight, "_filesystem_usage", lambda _path: (1, 29_000_000_000)
    )

    result = _preflight.preflight(
        sample_root / "requirements.json",
        processes=processes,
        base=sample_root,
        deployment=deployment,
        required_credentials=credentials,
    )

    assert result.ok
    assert {process.name for process in processes} == {
        "stt",
        "tts",
        "omni",
        "vlm",
        "embedding",
    }
    assert _result(result, "disk:runtime:1").ok
    assert any(check.status == "deferred" for check in result.checks)


def test_local_hub_and_remote_vlm_can_share_port_8000(
    tmp_path: Path,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "hub", "port": 8000}]},
    )
    probe = EndpointProbe(
        name="vlm",
        role="vlm",
        endpoint_url="http://gpu-box:8000/v1",
        health_url="http://gpu-box:8000/health",
        ownership="external",
        readiness="none",
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("hub", ".", "hub", port=8000),),
        endpoints=(probe,),
        base=tmp_path,
    )

    assert result.ok
    assert {
        (service.name, service.ownership, service.endpoint, service.port)
        for service in result.services
    } == {
        ("hub", "own", None, 8000),
        ("vlm", "external", "http://gpu-box:8000/v1", 8000),
    }


def test_standalone_contract_keeps_local_named_ports_with_external_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "hub", "port": 8000}]},
    )
    probe = EndpointProbe(
        name="vlm",
        role="vlm",
        endpoint_url="https://models.example.test/v1",
        health_url="https://models.example.test/health",
        ownership="external",
        readiness="none",
    )
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible tcp socket"),
    )

    result = _preflight.preflight(contract, endpoints=(probe,))

    port = _result(result, "port:hub:tcp:8000")
    assert not port.ok
    assert "xr-ai-livekit-server" in port.remediation
    assert {(service.name, service.ownership) for service in result.services} == {
        ("hub", "own"),
        ("vlm", "external"),
    }


def test_bare_contract_still_runs_docker_capability_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"docker": "24", "nvidia_container_toolkit": True},
    )
    monkeypatch.setattr(_preflight, "_docker_version", lambda: ("28.0", None))
    monkeypatch.setattr(
        _preflight,
        "_command_output",
        lambda command, **_kwargs: (
            (True, '{"nvidia": {}}')
            if command[:3] == ["docker", "info", "--format"]
            else (False, "unexpected command")
        ),
    )
    expensive: list[tuple[tuple[str, ...], bool]] = []

    def run_expensive(names, *, force):
        expensive.append((tuple(names), force))
        return [
            _preflight.CheckResult(
                "container_gpu",
                True,
                "visible",
                "visible",
                "",
                tier="expensive",
            )
        ]

    monkeypatch.setattr(_preflight, "_expensive_checks", run_expensive)

    result = _preflight.preflight(contract)

    assert result.ok
    assert _result(result, "docker").detected == "28.0"
    assert _result(result, "nvidia_container_toolkit").ok
    assert expensive == [(('container_gpu',), False)]


def test_arch_presence_override_differs_from_boolean_truthy_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = tmp_path / "requirements.json"
    monkeypatch.setattr(_preflight.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("ARCH_OVERRIDE", "false")
    _write(
        contract,
        {
            "arch": {
                "allowed": ["x86_64"],
                "unless_env_present": "ARCH_OVERRIDE",
            }
        },
    )

    present = _preflight.preflight(contract)

    assert _result(present, "arch").skipped
    _write(
        contract,
        {
            "arch": {
                "allowed": ["x86_64"],
                "unless_env_truthy": "ARCH_OVERRIDE",
            }
        },
    )
    truthy = _preflight.preflight(contract)
    assert not _result(truthy, "arch").ok
    assert not _result(truthy, "arch").skipped


def test_optional_credential_warns_and_endpoint_promotion_preserves_docs(
    tmp_path: Path,
) -> None:
    docs = "https://example.test/token-help"
    contract = _write(
        tmp_path / "requirements.json",
        {"env": [{"name": "MODEL_TOKEN", "required": False, "docs": docs}]},
    )

    optional = _preflight.preflight(contract)
    warning = _result(optional, "env:MODEL_TOKEN")
    assert warning.ok
    assert warning.status == "warning"
    assert warning.remediation == docs

    promoted = _preflight.preflight(
        contract,
        endpoints=(
            EndpointProbe(
                name="vlm",
                health_url="https://models.test/health",
                ownership="external",
                readiness="none",
                api_key_env="MODEL_TOKEN",
            ),
        ),
    )
    required = _result(promoted, "env:MODEL_TOKEN")
    assert not required.ok
    assert required.status == "failed"
    assert required.required == "set"
    assert required.remediation == docs


def test_resolved_service_keeps_base_endpoint_separate_from_health_url(
    tmp_path: Path,
) -> None:
    probe = EndpointProbe(
        name="vlm",
        role="vlm",
        endpoint_url="https://user:secret@models.test/v1?token=secret",
        health_url="https://user:secret@models.test/ready?token=secret",
        ownership="external",
    )

    services = _preflight._resolve_services(
        {}, (), (probe,), tmp_path, suppress_unprofiled_reuse=False
    )

    assert services[0].endpoint == "https://models.test/v1"
    assert services[0].health == "https://models.test/ready"


def test_driver_and_docker_probe_failures_preserve_error_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"nvidia_driver": "580", "docker": "24"},
    )
    monkeypatch.setattr(
        _preflight,
        "_driver_version",
        lambda: (None, "nvidia-smi failed: driver/library mismatch"),
    )
    monkeypatch.setattr(
        _preflight,
        "_docker_version",
        lambda: (None, "permission denied opening the Docker socket"),
    )

    result = _preflight.preflight(contract)

    assert _result(result, "nvidia_driver").detected == (
        "nvidia-smi failed: driver/library mismatch"
    )
    assert _result(result, "docker").detected == (
        "permission denied opening the Docker socket"
    )


@pytest.mark.parametrize("stdout", ["  \n", "Docker client 28.0\n"])
def test_failed_command_prefers_stripped_stderr_over_stdout(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    monkeypatch.setattr(
        _preflight,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["docker", "version"],
            1,
            stdout=stdout,
            stderr=" permission denied opening the Docker socket \n",
        ),
    )

    ok, output = _preflight._command_output(["docker", "version"])

    assert not ok
    assert output == "permission denied opening the Docker socket"


def test_disk_policy_without_process_context_is_deferred(
    tmp_path: Path,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        },
    )

    result = _preflight.preflight(contract)

    check = _result(result, "disk:resolution:unknown")
    assert check.ok is None
    assert check.status == "deferred"
    assert check.skipped
    assert result.ok
    assert result.to_dict()["ok"] is True


def test_disk_preparation_without_inventory_is_deferred_and_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        },
    )
    config = tmp_path / "service.yaml"
    config.write_text("model_cache: ./configured-cache\n", encoding="utf-8")
    monkeypatch.setattr(
        _preflight, "_filesystem_usage", lambda _path: (7, 10_000_000_000)
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("vlm", ".", "vlm", config=config),),
        base=tmp_path,
    )

    assert _result(result, "disk:runtime:7").ok
    unknown = _result(result, "disk:preparation:vlm:unknown")
    assert unknown.ok is None
    assert unknown.status == "deferred"
    assert unknown.skipped
    assert unknown.to_dict()["ok"] is None
    assert result.ok


def test_disk_inventory_uses_effective_paths_instead_of_configured_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        },
    )
    config = tmp_path / "service.yaml"
    config.write_text("model_cache: ./configured-cache\n", encoding="utf-8")
    monkeypatch.setattr(
        _preflight, "_filesystem_usage", lambda _path: (9, 10_000_000_000)
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("vlm", ".", "vlm", config=config),),
        base=tmp_path,
        preparation_space=(
            _preflight.PreparationSpace("vlm", "effective-cache", 1_000_000_000),
        ),
    )

    runtime = _result(result, "disk:runtime:9")
    assert str(tmp_path / "effective-cache") in runtime.detected
    assert "configured-cache" not in runtime.detected
    assert _result(result, "disk:preparation:9").ok


@pytest.mark.parametrize(
    ("free_gb", "remaining_gb", "expected"),
    [
        (29, 0, True),
        (55, 75, False),
    ],
)
def test_disk_inventory_distinguishes_warm_and_cold_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gb: int,
    remaining_gb: int,
    expected: bool,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        },
    )
    config = tmp_path / "service.yaml"
    config.write_text("model_cache: ./configured-cache\n", encoding="utf-8")
    monkeypatch.setattr(
        _preflight,
        "_filesystem_usage",
        lambda _path: (13, free_gb * 1_000_000_000),
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("vlm", ".", "vlm", config=config),),
        base=tmp_path,
        preparation_space=(
            _preflight.PreparationSpace(
                "vlm", "effective-cache", remaining_gb * 1_000_000_000
            ),
        ),
    )

    assert _result(result, "disk:runtime:13").ok
    assert _result(result, "disk:preparation:13").ok is expected
    assert result.ok is expected


def test_disk_inventory_aggregates_same_filesystem_remaining_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {
            "disk_gb_free": {
                "config_keys": ["model_cache"],
                "runtime_gb": 5,
                "preparation": True,
            }
        },
    )
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    first_config.write_text("model_cache: ./unused-first\n", encoding="utf-8")
    second_config.write_text("model_cache: ./unused-second\n", encoding="utf-8")
    monkeypatch.setattr(
        _preflight, "_filesystem_usage", lambda _path: (11, 7_000_000_000)
    )

    result = _preflight.preflight(
        contract,
        processes=(
            Process("vlm", ".", "vlm", config=first_config),
            Process("llm", ".", "llm", config=second_config),
        ),
        base=tmp_path,
        preparation_space=(
            _preflight.PreparationSpace("vlm", "effective-vlm", 1_000_000_000),
            _preflight.PreparationSpace("llm", "effective-llm", 2_000_000_000),
        ),
    )

    runtime = [check for check in result.checks if check.name.startswith("disk:runtime")]
    preparation = _result(result, "disk:preparation:11")
    assert len(runtime) == 1
    assert runtime[0].required == ">=5 GB free runtime headroom"
    assert preparation.required.startswith(">=8.0 GB")
    assert "3.0 GB of selected artifacts remain" in preparation.detected
    assert not preparation.ok
    assert not result.ok


def test_disk_inventory_rejects_duplicate_invalid_unknown_and_reused_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {
        "config_keys": ["model_cache"],
        "runtime_gb": 5,
        "preparation": True,
    }
    contract = _write(tmp_path / "requirements.json", {"disk_gb_free": policy})
    config = tmp_path / "service.yaml"
    config.write_text("model_cache: ./cache\n", encoding="utf-8")
    owned = Process("vlm", ".", "vlm", config=config)
    monkeypatch.setattr(
        _preflight, "_filesystem_usage", lambda _path: (1, 10_000_000_000)
    )

    duplicate = _preflight.PreparationSpace("vlm", "cache", 0)
    with pytest.raises(_preflight.ContractError, match="duplicate preparation-space"):
        _preflight.preflight(
            contract,
            processes=(owned,),
            base=tmp_path,
            preparation_space=(duplicate, duplicate),
        )
    with pytest.raises(_preflight.ContractError, match="non-negative integer"):
        _preflight.preflight(
            contract,
            processes=(owned,),
            base=tmp_path,
            preparation_space=(
                _preflight.PreparationSpace("vlm", "cache", True),
            ),
        )
    with pytest.raises(_preflight.ContractError, match="not launcher-owned"):
        _preflight.preflight(
            contract,
            processes=(owned,),
            base=tmp_path,
            preparation_space=(
                _preflight.PreparationSpace("unknown", "cache", 0),
            ),
        )
    reused = Process("vlm", ".", "vlm", config=config, launch_mode="reuse")
    with pytest.raises(_preflight.ContractError, match="not launcher-owned"):
        _preflight.preflight(
            contract,
            processes=(reused,),
            base=tmp_path,
            preparation_space=(
                _preflight.PreparationSpace("vlm", "cache", 0),
            ),
        )
    invalid = _write(
        tmp_path / "invalid.json",
        {"disk_gb_free": {**policy, "config_keys": ["model_cache", "model_cache"]}},
    )
    with pytest.raises(_preflight.ContractError, match="unique strings"):
        _preflight.load_contract(invalid)


def test_nvenc_probe_uses_hardware_compatible_frame_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(_preflight.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def output(command, **_kwargs):
        commands.append(list(command))
        return True, ""

    monkeypatch.setattr(_preflight, "_command_output", output)

    result = _preflight._nvenc_check()

    assert result.ok
    assert "color=size=256x256:rate=1" in commands[0]


def test_livekit_port_conflict_has_container_specific_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write(
        tmp_path / "requirements.json",
        {"ports": [{"name": "hub", "port": 9002, "proto": "udp"}]},
    )
    monkeypatch.setattr(
        _preflight,
        "_port_is_free",
        lambda *_args: _inspection("occupied", "visible udp socket"),
    )

    result = _preflight.preflight(
        contract,
        processes=(Process("hub", ".", "device_io_hub", port=9002),),
        base=tmp_path,
    )

    remediation = _result(result, "port:hub:udp:9002").remediation
    assert "xr-ai-livekit-server" in remediation
    assert "different port" in remediation


def test_loopback_external_failure_uses_service_specific_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _preflight._LOCAL_HEALTH_OPENER,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("refused")),
    )
    probe = EndpointProbe(
        name="vlm",
        role="vision",
        health_url="http://127.0.0.1:8100/health",
        ownership="external",
    )

    result = _preflight.preflight(
        tmp_path / "requirements.json", endpoints=(probe,)
    )

    remediation = _result(result, "endpoint:vision").remediation
    assert "shared model service 'vision'" in remediation
    assert "127.0.0.1:8100" in remediation
    assert "#preflight-reused-service" in remediation
