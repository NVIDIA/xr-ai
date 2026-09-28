# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
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


@pytest.fixture(autouse=True)
def _isolate_host_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_preflight, "load_credentials", lambda: None)
    monkeypatch.setattr(_preflight, "_port_is_free", lambda *_args: (True, "free"))
    for name in (
        "HF_TOKEN",
        "NGC_API_KEY",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
    ):
        monkeypatch.delenv(name, raising=False)


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
    monkeypatch.setattr(_preflight, "_port_is_free", lambda *_args: (False, "occupied"))
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
    monkeypatch.setattr(_preflight, "_port_is_free", lambda *_args: (False, "occupied"))
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
    monkeypatch.setattr(_preflight, "_port_is_free", lambda *_args: (False, "occupied"))

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
        lambda *args: probe_calls.append(args) or (True, "free"),
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


def test_tcp_port_probe_enables_reuse_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    class Candidate:
        def __init__(self, family, sock_type):
            self.family = family
            events.append(("socket", family, sock_type))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, level, option, value):
            events.append(("setsockopt", self.family, level, option, value))

        def bind(self, address):
            events.append(("bind", self.family, address))

        def listen(self, backlog):
            events.append(("listen", self.family, backlog))

    monkeypatch.setattr(_preflight.socket, "socket", Candidate)

    assert _REAL_PORT_IS_FREE(8100, "tcp", "0.0.0.0") == (True, "free")
    assert ("listen", socket.AF_INET, 1) in events
    for index, event in enumerate(events):
        if event[0] == "bind":
            family = event[1]
            assert any(
                earlier[:2] == ("setsockopt", family)
                and earlier[3:] == (socket.SO_REUSEADDR, 1)
                for earlier in events[:index]
            )


def test_port_probe_checks_ipv6_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families: list[int] = []
    listens: list[tuple[int, int]] = []

    class Candidate:
        def __init__(self, family, _sock_type):
            self.family = family
            families.append(family)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            if self.family == socket.AF_INET6:
                raise OSError(errno.EADDRINUSE, "address already in use")

        def listen(self, backlog):
            listens.append((self.family, backlog))

    monkeypatch.setattr(_preflight.socket, "socket", Candidate)

    free, detected = _REAL_PORT_IS_FREE(8100, "tcp", "::")

    assert not free
    assert "occupied" in detected
    assert families == [socket.AF_INET6]
    assert listens == []


def test_ipv6_wildcard_port_probe_allows_an_ipv4_only_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families: list[int] = []
    listens: list[tuple[int, int]] = []

    class Candidate:
        def __init__(self, family, _sock_type):
            self.family = family
            families.append(family)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            pass

        def listen(self, backlog):
            listens.append((self.family, backlog))

    monkeypatch.setattr(_preflight.socket, "socket", Candidate)

    assert _REAL_PORT_IS_FREE(8100, "tcp", "::") == (True, "free")
    assert families == [socket.AF_INET6]
    assert listens == [(socket.AF_INET6, 1)]


def test_port_probe_distinguishes_an_unresolvable_bind_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Candidate:
        def __init__(self, _family, _sock_type):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(_preflight.socket, "socket", Candidate)

    free, detected = _REAL_PORT_IS_FREE(8100, "tcp", "not-a-host.invalid")

    assert not free
    assert detected == (
        "could not resolve bind host 'not-a-host.invalid' "
        "(Name or service not known)"
    )


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
    monkeypatch.setattr(_preflight, "_port_is_free", lambda *_args: (True, "free"))

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
        _preflight, "_port_is_free", lambda *_args: (False, "occupied")
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
        _preflight, "_port_is_free", lambda *_args: (False, "occupied")
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
        _preflight, "_port_is_free", lambda *_args: (False, "occupied")
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
