# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-profile MPS validation and transport without loading inference models."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from xr_ai_launcher import CUDA_MPS_ENV_KEYS, _mps, configure_cuda_mps, cuda_mps_identity
from xr_ai_model_server_stack import _stack
from xr_ai_vllm import setup_hf_env
from xr_ai_vllm._docker import build_run_argv


@pytest.fixture(autouse=True)
def restore_hub_environment(monkeypatch):
    for key in ("HF_HOME", "HF_TOKEN", "HF_XET_HIGH_PERFORMANCE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


@pytest.fixture
def controller(monkeypatch):
    # AF_UNIX paths must fit the platform's short socket-path limit.
    with tempfile.TemporaryDirectory(prefix="xr-mps-test-") as directory:
        pipe = Path(directory)
        with socket.socket(socket.AF_UNIX) as control:
            control.bind(str(pipe / "control"))
            monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(pipe))
            monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "10")
            monkeypatch.setenv("CUDA_MPS_CLIENT_PRIORITY", "1")
            calls = []

            def probe(argv, **kwargs):
                calls.append((argv, kwargs))
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            monkeypatch.setattr(_mps.subprocess, "run", probe)
            yield pipe, calls


@pytest.mark.parametrize("settings, message", [
    (None, "mps must be a mapping containing only"),
    (True, "mps must be a mapping containing only"),
    ([], "mps must be a mapping containing only"),
    ({}, "mps.enabled must be a boolean"),
    ({"enabled": "true"}, "mps.enabled must be a boolean"),
    ({"enabled": True, "typo": 1}, "mps must be a mapping containing only"),
    ({"enabled": False, "client_priority": 1}, "disabled mps must contain only enabled: false"),
    *[({"enabled": True, "active_thread_percentage": value},
       "mps.active_thread_percentage must be an integer from 1 to 100")
      for value in (True, 0, 101, "75")],
    *[({"enabled": True, "client_priority": value},
       "mps.client_priority must be an integer from 0 to 1")
      for value in (-1, 2)],
])
def test_invalid_settings_do_not_change_environment_or_probe(controller, settings, message, monkeypatch):
    before = dict(os.environ)
    monkeypatch.setattr(_mps.subprocess, "run", lambda *_a, **_k: pytest.fail("invalid settings probed CUDA"))
    with pytest.raises(ValueError, match=message):
        configure_cuda_mps(settings)
    assert dict(os.environ) == before


def test_explicit_mps_uses_defaults_and_records_effective_controller(controller):
    pipe, calls = controller
    identity = configure_cuda_mps({"enabled": True})
    assert identity == {
        "enabled": True, "pipe_directory": str(pipe),
        "active_thread_percentage": 100, "client_priority": 0,
        "controller": [getattr((pipe / "control").stat(), key) for key in ("st_dev", "st_ino", "st_ctime_ns")],
    }
    assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "100"
    assert os.environ["CUDA_MPS_CLIENT_PRIORITY"] == "0"
    assert calls[0][0] == ["nvidia-cuda-mps-control"]
    assert calls[0][1]["input"] == "get_server_list\n"


def test_disabled_mps_clears_inherited_environment_without_probing(controller, monkeypatch):
    monkeypatch.setattr(_mps.subprocess, "run", lambda *_a, **_k: pytest.fail("disabled MPS probed CUDA"))
    assert configure_cuda_mps({"enabled": False}) == {"enabled": False}
    assert all(key not in os.environ for key in CUDA_MPS_ENV_KEYS)


@pytest.mark.parametrize("pipe, error, message", [
    (None, ValueError, "mps.enabled requires CUDA_MPS_PIPE_DIRECTORY"),
    ("relative", ValueError, "CUDA MPS requires an absolute CUDA_MPS_PIPE_DIRECTORY"),
    ("/tmp/bad:pipe", ValueError, "CUDA_MPS_PIPE_DIRECTORY cannot contain ':'"),
    ("/missing/xr-mps-test", RuntimeError, "Start a CUDA MPS controller"),
])
def test_missing_controller_is_actionable(pipe, error, message, monkeypatch):
    if pipe is not None:
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", pipe)
    with pytest.raises(error, match=message):
        configure_cuda_mps({"enabled": True})


@pytest.mark.parametrize("failure", [FileNotFoundError("missing CLI"), subprocess.TimeoutExpired("mps", 5)])
def test_controller_cli_failure_is_actionable(controller, monkeypatch, failure):
    def fail(*_a, **_k):
        raise failure

    monkeypatch.setattr(_mps.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="nvidia-cuda-mps-control failed"):
        configure_cuda_mps({"enabled": True})


def test_failed_probe_does_not_change_client_settings(controller, monkeypatch):
    before = dict(os.environ)
    monkeypatch.setattr(_mps.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stderr="unreachable"))
    with pytest.raises(RuntimeError, match="Cannot contact.*unreachable"):
        configure_cuda_mps({"enabled": True})
    assert dict(os.environ) == before


def test_profile_settings_reach_speech_and_docker_with_separate_shares(controller, tmp_path):
    pipe, _ = controller
    overrides = {
        name: {"mps": {"enabled": True, "active_thread_percentage": share, "client_priority": priority}}
        for name, share, priority in (("speaker-stt", 100, 0), ("tts", 25, 1), ("vlm", 75, 1))
    }
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text(yaml.safe_dump(overrides))
    configs = _stack._configs(tmp_path, tuple(overrides), "custom")
    output = tmp_path / "merged"
    output.mkdir()
    processes = _stack._processes(configs, output, tmp_path)
    for process in processes:
        cfg = yaml.safe_load(Path(process.config).read_text())
        assert cfg["mps"] == overrides[process.name]["mps"]
        if process.name == "vlm":
            setup_hf_env(cfg, tmp_path)
        else:
            configure_cuda_mps(cfg["mps"])
        assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == str(cfg["mps"]["active_thread_percentage"])
        assert os.environ["CUDA_MPS_CLIENT_PRIORITY"] == str(cfg["mps"]["client_priority"])
    argv = build_run_argv(
        image="test/image", container_name="test-mps", port=12345,
        model_cache=tmp_path, hf_token=None, cuda_visible_devices="0",
        extra_env=None, extra_pip=None, vllm_argv=["vllm", "serve", "test-model"],
    )
    assert f"{pipe}:{pipe}" in argv
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=75" in argv
    assert "CUDA_MPS_CLIENT_PRIORITY=1" in argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_omitted_mps_preserves_inherited_settings(controller, monkeypatch, tmp_path):
    before = {key: os.environ[key] for key in CUDA_MPS_ENV_KEYS}
    monkeypatch.setattr(_mps.subprocess, "run", lambda *_a, **_k: pytest.fail("omitted MPS probed CUDA"))
    setup_hf_env({}, tmp_path)
    assert {key: os.environ[key] for key in CUDA_MPS_ENV_KEYS} == before


def test_pip_vllm_rejects_explicit_mps_before_environment_changes(controller, tmp_path):
    before = dict(os.environ)
    with pytest.raises(ValueError, match="requires vllm_backend: docker"):
        setup_hf_env({"vllm_backend": "pip", "mps": {"enabled": True}}, tmp_path)
    assert dict(os.environ) == before


@pytest.mark.parametrize("service", ["stt", "clef"])
def test_stack_preserves_mps_opt_out_for_every_service(tmp_path, service):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text(f"{service}: {{mps: {{enabled: false}}}}\n")
    assert _stack._configs(tmp_path, (service,), "custom")[service]["mps"] == {"enabled": False}


@pytest.mark.parametrize("backend", [None, "pip"])
def test_explicit_mps_requires_docker_backend(controller, tmp_path, backend):
    cfg = {"mps": {"enabled": True}}
    if backend is not None:
        cfg["vllm_backend"] = backend
    with pytest.raises(ValueError, match="mps.enabled requires vllm_backend: docker"):
        setup_hf_env(cfg, tmp_path)


def test_yaml_opt_out_reaches_docker_without_mps_flags(controller, tmp_path):
    setup_hf_env({"vllm_backend": "docker", "mps": {"enabled": False}}, tmp_path)
    argv = build_run_argv(
        image="test/image", container_name="test-off", port=12345,
        model_cache=tmp_path, hf_token=None, cuda_visible_devices="0",
        extra_env=None, extra_pip=None, vllm_argv=["vllm", "serve", "test-model"],
    )
    assert "--user" not in argv
    assert all("CUDA_MPS_" not in value for value in argv)
    assert argv.count("-v") == 1
    assert str(controller[0]) not in " ".join(argv)


def test_effective_inherited_state_is_probed_once_and_matches_explicit_state(controller):
    inherited = cuda_mps_identity()
    explicit = configure_cuda_mps({"enabled": True, "active_thread_percentage": 10, "client_priority": 1})
    assert inherited == explicit


@pytest.mark.parametrize("project, module", [
    ("speaker-stt", "speaker_stt"), ("pocket-tts", "pocket_tts_server"),
    ("stt-server", "stt_server"), ("clef-server", "clef_server"),
])
@pytest.mark.parametrize("pipe, expected", [
    (None, "mps.enabled requires CUDA_MPS_PIPE_DIRECTORY"),
    ("/missing/xr-mps-test", "Start a CUDA MPS controller"),
])
def test_service_mps_errors_are_one_line_exits(tmp_path, project, module, pipe, expected):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "service.yaml"
    config.write_text(yaml.safe_dump({"mps": {"enabled": True}, "model_cache": str(root / "models")}))
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(root / "services" / project), os.environ.get("PYTHONPATH", ""))),
        "XR_AI_LOG_ROOT": str(tmp_path / "logs"),
    }
    if pipe is not None:
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
    result = subprocess.run(
        [sys.executable, "-m", module, "--config", str(config)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert f"[{module}]" in result.stderr and expected in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("value", [True, None])
def test_clef_rejects_malformed_mps_with_a_concise_exit(tmp_path, value):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "clef.yaml"
    config.write_text(yaml.safe_dump({"mps": value}))
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(root / "services/clef-server"), os.environ.get("PYTHONPATH", ""))),
    }
    result = subprocess.run(
        [sys.executable, "-m", "clef_server", "--config", str(config)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert result.stderr.strip() == "[clef_server] mps must be a mapping"


@pytest.mark.parametrize("name", ["stt", "clef"])
def test_additional_cuda_services_apply_opt_out_before_reuse(controller, tmp_path, monkeypatch, name):
    root = Path(__file__).resolve().parents[1]
    project, command = {"stt": ("stt-server", "stt_server"), "clef": ("clef-server", "clef_server")}[name]
    monkeypatch.syspath_prepend(str(root / "services" / project))
    from importlib import import_module

    service = import_module(f"{command}.__main__")
    config = tmp_path / "service.yaml"
    config.write_text("mps: {enabled: false}\n")
    monkeypatch.setattr(service, "setup_logging", lambda *_a: None)
    monkeypatch.setattr(sys, "argv", [command, "--config", str(config)])
    inspected = []

    def reuse(*_args):
        inspected.append(True)
        assert all(key not in os.environ for key in CUDA_MPS_ENV_KEYS)
        return True

    if name == "clef":
        monkeypatch.setenv("XR_AI_VLLM_MANAGED", "1")
        monkeypatch.setenv("XR_AI_VLLM_PORT", "8120")
        monkeypatch.setattr(service, "_reuse_ready_server", reuse)
    else:
        monkeypatch.setattr(service, "_health_url_ok", reuse)
        monkeypatch.setattr(service, "_idle_until_stopped", lambda *_a: None)
    service.run()
    assert inspected == [True]



@pytest.mark.parametrize("existing", ["no-mps", "different-share", "different-controller", "matching"])
def test_speaker_wrapper_checks_effective_mps_before_reuse(controller, tmp_path, monkeypatch, existing):
    import sys

    from speaker_stt import __main__ as speaker

    settings = {"enabled": True, "active_thread_percentage": 100, "client_priority": 0}
    config = tmp_path / "speaker.yaml"
    cfg = {"mps": settings, "model_cache": str(tmp_path), "port": 8102}
    config.write_text(yaml.safe_dump(cfg))
    desired = configure_cuda_mps(settings)
    if existing != "no-mps":
        old = dict(desired)
        if existing == "different-share":
            old["active_thread_percentage"] = 75
        if existing == "different-controller":
            old["controller"] = [1, 2, 3]
        cfg["_mps_identity"] = old
    status = speaker._identity(cfg)

    async def health(_url):
        return status

    monkeypatch.setattr(sys, "argv", ["speaker_stt", "--config", str(config)])
    monkeypatch.setenv(speaker._READY_PROCESS_MAY_EXIT_ENV, "1")
    monkeypatch.setattr(speaker, "setup_logging", lambda *_a: None)
    monkeypatch.setattr(speaker, "pid_on_port_checked", lambda _port: (123, True, True))
    monkeypatch.setattr(speaker, "has_ownership_marker", lambda *_a: True)
    monkeypatch.setattr(speaker, "_status", health)
    monkeypatch.setattr(speaker.os, "execvpe", lambda *_a: pytest.fail("occupied service was replaced"))
    if existing == "matching":
        speaker._run()
    else:
        with pytest.raises(SystemExit, match="configuration changed"):
            speaker._run()
