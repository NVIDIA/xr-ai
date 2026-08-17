# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_vllm._docker pure helpers."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from xr_ai_vllm._docker import (
    _CONFIG_LABEL,
    _already_logged_in,
    _launch_fingerprint,
    _LogStreamer,
    _registry_for,
    build_run_argv,
    container_exists,
    container_label,
    container_running,
    pid_on_port,
    run,
)


class TestRegistryFor:
    def test_nvcr_registry(self):
        assert _registry_for("nvcr.io/nvidia/vllm:26.04-py3") == "nvcr.io"

    def test_unqualified_name_no_registry(self):
        # A bare name with no slash and no dot/colon in the first component.
        assert _registry_for("myimage") is None

    def test_registry_with_port(self):
        # host:port/image has a colon in the first segment
        assert _registry_for("localhost:5000/myimage:latest") == "localhost:5000"

    def test_tagged_unqualified_name_no_registry(self):
        # A bare image with a tag must not be misread as a registry just
        # because the tag's `:` looks like a host:port marker.
        assert _registry_for("myimage:latest") is None

    def test_namespace_no_registry(self):
        # slash present but first segment has no dot or colon — Docker Hub library namespace
        assert _registry_for("library/myimage") is None

    def test_tagged_namespace_no_registry(self):
        # same as above with an explicit tag — still not a registry reference
        assert _registry_for("library/myimage:latest") is None


class TestAlreadyLoggedIn:
    def test_no_docker_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("xr_ai_vllm._docker._DOCKER_CONFIG", tmp_path / "config.json")
        assert not _already_logged_in("nvcr.io")

    def test_registry_in_auths(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"auths": {"nvcr.io": {"auth": "dG9rZW4="}}}))
        monkeypatch.setattr("xr_ai_vllm._docker._DOCKER_CONFIG", cfg)
        assert _already_logged_in("nvcr.io")

    def test_other_registry_not_present(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"auths": {"docker.io": {}}}))
        monkeypatch.setattr("xr_ai_vllm._docker._DOCKER_CONFIG", cfg)
        assert not _already_logged_in("nvcr.io")

    def test_corrupt_config_returns_false(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text("not json{{{")
        monkeypatch.setattr("xr_ai_vllm._docker._DOCKER_CONFIG", cfg)
        assert not _already_logged_in("nvcr.io")


class TestBuildRunArgv:
    def _base_kwargs(self, tmp_path: Path) -> dict:
        return dict(
            image="nvcr.io/nvidia/vllm:26.04-py3",
            container_name="xr-ai-vllm-vlm",
            port=8100,
            model_cache=tmp_path / "models",
            hf_token="tok123",
            cuda_visible_devices=None,
            extra_env=None,
            extra_pip=None,
            vllm_argv=["vllm", "serve", "my-model", "--host", "0.0.0.0", "--port", "8100"],
        )

    def test_contains_docker_run(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        assert argv[0] == "docker"
        assert argv[1] == "run"

    def test_container_name_present(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        assert "--name" in argv
        idx = argv.index("--name")
        assert argv[idx + 1] == "xr-ai-vllm-vlm"

    def test_port_label_set(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        labels = [argv[index + 1] for index, value in enumerate(argv) if value == "--label"]
        assert "xr-ai-vllm.port=8100" in labels
        assert any(label.startswith(f"{_CONFIG_LABEL}=") for label in labels)

    def test_configuration_fingerprint_changes_with_vllm_arguments(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        first = build_run_argv(**kwargs)
        kwargs["vllm_argv"] = [*kwargs["vllm_argv"], "--gpu-memory-utilization", "0.78"]
        second = build_run_argv(**kwargs)

        def fingerprint(argv):
            labels = [argv[index + 1] for index, value in enumerate(argv) if value == "--label"]
            return next(label for label in labels if label.startswith(f"{_CONFIG_LABEL}="))

        assert fingerprint(first) != fingerprint(second)

    def test_configuration_fingerprint_changes_with_contract_version(
        self,
        tmp_path,
        monkeypatch,
    ):
        kwargs = self._base_kwargs(tmp_path)
        first = build_run_argv(**kwargs)
        monkeypatch.setattr("xr_ai_vllm._docker._LAUNCH_CONTRACT_VERSION", 2)
        second = build_run_argv(**kwargs)

        def fingerprint(argv):
            labels = [argv[index + 1] for index, value in enumerate(argv) if value == "--label"]
            return next(label for label in labels if label.startswith(f"{_CONFIG_LABEL}="))

        assert fingerprint(first) != fingerprint(second)

    def test_network_host(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        assert "--network" in argv
        assert argv[argv.index("--network") + 1] == "host"

    def test_nvidia_runtime_all_gpus_when_no_cuda_filter(self, tmp_path):
        # nvidia runtime (not legacy --gpus) so the launch works under both
        # legacy and CDI toolkit modes; "all" requests every GPU.
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        assert "--gpus" not in argv
        assert argv[argv.index("--runtime") + 1] == "nvidia"
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert "NVIDIA_VISIBLE_DEVICES=all" in env_flags

    def test_cuda_visible_devices_applied(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["cuda_visible_devices"] = "0,1"
        argv = build_run_argv(**kwargs)
        assert argv[argv.index("--runtime") + 1] == "nvidia"
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert "NVIDIA_VISIBLE_DEVICES=0,1" in env_flags

    def test_hf_token_in_env(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert any(f.startswith("HF_TOKEN=") for f in env_flags)

    def test_no_hf_token_when_none(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["hf_token"] = None
        argv = build_run_argv(**kwargs)
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert not any(f.startswith("HF_TOKEN=") for f in env_flags)

    def test_extra_env_included(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["extra_env"] = {"MY_VAR": "my_val"}
        argv = build_run_argv(**kwargs)
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert any(f == "MY_VAR=my_val" for f in env_flags)

    def test_model_cache_volume_mounted(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        argv = build_run_argv(**kwargs)
        cache = str(kwargs["model_cache"])
        assert "-v" in argv
        idx = argv.index("-v")
        assert argv[idx + 1] == f"{cache}:{cache}"

    def test_image_present(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        assert "nvcr.io/nvidia/vllm:26.04-py3" in argv

    def test_shell_overrides_image_entrypoint(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        image = "vllm/vllm-openai:v0.20.0"
        kwargs["image"] = image
        argv = build_run_argv(**kwargs)
        image_index = argv.index(image)
        assert argv[image_index - 2 : image_index] == ["--entrypoint", "/bin/bash"]
        assert argv[image_index + 1] == "-c"

    def test_no_extra_pip_runs_only_hf_transfer_install(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        # /bin/bash -c "<install>... && vllm serve ..." — the install is the last
        # argv entry; with no extra_pip there must be exactly one pip line.
        bash_cmd = argv[-1]
        assert bash_cmd.count("pip install ") == 1
        assert "hf_transfer" in bash_cmd
        assert "--no-build-isolation" not in bash_cmd

    def test_extra_pip_uses_no_build_isolation(self, tmp_path):
        # mamba-ssm and causal-conv1d both `import torch` from setup.py at
        # config time — pip's default isolated build env has no torch and
        # the source build aborts. The extra_pip install path must pass
        # --no-build-isolation so the build sees the container's torch.
        kwargs = self._base_kwargs(tmp_path)
        kwargs["extra_pip"] = ["mamba-ssm", "causal-conv1d"]
        argv = build_run_argv(**kwargs)
        bash_cmd = argv[-1]
        assert "pip install -q hf_transfer" in bash_cmd
        assert "pip install -q --no-build-isolation mamba-ssm causal-conv1d" in bash_cmd


class _FakeLogProc:
    """Stands in for the `docker logs -f` Popen inside _LogStreamer."""

    def __init__(self):
        self._done = threading.Event()

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return 0

    def poll(self):
        return 0 if self._done.is_set() else None

    def terminate(self):
        self._done.set()

    kill = terminate


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestLogStreamer:
    def _make(self, monkeypatch, tmp_path, exists):
        attached: list[_FakeLogProc] = []
        monkeypatch.setattr(
            "xr_ai_vllm._docker._container_log_path",
            lambda name: tmp_path / f"{name}.log",
        )
        monkeypatch.setattr(
            "xr_ai_vllm._docker.container_exists", lambda name: exists.is_set(),
        )

        def _fake_attach(self):
            proc = _FakeLogProc()
            attached.append(proc)
            return proc

        monkeypatch.setattr(_LogStreamer, "_attach", _fake_attach)
        return _LogStreamer("fake-container"), attached

    def test_attach_waits_for_container(self, monkeypatch, tmp_path):
        exists = threading.Event()
        streamer, attached = self._make(monkeypatch, tmp_path, exists)
        try:
            time.sleep(0.3)
            assert not attached  # container absent — must not attach yet
            exists.set()
            assert _wait_for(lambda: len(attached) == 1)
        finally:
            streamer.stop()

    def test_reattaches_after_streamer_exit(self, monkeypatch, tmp_path):
        exists = threading.Event()
        exists.set()
        streamer, attached = self._make(monkeypatch, tmp_path, exists)
        try:
            assert _wait_for(lambda: len(attached) == 1)
            attached[0].terminate()  # simulate "docker logs -f" dying
            assert _wait_for(lambda: len(attached) == 2)
        finally:
            streamer.stop()

    def test_stop_terminates_streamer(self, monkeypatch, tmp_path):
        exists = threading.Event()
        exists.set()
        streamer, attached = self._make(monkeypatch, tmp_path, exists)
        assert _wait_for(lambda: len(attached) == 1)
        streamer.stop()
        assert attached[0].poll() is not None
        assert not streamer._thread.is_alive()

    def test_stop_before_container_exists(self, monkeypatch, tmp_path):
        exists = threading.Event()
        streamer, attached = self._make(monkeypatch, tmp_path, exists)
        streamer.stop()
        assert not streamer._thread.is_alive()
        assert not attached

    def test_reattach_passes_since(self, monkeypatch, tmp_path):
        """A re-attach must not replay the container log from the start."""
        exists = threading.Event()
        exists.set()
        streamer, attached = self._make(monkeypatch, tmp_path, exists)
        try:
            assert _wait_for(lambda: len(attached) == 1)
            assert streamer._since is None
            attached[0].terminate()
            assert _wait_for(lambda: len(attached) == 2)
            assert streamer._since is not None
        finally:
            streamer.stop()

    def test_unwritable_log_path_ends_supervisor(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "xr_ai_vllm._docker._container_log_path",
            lambda name: tmp_path / "missing-dir" / f"{name}.log",
        )
        monkeypatch.setattr("xr_ai_vllm._docker.container_exists", lambda name: True)
        streamer = _LogStreamer("fake-container")
        assert _wait_for(lambda: not streamer._thread.is_alive())
        streamer.stop()

    def test_popen_oserror_ends_supervisor(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "xr_ai_vllm._docker._container_log_path",
            lambda name: tmp_path / f"{name}.log",
        )
        monkeypatch.setattr("xr_ai_vllm._docker.container_exists", lambda name: True)
        monkeypatch.setattr(
            "xr_ai_vllm._docker.subprocess.Popen",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("no fds")),
        )
        streamer = _LogStreamer("fake-container")
        assert _wait_for(lambda: not streamer._thread.is_alive())
        streamer.stop()


class TestContainerHelpers:
    def test_container_exists_false_when_docker_missing(self):
        with patch(
            "xr_ai_vllm._docker.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert not container_exists("some-name")

    def test_container_running_false_when_docker_missing(self):
        with patch(
            "xr_ai_vllm._docker.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert not container_running("some-name")

    def test_container_label_returns_inspected_value(self):
        with patch(
            "xr_ai_vllm._docker.subprocess.check_output",
            return_value="abc123\n",
        ):
            assert container_label("some-name", _CONFIG_LABEL) == "abc123"

    def test_pid_on_port_returns_none_when_tools_missing(self):
        with patch(
            "xr_ai_vllm._docker.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert pid_on_port(8100) is None


def _run_kwargs(tmp_path):
    return dict(
        image="vllm/vllm-openai:v0.20.0",
        container_name="xr-ai-vllm-test",
        log_prefix="test",
        vllm_argv=["vllm", "serve", "model", "--gpu-memory-utilization", "0.78"],
        host="0.0.0.0",
        port=8107,
        model_cache=tmp_path / "models",
        hf_token=None,
        cuda_visible_devices="1",
        extra_env=None,
        extra_pip=None,
        ready_file=None,
    )


def _expected_fingerprint(kwargs):
    return _launch_fingerprint(
        image=kwargs["image"],
        port=kwargs["port"],
        model_cache=kwargs["model_cache"],
        cuda_visible_devices=kwargs["cuda_visible_devices"],
        extra_env=kwargs["extra_env"],
        extra_pip=kwargs["extra_pip"],
        vllm_argv=kwargs["vllm_argv"],
    )


class TestRun:
    def test_healthy_unowned_listener_is_rejected(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=True),
            patch("xr_ai_vllm._docker.container_exists", return_value=False),
            patch("xr_ai_vllm._docker.stop_container") as stop,
            patch("xr_ai_vllm._docker.remove_container") as remove,
            patch("xr_ai_vllm._docker.subprocess.Popen") as popen,
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
            pytest.raises(SystemExit, match="1"),
        ):
            run(**kwargs)

        stop.assert_not_called()
        remove.assert_not_called()
        popen.assert_not_called()

    def test_healthy_stale_container_is_recreated(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        state = {"exists": True}
        process = MagicMock()
        process.poll.return_value = None
        argv = ["docker", "run", "fresh-container"]

        def remove(_name):
            state["exists"] = False
            return True

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=True),
            patch("xr_ai_vllm._docker.container_exists", side_effect=lambda _name: state["exists"]),
            patch("xr_ai_vllm._docker.container_running", return_value=True),
            patch("xr_ai_vllm._docker.container_label", return_value="stale"),
            patch("xr_ai_vllm._docker.stop_container", return_value=True) as stop,
            patch("xr_ai_vllm._docker.remove_container", side_effect=remove) as remove_mock,
            patch("xr_ai_vllm._docker._maybe_ngc_login"),
            patch("xr_ai_vllm._docker.build_run_argv", return_value=argv),
            patch("xr_ai_vllm._docker.subprocess.Popen", return_value=process) as popen,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        stop.assert_called_once_with("xr-ai-vllm-test")
        remove_mock.assert_called_once_with("xr-ai-vllm-test")
        popen.assert_called_once_with(argv, start_new_session=True)

    def test_stale_stopped_container_is_recreated(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        state = {"exists": True}
        process = MagicMock()
        process.poll.return_value = None
        argv = ["docker", "run", "fresh-container"]

        def remove(_name):
            state["exists"] = False
            return True

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=False),
            patch("xr_ai_vllm._docker.container_exists", side_effect=lambda _name: state["exists"]),
            patch("xr_ai_vllm._docker.container_running", return_value=False),
            patch("xr_ai_vllm._docker.container_label", return_value="stale"),
            patch("xr_ai_vllm._docker.remove_container", side_effect=remove) as remove_mock,
            patch("xr_ai_vllm._docker._maybe_ngc_login"),
            patch("xr_ai_vllm._docker.build_run_argv", return_value=argv),
            patch("xr_ai_vllm._docker.subprocess.Popen", return_value=process) as popen,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        remove_mock.assert_called_once_with("xr-ai-vllm-test")
        popen.assert_called_once_with(argv, start_new_session=True)

    def test_matching_stopped_container_is_restarted(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        wait_handle = MagicMock()
        wait_handle.poll.return_value = None

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=False),
            patch("xr_ai_vllm._docker.container_exists", return_value=True),
            patch("xr_ai_vllm._docker.container_running", return_value=False),
            patch("xr_ai_vllm._docker.container_label", return_value=_expected_fingerprint(kwargs)),
            patch("xr_ai_vllm._docker.start_container", return_value=True) as start,
            patch("xr_ai_vllm._docker._wait_for_container", return_value=wait_handle) as wait,
            patch("xr_ai_vllm._docker.build_run_argv") as build,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        start.assert_called_once_with("xr-ai-vllm-test")
        wait.assert_called_once_with("xr-ai-vllm-test")
        build.assert_not_called()

    def test_matching_running_container_continues_startup(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        wait_handle = MagicMock()
        wait_handle.poll.return_value = None

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=False),
            patch("xr_ai_vllm._docker.container_exists", return_value=True),
            patch("xr_ai_vllm._docker.container_running", return_value=True),
            patch("xr_ai_vllm._docker.container_label", return_value=_expected_fingerprint(kwargs)),
            patch("xr_ai_vllm._docker._wait_for_container", return_value=wait_handle) as wait,
            patch("xr_ai_vllm._docker.start_container") as start,
            patch("xr_ai_vllm._docker.build_run_argv") as build,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        wait.assert_called_once_with("xr-ai-vllm-test")
        start.assert_not_called()
        build.assert_not_called()
