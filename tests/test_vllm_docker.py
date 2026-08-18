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
from xr_ai_vllm import _docker
from xr_ai_vllm._docker import (
    _CONFIG_LABEL,
    _already_logged_in,
    _LogStreamer,
    _registry_for,
    build_run_argv,
    container_exists,
    container_label,
    container_running,
    launch_fingerprint,
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

    def test_fingerprint_changes_when_hf_token_rotates(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["hf_token"] = "hf_first"
        first = _fingerprint_from_argv(build_run_argv(**kwargs))
        kwargs["hf_token"] = "hf_second"
        second = _fingerprint_from_argv(build_run_argv(**kwargs))
        assert first != second
        # The digest is one-way: the token value never reaches the label.
        assert "hf_first" not in first and "hf_second" not in second

    def test_tokenless_fingerprint_omits_the_credential_key(self, tmp_path):
        # No hf_token → no digest key, so fingerprints stay compatible with
        # containers created by code that predates credential digests.
        kwargs = self._base_kwargs(tmp_path)
        kwargs["hf_token"] = None
        first = _fingerprint_from_argv(build_run_argv(**kwargs))
        second = _fingerprint_from_argv(build_run_argv(**kwargs))
        assert first == second

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

    def test_hf_token_passed_by_name_only(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["hf_token"] = "hf_secret"
        argv = build_run_argv(**kwargs)
        # Name-only -e keeps the token off the ps-visible argv.
        assert "HF_TOKEN" in argv
        assert not any("hf_secret" in a for a in argv)

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


def _fingerprint_from_argv(argv):
    labels = [argv[i + 1] for i, v in enumerate(argv) if v == "--label"]
    tagged = next(x for x in labels if x.startswith(f"{_CONFIG_LABEL}="))
    return tagged.split("=", 1)[1]


def _expected_fingerprint(kwargs):
    return launch_fingerprint({
        "image": kwargs["image"],
        "port": kwargs["port"],
        "model_cache": str(kwargs["model_cache"]),
        "cuda_visible_devices": kwargs["cuda_visible_devices"],
        "extra_env": kwargs["extra_env"] or {},
        "extra_pip": kwargs["extra_pip"] or [],
        "vllm_argv": kwargs["vllm_argv"],
    })


class TestRun:
    def test_healthy_unowned_listener_is_rejected(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)
        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker.container_on_port_checked",
                  return_value=(None, True)),
            patch("xr_ai_vllm._docker.evict_local_listener"),
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
            patch("xr_ai_vllm._docker.container_on_port_checked",
                  return_value=(None, True)),
            patch("xr_ai_vllm._docker.evict_local_listener"),
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
            patch("xr_ai_vllm._docker.container_on_port_checked",
                  return_value=(None, True)),
            patch("xr_ai_vllm._docker.evict_local_listener"),
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

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker.container_on_port_checked",
                  return_value=(None, True)),
            patch("xr_ai_vllm._docker.evict_local_listener"),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=False),
            patch("xr_ai_vllm._docker.container_exists", return_value=True),
            patch("xr_ai_vllm._docker.container_running", return_value=False),
            patch("xr_ai_vllm._docker.container_label",
                  side_effect=lambda name, label: _fingerprint_from_argv(
                      build_run_argv(
                          image=kwargs["image"],
                          container_name=kwargs["container_name"],
                          port=kwargs["port"],
                          model_cache=kwargs["model_cache"],
                          hf_token=kwargs["hf_token"],
                          cuda_visible_devices=kwargs["cuda_visible_devices"],
                          extra_env=kwargs["extra_env"],
                          extra_pip=kwargs["extra_pip"],
                          vllm_argv=kwargs["vllm_argv"],
                      )
                  )),
            patch("xr_ai_vllm._docker.start_container", return_value=True) as start,
            patch("xr_ai_vllm._docker.subprocess.Popen") as popen,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        start.assert_called_once_with("xr-ai-vllm-test")
        popen.assert_not_called()

    def test_matching_running_container_continues_startup(self, tmp_path):
        kwargs = _run_kwargs(tmp_path)

        with (
            patch("xr_ai_vllm._docker._docker_available", return_value=True),
            patch("xr_ai_vllm._docker.container_on_port_checked",
                  return_value=("xr-ai-vllm-test", True)),
            patch("xr_ai_vllm._docker.evict_local_listener"),
            patch("xr_ai_vllm._docker._lifecycle.health_ok", return_value=False),
            patch("xr_ai_vllm._docker.container_exists", return_value=True),
            patch("xr_ai_vllm._docker.container_running", return_value=True),
            patch("xr_ai_vllm._docker.container_label",
                  side_effect=lambda name, label: _fingerprint_from_argv(
                      build_run_argv(
                          image=kwargs["image"],
                          container_name=kwargs["container_name"],
                          port=kwargs["port"],
                          model_cache=kwargs["model_cache"],
                          hf_token=kwargs["hf_token"],
                          cuda_visible_devices=kwargs["cuda_visible_devices"],
                          extra_env=kwargs["extra_env"],
                          extra_pip=kwargs["extra_pip"],
                          vllm_argv=kwargs["vllm_argv"],
                      )
                  )),
            patch("xr_ai_vllm._docker.start_container") as start,
            patch("xr_ai_vllm._docker.subprocess.Popen") as popen,
            patch("xr_ai_vllm._docker._LogStreamer", return_value=MagicMock()),
            patch("xr_ai_vllm._docker._lifecycle.wait_until_healthy"),
            patch("xr_ai_vllm._docker._lifecycle.idle_until_stopped"),
            patch("xr_ai_vllm._docker.signal.getsignal", return_value=None),
            patch("xr_ai_vllm._docker.signal.signal"),
        ):
            run(**kwargs)

        start.assert_not_called()
        popen.assert_not_called()


class TestRunContainer:
    """Lifecycle branches of the shared run_container flow (no docker daemon)."""

    class _FakeStreamer:
        log_path = None

        def __init__(self, name):
            pass

        def stop(self):
            pass

    def _kwargs(self, tmp_path: Path) -> dict:
        return dict(
            argv=["docker", "run", "some-image"],
            image="some-image",
            container_name="xr-ai-test-ctr",
            log_prefix="test",
            port=1,
            health_url="http://127.0.0.1:1/health",
            launch_banner="launching",
            reuse_banner="reusing",
            ready_banner="ready",
            ready_file=tmp_path / "ready",
        )

    def _common_stubs(self, monkeypatch, d) -> dict:
        captured: dict = {}
        monkeypatch.setattr(d, "_docker_available", lambda: True)
        monkeypatch.setattr(d, "container_on_port_checked", lambda port: (None, True))
        monkeypatch.setattr(d, "evict_local_listener", lambda port, log_prefix: None)
        monkeypatch.setattr(d, "container_label", lambda name, label: None)
        monkeypatch.setattr(d, "start_container", lambda name: True)
        monkeypatch.setattr(d, "container_running", lambda name: False)
        monkeypatch.setattr(d, "container_exists", lambda name: False)
        monkeypatch.setattr(d, "_LogStreamer", self._FakeStreamer)
        monkeypatch.setattr(d, "_maybe_ngc_login", lambda image: None)
        monkeypatch.setattr(d._lifecycle, "idle_until_stopped", lambda *a, **k: None)

        def _wait(url, *, is_alive, **kw):
            captured["is_alive"] = is_alive

        monkeypatch.setattr(d._lifecycle, "wait_until_healthy", _wait)
        return captured

    def test_reuse_healthy_touches_ready_file_without_popen(self, monkeypatch, tmp_path):
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(_docker, "container_exists", lambda name: True)
        monkeypatch.setattr(_docker, "container_running", lambda name: True)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: True)

        def _no_popen(*a, **kw):
            raise AssertionError("Popen must not be called on healthy reuse")

        monkeypatch.setattr(_docker.subprocess, "Popen", _no_popen)
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert kwargs["ready_file"].exists()

    def test_adopt_running_container_aliveness_follows_container(
        self, monkeypatch, tmp_path,
    ):
        captured = self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)
        running = {"v": True}
        monkeypatch.setattr(_docker, "container_running", lambda name: running["v"])
        monkeypatch.setattr(_docker, "container_exists", lambda name: True)

        def _no_popen(*a, **kw):
            raise AssertionError("adopting a running container must not docker run/start")

        monkeypatch.setattr(_docker.subprocess, "Popen", _no_popen)
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert kwargs["ready_file"].exists()
        is_alive = captured["is_alive"]
        assert is_alive() is True
        running["v"] = False
        assert is_alive() is False

    def test_matching_stopped_container_is_restarted(self, monkeypatch, tmp_path):
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)
        state = {"running": False}
        monkeypatch.setattr(_docker, "container_running", lambda name: state["running"])
        monkeypatch.setattr(_docker, "container_exists", lambda name: True)
        started: list[str] = []

        def _start(name):
            started.append(name)
            state["running"] = True
            return True

        monkeypatch.setattr(_docker, "start_container", _start)

        def _no_popen(*a, **kw):
            raise AssertionError("matching stopped container must docker start, not run")

        monkeypatch.setattr(_docker.subprocess, "Popen", _no_popen)
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert started == ["xr-ai-test-ctr"]
        assert kwargs["ready_file"].exists()

    def test_stale_stopped_container_is_recreated(self, monkeypatch, tmp_path):
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)
        monkeypatch.setattr(_docker, "container_running", lambda name: False)
        state = {"exists": True}
        monkeypatch.setattr(_docker, "container_exists", lambda name: state["exists"])
        monkeypatch.setattr(_docker, "container_label", lambda name, label: "stale")
        removed: list[str] = []

        def _remove(name):
            removed.append(name)
            state["exists"] = False
            return True

        monkeypatch.setattr(_docker, "remove_container", _remove)
        popen_argvs: list[list[str]] = []

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        monkeypatch.setattr(
            _docker.subprocess, "Popen",
            lambda argv, **kw: popen_argvs.append(list(argv)) or _FakeProc(),
        )
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert removed == ["xr-ai-test-ctr"]
        assert popen_argvs == [["docker", "run", "some-image"]]
        assert kwargs["ready_file"].exists()

    def test_own_container_on_port_is_not_evicted(self, monkeypatch, tmp_path):
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(
            _docker, "container_on_port_checked",
            lambda port: ("xr-ai-test-ctr", True),
        )
        monkeypatch.setattr(_docker, "container_exists", lambda name: True)
        monkeypatch.setattr(_docker, "container_running", lambda name: True)

        def _no_evict(*a, **kw):
            raise AssertionError("must not evict our own container")

        monkeypatch.setattr(_docker, "stop_container", _no_evict)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: True)
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert kwargs["ready_file"].exists()

    def test_unchecked_port_inspection_skips_eviction(self, monkeypatch, tmp_path):
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(
            _docker, "container_on_port_checked",
            lambda port: (None, False),
        )
        monkeypatch.setattr(_docker, "container_exists", lambda name: True)
        monkeypatch.setattr(_docker, "container_running", lambda name: True)

        def _no_evict(*a, **kw):
            raise AssertionError("must not evict on failed inspection")

        monkeypatch.setattr(_docker, "stop_container", _no_evict)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: True)
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert kwargs["ready_file"].exists()

    def test_running_container_with_changed_config_is_recreated(
        self, monkeypatch, tmp_path,
    ):
        # A profile switch that moves GPUs (or bumps the image pin) reuses the
        # container name; the creation-time contract is immutable, so the
        # fingerprint label mismatch forces a recreate.
        self._common_stubs(monkeypatch, _docker)
        state = {"running": True}
        monkeypatch.setattr(_docker, "container_running", lambda name: state["running"])
        monkeypatch.setattr(_docker, "container_exists", lambda name: state["running"])
        monkeypatch.setattr(_docker, "container_label", lambda name, label: "stale")
        removed: list[str] = []

        def _remove(name):
            removed.append(name)
            state["running"] = False
            return True

        monkeypatch.setattr(_docker, "stop_container", lambda name, **kw: True)
        monkeypatch.setattr(_docker, "remove_container", _remove)
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)
        popen_argvs: list[list[str]] = []

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        monkeypatch.setattr(
            _docker.subprocess, "Popen",
            lambda argv, **kw: popen_argvs.append(list(argv)) or _FakeProc(),
        )
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert removed == ["xr-ai-test-ctr"]
        assert popen_argvs == [["docker", "run", "some-image"]]

    def test_foreign_port_holder_is_evicted_before_launch(self, monkeypatch, tmp_path):
        # A profile switch leaves a different persistent xr-ai container on
        # this port (e.g. a NIM where the local vLLM belongs); it must be
        # stopped and removed, then our container launched.
        self._common_stubs(monkeypatch, _docker)
        monkeypatch.setattr(
            _docker, "container_on_port_checked",
            lambda port: ("xr-ai-nim-cosmos-reason1-7b", True),
        )
        evicted: list[tuple[str, str]] = []
        monkeypatch.setattr(
            _docker, "stop_container",
            lambda name, **kw: evicted.append(("stop", name)) or True,
        )
        monkeypatch.setattr(
            _docker, "remove_container",
            lambda name: evicted.append(("rm", name)) or True,
        )
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)
        monkeypatch.setattr(_docker, "container_running", lambda name: False)
        monkeypatch.setattr(_docker, "container_exists", lambda name: False)
        popen_argvs: list[list[str]] = []

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        monkeypatch.setattr(
            _docker.subprocess, "Popen",
            lambda argv, **kw: popen_argvs.append(list(argv)) or _FakeProc(),
        )
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert evicted == [
            ("stop", "xr-ai-nim-cosmos-reason1-7b"),
            ("rm", "xr-ai-nim-cosmos-reason1-7b"),
        ]
        assert popen_argvs == [["docker", "run", "some-image"]]

    def test_local_pip_listener_is_evicted_when_no_container_holds_port(
        self, monkeypatch, tmp_path,
    ):
        # A pip-mode vLLM left by a profile switch can answer the health
        # probe and be mistaken for a reusable container.
        self._common_stubs(monkeypatch, _docker)
        evictions: list[int] = []
        monkeypatch.setattr(
            _docker, "evict_local_listener",
            lambda port, log_prefix: evictions.append(port),
        )
        monkeypatch.setattr(_docker._lifecycle, "health_ok", lambda url, **kw: False)

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        monkeypatch.setattr(
            _docker.subprocess, "Popen", lambda argv, **kw: _FakeProc(),
        )
        kwargs = self._kwargs(tmp_path)
        _docker.run_container(**kwargs)
        assert evictions == [1]


class TestEvictLocalListener:
    def test_xr_ai_listener_is_terminated(self, monkeypatch):
        monkeypatch.setattr(
            _docker, "pid_on_port_checked", lambda port: (4242, True, True),
        )
        monkeypatch.setattr(
            _docker, "is_xr_ai_server_process", lambda pid, label, port: True,
        )
        sent: list[int] = []

        def _kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError
            sent.append(sig)

        monkeypatch.setattr(_docker.os, "kill", _kill)
        monkeypatch.setattr(_docker.time, "sleep", lambda s: None)
        _docker.evict_local_listener(8100, "test")
        assert sent == [_docker.signal.SIGTERM]

    def test_non_xr_ai_listener_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(
            _docker, "pid_on_port_checked", lambda port: (4242, True, True),
        )
        monkeypatch.setattr(
            _docker, "is_xr_ai_server_process", lambda pid, label, port: False,
        )

        def _no_kill(pid, sig):
            raise AssertionError("must not signal an unrelated process")

        monkeypatch.setattr(_docker.os, "kill", _no_kill)
        _docker.evict_local_listener(8100, "test")

    def test_free_port_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            _docker, "pid_on_port_checked", lambda port: (None, True, False),
        )

        def _no_kill(pid, sig):
            raise AssertionError("nothing to signal on a free port")

        monkeypatch.setattr(_docker.os, "kill", _no_kill)
        _docker.evict_local_listener(8100, "test")


class TestPipEviction:
    def test_pip_run_evicts_container_holding_its_port(self, monkeypatch):
        from xr_ai_vllm import _pip

        evicted: list[tuple[str, str]] = []
        monkeypatch.setattr(
            _pip._docker, "container_on_port_checked",
            lambda port: ("xr-ai-nim-cosmos-reason1-7b", True),
        )
        monkeypatch.setattr(
            _pip._docker, "stop_container",
            lambda name, **kw: evicted.append(("stop", name)) or True,
        )
        monkeypatch.setattr(
            _pip._docker, "remove_container",
            lambda name: evicted.append(("rm", name)) or True,
        )
        monkeypatch.setattr(_pip._lifecycle, "health_ok", lambda url, **kw: True)
        monkeypatch.setattr(
            _pip._lifecycle, "idle_until_stopped", lambda *a, **kw: None,
        )

        def _no_popen(*a, **kw):
            raise AssertionError("reuse path must not spawn vllm")

        monkeypatch.setattr(_pip.subprocess, "Popen", _no_popen)
        _pip.run(
            persistent=True,
            log_prefix="test",
            vllm_argv=["vllm", "serve", "m"],
            host="0.0.0.0",
            port=8100,
            ready_file=None,
        )
        assert evicted == [
            ("stop", "xr-ai-nim-cosmos-reason1-7b"),
            ("rm", "xr-ai-nim-cosmos-reason1-7b"),
        ]
