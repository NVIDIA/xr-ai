# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_vllm._docker pure helpers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from xr_ai_vllm._docker import (
    _already_logged_in,
    _registry_for,
    build_run_argv,
    container_exists,
    container_running,
    pid_on_port,
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
        assert "--label" in argv
        idx = argv.index("--label")
        assert argv[idx + 1] == "xr-ai-vllm.port=8100"

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

    def test_no_extra_pip_runs_only_hf_transfer_install(self, tmp_path):
        argv = build_run_argv(**self._base_kwargs(tmp_path))
        # bash -c "<install>... && vllm serve ..." — the install is the last
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

    def test_pid_on_port_returns_none_when_tools_missing(self):
        with patch(
            "xr_ai_vllm._docker.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert pid_on_port(8100) is None
