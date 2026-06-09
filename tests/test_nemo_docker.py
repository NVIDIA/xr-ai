# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in NeMo docker backend.

Two layers, both hermetic (no docker/GPU):

* ``xr_ai_nemo_runtime._docker`` pure helpers — the ``docker run`` argv builder
  and the registry helpers.
* The backend toggle in each NeMo server's ``run()``: ``backend: pip`` (default
  / unset) takes the in-venv path; ``backend: docker`` routes to the runner.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

from xr_ai_nemo_runtime import DEFAULT_IMAGE
from xr_ai_nemo_runtime._docker import (
    _already_logged_in,
    _registry_for,
    build_run_argv,
    container_exists,
)

_REPO = Path(__file__).resolve().parents[1]


def _base_kwargs(tmp_path: Path) -> dict:
    return dict(
        image=DEFAULT_IMAGE,
        container_name="xr-ai-nemo-stt-server",
        port=8103,
        repo_root=_REPO,
        server_pkg_dir=_REPO / "ai-services" / "stt-server",
        server_module="stt_server",
        config_path=_REPO / "ai-services" / "stt-server" / "stt_server.yaml",
        model_cache=tmp_path / "models",
        nemo_cache_dir=tmp_path / "models" / "nemo",
        hf_token="tok123",
        cuda_visible_devices=None,
        extra_pip=["python-multipart"],
        extra_env=None,
    )


class TestBuildRunArgv:
    def test_docker_run_prefix(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[0] == "docker"
        assert argv[1] == "run"

    def test_image_present(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert DEFAULT_IMAGE in argv

    def test_container_name(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[argv.index("--name") + 1] == "xr-ai-nemo-stt-server"

    def test_port_label(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[argv.index("--label") + 1] == "xr-ai-nemo.port=8103"

    def test_network_host(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[argv.index("--network") + 1] == "host"

    def test_ipc_host(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[argv.index("--ipc") + 1] == "host"

    def test_gpus_all_when_no_filter(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[argv.index("--gpus") + 1] == "all"

    def test_cuda_visible_devices_applied(self, tmp_path):
        kwargs = _base_kwargs(tmp_path)
        kwargs["cuda_visible_devices"] = "1"
        argv = build_run_argv(**kwargs)
        assert argv[argv.index("--gpus") + 1] == "device=1"

    def test_hf_transfer_env_on(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert "HF_HUB_ENABLE_HF_TRANSFER=1" in env_flags

    def test_hf_token_in_env_when_provided(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert any(f.startswith("HF_TOKEN=") for f in env_flags)

    def test_no_hf_token_when_none(self, tmp_path):
        kwargs = _base_kwargs(tmp_path)
        kwargs["hf_token"] = None
        argv = build_run_argv(**kwargs)
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert not any(f.startswith("HF_TOKEN=") for f in env_flags)

    def test_nemo_cache_dir_in_env(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert any(f.startswith("NEMO_CACHE_DIR=") for f in env_flags)

    def test_model_cache_mounted(self, tmp_path):
        kwargs = _base_kwargs(tmp_path)
        argv = build_run_argv(**kwargs)
        cache = str(kwargs["model_cache"])
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        assert f"{cache}:{cache}" in mounts

    def test_repo_mounted_read_only(self, tmp_path):
        argv = build_run_argv(**_base_kwargs(tmp_path))
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        assert f"{_REPO}:{_REPO}:ro" in mounts

    def test_bash_installs_light_deps_not_pyproject(self, tmp_path):
        # The container has torch+nemo already; we must install only the light
        # missing deps and never `pip install` our pyproject (which drags
        # nemo_toolkit/torch and conflicts).
        argv = build_run_argv(**_base_kwargs(tmp_path))
        assert argv[-3] == "bash"
        assert argv[-2] == "-c"
        bash_cmd = argv[-1]
        assert bash_cmd.startswith("pip install -q ")
        assert "fastapi" in bash_cmd
        assert "uvicorn[standard]" in bash_cmd  # shlex may quote the brackets
        assert "hf_transfer" in bash_cmd
        assert "loguru" in bash_cmd
        assert "pyyaml" in bash_cmd
        # per-server extra_pip merged in
        assert "python-multipart" in bash_cmd
        # never our own package / pyproject
        assert "nemo_toolkit" not in bash_cmd
        assert "pip install -e" not in bash_cmd
        assert "pyproject" not in bash_cmd

    def test_bash_runs_module_with_serve_flag(self, tmp_path):
        # In-container entry must use --_serve so the server does NOT re-read
        # `backend: docker` and spawn another container.
        argv = build_run_argv(**_base_kwargs(tmp_path))
        bash_cmd = argv[-1]
        assert "python -m stt_server" in bash_cmd
        assert "--_serve" in bash_cmd
        assert "--config" in bash_cmd

    def test_bash_sets_pythonpath_to_mounts(self, tmp_path):
        # PYTHONPATH points at the mounted server package + xr-ai-logging so
        # imports resolve from the mount without a pip install.
        kwargs = _base_kwargs(tmp_path)
        argv = build_run_argv(**kwargs)
        bash_cmd = argv[-1]
        assert "PYTHONPATH=" in bash_cmd
        assert str(kwargs["server_pkg_dir"]) in bash_cmd
        assert str(_REPO / "utils" / "xr-ai-logging") in bash_cmd

    def test_magpie_extra_pip_includes_soundfile(self, tmp_path):
        kwargs = _base_kwargs(tmp_path)
        kwargs["extra_pip"] = ["soundfile", "numpy"]
        kwargs["server_module"] = "magpie_tts_server"
        argv = build_run_argv(**kwargs)
        bash_cmd = argv[-1]
        assert "soundfile" in bash_cmd
        assert "python -m magpie_tts_server" in bash_cmd
        # magpie does not need python-multipart (no multipart endpoint)
        assert "python-multipart" not in bash_cmd


class TestRegistryHelpers:
    def test_nvcr_registry(self):
        assert _registry_for("nvcr.io/nvidia/nemo:25.04") == "nvcr.io"

    def test_unqualified_name_no_registry(self):
        assert _registry_for("myimage") is None

    def test_tagged_unqualified_name_no_registry(self):
        assert _registry_for("myimage:latest") is None

    def test_already_logged_in_false_without_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "xr_ai_nemo_runtime._docker._DOCKER_CONFIG", tmp_path / "config.json"
        )
        assert not _already_logged_in("nvcr.io")


class TestContainerHelpers:
    def test_container_exists_false_when_docker_missing(self):
        with patch(
            "xr_ai_nemo_runtime._docker.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert not container_exists("some-name")


# ── backend toggle: server run() routes pip vs docker ────────────────────────


def _load_server_main(pkg_dir: Path, module_name: str):
    """Import a NeMo server's __main__ from its source tree.

    The servers aren't tests deps (that would drag nemo_toolkit), but their
    top-level imports are light (yaml/loguru/xr_ai_logging only), so the
    module loads fine once its package dir is on sys.path.
    """
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    spec = importlib.util.find_spec(f"{module_name}.__main__")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_with_argv(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    mod.run()


class TestSttBackendToggle:
    def _mod(self):
        return _load_server_main(_REPO / "ai-services" / "stt-server", "stt_server")

    def test_default_pip_does_not_call_runner(self, tmp_path, monkeypatch):
        mod = self._mod()
        cfg = _REPO / "ai-services" / "stt-server" / "stt_server.yaml"
        called = {"docker": False}
        monkeypatch.setattr(mod, "_run_docker",
                            lambda *a, **k: called.__setitem__("docker", True))
        # Stop the in-venv wrapper path immediately after the toggle decision.
        sentinel = RuntimeError("pip path reached")

        def _boom(*a, **k):
            raise sentinel
        # No server is up: keep the reuse-probe hermetic (no live socket) and
        # explode on the first in-venv side effect after the toggle decision.
        monkeypatch.setattr(mod.urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
        monkeypatch.setattr(mod.subprocess, "Popen", _boom)
        # default config has backend: pip
        try:
            _run_with_argv(mod, ["stt_server", "--config", str(cfg)], monkeypatch)
        except RuntimeError as exc:
            assert exc is sentinel  # reached the pip path, not docker
        assert called["docker"] is False

    def test_docker_routes_to_runner(self, tmp_path, monkeypatch):
        mod = self._mod()
        cfg = tmp_path / "stt_server.yaml"
        cfg.write_text("model: nvidia/parakeet-tdt-0.6b-v3\nbackend: docker\nport: 8103\n")
        seen = {}
        monkeypatch.setattr(mod, "_run_docker",
                            lambda c, y, n: seen.update(cfg=c, ns=n))
        # in-venv path must NOT run
        monkeypatch.setattr(mod.subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("pip path invoked under backend: docker")))
        _run_with_argv(mod, ["stt_server", "--config", str(cfg)], monkeypatch)
        assert seen["cfg"]["backend"] == "docker"

    def test_serve_flag_bypasses_docker_dispatch(self, tmp_path, monkeypatch):
        # The in-container entry (--_serve) must never re-route to docker.
        mod = self._mod()
        cfg = tmp_path / "stt_server.yaml"
        cfg.write_text("model: m\nbackend: docker\nport: 8103\n")
        monkeypatch.setattr(mod, "_run_docker",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("--_serve re-routed to docker")))
        # _serve calls asyncio.run(_run(...)); stub _run so we don't load nemo.
        async def _fake_run(*a, **k):
            return None
        monkeypatch.setattr(mod, "_run", _fake_run)
        _run_with_argv(mod, ["stt_server", "--_serve", "--config", str(cfg)], monkeypatch)


class TestMagpieBackendToggle:
    def _mod(self):
        return _load_server_main(_REPO / "ai-services" / "tts" / "magpie",
                                 "magpie_tts_server")

    def test_default_pip_does_not_call_runner(self, tmp_path, monkeypatch):
        mod = self._mod()
        cfg = _REPO / "ai-services" / "tts" / "magpie" / "magpie_tts_server.yaml"
        called = {"docker": False}
        monkeypatch.setattr(mod, "_run_docker",
                            lambda *a, **k: called.__setitem__("docker", True))
        sentinel = RuntimeError("pip path reached")

        async def _fake_run(*a, **k):
            raise sentinel
        monkeypatch.setattr(mod, "_run", _fake_run)
        try:
            _run_with_argv(mod, ["magpie_tts_server", "--config", str(cfg)], monkeypatch)
        except RuntimeError as exc:
            assert exc is sentinel
        assert called["docker"] is False

    def test_docker_routes_to_runner(self, tmp_path, monkeypatch):
        mod = self._mod()
        cfg = tmp_path / "magpie_tts_server.yaml"
        cfg.write_text("model: nvidia/magpie_tts_multilingual_357m\nbackend: docker\nport: 8104\n")
        seen = {}
        monkeypatch.setattr(mod, "_run_docker",
                            lambda c, y, n: seen.update(cfg=c, ns=n))

        async def _no_pip(*a, **k):
            raise AssertionError("pip path invoked under backend: docker")
        monkeypatch.setattr(mod, "_run", _no_pip)
        _run_with_argv(mod, ["magpie_tts_server", "--config", str(cfg)], monkeypatch)
        assert seen["cfg"]["backend"] == "docker"

    def test_serve_flag_bypasses_docker_dispatch(self, tmp_path, monkeypatch):
        mod = self._mod()
        cfg = tmp_path / "magpie_tts_server.yaml"
        cfg.write_text("model: m\nbackend: docker\nport: 8104\n")
        monkeypatch.setattr(mod, "_run_docker",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("--_serve re-routed to docker")))

        async def _fake_run(*a, **k):
            return None
        monkeypatch.setattr(mod, "_run", _fake_run)
        _run_with_argv(mod, ["magpie_tts_server", "--_serve", "--config", str(cfg)], monkeypatch)
