# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_vllm._nim (self-hosted NIM container backend) and
the shipped NIM deployment profiles. Pure argv/config coverage — no
docker daemon or GPU."""
from __future__ import annotations

from pathlib import Path

import pytest
from xr_ai_models import load_models_config
from xr_ai_vllm import _docker
from xr_ai_vllm._nim import build_nim_run_argv, serve_nim

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBuildNimRunArgv:
    def _base_kwargs(self, tmp_path: Path) -> dict:
        return dict(
            image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
            container_name="xr-ai-nim-llama-3.1-8b-instruct",
            http_port=8106,
            grpc_port=None,
            nim_cache=tmp_path / "nim",
            cuda_visible_devices=None,
            extra_env=None,
        )

    def _env_flags(self, argv: list[str]) -> list[str]:
        return [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]

    def test_no_command_override_after_image(self, tmp_path):
        # The NIM entrypoint is the server — the image must be argv's last
        # element (unlike the vLLM argv, which appends `bash -c …`).
        argv = build_nim_run_argv(**self._base_kwargs(tmp_path))
        assert argv[-1] == "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"

    def test_nim_env_and_llm_port_mapping(self, tmp_path):
        argv = build_nim_run_argv(**self._base_kwargs(tmp_path))
        env = self._env_flags(argv)
        # The key is passed by name only so its value stays off the
        # ps-visible argv; docker reads it from the wrapper's environment.
        assert "NGC_API_KEY" in env
        assert not [e for e in env if e.startswith("NGC_API_KEY=")]
        assert "NIM_CACHE_PATH=/opt/nim/.cache" in env
        # Bridge networking: map to the LLM NIM's internal default (8000).
        assert "--network" not in argv
        assert argv[argv.index("-p") + 1] == "8106:8000"

    def test_grpc_port_mapping_for_speech_nims(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["grpc_port"] = 50052
        kwargs["http_port"] = 9011
        argv = build_nim_run_argv(**kwargs)
        mappings = [argv[i + 1] for i, a in enumerate(argv) if a == "-p"]
        # Riva internal defaults: gRPC 50051, HTTP health 9000.
        assert mappings == ["50052:50051", "9011:9000"]

    def test_cache_volume_and_default_user(self, tmp_path):
        # No -u override: Riva NIMs write image-internal paths the host uid
        # cannot (see build_nim_run_argv).
        argv = build_nim_run_argv(**self._base_kwargs(tmp_path))
        assert f"{tmp_path / 'nim'}:/opt/nim/.cache" in argv
        assert "-u" not in argv

    def test_port_label_and_nvidia_runtime(self, tmp_path):
        argv = build_nim_run_argv(**self._base_kwargs(tmp_path))
        assert argv[argv.index("--label") + 1] == "xr-ai-vllm.port=8106"
        assert argv[argv.index("--runtime") + 1] == "nvidia"
        assert "--gpus" not in argv

    def test_extra_env_passthrough(self, tmp_path):
        kwargs = self._base_kwargs(tmp_path)
        kwargs["extra_env"] = {"NIM_LOG_LEVEL": "DEBUG"}
        env = self._env_flags(build_nim_run_argv(**kwargs))
        assert "NIM_LOG_LEVEL=DEBUG" in env


def test_serve_nim_uses_world_writable_per_container_cache(tmp_path, monkeypatch):
    # Riva NIMs write the cache as root, LLM/VLM NIMs as uid 1000 — each
    # container gets its own world-writable subdir so neither blocks the other.
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    captured: dict = {}
    monkeypatch.setattr(_docker, "run_container",
                        lambda **kw: captured.update(kw))
    serve_nim(
        image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
        container_name="xr-ai-nim-llama",
        log_prefix="x",
        http_port=8106,
        nim_cache=tmp_path / "nim",
    )
    leaf = tmp_path / "nim" / "xr-ai-nim-llama"
    assert leaf.stat().st_mode & 0o777 == 0o777
    assert f"{leaf}:/opt/nim/.cache" in captured["argv"]


def test_serve_nim_tolerates_foreign_cache_dir_if_world_writable(
    tmp_path, monkeypatch,
):
    # Shared machine: the per-container cache dir belongs to another OS user
    # (chmod raises), but they already made it world-writable, so proceed.
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    cache = tmp_path / "nim" / "xr-ai-nim-llama"
    cache.mkdir(parents=True)
    cache.chmod(0o777)
    monkeypatch.setattr(
        Path, "chmod",
        lambda self, mode: (_ for _ in ()).throw(PermissionError()),
    )
    captured: dict = {}
    monkeypatch.setattr(_docker, "run_container", lambda **kw: captured.update(kw))
    serve_nim(
        image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
        container_name="xr-ai-nim-llama",
        log_prefix="x",
        http_port=8106,
        nim_cache=tmp_path / "nim",
    )
    assert f"{cache}:/opt/nim/.cache" in captured["argv"]


def test_serve_nim_exits_on_unwritable_foreign_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    cache = tmp_path / "nim" / "xr-ai-nim-llama"
    cache.mkdir(parents=True)
    cache.chmod(0o755)
    real_chmod = Path.chmod

    def _deny(self, mode):
        if self == cache:
            raise PermissionError()
        return real_chmod(self, mode)

    monkeypatch.setattr(Path, "chmod", _deny)
    with pytest.raises(SystemExit):
        serve_nim(
            image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
            container_name="xr-ai-nim-llama",
            log_prefix="x",
            http_port=8106,
            nim_cache=tmp_path / "nim",
        )


def test_serve_nim_exits_when_cache_dir_cannot_be_created(tmp_path, monkeypatch):
    # Shared machine: the parent nim_cache belongs to another OS user and the
    # per-container subdir does not exist yet, so mkdir raises.
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    monkeypatch.setattr(
        Path, "mkdir",
        lambda self, *a, **kw: (_ for _ in ()).throw(PermissionError()),
    )
    with pytest.raises(SystemExit):
        serve_nim(
            image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
            container_name="xr-ai-nim-llama",
            log_prefix="x",
            http_port=8106,
            nim_cache=tmp_path / "nim",
        )


def test_serve_nim_exits_without_ngc_key(tmp_path, monkeypatch):
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        serve_nim(
            image="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
            container_name="x",
            log_prefix="x",
            http_port=8106,
            nim_cache=tmp_path / "nim",
        )


# ── shipped profiles parse through the real loader ─────────────────────────


_MS_YAML = _REPO_ROOT / "agent-samples" / "model-servers" / "yaml"


def test_model_servers_nim_profile_parses() -> None:
    cfg = load_models_config(_MS_YAML / "models.vlm_llm_nim.json")
    llm = cfg.llm("llm")
    assert llm.kind == "openai_compat"
    assert llm.model_name == "nvidia/nemotron-3-nano"
    # The NIM chat template defaults thinking ON (local vLLM off); without
    # the pin, non-thinking agent calls truncate mid-reasoning.
    assert llm.default_extras["chat_template_kwargs"] == {"enable_thinking": False}
    vlm = cfg.vlm("vlm")
    assert vlm.model_name == "nvidia/cosmos-reason1-7b"
    # NIM's health route is /v1/health/ready, not /health; the container
    # gates readiness, so the client must not probe.
    assert vlm.health_check is False
    assert cfg.stt("stt").kind == "openai_compat"


def test_model_servers_vlm_speech_nim_profile_parses() -> None:
    cfg = load_models_config(_MS_YAML / "models.vlm_speech_nim.json")
    # Speech rides the Riva gRPC kind: local container, no NVCF function id,
    # and a real channel-ready health probe (the default).
    for spec in (cfg.stt("stt"), cfg.tts("tts")):
        assert spec.kind == "riva_grpc"
        assert spec.function_id is None
        assert spec.health_check is True
    assert cfg.tts("tts").voice == "Magpie-Multilingual.EN-US.Sofia"


@pytest.mark.parametrize("sample", ["xr-render-demo", "simple-vlm-example"])
def test_sample_vlm_llm_nim_profiles_reuse_the_nim_stack(sample: str) -> None:
    path = _REPO_ROOT / "agent-samples" / sample / "yaml" / "models.vlm_llm_nim.json"
    cfg = load_models_config(path)
    vlm = cfg.vlm("vlm")
    assert vlm.kind == "openai_compat"
    assert vlm.health_check is False
    assert vlm.deployment.ownership == "reused"
    assert vlm.deployment.service == "vlm-nim"
    # Speech stays on the shared local servers in the 2x48 GB nim stack.
    assert cfg.stt("stt").deployment.ownership == "reused"
    assert cfg.tts("tts").deployment.ownership == "managed"


def test_simple_vlm_speech_nim_profile_reuses_riva_containers() -> None:
    path = (
        _REPO_ROOT / "agent-samples" / "simple-vlm-example"
        / "yaml" / "models.vlm_speech_nim.json"
    )
    cfg = load_models_config(path)
    for name, service in (("stt", "stt-nim"), ("tts", "tts-nim")):
        spec = cfg.stt(name) if name == "stt" else cfg.tts(name)
        assert spec.kind == "riva_grpc"
        assert spec.deployment.ownership == "reused"
        assert spec.deployment.service == service
