# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-only preparation tests for download-capable services."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import xr_ai_vllm
from xr_ai_launcher import _artifacts as launcher_artifacts
from xr_ai_launcher import prepare_or_exit
from xr_ai_vllm import _config as vllm_config
from xr_ai_vllm import _docker, _nim, _ownership, _pip
from xr_ai_vllm import _prepare as vllm_prepare
from xr_ai_vllm._config import (
    local_service_identity_env,
    prepare_requested,
    service_config_digest,
)

_ROOT = Path(__file__).resolve().parents[1]
_MISSING_MODULE = object()


def _described_identity(output: str) -> str:
    values = [
        line.removeprefix(_docker._LAUNCH_IDENTITY_PREFIX)
        for line in output.splitlines()
        if line.startswith(_docker._LAUNCH_IDENTITY_PREFIX)
    ]
    assert len(values) == 1
    return values[0]


@pytest.fixture(autouse=True)
def _isolate_credential_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HF_XET_CACHE",
        "HF_XET_HIGH_PERFORMANCE",
        "HUGGINGFACE_HUB_CACHE",
        "LOVR_BIN",
        "MAX_JOBS",
        "NEMO_CACHE_DIR",
        "NEMO_LOGGING_LEVEL",
        "NGC_API_KEY",
        "NUMEXPR_MAX_THREADS",
        "TRANSFORMERS_CACHE",
    ):
        if name not in os.environ:
            monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name, _MISSING_MODULE)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _fake_hf_api(
    revision: str,
    files: dict[str, int],
):
    class HfApi:
        def __init__(self, *, token: str | None = None) -> None:
            self.token = token

        def repo_info(self, *, repo_id: str, revision: str | None = None):
            return SimpleNamespace(sha=revision_value)

        def list_repo_tree(
            self,
            *,
            repo_id: str,
            revision: str,
            recursive: bool,
        ):
            assert recursive
            return [
                SimpleNamespace(path=path, size=size)
                for path, size in files.items()
            ]

    revision_value = revision
    return HfApi


def test_prepare_requested_coexists_with_config() -> None:
    assert prepare_requested(["--config", "service.yaml", "--prepare"])
    assert not prepare_requested(["--config", "service.yaml"])


def test_service_config_digest_protects_every_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "service.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    protected: list[str] = []
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("NGC_API_KEY", "ngc-secret")
    monkeypatch.setattr(
        vllm_config,
        "credential_digest",
        lambda value: protected.append(value) or f"pbkdf2:{value}",
    )

    digest = service_config_digest(config)

    assert digest is not None
    assert protected == ["hf-secret", "ngc-secret"]
    assert "hf-secret" not in digest
    assert "ngc-secret" not in digest


def test_service_config_digest_includes_the_canonical_config_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "service.yaml"
    second = tmp_path / "second" / "service.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    contents = "model: org/model\nmodel_cache: ./models\n"
    first.write_text(contents, encoding="utf-8")
    second.write_text(contents, encoding="utf-8")

    assert service_config_digest(first, env={}) != service_config_digest(
        second, env={}
    )


def test_local_service_wrappers_share_the_child_identity_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "NEMO_LOGGING_LEVEL",
        "NUMEXPR_MAX_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    stt = _load_module(
        "service_prepare_stt_digest",
        "services/stt-server/stt_server/__main__.py",
    )
    pocket = _load_module(
        "service_prepare_pocket_digest",
        "services/pocket-tts/pocket_tts_server/__main__.py",
    )

    assert stt.local_service_identity_env is local_service_identity_env
    assert pocket.local_service_identity_env is local_service_identity_env


def test_child_identity_env_removes_stale_digest_without_current_config() -> None:
    source = {
        "XR_AI_SERVICE_CONFIG_DIGEST": "stale-parent-value",
        "UNCHANGED": "value",
    }

    child = local_service_identity_env(None, "stt_server", env=source)

    assert "XR_AI_SERVICE_CONFIG_DIGEST" not in child
    assert child["XR_AI_SERVICE_IDENTITY"] == "stt_server"
    assert child["UNCHANGED"] == "value"
    assert source["XR_AI_SERVICE_CONFIG_DIGEST"] == "stale-parent-value"


def test_local_service_wrappers_resolve_relative_config_in_child_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt = _load_module(
        "service_prepare_stt_relative_identity",
        "services/stt-server/stt_server/__main__.py",
    )
    pocket = _load_module(
        "service_prepare_pocket_relative_identity",
        "services/pocket-tts/pocket_tts_server/__main__.py",
    )
    config = tmp_path / "service.yaml"
    config.write_text("voice: bill_boerst\nport: 8105\n", encoding="utf-8")
    expected_digest = service_config_digest(config)
    assert expected_digest is not None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XR_AI_SERVICE_CONFIG_DIGEST", "parent-value")

    stt_child: dict[str, str] = {}
    monkeypatch.setattr(stt, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stt, "_health_url_ok", lambda *_args: False)
    monkeypatch.setattr(stt, "_idle_until_stopped", lambda *_args: None)

    def start_stt(*_args: object, env: dict[str, str]):
        stt_child.update(env)
        return SimpleNamespace()

    monkeypatch.setattr(stt, "_start_persistent_server", start_stt)
    monkeypatch.setattr(sys, "argv", ["stt_server", "--config", "service.yaml"])
    stt.run()

    assert stt_child["XR_AI_SERVICE_CONFIG_DIGEST"] == expected_digest
    assert stt_child["XR_AI_SERVICE_IDENTITY"] == "stt_server"
    assert os.environ["XR_AI_SERVICE_CONFIG_DIGEST"] == "parent-value"

    pocket_child: dict[str, str] = {}
    monkeypatch.setattr(pocket, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pocket, "_health_url_ok", lambda *_args: False)
    monkeypatch.setattr(pocket, "_port_open", lambda *_args: False)
    monkeypatch.setattr(pocket, "_ensure_owned_process_group", lambda: None)
    monkeypatch.setattr(
        pocket.os,
        "execvpe",
        lambda _executable, _argv, env: pocket_child.update(env),
    )
    monkeypatch.setattr(
        sys, "argv", ["pocket_tts_server", "--config", "service.yaml"]
    )
    pocket.run()

    assert pocket_child["XR_AI_SERVICE_CONFIG_DIGEST"] == expected_digest
    assert pocket_child["XR_AI_SERVICE_IDENTITY"] == "pocket_tts_server"
    assert os.environ["XR_AI_SERVICE_CONFIG_DIGEST"] == "parent-value"


def test_vllm_wrapper_describes_resolved_mounts_and_defaults_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    embedding = _load_module(
        "service_prepare_embedding_describe",
        "services/embedding-server/embedding_server/__main__.py",
    )
    monkeypatch.setattr(embedding, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _docker,
        "run",
        lambda **_kwargs: pytest.fail("describe mode must not launch Docker"),
    )
    monkeypatch.setattr(
        _pip,
        "run",
        lambda **_kwargs: pytest.fail("describe mode must not launch pip vLLM"),
    )
    contents = (
        "model: org/model\n"
        "model_cache: ./models\n"
        "vllm_backend: docker\n"
    )
    first = tmp_path / "first" / "embedding.yaml"
    second = tmp_path / "second" / "embedding.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(contents, encoding="utf-8")
    second.write_text(contents, encoding="utf-8")
    ready_file = tmp_path / "ready"

    def describe(config: Path) -> str:
        for name in (
            "HF_HUB_DISABLE_XET",
            "HF_HUB_ENABLE_HF_TRANSFER",
            "HF_XET_HIGH_PERFORMANCE",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "embedding_server",
                "--config",
                str(config),
                "--ready-file",
                str(ready_file),
                "--describe-launch",
            ],
        )
        embedding.run()
        return _described_identity(capsys.readouterr().out)

    first_identity = describe(first)
    second_identity = describe(second)

    assert first_identity != second_identity
    assert not (first.parent / "models").exists()
    assert not (second.parent / "models").exists()
    assert not ready_file.exists()

    absolute_cache = tmp_path / "shared-models"
    first.write_text(
        f"model: org/model\nmodel_cache: {absolute_cache}\nvllm_backend: docker\n",
        encoding="utf-8",
    )
    second.write_text(
        f"# source-only change\nvllm_backend: docker\n"
        f"model_cache: {absolute_cache}\nmodel: org/model\n",
        encoding="utf-8",
    )
    stable_identity = describe(first)
    assert stable_identity == describe(second)

    monkeypatch.setattr(embedding, "_DEFAULT_GPU_MEM", 0.42)
    assert describe(first) != stable_identity


def test_nim_wrapper_describes_launch_without_credentials_or_cache_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nim = _load_module(
        "service_prepare_nim_describe",
        "services/nim-server/nim_server/__main__.py",
    )
    monkeypatch.setattr(nim, "setup_logging", lambda *_args, **_kwargs: None)
    config = tmp_path / "nim.yaml"
    config.write_text(
        "image: nvcr.io/nim/example/model:1\n"
        "http_port: 8100\n"
        "nim_cache: ./nim-cache\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NGC_API_KEY", "  ngc-secret\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["nim_server", "--config", str(config), "--describe-launch"],
    )

    nim.run()

    identity = _described_identity(capsys.readouterr().out)
    digest = service_config_digest(config)
    assert digest is not None
    assert identity == _nim.nim_launch_identity(
        image="nvcr.io/nim/example/model:1",
        http_port=8100,
        grpc_port=None,
        nim_cache=tmp_path / "nim-cache" / "xr-ai-nim-model",
        cuda_visible_devices=None,
        extra_env={},
        service_identity="nim_server",
    )
    monkeypatch.setenv("NGC_API_KEY", "ngc-secret")
    argv = _nim.build_nim_run_argv(
        image="nvcr.io/nim/example/model:1",
        container_name="xr-ai-nim-model",
        http_port=8100,
        grpc_port=None,
        nim_cache=tmp_path / "nim-cache" / "xr-ai-nim-model",
        cuda_visible_devices=None,
        extra_env={},
        config_digest=digest,
        service_identity="nim_server",
    )
    assert _docker._requested_fingerprint(argv) == identity
    assert not (tmp_path / "nim-cache").exists()


def test_nano_describe_derives_the_parser_path_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nano = _load_module(
        "service_prepare_nano_describe",
        "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py",
    )
    monkeypatch.setattr(nano, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nano, "gpu_compute_major", lambda: 8)
    monkeypatch.setattr(nano, "_gpu_is_dgx_spark", lambda: False)
    monkeypatch.setattr(
        nano,
        "_ensure_reasoning_parser",
        lambda *_args, **_kwargs: pytest.fail(
            "describe mode must not download the reasoning parser"
        ),
    )
    monkeypatch.setattr(
        _docker,
        "run",
        lambda **_kwargs: pytest.fail("describe mode must not launch Docker"),
    )
    config = tmp_path / "nano.yaml"
    config.write_text(
        "model_cache: ./models\n"
        "vllm_backend: docker\n"
        "spark_uma: false\n",
        encoding="utf-8",
    )
    ready_file = tmp_path / "ready"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron3_nano_llm_server",
            "--config",
            str(config),
            "--ready-file",
            str(ready_file),
            "--describe-launch",
        ],
    )

    nano.run()

    assert len(_described_identity(capsys.readouterr().out)) == 20
    assert not (tmp_path / "models").exists()
    assert not ready_file.exists()


def test_public_vllm_exports_are_complete() -> None:
    import xr_ai_vllm

    expected = {
        "managed_service_matches",
        "prepare_nim",
        "prepare_or_exit",
        "prepare_requested",
        "prepare_vllm",
        "report_prepare_status",
        "service_config_digest",
        "source_config_digest",
    }

    assert expected <= set(xr_ai_vllm.__all__)
    assert all(hasattr(xr_ai_vllm, name) for name in xr_ai_vllm.__all__)


def test_public_launcher_artifact_exports_are_complete() -> None:
    import xr_ai_launcher

    expected = {
        "ArtifactManifest",
        "format_size",
        "path_size",
        "prepare_or_exit",
        "read_artifact_manifest",
        "repair_hf_snapshot",
        "report_prepare_status",
        "write_artifact_manifest",
    }

    assert expected <= set(xr_ai_launcher.__all__)
    assert all(hasattr(xr_ai_launcher, name) for name in expected)


def test_prepare_image_reports_cached_size_without_pull(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(vllm_prepare, "_image_size", lambda _image: 2048)
    monkeypatch.setattr(
        vllm_prepare.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "docker pull must not run for a cached image"
        ),
    )

    vllm_prepare.prepare_image("example/image:1")

    assert capsys.readouterr().out == (
        "[prepare] container image example/image:1: cached (size: 2.0 KiB)\n"
    )


def test_docker_vllm_prepare_downloads_snapshot_without_starting_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(vllm_prepare, "prepare_image", lambda _image: None)
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "weights").write_bytes(b"weights")
    decoy = tmp_path / "hub" / "models--org--model" / "snapshots" / "newer"
    decoy.mkdir()
    (decoy / "weights").write_bytes(b"other weights")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        stdout = f"download log\n{snapshot}\n" if argv[:2] == ["docker", "run"] else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)
    vllm_prepare.prepare_vllm(
        backend="docker",
        image="example/vllm:1",
        model="org/model",
        model_cache=tmp_path,
        hf_token=None,
    )

    assert len(calls) == 2
    command = " ".join(calls[0])
    assert "snapshot_download" in command
    assert "vllm serve" not in command
    assert "--network" not in calls[0]
    env_flags = [
        calls[0][index + 1]
        for index, value in enumerate(calls[0])
        if value == "-e"
    ]
    assert f"HF_HUB_CACHE={tmp_path / 'hub'}" in env_flags
    download_name = calls[0][calls[0].index("--name") + 1]
    assert download_name.startswith("xr-ai-prepare-hf-org-model-")
    assert calls[1] == ["docker", "rm", "-f", download_name]
    marker = next((tmp_path / ".xr-ai-prepare").glob("hf-vllm-*"))
    assert json.loads(marker.read_text(encoding="utf-8"))["artifact_root"] == str(
        snapshot
    )
    assert capsys.readouterr().out == (
        "[prepare] Hugging Face model org/model: downloading (size: unknown)\n"
    )


def test_docker_vllm_repair_forces_download_and_preserves_the_invalid_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_prepare, "prepare_image", lambda _image: None)
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "revision"
    weights = snapshot / "weights"
    snapshot.mkdir(parents=True)
    weights.write_bytes(b"weights")
    marker = vllm_prepare._snapshot_marker(
        tmp_path,
        "org/model",
        backend="docker",
        hub_cache=tmp_path / "hub",
    )
    vllm_prepare._write_snapshot_marker(marker, snapshot)
    marker_before = marker.read_bytes()
    weights.write_bytes(b"corrupt-cache")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            4 if argv[:2] == ["docker", "run"] else 0,
            stdout="",
        )

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exited with status 4"):
        vllm_prepare.prepare_vllm(
            backend="docker",
            image="example/vllm:1",
            model="org/model",
            model_cache=tmp_path,
            hf_token=None,
        )

    command = calls[0][-1]
    assert "api.list_repo_tree" in command
    assert "org/model 1" in command
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert marker.read_bytes() == marker_before


def test_docker_snapshot_rejects_missing_or_foreign_output(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="without reporting its snapshot"):
        vllm_prepare._snapshot_from_output(
            "\n",
            hub_cache=tmp_path / "hub",
            model="org/model",
        )

    foreign = tmp_path / "other" / "snapshot"
    foreign.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="invalid snapshot path"):
        vllm_prepare._snapshot_from_output(
            str(foreign),
            hub_cache=tmp_path / "hub",
            model="org/model",
        )


def test_pip_vllm_prepare_repairs_an_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "revision"
    weights = snapshot / "weights"
    calls: list[tuple[bool, str | None]] = []

    def download(
        *,
        repo_id: str,
        revision: str | None,
        token: str | None,
        cache_dir: Path,
        force_download: bool,
    ) -> str:
        assert repo_id == "org/model"
        assert token == "token"
        assert cache_dir == tmp_path / "hub"
        calls.append((force_download, revision))
        snapshot.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"weights")
        if force_download:
            ref = cache_dir / "models--org--model" / "refs" / str(revision)
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("revision", encoding="utf-8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=_fake_hf_api("revision", {"weights": len(b"weights")}),
            snapshot_download=download,
        ),
    )

    kwargs = {
        "backend": "pip",
        "image": "unused",
        "model": "org/model",
        "model_cache": tmp_path,
        "hf_token": "token",
    }
    vllm_prepare.prepare_vllm(**kwargs)
    vllm_prepare.prepare_vllm(**kwargs)
    weights.write_bytes(b"truncated")
    vllm_prepare.prepare_vllm(**kwargs)

    assert calls == [(False, None), (True, "main")]
    assert weights.read_bytes() == b"weights"


def test_pip_vllm_repair_rejects_a_corrupt_offline_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "revision"
    weights = snapshot / "weights"

    def download(**_kwargs) -> str:
        snapshot.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            weights.write_bytes(b"weights")
            ref = tmp_path / "hub" / "models--org--model" / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("revision", encoding="utf-8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=_fake_hf_api("revision", {"weights": len(b"weights")}),
            snapshot_download=download,
        ),
    )
    kwargs = {
        "backend": "pip",
        "image": "unused",
        "model": "org/model",
        "model_cache": tmp_path,
        "hf_token": "token",
    }
    vllm_prepare.prepare_vllm(**kwargs)
    marker = next((tmp_path / ".xr-ai-prepare").glob("hf-vllm-*"))
    marker_before = marker.read_bytes()
    weights.write_bytes(b"corrupt-cache")

    with pytest.raises(RuntimeError, match="repair did not restore"):
        vllm_prepare.prepare_vllm(**kwargs)

    assert marker.read_bytes() == marker_before


@pytest.mark.parametrize(
    ("returned_revision", "cached_ref", "message"),
    [
        ("old", "old", "returned old instead of new"),
        ("new", "old", "left org/model@main at 'old'"),
    ],
)
def test_hf_repair_rejects_a_stale_branch_snapshot_or_ref(
    returned_revision: str,
    cached_ref: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_cache = tmp_path / "models--org--model"
    snapshot = repo_cache / "snapshots" / returned_revision
    snapshot.mkdir(parents=True)
    (snapshot / "weights").write_bytes(b"weights")
    ref = repo_cache / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(cached_ref, encoding="utf-8")
    calls: list[dict[str, object]] = []

    def download(**kwargs) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=_fake_hf_api("new", {"weights": len(b"weights")}),
            snapshot_download=download,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        launcher_artifacts.repair_hf_snapshot("org/model", tmp_path)

    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "main",
            "token": None,
            "cache_dir": tmp_path,
            "force_download": True,
        }
    ]


def test_pip_vllm_rejects_a_foreign_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = tmp_path / "foreign" / "snapshot"
    foreign.mkdir(parents=True)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(foreign)),
    )

    with pytest.raises(RuntimeError, match="invalid snapshot path"):
        vllm_prepare.prepare_vllm(
            backend="pip",
            image="unused",
            model="org/model",
            model_cache=tmp_path,
            hf_token=None,
        )


@pytest.mark.parametrize(
    ("returncode", "detail"),
    [(3, "exited with status 3"), (-signal.SIGTERM, "interrupted by signal 15")],
)
def test_docker_snapshot_failures_remove_the_download_container(
    returncode: int,
    detail: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_prepare, "prepare_image", lambda _image: None)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        code = returncode if argv[:2] == ["docker", "run"] else 0
        return subprocess.CompletedProcess(argv, code, stdout="")

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=detail):
        vllm_prepare.prepare_vllm(
            backend="docker",
            image="example/vllm:1",
            model="org/model",
            model_cache=tmp_path,
            hf_token=None,
        )

    name = calls[0][calls[0].index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "-f", name]


def test_missing_docker_still_attempts_container_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise FileNotFoundError("docker")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="requires docker"):
        vllm_prepare._run_download_container(
            ["docker", "run", "image"],
            "download-name",
        )

    assert calls[-1] == ["docker", "rm", "-f", "download-name"]


def test_download_container_defers_interrupt_until_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        events.append("cleanup" if argv[1:3] == ["rm", "-f"] else "download")
        return subprocess.CompletedProcess(argv, 0)

    def fake_sigmask(operation: int, _signals: object) -> set[signal.Signals]:
        if operation == signal.SIG_BLOCK:
            events.append("blocked")
            return set()
        events.append("restored")
        raise KeyboardInterrupt

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)
    monkeypatch.setattr(vllm_prepare.signal, "pthread_sigmask", fake_sigmask)

    with pytest.raises(KeyboardInterrupt):
        vllm_prepare._run_download_container(
            ["docker", "run", "image"],
            "download-name",
        )

    assert events == ["download", "blocked", "cleanup", "restored"]


def test_prepare_vllm_rejects_an_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown vllm_backend"):
        vllm_prepare.prepare_vllm(
            backend="other",
            image="unused",
            model="org/model",
            model_cache=tmp_path,
            hf_token=None,
        )


def test_nim_prepare_uses_download_utility_and_warm_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NGC_API_KEY", "secret")
    monkeypatch.setattr(vllm_prepare, "prepare_image", lambda _image: None)
    monkeypatch.setattr(vllm_prepare, "_image_id", lambda _image: "sha256:image")
    monkeypatch.setattr(vllm_prepare, "_gpu_identity", lambda _devices: "GPU-1")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            cache = tmp_path / "xr-ai-nim-model"
            (cache / "ngc" / "profile" / "weights").mkdir(
                parents=True, exist_ok=True
            )
            (cache / "ngc" / "profile" / "weights" / "model.bin").write_bytes(
                b"weights"
            )
            (cache / "ngc" / "hub" / "repo" / "refs").mkdir(
                parents=True, exist_ok=True
            )
            (cache / "ngc" / "hub" / "repo" / "refs" / "main").write_text(
                "revision", encoding="utf-8"
            )
            (cache / "ngc" / "hub" / ".locks").mkdir(parents=True, exist_ok=True)
            (cache / "ngc" / "hub" / ".locks" / "download.lock").touch()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(vllm_prepare.subprocess, "run", fake_run)
    kwargs = {
        "image": "nvcr.io/nim/nvidia/model:1",
        "container_name": "xr-ai-nim-model",
        "nim_cache": tmp_path,
        "cuda_visible_devices": "1",
        "extra_env": {"NIM_MODEL_PROFILE": "default"},
    }
    vllm_prepare.prepare_nim(**kwargs)

    assert len(calls) == 2
    argv = calls[0]
    assert "download-to-cache" in argv[-1]
    assert "-p" not in argv
    assert "8000" not in argv
    download_name = argv[argv.index("--name") + 1]
    assert download_name.startswith("xr-ai-prepare-nim-xr-ai-nim-model-")
    assert calls[1] == ["docker", "rm", "-f", download_name]

    calls.clear()
    vllm_prepare.prepare_nim(**kwargs)
    assert calls == []

    marker = next((tmp_path / "xr-ai-nim-model").glob(".xr-ai-prepare-*"))
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert Path(manifest["artifact_root"]) == (
        tmp_path / "xr-ai-nim-model" / "ngc"
    )
    assert manifest["files"] == [["profile/weights/model.bin", len(b"weights")]]

    cache = tmp_path / "xr-ai-nim-model"
    (cache / "ngc/hub/repo/refs/main").write_text(
        "different revision metadata", encoding="utf-8"
    )
    (cache / "ngc/hub/.locks/download.lock").unlink()
    (cache / "compile-cache").mkdir()
    (cache / "compile-cache" / "kernel.bin").write_bytes(b"compiled")
    (cache / "scratch.sock").touch()
    vllm_prepare.prepare_nim(**kwargs)
    assert calls == []

    (cache / "ngc/profile/weights/model.bin").write_bytes(b"x")
    vllm_prepare.prepare_nim(**kwargs)
    assert len(calls) == 2
    output = capsys.readouterr().out
    assert "NIM model profile nvcr.io/nim/nvidia/model:1: downloading" in output
    assert "NIM model profile nvcr.io/nim/nvidia/model:1: cached" in output


def test_managed_service_probe_ignores_source_only_changes_and_rejects_effective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n")
    health: list[tuple[int, str]] = []
    monkeypatch.setattr(
        _ownership,
        "_health_ok",
        lambda port, path: health.append((port, path)) or True,
    )
    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, **_kwargs: (("arbitrary-generated-name",), True),
    )
    launch_identity = "1" * 20
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda _name, **_kwargs: (
            _docker.ContainerOwnershipSnapshot(
                container_id="a" * 64,
                running=True,
                config_label=launch_identity,
                service_label="vlm_server",
            ),
            True,
        ),
    )
    monkeypatch.setattr(
        _ownership,
        "_describe_launch_identity",
        lambda *_args: launch_identity,
    )

    assert _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        needs_docker=True,
        health_path="/ready",
        project=tmp_path,
    )
    assert health == [(8100, "/ready")]
    config.write_text("# formatting-only source change\nmodel: org/model\n")
    assert _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        needs_docker=True,
        project=tmp_path,
    )

    monkeypatch.setattr(
        _ownership,
        "_describe_launch_identity",
        lambda *_args: "2" * 20,
    )
    with pytest.raises(_ownership.OwnershipProbeMismatch) as mismatch:
        _ownership.managed_service_matches(
            config,
            "vlm_server",
            8100,
            needs_docker=True,
            mismatch_remediation="restart the managed stack",
        )
    assert "different effective launch identity" in mismatch.value.detected
    assert mismatch.value.remediation == "restart the managed stack"

    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, **_kwargs: (("unrelated",), True),
    )
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda _name, **_kwargs: (
            _docker.ContainerOwnershipSnapshot(
                container_id="b" * 64,
                running=True,
                config_label=None,
                service_label=None,
            ),
            True,
        ),
    )
    assert not _ownership.managed_service_matches(
        config, "vlm_server", 8100, needs_docker=True
    )


def test_managed_service_probe_identifies_a_legacy_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, **_kwargs: (("xr-ai-vllm-vlm",), True),
    )
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda _name, **_kwargs: (
            _docker.ContainerOwnershipSnapshot(
                container_id="c" * 64,
                running=True,
                config_label="old-fingerprint",
                service_label=None,
            ),
            True,
        ),
    )

    with pytest.raises(_ownership.OwnershipProbeMismatch) as mismatch:
        _ownership.managed_service_matches(
            config,
            "vlm_server",
            8100,
            needs_docker=True,
            mismatch_remediation="stop every persisted model stack, then rerun",
        )

    assert mismatch.value.detected == (
        "legacy managed container 'xr-ai-vllm-vlm' "
        "(id cccccccccccc) lacks launch identity labels"
    )
    assert mismatch.value.remediation == (
        "stop every persisted model stack, then rerun"
    )


def test_managed_service_probe_bounds_docker_queries_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    calls: list[tuple[str, float | None]] = []
    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, *, timeout=None: (
            calls.append(("discover", timeout)) or ("xr-ai-vllm-test",),
            True,
        ),
    )
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda _name, *, timeout=None: (
            calls.append(("inspect", timeout)) or None,
            False,
        ),
    )

    assert not _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        needs_docker=True,
        project=tmp_path,
    )
    assert calls == [
        ("discover", _ownership._DOCKER_PROBE_TIMEOUT_S),
        ("inspect", _ownership._DOCKER_PROBE_TIMEOUT_S),
    ]


def test_managed_service_probe_rejects_ambiguous_container_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, **_kwargs: (("first", "second"), True),
    )
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous ownership must not inspect one candidate")
        ),
    )

    assert not _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        needs_docker=True,
        project=tmp_path,
    )


def test_managed_service_probe_consumes_labels_from_container_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    digest = service_config_digest(config)
    assert digest is not None
    argv = _docker.build_run_argv(
        image="example/vllm:1",
        container_name="xr-ai-vllm-test",
        port=8100,
        model_cache=tmp_path / "models",
        hf_token=None,
        cuda_visible_devices=None,
        extra_env=None,
        extra_pip=None,
        vllm_argv=["vllm", "serve", "org/model"],
        config_digest=digest,
        service_identity="vlm_server",
    )
    labels = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for index, item in enumerate(argv)
        if item == "--label"
        for value in (argv[index + 1],)
    }
    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "containers_on_port_checked",
        lambda _port, **_kwargs: (("xr-ai-vllm-test",), True),
    )
    monkeypatch.setattr(
        _ownership._docker,
        "container_ownership_snapshot_checked",
        lambda _name, **_kwargs: (
            _docker.ContainerOwnershipSnapshot(
                container_id="d" * 64,
                running=True,
                config_label=labels.get(_docker._CONFIG_LABEL),
                service_label=labels.get(_docker._SERVICE_LABEL),
            ),
            True,
        ),
    )
    monkeypatch.setattr(
        _ownership,
        "_describe_launch_identity",
        lambda *_args: labels[_docker._CONFIG_LABEL],
    )

    assert _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        needs_docker=True,
        project=tmp_path,
    )

    monkeypatch.setattr(
        _ownership,
        "_describe_launch_identity",
        lambda *_args: "f" * 20,
    )
    with pytest.raises(_ownership.OwnershipProbeMismatch) as mismatch:
        _ownership.managed_service_matches(
            config,
            "vlm_server",
            8100,
            needs_docker=True,
            project=tmp_path,
        )
    assert "different effective launch identity" in mismatch.value.detected
    assert "managed container 'xr-ai-vllm-test' (id dddddddddddd)" in (
        mismatch.value.remediation
    )
    assert "port 8100" in mismatch.value.remediation.lower()
    assert "model_servers --stop" in mismatch.value.remediation
    assert "kill" not in mismatch.value.remediation
    assert "docker rm" not in mismatch.value.remediation


def test_launch_identity_query_uses_the_service_project_without_syncing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    project = tmp_path / "service"
    project.mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(_ownership.shutil, "which", lambda _name: "/usr/bin/uv")

    class Process:
        pid = 4321
        returncode = 0

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 15
            return (
                f"log output\n{_docker._LAUNCH_IDENTITY_PREFIX}{'a' * 20}\n",
                "",
            )

    def popen(argv: list[str], **kwargs: object) -> Process:
        calls.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(_ownership.subprocess, "Popen", popen)

    assert _ownership._describe_launch_identity(
        config,
        "vlm_server",
        project,
        {"CUDA_VISIBLE_DEVICES": "1"},
    ) == "a" * 20
    argv, kwargs = calls[0]
    assert argv == [
        "/usr/bin/uv",
        "run",
        "--quiet",
        "--offline",
        "--no-sync",
        "--project",
        str(project),
        "vlm_server",
        "--config",
        str(config),
        "--describe-launch",
    ]
    assert kwargs["env"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert kwargs["stdout"] is _ownership.subprocess.PIPE
    assert kwargs["stderr"] is _ownership.subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["start_new_session"] is True


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("uv", 15), KeyboardInterrupt()])
def test_launch_identity_query_cleans_up_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    project = tmp_path / "service"
    project.mkdir()
    communicates: list[float] = []
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 4321
        returncode = None

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            communicates.append(timeout)
            if len(communicates) == 1:
                raise failure
            self.returncode = -signal.SIGKILL
            return "", ""

    monkeypatch.setattr(_ownership.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        _ownership.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        _ownership.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            _ownership._describe_launch_identity(
                config, "vlm_server", project, None
            )
    else:
        assert _ownership._describe_launch_identity(
            config, "vlm_server", project, None
        ) is None
    assert signals == [(4321, signal.SIGKILL)]
    assert communicates == [15, 1]


def test_launch_identity_cleanup_closes_stuck_owned_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    project = tmp_path / "service"
    project.mkdir()
    class Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    stdout = Stream()
    stderr = Stream()

    class Process:
        pid = 4321
        returncode = None

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            raise subprocess.TimeoutExpired("uv", timeout)

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1
            return self.returncode

    process = Process()
    process.stdout = stdout
    process.stderr = stderr
    monkeypatch.setattr(_ownership.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        _ownership.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(_ownership.os, "killpg", lambda *_args: None)

    assert _ownership._describe_launch_identity(
        config, "vlm_server", project, None
    ) is None
    assert stdout.closed
    assert stderr.closed


def test_pip_launch_identity_round_trips_through_the_serving_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    digest = service_config_digest(config, env={})
    assert digest is not None
    spawned: dict[str, object] = {}

    class Child:
        def poll(self) -> None:
            return None

    def popen(argv: list[str], **kwargs: object) -> Child:
        spawned.update(argv=list(argv), **kwargs)
        return Child()

    monkeypatch.setattr(
        _pip._docker,
        "containers_on_port_checked",
        lambda _port: ((), True),
    )
    monkeypatch.setattr(_pip._lifecycle, "health_ok", lambda _url: False)
    monkeypatch.setattr(
        _pip._lifecycle,
        "wait_until_healthy",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        _pip._lifecycle,
        "idle_until_stopped",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(_pip.subprocess, "Popen", popen)

    xr_ai_vllm.serve(
        backend="pip",
        persistent=True,
        container_name="unused",
        log_prefix="vlm_server",
        model="org/model",
        extra_serve_args=[],
        host="127.0.0.1",
        port=8100,
        model_cache=tmp_path / "models",
        config_digest=digest,
        service_identity="vlm_server",
    )

    child_argv = spawned["argv"]
    child_env = spawned["env"]
    assert isinstance(child_argv, list)
    assert isinstance(child_env, dict)
    assert child_argv[:3] == ["vllm", "serve", "org/model"]
    assert child_env["XR_AI_SERVICE_CONFIG_DIGEST"] == digest
    assert child_env["XR_AI_SERVICE_IDENTITY"] == "vlm_server"
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/4321/cmdline":
            return b"\0".join(item.encode() for item in child_argv) + b"\0"
        if str(path) == "/proc/4321/environ":
            return b"\0".join(
                f"{key}={value}".encode()
                for key, value in child_env.items()
            ) + b"\0"
        return original_read_bytes(path)

    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "pid_on_port_checked",
        lambda _port: (4321, True, True),
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert _ownership.managed_service_matches(
        config,
        "vlm_server",
        8100,
        env={},
        needs_docker=False,
    )
    config.write_text("model: org/changed\n", encoding="utf-8")
    with pytest.raises(_ownership.OwnershipProbeMismatch, match="launch identity"):
        _ownership.managed_service_matches(
            config,
            "vlm_server",
            8100,
            env={},
            needs_docker=False,
        )


def test_local_service_probe_verifies_proc_command_config_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "stt.yaml"
    config.write_text("model: org/model\n", encoding="utf-8")
    digest = service_config_digest(config)
    assert digest is not None
    original_read_bytes = Path.read_bytes
    process_config = [config.resolve()]
    process_digest = [digest]

    def read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/4321/cmdline":
            return (
                b"uv\0stt_server\0--config\0"
                + str(process_config[0]).encode()
                + b"\0"
            )
        if str(path) == "/proc/4321/environ":
            return f"XR_AI_SERVICE_CONFIG_DIGEST={process_digest[0]}\0".encode()
        return original_read_bytes(path)

    monkeypatch.setattr(_ownership, "_health_ok", lambda *_args: True)
    monkeypatch.setattr(
        _ownership._docker,
        "pid_on_port_checked",
        lambda _port: (4321, True, True),
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert _ownership.managed_service_matches(
        config, "stt_server", 8103, needs_docker=False
    )

    process_config[0] = tmp_path / "other.yaml"
    with pytest.raises(_ownership.OwnershipProbeMismatch) as path_mismatch:
        _ownership.managed_service_matches(
            config,
            "stt_server",
            8103,
            needs_docker=False,
            mismatch_remediation="restart the local service",
        )
    assert path_mismatch.value.detected == (
        "managed process 4321 uses a different service config"
    )
    assert path_mismatch.value.remediation == "restart the local service"

    process_config[0] = config.resolve()
    process_digest[0] = "different-digest"
    with pytest.raises(_ownership.OwnershipProbeMismatch) as digest_mismatch:
        _ownership.managed_service_matches(
            config,
            "stt_server",
            8103,
            needs_docker=False,
            mismatch_remediation="restart the local service",
        )
    assert digest_mismatch.value.detected == (
        "managed process 4321 has a different launch identity"
    )
    assert digest_mismatch.value.remediation == "restart the local service"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("download failed"), "[example_service] download failed"),
        (KeyboardInterrupt(), 130),
    ],
)
def test_prepare_or_exit_translates_failures(
    failure: BaseException,
    expected: str | int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise failure

    with pytest.raises(SystemExit) as error:
        prepare_or_exit("example_service", fail)

    if isinstance(expected, int):
        assert error.value.code == expected
        assert "[example_service] artifact preparation interrupted" in capsys.readouterr().err
    else:
        assert str(error.value) == expected


def test_prepare_or_exit_translates_sigterm_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handlers: dict[int, object] = {}

    def install(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(
        launcher_artifacts.signal,
        "getsignal",
        lambda _signum: signal.SIG_DFL,
    )
    monkeypatch.setattr(launcher_artifacts.signal, "signal", install)

    def terminate() -> None:
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)

    with pytest.raises(SystemExit) as error:
        prepare_or_exit("example_service", terminate)

    assert error.value.code == 143
    assert handlers[signal.SIGTERM] is signal.SIG_DFL
    assert capsys.readouterr().err == (
        "[example_service] artifact preparation interrupted by signal 15\n"
    )


def test_stt_prepare_downloads_once_and_repairs_a_changed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "NEMO_CACHE_DIR",
        "HF_HOME",
        "HF_XET_HIGH_PERFORMANCE",
        "NEMO_LOGGING_LEVEL",
        "NUMEXPR_MAX_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    stt = _load_module(
        "service_prepare_stt",
        "services/stt-server/stt_server/__main__.py",
    )
    calls: list[tuple[str, str | None, Path, bool]] = []
    weights: Path | None = None

    def download(
        *,
        repo_id: str,
        revision: str | None,
        cache_dir: Path,
        force_download: bool,
        token: str | None = None,
    ) -> str:
        nonlocal weights
        calls.append((repo_id, revision, cache_dir, force_download))
        snapshot = cache_dir / "models--nvidia--parakeet" / "snapshots" / "revision"
        weights = snapshot / "weights.nemo"
        snapshot.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"weights")
        if force_download:
            ref = cache_dir / "models--nvidia--parakeet" / "refs" / str(revision)
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("revision", encoding="utf-8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=_fake_hf_api("revision", {"weights.nemo": len(b"weights")}),
            snapshot_download=download,
        ),
    )
    cfg = {"model": "nvidia/parakeet", "model_cache": str(tmp_path / "cache")}

    stt._prepare(cfg, tmp_path)
    stt._prepare(cfg, tmp_path)
    assert weights is not None
    weights.write_bytes(b"changed-weights")
    stt._prepare(cfg, tmp_path)

    hub_cache = tmp_path / "cache" / "huggingface" / "hub"
    assert calls == [
        ("nvidia/parakeet", None, hub_cache, False),
        ("nvidia/parakeet", "main", hub_cache, True),
    ]
    assert weights.read_bytes() == b"weights"
    output = capsys.readouterr().out
    assert output.count("downloading") == 2
    assert output.count("cached") == 1


def test_stt_prepare_rejects_a_foreign_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt = _load_module(
        "service_prepare_stt_foreign",
        "services/stt-server/stt_server/__main__.py",
    )
    foreign = tmp_path / "foreign" / "snapshot"
    foreign.mkdir(parents=True)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(foreign)),
    )

    with pytest.raises(RuntimeError, match="invalid snapshot path"):
        stt._prepare(
            {"model": "nvidia/parakeet", "model_cache": str(tmp_path / "cache")},
            tmp_path,
        )


def test_magpie_prepare_pins_revision_and_validates_the_cached_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("NEMO_CACHE_DIR", "HF_HOME", "HF_XET_HIGH_PERFORMANCE"):
        monkeypatch.delenv(name, raising=False)
    magpie = _load_module(
        "service_prepare_magpie",
        "services/magpie-tts/magpie_tts_server/__main__.py",
    )
    artifact: Path | None = None
    calls: list[tuple[str, str, str, Path, bool]] = []

    def download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: Path,
        force_download: bool,
    ) -> str:
        nonlocal artifact
        calls.append((repo_id, filename, revision, cache_dir, force_download))
        artifact = (
            cache_dir
            / "models--nvidia--magpie"
            / "snapshots"
            / revision
            / filename
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"weights")
        return str(artifact)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=download,
            snapshot_download=lambda **_kwargs: pytest.fail(
                "a pinned .nemo model must use hf_hub_download"
            ),
        ),
    )
    cfg = {
        "model": "nvidia/magpie",
        "model_revision": "abc123",
        "model_cache": str(tmp_path / "cache"),
    }

    magpie._prepare(cfg, tmp_path)
    magpie._prepare(cfg, tmp_path)
    assert artifact is not None
    artifact.write_bytes(b"changed-weights")
    magpie._prepare(cfg, tmp_path)

    hub_cache = tmp_path / "cache" / "huggingface" / "hub"
    assert calls == [
        ("nvidia/magpie", "magpie.nemo", "abc123", hub_cache, False),
        ("nvidia/magpie", "magpie.nemo", "abc123", hub_cache, True),
    ]
    assert artifact.read_bytes() == b"weights"


def test_magpie_prepare_accepts_a_snapshot_symlink_to_the_blob_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    magpie = _load_module(
        "service_prepare_magpie_symlink",
        "services/magpie-tts/magpie_tts_server/__main__.py",
    )
    hub_cache = tmp_path / "cache" / "huggingface" / "hub"
    repo_cache = hub_cache / "models--nvidia--magpie"
    blob = repo_cache / "blobs" / "etag"
    artifact = repo_cache / "snapshots" / "abc123" / "magpie.nemo"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    artifact.parent.mkdir(parents=True)
    artifact.symlink_to(Path("../../blobs/etag"))
    calls: list[bool] = []

    def download(**kwargs) -> str:
        calls.append(kwargs["force_download"])
        return str(artifact)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=download,
            snapshot_download=lambda **_kwargs: pytest.fail(
                "a pinned .nemo model must use hf_hub_download"
            ),
        ),
    )
    cfg = {
        "model": "nvidia/magpie",
        "model_revision": "abc123",
        "model_cache": str(tmp_path / "cache"),
    }

    magpie._prepare(cfg, tmp_path)
    magpie._prepare(cfg, tmp_path)

    assert calls == [False]


def test_magpie_unpinned_repair_rejects_a_corrupt_offline_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    magpie = _load_module(
        "service_prepare_magpie_offline",
        "services/magpie-tts/magpie_tts_server/__main__.py",
    )
    hub_cache = tmp_path / "cache" / "huggingface" / "hub"
    snapshot = hub_cache / "models--nvidia--magpie" / "snapshots" / "revision"
    weights = snapshot / "weights.nemo"

    def download(**_kwargs) -> str:
        snapshot.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            weights.write_bytes(b"weights")
            ref = hub_cache / "models--nvidia--magpie" / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("revision", encoding="utf-8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=_fake_hf_api(
                "revision",
                {"weights.nemo": len(b"weights")},
            ),
            hf_hub_download=lambda **_kwargs: pytest.fail(
                "an unpinned model must use snapshot_download"
            ),
            snapshot_download=download,
        ),
    )
    cfg = {
        "model": "nvidia/magpie",
        "model_cache": str(tmp_path / "cache"),
    }
    magpie._prepare(cfg, tmp_path)
    marker = next((tmp_path / "cache" / ".xr-ai-prepare").glob("hf-magpie-*"))
    marker_before = marker.read_bytes()
    weights.write_bytes(b"corrupt-cache")

    with pytest.raises(RuntimeError, match="repair did not restore"):
        magpie._prepare(cfg, tmp_path)

    assert marker.read_bytes() == marker_before


def test_reasoning_parser_replaces_a_zero_length_file_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano = _load_module(
        "service_prepare_nano_parser",
        "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py",
    )
    target = tmp_path / nano._PARSER_FILENAME
    target.touch()
    replacements: list[tuple[Path, Path]] = []
    real_replace = nano.os.replace

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return b"parser source"

    def replace(source: Path, destination: Path) -> None:
        source = Path(source)
        destination = Path(destination)
        assert source.parent == tmp_path
        assert destination == target
        assert target.read_bytes() == b""
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(nano.urllib.request, "urlopen", lambda _url: Response())
    monkeypatch.setattr(nano.os, "replace", replace)

    assert nano._ensure_reasoning_parser(tmp_path, "https://example/parser") == target
    assert target.read_bytes() == b"parser source"
    assert len(replacements) == 1
    assert not list(tmp_path.glob(f".{nano._PARSER_FILENAME}.*"))


def test_reasoning_parser_rejects_an_empty_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano = _load_module(
        "service_prepare_nano_empty_parser",
        "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py",
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return b""

    monkeypatch.setattr(nano.urllib.request, "urlopen", lambda _url: Response())

    with pytest.raises(RuntimeError, match="is empty"):
        nano._ensure_reasoning_parser(tmp_path, "https://example/parser")

    assert not (tmp_path / nano._PARSER_FILENAME).exists()


def test_device_hub_prepare_prefixes_missing_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import device_io_hub.__main__ as hub_main

    monkeypatch.setattr(hub_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        hub_main,
        "load_artifact_config",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing hub.yaml")),
    )
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--prepare"])

    with pytest.raises(SystemExit) as error:
        hub_main.run()

    assert str(error.value) == "[device_io_hub] missing hub.yaml"


def test_pocket_prepare_loads_artifacts_on_cpu_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    external_hf_home = tmp_path / "external-hf"
    monkeypatch.setenv("HF_HOME", str(external_hf_home))
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    pocket = _load_module(
        "service_prepare_pocket",
        "services/pocket-tts/pocket_tts_server/__main__.py",
    )
    calls: list[tuple[str, str]] = []
    external_hf_home.mkdir()
    unrelated = external_hf_home / "unrelated.bin"
    unrelated.write_bytes(b"unrelated")
    hub_cache = external_hf_home / "hub"
    gated = hub_cache / "models" / "gated.bin"
    fallback = hub_cache / "models" / "fallback.bin"
    tokenizer = hub_cache / "tokenizer" / "tokenizer.model"
    voice = hub_cache / "voices" / "bill_boerst.safetensors"
    gated.parent.mkdir(parents=True)
    gated.write_bytes(b"stale gated weights")
    monkeypatch.setattr(
        pocket,
        "_pocket_artifact_paths",
        lambda *_args: (gated, fallback, tokenizer, voice),
    )
    monkeypatch.setattr(pocket, "_pocket_version", lambda: "3.0.2")
    monkeypatch.setattr(pocket, "_pocket_auth_identity", lambda: "anonymous")

    def fake_load(self) -> None:
        if self._model is not None:
            return
        calls.append((self._requested_device, self._language))
        self._model = SimpleNamespace(has_voice_cloning=False)
        for artifact in (fallback, tokenizer, voice):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"weights")

    monkeypatch.setattr(pocket._PocketTTSBackend, "_ensure_loaded", fake_load)
    cfg = {"voice": "bill_boerst", "language": "english", "model_cache": str(tmp_path)}
    pocket._prepare(cfg, tmp_path)
    marker = pocket._pocket_marker(
        tmp_path / "pocket",
        version="3.0.2",
        hub_cache=hub_cache,
        language="english",
        voice="bill_boerst",
        auth_identity="anonymous",
        selection="fallback",
    )
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert Path(manifest["artifact_root"]) == hub_cache
    assert {entry[0] for entry in manifest["files"]} == {
        "models/fallback.bin",
        "tokenizer/tokenizer.model",
        "voices/bill_boerst.safetensors",
    }
    assert all(entry[1] == len(b"weights") for entry in manifest["files"])

    unrelated.write_bytes(b"unrelated changed after preparation")
    pocket._prepare(cfg, tmp_path)

    fallback.write_bytes(b"changed weights")
    pocket._prepare(cfg, tmp_path)

    assert calls == [("cpu", "english"), ("cpu", "english")]
    assert marker.is_file()
    assert unrelated.is_file()
    output = capsys.readouterr().out
    assert "Pocket TTS english/bill_boerst: downloading" in output
    assert "Pocket TTS english/bill_boerst: cached" in output


def test_pocket_prepare_keys_the_selected_weights_by_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pocket = _load_module(
        "service_prepare_pocket_auth",
        "services/pocket-tts/pocket_tts_server/__main__.py",
    )
    hub_cache = tmp_path / "pocket" / "huggingface" / "hub"
    gated = hub_cache / "models" / "gated.bin"
    fallback = hub_cache / "models" / "fallback.bin"
    tokenizer = hub_cache / "tokenizer" / "tokenizer.model"
    voice = hub_cache / "voices" / "bill_boerst.safetensors"
    auth = {"identity": "anonymous", "gated": False}
    loads: list[bool] = []

    monkeypatch.setattr(
        pocket,
        "_pocket_artifact_paths",
        lambda *_args: (gated, fallback, tokenizer, voice),
    )
    monkeypatch.setattr(pocket, "_pocket_version", lambda: "3.0.2")
    monkeypatch.setattr(
        pocket,
        "_pocket_auth_identity",
        lambda: auth["identity"],
    )

    def fake_load(self) -> None:
        if self._model is not None:
            return
        selected = gated if auth["gated"] else fallback
        loads.append(auth["gated"])
        self._model = SimpleNamespace(has_voice_cloning=auth["gated"])
        for artifact in (selected, tokenizer, voice):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"weights")

    monkeypatch.setattr(pocket._PocketTTSBackend, "_ensure_loaded", fake_load)
    cfg = {
        "voice": "bill_boerst",
        "language": "english",
        "model_cache": str(tmp_path),
    }

    pocket._prepare(cfg, tmp_path)
    pocket._prepare(cfg, tmp_path)
    auth.update(identity="token-digest", gated=True)
    pocket._prepare(cfg, tmp_path)

    assert loads == [False, True]
    assert len(list((tmp_path / "pocket").glob(".xr-ai-prepare-*"))) == 2


@pytest.mark.parametrize(
    ("module_name", "relative_path", "service_name", "prepare_name"),
    [
        (
            "service_prepare_cli_embedding",
            "services/embedding-server/embedding_server/__main__.py",
            "embedding_server",
            "prepare_vllm",
        ),
        (
            "service_prepare_cli_llama",
            "services/llama-nemotron-llm/llama_nemotron_llm_server/__main__.py",
            "llama_nemotron_llm_server",
            "prepare_vllm",
        ),
        (
            "service_prepare_cli_magpie",
            "services/magpie-tts/magpie_tts_server/__main__.py",
            "magpie_tts_server",
            "_prepare",
        ),
        (
            "service_prepare_cli_omni",
            "services/nemotron-omni-llm/nemotron_omni_llm_server/__main__.py",
            "nemotron_omni_llm_server",
            "prepare_vllm",
        ),
        (
            "service_prepare_cli_nano",
            "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py",
            "nemotron3_nano_llm_server",
            "prepare_vllm",
        ),
        (
            "service_prepare_cli_nim",
            "services/nim-server/nim_server/__main__.py",
            "nim_server",
            "prepare_nim",
        ),
        (
            "service_prepare_cli_pocket",
            "services/pocket-tts/pocket_tts_server/__main__.py",
            "pocket_tts_server",
            "_prepare",
        ),
        (
            "service_prepare_cli_stt",
            "services/stt-server/stt_server/__main__.py",
            "stt_server",
            "_prepare",
        ),
        (
            "service_prepare_cli_vlm",
            "services/vlm-server/vlm_server/__main__.py",
            "vlm_server",
            "prepare_vllm",
        ),
    ],
)
def test_download_capable_service_cli_dispatches_prepare_without_serving(
    module_name: str,
    relative_path: str,
    service_name: str,
    prepare_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(module_name, relative_path)
    prepared: list[str] = []
    dispatched: list[str] = []
    prepare_signature = inspect.signature(getattr(module, prepare_name))

    def fake_prepare(*args, **kwargs) -> None:
        prepare_signature.bind(*args, **kwargs)
        prepared.append(prepare_name)

    monkeypatch.setattr(module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "prepare_or_exit",
        lambda name, action: (dispatched.append(name), action()),
    )
    monkeypatch.setattr(
        module,
        prepare_name,
        fake_prepare,
    )
    if hasattr(module, "load_config"):
        monkeypatch.setattr(
            module,
            "load_config",
            lambda: (
                {
                    "image": "example/image:1",
                    "http_port": 8000,
                    "model": "example/model",
                    "use_bf16": True,
                },
                tmp_path,
                None,
            ),
        )
    if hasattr(module, "source_config_digest"):
        monkeypatch.setattr(module, "source_config_digest", lambda: "digest")
    if hasattr(module, "resolve_model_cache"):
        monkeypatch.setattr(
            module,
            "resolve_model_cache",
            lambda *_args, **_kwargs: tmp_path,
        )
    if hasattr(module, "setup_hf_env"):
        monkeypatch.setattr(module, "setup_hf_env", lambda *_args, **_kwargs: None)
    if hasattr(module, "gpu_compute_major"):
        monkeypatch.setattr(module, "gpu_compute_major", lambda: 8)
    if hasattr(module, "_ensure_reasoning_parser"):
        monkeypatch.setattr(
            module,
            "_ensure_reasoning_parser",
            lambda *_args, **_kwargs: tmp_path / "reasoning_parser.py",
        )
    if hasattr(module, "serve"):
        monkeypatch.setattr(
            module,
            "serve",
            lambda **_kwargs: pytest.fail("--prepare must not start the service"),
        )
    if hasattr(module, "serve_nim"):
        monkeypatch.setattr(
            module,
            "serve_nim",
            lambda **_kwargs: pytest.fail("--prepare must not start the service"),
        )
    if service_name in {"stt_server", "pocket_tts_server"}:
        monkeypatch.setattr(
            module,
            "_health_url_ok",
            lambda *_args, **_kwargs: pytest.fail(
                "--prepare must not probe or start the service"
            ),
        )
    if service_name == "magpie_tts_server":
        monkeypatch.setattr(
            module.asyncio,
            "run",
            lambda _coroutine: pytest.fail("--prepare must not start the service"),
        )
    monkeypatch.setattr(sys, "argv", [service_name, "--prepare"])

    module.run()

    assert dispatched == [service_name]
    assert prepared == [prepare_name]


def test_device_hub_prepare_dispatches_the_real_artifact_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import device_io_hub.__main__ as hub_main
    from device_io_hub import _prepare as hub_prepare
    from device_io_hub._config_loader import DeviceIOHubArtifactConfig

    config = DeviceIOHubArtifactConfig(
        web_client_dir=str(tmp_path / "web"),
        web_xr_vendor_build_script=str(tmp_path / "build.sh"),
    )
    calls: list[object] = []
    monkeypatch.setattr(hub_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub_main, "load_artifact_config", lambda: config)
    monkeypatch.setattr(
        hub_main,
        "load_config",
        lambda: pytest.fail("artifact preparation must not load runtime credentials"),
    )
    monkeypatch.setattr(
        hub_main,
        "prepare_or_exit",
        lambda name, action: (calls.append(name), action()),
    )
    monkeypatch.setattr(
        hub_prepare,
        "prepare_livekit_image",
        lambda: calls.append("image"),
    )
    monkeypatch.setattr(
        hub_prepare,
        "prepare_web_xr_vendor",
        lambda web_client_dir, build_script: calls.append(
            (web_client_dir, build_script)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["device_io_hub", "--prepare"])

    hub_main.run()

    assert calls == [
        "device_io_hub",
        "image",
        (config.web_client_dir, config.web_xr_vendor_build_script),
    ]


def test_scene_cli_dispatches_prepare_without_serving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xr_render_scene import __main__ as scene_main

    config = tmp_path / "scene.yaml"
    config.write_text("{}\n", encoding="utf-8")
    lovr = tmp_path / "lovr"
    dispatched: list[str] = []
    prepared: list[Path] = []
    monkeypatch.delenv("LOVR_BIN", raising=False)
    monkeypatch.setattr(scene_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scene_main,
        "prepare_or_exit",
        lambda name, action: (dispatched.append(name), action()),
    )
    monkeypatch.setattr(
        scene_main,
        "prepare_lovr",
        lambda path, **_kwargs: prepared.append(path) or lovr,
    )
    monkeypatch.setattr(
        scene_main.asyncio,
        "run",
        lambda _coroutine: pytest.fail("--prepare must not start the scene service"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["xr_render_scene", "--config", str(config), "--prepare"],
    )

    scene_main.run()

    assert dispatched == ["xr-render-scene"]
    assert prepared == [config]
    assert "LOVR_BIN" not in scene_main.os.environ
