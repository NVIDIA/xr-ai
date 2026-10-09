# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first model resolution for the Docker vLLM backend."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest
from _helpers_hf import COMMIT, block_network, cache_file, use_hub_cache
from xr_ai_vllm import _docker, _hf_snapshot
from xr_ai_vllm._docker import build_run_argv

_REPO = "org/model"


def _complete_snapshot(cache: Path, **kwargs) -> Path:
    cache_file(cache, _REPO, "config.json", "{}", **kwargs)
    cache_file(cache, _REPO, "tokenizer.json", "{}", **kwargs)
    return cache_file(cache, _REPO, "model.safetensors", "w", **kwargs).parent


class TestLocalResolution:
    @pytest.mark.parametrize("offline", [False, True])
    def test_cache_hit_makes_no_network_request(self, tmp_path, monkeypatch, capsys, offline):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=offline)
        snapshot = _complete_snapshot(cache)
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.main(["local", _REPO, ""]) == 0
        assert capsys.readouterr().out == f"{snapshot}\n"
        assert attempts == []

    def test_snapshot_contents_are_left_to_the_loader(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        snapshot = cache_file(cache, _REPO, "params.json", "{}").parent
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, None) == snapshot
        assert attempts == []

    def test_pinned_revision_resolves_its_snapshot(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        snapshot = _complete_snapshot(cache, revision=None)
        block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, COMMIT) == snapshot
        assert _hf_snapshot.resolve_local(_REPO, "other") is None

    def test_local_model_directory_is_served_as_configured(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)
        assert _hf_snapshot.resolve_local(str(tmp_path), None) == tmp_path

    def test_absent_snapshot_is_a_miss(self, tmp_path, monkeypatch, capsys):
        use_hub_cache(monkeypatch, tmp_path / "hub")
        attempts = block_network(monkeypatch)
        assert _hf_snapshot.main(["local", _REPO, ""]) == _hf_snapshot.CACHE_MISS
        assert capsys.readouterr().out == ""
        assert attempts == []

    def test_explicit_offline_miss_names_model_revision_and_cache(self, tmp_path, monkeypatch, capsys):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)
        attempts = block_network(monkeypatch)
        assert _hf_snapshot.main(["local", _REPO, "v1.2"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "org/model (revision v1.2) is not cached" in captured.err
        assert str(cache) in captured.err
        assert "HF_HUB_OFFLINE" in captured.err
        assert attempts == []


class TestDownload:
    def test_downloads_flushes_and_prints_snapshot(self, tmp_path, monkeypatch, capsys):
        snapshot = _complete_snapshot(tmp_path / "hub")
        fetch = Mock(return_value=str(snapshot))
        sync = Mock()
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
        monkeypatch.setattr(_hf_snapshot.os, "sync", sync, raising=False)
        assert _hf_snapshot.main(["download", _REPO, "v1.2"]) == 0
        fetch.assert_called_once_with(repo_id=_REPO, revision="v1.2")
        sync.assert_called_once_with()
        assert capsys.readouterr().out == f"{snapshot}\n"


class TestContainerInvocation:
    """The embedded source as the container runs it: ``python3 -c``."""

    def _run(self, cache: Path, *args: str, offline: bool) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "HF_HUB_CACHE": str(cache),
            # Any Hub request would fail against this endpoint.
            "HF_ENDPOINT": "http://127.0.0.1:9",
        }
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        return subprocess.run(
            [sys.executable, "-c", _docker._HF_SNAPSHOT_CODE, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_cache_hit_prints_snapshot(self, tmp_path):
        snapshot = _complete_snapshot(tmp_path / "hub")

        result = self._run(tmp_path / "hub", "local", _REPO, "", offline=False)

        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{snapshot}\n"

    def test_offline_miss_fails_with_model_and_revision(self, tmp_path):
        result = self._run(tmp_path / "hub", "local", _REPO, "", offline=True)

        assert result.returncode == 1
        assert result.stdout == ""
        assert "org/model (revision main) is not cached" in result.stderr


_FAKE_PYTHON = """#!/bin/sh
# Container python3 stand-in: python3 -c <code> <mode> <repo> <revision>.
case "$3" in
  local)
    echo "local $4 $5" >> "$CALLS"
    [ -n "$CACHED" ] && { echo "$CACHED"; exit 0; }
    exit "$LOCAL_STATUS" ;;
  download)
    echo "download $4 $5" >> "$CALLS"
    echo "$DOWNLOADED" ;;
  *)
    echo "xet" >> "$CALLS" ;;
esac
"""

_FAKE_VLLM = """#!/bin/sh
printf '%s\\n' "$@" > "$VLLM_ARGS"
"""


class TestContainerCommand:
    """The bootstrap shell as Docker runs it, with stand-in executables."""

    def _command(self, tmp_path: Path, serve_args: list[str]) -> str:
        return build_run_argv(
            image="nvcr.io/nvidia/vllm:26.09-py3",
            container_name="xr-ai-vllm-test",
            port=8100,
            model_cache=tmp_path / "models",
            hf_token=None,
            cuda_visible_devices=None,
            extra_env=None,
            extra_pip=None,
            vllm_argv=["vllm", "serve", _REPO, *serve_args],
        )[-1]

    def _run(self, tmp_path: Path, command: str, **env: str):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (("python3", _FAKE_PYTHON), ("vllm", _FAKE_VLLM)):
            (bin_dir / name).write_text(script)
            (bin_dir / name).chmod(0o755)
        calls, vllm_args = tmp_path / "calls", tmp_path / "vllm_args"
        result = subprocess.run(
            ["bash", "-c", command],
            env={
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "CALLS": str(calls),
                "VLLM_ARGS": str(vllm_args),
                "CACHED": "",
                "LOCAL_STATUS": str(_hf_snapshot.CACHE_MISS),
                "DOWNLOADED": "",
                **env,
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        served = vllm_args.read_text().splitlines() if vllm_args.exists() else None
        steps = calls.read_text().splitlines() if calls.exists() else []
        return result, steps, served

    def test_cache_hit_serves_snapshot_without_download(self, tmp_path):
        command = self._command(tmp_path, ["--served-model-name", "served", "--port", "8100"])

        result, steps, served = self._run(tmp_path, command, CACHED="/cache/snapshot")

        assert result.returncode == 0, result.stderr
        assert steps == [f"local {_REPO} "]
        assert served == [
            "serve", "/cache/snapshot", "--served-model-name", "served", "--port", "8100",
        ]

    def test_cache_miss_downloads_then_serves_downloaded_snapshot(self, tmp_path):
        command = self._command(tmp_path, ["--served-model-name", "served"])

        result, steps, served = self._run(tmp_path, command, DOWNLOADED="/cache/fresh")

        assert result.returncode == 0, result.stderr
        assert steps == [f"local {_REPO} ", "xet", "xet", f"download {_REPO} "]
        assert served == ["serve", "/cache/fresh", "--served-model-name", "served"]

    def test_resolver_failure_stops_before_download_and_vllm(self, tmp_path):
        command = self._command(tmp_path, ["--served-model-name", "served"])

        result, steps, served = self._run(tmp_path, command, LOCAL_STATUS="1")

        assert result.returncode == 1
        assert steps == [f"local {_REPO} "]
        assert served is None

    def test_revision_selects_the_resolved_snapshot(self, tmp_path):
        command = self._command(
            tmp_path, ["--served-model-name", "served", "--revision", COMMIT]
        )

        _result, steps, served = self._run(tmp_path, command, CACHED="/cache/pinned")

        assert steps == [f"local {_REPO} {COMMIT}"]
        assert served[:2] == ["serve", "/cache/pinned"]

    def test_served_name_defaults_to_model_id(self, tmp_path):
        command = self._command(tmp_path, ["--port", "8100"])

        _result, _steps, served = self._run(tmp_path, command, CACHED="/cache/snapshot")

        assert served == [
            "serve", "/cache/snapshot", "--served-model-name", _REPO, "--port", "8100",
        ]
