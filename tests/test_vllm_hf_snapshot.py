# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first model resolution for the Docker vLLM backend."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call

import huggingface_hub
import pytest
from _helpers_hf import COMMIT, block_network, cache_file, use_hub_cache
from xr_ai_vllm import _docker, _hf_snapshot
from xr_ai_vllm._docker import build_run_argv

_REPO = "org/model"


@pytest.mark.parametrize("offline,cached", [(False, True), (True, True), (False, False), (True, False)])
def test_local_resolution(tmp_path, monkeypatch, capsys, offline, cached):
    cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=offline)
    # The resolver does not impose a model layout or prove completeness.
    snapshot = cache_file(cache, _REPO, "params.json").parent if cached else None
    attempts = block_network(monkeypatch)
    status = _hf_snapshot.main(["local", _REPO, ""])
    output = capsys.readouterr()
    assert attempts == []
    if cached:
        assert status == 0 and output.out == f"{snapshot}\n"
    else:
        assert status == (1 if offline else _hf_snapshot.CACHE_MISS)
        assert output.out == ""
        if offline:
            assert _REPO in output.err and "revision main" in output.err
            assert str(cache) in output.err and "HF_HUB_OFFLINE" in output.err


def test_revision_and_explicit_directory(tmp_path, monkeypatch):
    cache = use_hub_cache(monkeypatch, tmp_path / "hub")
    snapshot = cache_file(cache, _REPO, "params.json", revision=None).parent
    attempts = block_network(monkeypatch)
    assert _hf_snapshot.resolve_local(_REPO, COMMIT) == snapshot
    assert _hf_snapshot.resolve_local(_REPO, "other") is None
    assert _hf_snapshot.resolve_local(str(tmp_path), None) == tmp_path
    assert attempts == []


def test_download_syncs_before_returning_path(tmp_path, monkeypatch, capsys):
    calls = Mock()

    def fetch(**kwargs):
        calls.fetch(**kwargs)
        print("download log")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    monkeypatch.setattr(_hf_snapshot.os, "sync", calls.sync, raising=False)
    assert _hf_snapshot.main(["download", _REPO, COMMIT]) == 0
    assert calls.mock_calls == [call.fetch(repo_id=_REPO, revision=COMMIT), call.sync()]
    output = capsys.readouterr()
    assert output.out == f"{tmp_path}\n" and "download log" in output.err


def test_embedded_source_runs_in_container_python(tmp_path):
    snapshot = cache_file(tmp_path, _REPO, "params.json").parent
    env = dict(os.environ)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", _docker._HF_SNAPSHOT_CODE, "local", _REPO, ""],
        env=env | {"HF_HUB_CACHE": str(tmp_path), "HF_ENDPOINT": "http://127.0.0.1:9"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{snapshot}\n"


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

    @pytest.mark.parametrize("cached,status", [(True, 0), (False, _hf_snapshot.CACHE_MISS), (False, 1)])
    @pytest.mark.parametrize("name_args", [[], ["--served-model-name", "served"]])
    def test_resolve_then_serve(self, tmp_path, cached, status, name_args):
        command = self._command(tmp_path, [*name_args, "--revision", COMMIT])
        result, steps, served = self._run(
            tmp_path, command, CACHED="/cache/model" if cached else "",
            LOCAL_STATUS=str(status), DOWNLOADED="/cache/model",
        )
        expected_steps = [f"local {_REPO} {COMMIT}"]
        if status == _hf_snapshot.CACHE_MISS:
            expected_steps += ["xet", "xet", f"download {_REPO} {COMMIT}"]
        assert steps == expected_steps
        if status == 1:
            assert result.returncode == 1 and served is None
        else:
            assert result.returncode == 0, result.stderr
            assert served == [
                "serve", "/cache/model", "--served-model-name",
                "served" if name_args else _REPO, "--revision", COMMIT,
            ]
