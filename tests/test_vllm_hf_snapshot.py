# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first model resolution for the Docker vLLM backend."""
from __future__ import annotations

import json
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


_SHARD_INDEX = json.dumps(
    {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
)
_PARTIAL_CACHES = {
    "no-config": {"model.safetensors": "w", "tokenizer.json": "{}"},
    "no-weights": {"config.json": "{}", "tokenizer.json": "{}"},
    "no-tokenizer": {"config.json": "{}", "model.safetensors": "w"},
    "vocab-without-merges": {
        "config.json": "{}",
        "vocab.json": "{}",
        "model.safetensors": "w",
    },
    "no-processor": {
        "config.json": '{"vision_config": {}}',
        "tokenizer.json": "{}",
        "model.safetensors": "w",
    },
    "no-processor-for-sound": {
        "config.json": '{"sound_config": {}}',
        "tokenizer.json": "{}",
        "model.safetensors": "w",
    },
    "no-remote-processor-code": {
        "config.json": '{"vision_config": {}}',
        "preprocessor_config.json": json.dumps(
            {"auto_map": {"AutoProcessor": "processing.Processor"}}
        ),
        "tokenizer.json": "{}",
        "model.safetensors": "w",
    },
    "no-remote-tokenizer-code": {
        "config.json": "{}",
        "tokenizer_config.json": json.dumps(
            {"auto_map": {"AutoTokenizer": ["tok.Slow", None]}}
        ),
        "tokenizer.json": "{}",
        "model.safetensors": "w",
    },
    "no-remote-helper-module": {
        "config.json": json.dumps({"auto_map": {"AutoModel": "modeling.Model"}}),
        "modeling.py": "from .modeling_utils import helper\n",
        "tokenizer.json": "{}",
        "model.safetensors": "w",
    },
    "missing-shard": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "model.safetensors.index.json": _SHARD_INDEX,
        "model-1.safetensors": "w",
    },
    "missing-index-and-shard": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "model-00001-of-00002.safetensors": "w",
    },
    "subfolder-shards-in-root-index": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": "transformer/m-1.safetensors", "b": "vision/m.safetensors"}}
        ),
        "transformer/m-1.safetensors": "w",
    },
    "subfolder-shards-without-index": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "transformer/m-00001-of-00002.safetensors": "w",
    },
    "trainer-output-only": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "training_args.bin": "w",
    },
}
_BIN_INDEX = json.dumps(
    {"weight_map": {"a": "pytorch_model-1.bin", "b": "pytorch_model-2.bin"}}
)
# Complete safetensors next to an incomplete format vLLM never loads.
_UNUSED_PARTIAL_FORMATS = {
    "partial-bin-index": {
        "pytorch_model.bin.index.json": _BIN_INDEX,
        "pytorch_model-1.bin": "w",
    },
    "partial-bin-shards": {"pytorch_model-00001-of-00002.bin": "w"},
    "partial-pt": {"model-00001-of-00003.pt": "w"},
}


class TestLocalResolution:
    def test_complete_cache_hit_makes_no_network_request(
        self, tmp_path, monkeypatch, capsys
    ):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        snapshot = _complete_snapshot(cache)
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.main(["local", _REPO, ""]) == 0

        assert capsys.readouterr().out == f"{snapshot}\n"
        assert attempts == []

    @pytest.mark.parametrize("offline", [False, True])
    @pytest.mark.parametrize(
        "files", _UNUSED_PARTIAL_FORMATS.values(), ids=_UNUSED_PARTIAL_FORMATS
    )
    def test_unused_partial_format_does_not_reject_selected_weights(
        self, tmp_path, monkeypatch, capsys, files, offline
    ):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=offline)
        snapshot = _complete_snapshot(cache)
        for name, content in files.items():
            cache_file(cache, _REPO, name, content)
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.main(["local", _REPO, ""]) == 0

        assert capsys.readouterr().out == f"{snapshot}\n"
        assert attempts == []

    def test_subfolder_weights_in_root_index_are_a_hit(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        index = {"weight_map": {"a": "transformer/m-1.safetensors", "b": "vision/m.safetensors"}}
        cache_file(cache, _REPO, "config.json", "{}")
        cache_file(cache, _REPO, "tokenizer.json", "{}")
        cache_file(cache, _REPO, "model.safetensors.index.json", json.dumps(index))
        cache_file(cache, _REPO, "transformer/m-1.safetensors")
        # Components outside the index are not loaded by vLLM.
        cache_file(cache, _REPO, "vae/m-00001-of-00002.safetensors")
        snapshot = cache_file(cache, _REPO, "vision/m.safetensors").parent.parent
        block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, None) == snapshot

    def test_complete_bin_weights_are_a_hit(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        cache_file(cache, _REPO, "config.json", "{}")
        cache_file(cache, _REPO, "tokenizer.json", "{}")
        cache_file(cache, _REPO, "training_args.bin")
        cache_file(cache, _REPO, "pytorch_model.bin.index.json", _BIN_INDEX)
        cache_file(cache, _REPO, "pytorch_model-1.bin")
        snapshot = cache_file(cache, _REPO, "pytorch_model-2.bin").parent
        block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, None) == snapshot

    def test_sharded_snapshot_with_every_shard_is_a_hit(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        index = {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
        cache_file(cache, _REPO, "config.json", "{}")
        cache_file(cache, _REPO, "tokenizer.json", "{}")
        cache_file(cache, _REPO, "model.safetensors.index.json", json.dumps(index))
        cache_file(cache, _REPO, "model-1.safetensors")
        snapshot = cache_file(cache, _REPO, "model-2.safetensors").parent

        assert _hf_snapshot.resolve_local(_REPO, None) == snapshot

    def test_remote_code_snapshot_with_its_modules_is_a_hit(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        auto_map = {"AutoModel": "modeling.Model", "AutoConfig": "other--remote.Config"}
        _complete_snapshot(cache)
        cache_file(cache, _REPO, "config.json", json.dumps({"auto_map": auto_map}))
        cache_file(cache, _REPO, "modeling.py", "from .utils import x\nimport torch\n")
        snapshot = cache_file(cache, _REPO, "utils.py").parent
        block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, None) == snapshot

    def test_pinned_revision_resolves_its_snapshot(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        snapshot = _complete_snapshot(cache, revision=None)
        block_network(monkeypatch)

        assert _hf_snapshot.resolve_local(_REPO, COMMIT) == snapshot

    @pytest.mark.parametrize("offline", [False, True])
    @pytest.mark.parametrize("files", _PARTIAL_CACHES.values(), ids=_PARTIAL_CACHES)
    def test_incomplete_cache_is_not_a_hit(
        self, tmp_path, monkeypatch, capsys, files, offline
    ):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=offline)
        for name, content in files.items():
            cache_file(cache, _REPO, name, content)
        attempts = block_network(monkeypatch)

        status = _hf_snapshot.main(["local", _REPO, ""])

        # Online falls through to the download path; offline fails explicitly.
        assert status == (1 if offline else _hf_snapshot.CACHE_MISS)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "incomplete cached snapshot" in captured.err
        if offline:
            assert "org/model (revision main) is not fully cached" in captured.err
        assert attempts == []

    def test_local_model_directory_is_served_as_configured(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)

        assert _hf_snapshot.resolve_local(str(tmp_path), None) == tmp_path

    def test_absent_snapshot_is_a_miss(self, tmp_path, monkeypatch, capsys):
        use_hub_cache(monkeypatch, tmp_path / "hub")
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.main(["local", _REPO, ""]) == _hf_snapshot.CACHE_MISS
        assert capsys.readouterr().out == ""
        assert attempts == []

    def test_explicit_offline_miss_names_model_and_revision(
        self, tmp_path, monkeypatch, capsys
    ):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)
        attempts = block_network(monkeypatch)

        assert _hf_snapshot.main(["local", _REPO, "v1.2"]) == 1

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "org/model (revision v1.2) is not fully cached" in captured.err
        assert "HF_HUB_OFFLINE" in captured.err
        assert attempts == []


class TestDownload:
    def test_downloads_flushes_and_prints_snapshot(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / "hub"
        snapshot = _complete_snapshot(cache)
        fetch = Mock(return_value=str(snapshot))
        sync = Mock()
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
        monkeypatch.setattr(_hf_snapshot.os, "sync", sync)

        assert _hf_snapshot.main(["download", _REPO, "v1.2"]) == 0

        fetch.assert_called_once_with(repo_id=_REPO, revision="v1.2")
        sync.assert_called_once_with()
        assert capsys.readouterr().out == f"{snapshot}\n"

    def test_incomplete_download_fails(self, tmp_path, monkeypatch):
        cache_file(tmp_path / "hub", _REPO, "tokenizer.json", "{}")
        partial = cache_file(tmp_path / "hub", _REPO, "config.json", "{}").parent
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", Mock(return_value=str(partial))
        )

        with pytest.raises(RuntimeError, match="missing: model weights"):
            _hf_snapshot.main(["download", _REPO, ""])


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
        assert "org/model (revision main) is not fully cached" in result.stderr


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
