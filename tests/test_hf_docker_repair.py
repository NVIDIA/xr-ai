# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker Hugging Face repair regressions without starting Docker."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from xr_ai_vllm import _prepare as vllm_prepare


def _fake_hub(
    hub_cache: Path,
    *,
    resolved: str,
    returned: str,
    artifact: bytes,
    cached_ref: str,
    calls: list[dict[str, object]],
) -> SimpleNamespace:
    class HfApi:
        def __init__(self, *, token: str | None = None) -> None:
            self.token = token

        def repo_info(self, *, repo_id: str, revision: str):
            assert repo_id == "org/model"
            assert revision == "main"
            return SimpleNamespace(sha=resolved)

        def list_repo_tree(
            self,
            *,
            repo_id: str,
            revision: str,
            recursive: bool,
        ):
            assert repo_id == "org/model"
            assert revision == resolved
            assert recursive
            return [SimpleNamespace(path="weights", size=len(b"weights"))]

    def snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        repo_cache = hub_cache / "models--org--model"
        snapshot = repo_cache / "snapshots" / returned
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "weights").write_bytes(artifact)
        ref = repo_cache / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(cached_ref, encoding="utf-8")
        return str(snapshot)

    return SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download)


def _run_download_code(
    monkeypatch: pytest.MonkeyPatch,
    hub: SimpleNamespace,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(sys, "argv", ["hf-repair", "org/model", "1"])
    monkeypatch.setattr(os, "sync", lambda: None)
    exec(compile(vllm_prepare._HF_DOWNLOAD_CODE, "<hf-repair>", "exec"), {})


def test_docker_download_code_repairs_and_publishes_the_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    hub = _fake_hub(
        tmp_path,
        resolved="new",
        returned="new",
        artifact=b"weights",
        cached_ref="new",
        calls=calls,
    )

    _run_download_code(monkeypatch, hub)

    snapshot = tmp_path / "models--org--model" / "snapshots" / "new"
    assert capsys.readouterr().out == f"{snapshot}\n"
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "main",
            "force_download": True,
        }
    ]


@pytest.mark.parametrize(
    ("artifact", "cached_ref", "message"),
    [
        (b"corrupt-cache", "new", "repair did not restore"),
        (b"weights", "old", "left org/model@main at 'old'"),
    ],
)
def test_docker_download_code_rejects_corrupt_or_stale_fallback(
    artifact: bytes,
    cached_ref: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = _fake_hub(
        tmp_path,
        resolved="new",
        returned="new",
        artifact=artifact,
        cached_ref=cached_ref,
        calls=[],
    )

    with pytest.raises(RuntimeError, match=message):
        _run_download_code(monkeypatch, hub)


@pytest.mark.parametrize("prepared_marker", [False, True])
@pytest.mark.parametrize(
    "reported_path",
    ["empty", "foreign", "missing", "wrong-repo"],
)
def test_docker_prepare_does_not_create_or_refresh_marker_for_invalid_output(
    prepared_marker: bool,
    reported_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_cache = tmp_path / "hub"
    expected = hub_cache / "models--org--model" / "snapshots" / "revision"
    weights = expected / "weights"
    marker = vllm_prepare._snapshot_marker(
        tmp_path,
        "org/model",
        backend="docker",
        hub_cache=hub_cache,
    )
    marker_before: bytes | None = None
    if prepared_marker:
        expected.mkdir(parents=True)
        weights.write_bytes(b"weights")
        vllm_prepare._write_snapshot_marker(marker, expected)
        marker_before = marker.read_bytes()
        weights.write_bytes(b"corrupt-cache")

    if reported_path == "empty":
        stdout = ""
    elif reported_path == "foreign":
        path = tmp_path / "foreign" / "snapshot"
        path.mkdir(parents=True)
        stdout = str(path)
    elif reported_path == "missing":
        stdout = str(hub_cache / "models--org--model" / "snapshots" / "missing")
    else:
        path = hub_cache / "models--other--model" / "snapshots" / "revision"
        path.mkdir(parents=True)
        stdout = str(path)

    monkeypatch.setattr(
        vllm_prepare,
        "_run_download_container",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout=stdout,
        ),
    )

    with pytest.raises(RuntimeError):
        vllm_prepare._prepare_docker_snapshot(
            image="example/vllm:1",
            model="org/model",
            model_cache=tmp_path,
            hf_token=None,
        )

    if marker_before is None:
        assert not marker.exists()
    else:
        assert marker.read_bytes() == marker_before
