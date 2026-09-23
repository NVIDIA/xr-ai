# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LOVR cache preparation without starting an XR session."""

from __future__ import annotations

import io
import os
from hashlib import sha256
from pathlib import Path

import pytest
from xr_render_scene import _prepare


@pytest.fixture
def scene_cache(tmp_path, monkeypatch):
    config = tmp_path / "scene.yaml"
    config.write_text("{}")
    cache_home = tmp_path / "cache"
    cache = cache_home / "xr-ai" / "lovr" / _prepare._VERSION
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.delenv("LOVR_BIN", raising=False)
    monkeypatch.setattr(_prepare.sys, "platform", "linux")
    monkeypatch.setattr(_prepare.platform, "machine", lambda: "x86_64")
    payload = b"executable"
    monkeypatch.setattr(_prepare, "_SHA256", sha256(payload).hexdigest())
    return config, cache, payload


def test_downloaded_lovr_is_executable_and_second_prepare_is_offline(scene_cache, monkeypatch, capsys):
    config, _, payload = scene_cache
    downloads = []

    def download(url, **kwargs):
        downloads.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(_prepare.urllib.request, "urlopen", download)
    executable = _prepare.prepare_lovr(config)
    assert os.access(executable, os.X_OK)
    assert _prepare.prepare_lovr(config) == executable
    assert len(downloads) == 1
    assert "cached (size: 10 B)" in capsys.readouterr().out


def test_partial_download_is_not_promoted(scene_cache, monkeypatch):
    config, cache, _ = scene_cache
    monkeypatch.setattr(_prepare.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO())
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _prepare.prepare_lovr(config)
    assert not (cache / _prepare._ASSET).exists()


def test_corrupt_cached_lovr_is_replaced(scene_cache, monkeypatch):
    config, cache, payload = scene_cache
    executable = cache / _prepare._ASSET
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"corrupt")
    monkeypatch.setattr(
        _prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    assert _prepare.prepare_lovr(config) == executable
    assert executable.read_bytes() == payload


def test_executable_cached_lovr_does_not_require_chmod(scene_cache, monkeypatch):
    config, cache, payload = scene_cache
    executable = cache / _prepare._ASSET
    executable.parent.mkdir(parents=True)
    executable.write_bytes(payload)
    executable.chmod(0o555)

    def reject_chmod(self, _mode):
        if self == executable:
            raise OSError("read-only filesystem")
        raise AssertionError(f"unexpected chmod: {self}")

    monkeypatch.setattr(Path, "chmod", reject_chmod)
    monkeypatch.setattr(
        _prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("an executable cached LOVR must remain offline"),
    )

    assert _prepare.resolve_lovr(config) == executable
    assert _prepare.prepare_lovr(config) == executable


def test_custom_lovr_allows_aarch64(scene_cache, monkeypatch, tmp_path):
    config, _, _ = scene_cache
    executable = tmp_path / "custom-lovr"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    config.write_text(f"lovr_bin: {executable}\n")
    monkeypatch.setattr(_prepare.platform, "machine", lambda: "aarch64")
    assert _prepare.prepare_lovr(config) == executable


def test_missing_custom_binary_fails_before_download(scene_cache, monkeypatch):
    config, _, _ = scene_cache
    monkeypatch.setenv("LOVR_BIN", "/missing/custom-lovr")
    with pytest.raises(RuntimeError, match="required an executable"):
        _prepare.prepare_lovr(config)


def test_yaml_cache_override_is_relative_to_the_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "scene.yaml"
    config.write_text("lovr_cache_dir: ../artifacts/lovr\n")
    monkeypatch.delenv("LOVR_BIN", raising=False)
    monkeypatch.setattr(_prepare.sys, "platform", "linux")
    monkeypatch.setattr(_prepare.platform, "machine", lambda: "x86_64")
    payload = b"executable"
    monkeypatch.setattr(_prepare, "_SHA256", sha256(payload).hexdigest())
    monkeypatch.setattr(
        _prepare.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    executable = _prepare.prepare_lovr(config)

    assert executable.parent == (tmp_path / "artifacts" / "lovr").resolve()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(RuntimeError("download failed"), "[xr-render-scene] download failed"), (KeyboardInterrupt(), 130)],
)
def test_scene_cli_translates_prepare_failures(
    tmp_path,
    monkeypatch,
    failure,
    expected,
):
    from xr_render_scene import __main__ as scene_main

    config = tmp_path / "scene.yaml"
    config.write_text("{}")
    monkeypatch.setattr(scene_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scene_main,
        "prepare_lovr",
        lambda _config: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        scene_main.sys,
        "argv",
        ["xr_render_scene", "--prepare", "--config", str(config)],
    )

    with pytest.raises(SystemExit) as error:
        scene_main.run()

    assert error.value.code == expected


def test_scene_normal_start_reports_unprepared_lovr_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xr_render_scene import __main__ as scene_main

    config = tmp_path / "scene.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(scene_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scene_main,
        "resolve_lovr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("LOVR is not prepared; run --prepare")),
    )
    monkeypatch.setattr(
        scene_main.asyncio,
        "run",
        lambda _coroutine: pytest.fail("LOVR failure must prevent service startup"),
    )
    monkeypatch.setattr(
        scene_main.sys,
        "argv",
        ["xr_render_scene", "--config", str(config)],
    )

    with pytest.raises(
        SystemExit,
        match="xr-render-scene: LOVR is not prepared; run --prepare",
    ):
        scene_main.run()


def test_scene_normal_start_uses_cached_lovr_without_preparing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xr_render_scene import __main__ as scene_main

    config = tmp_path / "scene.yaml"
    config.write_text("{}\n", encoding="utf-8")
    executable = tmp_path / "lovr"
    executable.write_bytes(b"prepared")
    monkeypatch.setenv("LOVR_BIN", "test-placeholder")
    monkeypatch.setattr(scene_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scene_main, "resolve_lovr", lambda _config: executable)
    monkeypatch.setattr(
        scene_main,
        "prepare_lovr",
        lambda *_args, **_kwargs: pytest.fail("normal start must not prepare LOVR"),
    )
    calls: list[tuple[Path, Path | None]] = []

    async def fake_serve(config_path: Path, ready_file: Path | None) -> None:
        calls.append((config_path, ready_file))

    monkeypatch.setattr(scene_main, "_serve", fake_serve)
    monkeypatch.setattr(
        scene_main.sys,
        "argv",
        ["xr_render_scene", "--config", str(config)],
    )

    scene_main.run()

    assert scene_main.os.environ["LOVR_BIN"] == str(executable)
    assert calls == [(config, None)]
