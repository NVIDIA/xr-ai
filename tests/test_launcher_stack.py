# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._stack data structures and pure helpers."""
from __future__ import annotations

import os

import pytest

from xr_ai_launcher._stack import Parallel, Process, _strip_conflicting_cudnn


class TestProcessDataclass:
    def test_defaults(self):
        p = Process("hub", "../../server-runtime", "xr_media_hub")
        assert p.name == "hub"
        assert p.project == "../../server-runtime"
        assert p.command == "xr_media_hub"
        assert p.config is None
        assert p.gpu is None
        assert p.launch_mode == "own"
        assert p.port is None

    def test_all_fields(self):
        p = Process(
            "vlm", "../../ai-services/vlm-server", "vlm_server",
            config="yaml/vlm.yaml",
            gpu="0",
            launch_mode="persist",
            port=8100,
        )
        assert p.config == "yaml/vlm.yaml"
        assert p.gpu == "0"
        assert p.launch_mode == "persist"
        assert p.port == 8100

    def test_frozen_immutability(self):
        p = Process("hub", "../../server-runtime", "xr_media_hub")
        with pytest.raises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]

    def test_reuse_launch_mode(self):
        p = Process("stt", "../../ai-services/stt-server", "stt_server",
                    launch_mode="reuse")
        assert p.launch_mode == "reuse"


class TestParallelDataclass:
    def test_stores_processes_as_tuple(self):
        p1 = Process("stt", "../../ai-services/stt-server", "stt_server")
        p2 = Process("tts", "../../ai-services/tts/piper", "piper_tts_server")
        group = Parallel([p1, p2])
        assert isinstance(group.processes, tuple)
        assert group.processes == (p1, p2)

    def test_accepts_single_process(self):
        p = Process("stt", "../../ai-services/stt-server", "stt_server")
        group = Parallel([p])
        assert len(group.processes) == 1

    def test_empty_parallel_ok(self):
        group = Parallel([])
        assert group.processes == ()

    def test_frozen_immutability(self):
        group = Parallel([])
        with pytest.raises((AttributeError, TypeError)):
            group.processes = ()  # type: ignore[misc]


class TestStripConflictingCudnn:
    """LD_LIBRARY_PATH sanitization so a host cuDNN can't shadow the venv one."""

    def _make_cudnn_dir(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        (d / "libcudnn.so.9").touch()
        return str(d)

    def test_none_and_empty_pass_through(self):
        assert _strip_conflicting_cudnn(None) == (None, [])
        assert _strip_conflicting_cudnn("") == ("", [])

    def test_no_cudnn_dirs_unchanged(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        value = os.pathsep.join([str(plain), "/usr/lib"])
        assert _strip_conflicting_cudnn(value) == (value, [])

    def test_drops_only_the_cudnn_dir(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        keep = tmp_path / "keep"
        keep.mkdir()
        value = os.pathsep.join([cudnn, str(keep)])
        cleaned, dropped = _strip_conflicting_cudnn(value)
        assert cleaned == str(keep)
        assert dropped == [cudnn]

    def test_returns_none_when_only_cudnn_dir(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        cleaned, dropped = _strip_conflicting_cudnn(cudnn)
        assert cleaned is None
        assert dropped == [cudnn]

    def test_preserves_empty_cwd_entry(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        # Leading "" => current-directory entry; must survive untouched.
        value = os.pathsep.join(["", cudnn, "/usr/lib"])
        cleaned, dropped = _strip_conflicting_cudnn(value)
        assert cleaned == os.pathsep.join(["", "/usr/lib"])
        assert dropped == [cudnn]

    def test_matches_versioned_soname(self, tmp_path):
        # glob libcudnn.so* must catch libcudnn.so.9.13.1 etc.
        d = tmp_path / "lib"
        d.mkdir()
        (d / "libcudnn.so.9.13.1").touch()
        cleaned, dropped = _strip_conflicting_cudnn(str(d))
        assert cleaned is None
        assert dropped == [str(d)]
