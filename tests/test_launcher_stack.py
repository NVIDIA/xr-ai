# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._stack data structures and pure helpers."""
from __future__ import annotations

import pytest

from xr_ai_launcher._stack import Parallel, Process


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
