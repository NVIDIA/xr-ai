# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_vllm._lifecycle pure helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xr_ai_vllm._lifecycle import health_ok, health_url, wait_until_healthy


class TestHealthUrl:
    def test_always_probes_localhost(self):
        # The host param is intentionally ignored — always 127.0.0.1.
        assert health_url("0.0.0.0", 8100) == "http://127.0.0.1:8100/health"
        assert health_url("192.168.1.1", 9000) == "http://127.0.0.1:9000/health"


class TestHealthOk:
    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch("xr_ai_vllm._lifecycle.urllib.request.urlopen", return_value=mock_resp):
            assert health_ok("http://127.0.0.1:8100/health")

    def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 503
        with patch("xr_ai_vllm._lifecycle.urllib.request.urlopen", return_value=mock_resp):
            assert not health_ok("http://127.0.0.1:8100/health")

    def test_returns_false_on_connection_error(self):
        with patch(
            "xr_ai_vllm._lifecycle.urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            assert not health_ok("http://127.0.0.1:8100/health")


class TestWaitUntilHealthy:
    def test_returns_immediately_when_healthy(self):
        with patch("xr_ai_vllm._lifecycle.health_ok", return_value=True):
            # is_alive always True; health immediately OK → should not raise.
            wait_until_healthy("http://127.0.0.1:8100/health", is_alive=lambda: True)

    def test_raises_system_exit_when_process_dies(self):
        # health_ok always False; is_alive returns False on first call.
        call_count = [0]

        def _dead():
            call_count[0] += 1
            return False

        with patch("xr_ai_vllm._lifecycle.health_ok", return_value=False), \
             patch("xr_ai_vllm._lifecycle.time.sleep"):
            with pytest.raises(SystemExit):
                wait_until_healthy("http://127.0.0.1:8100/health", is_alive=_dead)

    def test_polls_until_healthy(self):
        """health_ok returns False once, then True — must not raise."""
        results = [False, True]

        def _health(_url, **_kw):
            return results.pop(0) if results else True

        with patch("xr_ai_vllm._lifecycle.health_ok", side_effect=_health), \
             patch("xr_ai_vllm._lifecycle.time.sleep"):
            wait_until_healthy("http://127.0.0.1:8100/health", is_alive=lambda: True)
