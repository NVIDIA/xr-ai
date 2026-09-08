# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_vllm configuration and lifecycle helpers."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from xr_ai_vllm._config import parse_config_bool, setup_hf_env
from xr_ai_vllm._lifecycle import health_ok, health_url, wait_until_healthy


class TestParseConfigBool:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("true", True),
            ("YES", True),
            ("1", True),
            ("on", True),
            ("false", False),
            ("No", False),
            ("0", False),
            (" off ", False),
        ],
    )
    def test_accepts_booleans_and_known_strings(self, value, expected):
        assert parse_config_bool(value, "enforce_eager") is expected

    @pytest.mark.parametrize("value", ["sometimes", 0, None])
    def test_rejects_other_values(self, value):
        with pytest.raises(ValueError, match="enforce_eager"):
            parse_config_bool(value, "enforce_eager")


class TestSetupHfEnv:
    _ENV_KEYS = (
        "HF_TOKEN",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    )

    @pytest.fixture(autouse=True)
    def _guard_hf_environment(self, monkeypatch):
        # monkeypatch only restores keys that existed at setup; setup_hf_env
        # writes new ones, so pop those explicitly.
        for key in self._ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        yield
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_env_token_beats_yaml_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_tok")

        setup_hf_env({"hf_token": "yaml_tok"}, tmp_path)

        assert os.environ["HF_TOKEN"] == "env_tok"

    def test_yaml_token_used_when_env_unset(self, tmp_path):
        setup_hf_env({"hf_token": "yaml_tok"}, tmp_path)

        assert os.environ["HF_TOKEN"] == "yaml_tok"

    def test_empty_env_token_falls_back_to_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")

        setup_hf_env({"hf_token": "yaml_tok"}, tmp_path)

        assert os.environ["HF_TOKEN"] == "yaml_tok"

    def test_no_token_leaves_env_unset(self, tmp_path):
        setup_hf_env({"hf_token": ""}, tmp_path)

        assert "HF_TOKEN" not in os.environ

    def test_env_hf_home_beats_model_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/external/cache")

        setup_hf_env({}, tmp_path)

        assert os.environ["HF_HOME"] == "/external/cache"

    def test_preserves_legacy_transfer_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")

        setup_hf_env({}, tmp_path)

        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
        assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
        assert os.environ["HF_HOME"] == str(tmp_path)
        assert os.environ["TRANSFORMERS_CACHE"] == str(tmp_path)

    def test_preserves_xet_high_performance_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")

        setup_hf_env({}, tmp_path)

        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"

    def test_preserves_xet_disable_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

        setup_hf_env({}, tmp_path)

        assert os.environ["HF_HUB_DISABLE_XET"] == "1"


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
