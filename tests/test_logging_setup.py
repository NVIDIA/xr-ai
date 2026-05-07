# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_logging.setup_logging."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from loguru import logger

from xr_ai_logging import setup_logging


@pytest.fixture(autouse=True)
def _clean_log_env(monkeypatch, tmp_path):
    """Redirect log output to tmp_path and clear timestamp/namespace stamps.

    `setup_logging` calls `logger.remove()` to wipe loguru's default sink.
    Add a teardown that does the same so each test leaves the global loguru
    state empty — otherwise a sink added by one test would still receive
    records emitted by the next.
    """
    monkeypatch.delenv("XR_AI_LOG_NAMESPACE", raising=False)
    monkeypatch.delenv("XR_AI_LOG_TIMESTAMP", raising=False)
    monkeypatch.setenv("XR_AI_LOG_ROOT", str(tmp_path))
    yield
    logger.remove()


class TestSetupLoggingReturnValue:
    def test_returns_path_object(self, tmp_path):
        result = setup_logging("test-proc")
        assert isinstance(result, Path)

    def test_log_file_created(self, tmp_path):
        # Loguru's `add(path, enqueue=True)` opens the file synchronously
        # before returning, so the path exists immediately even though no
        # record has been written yet.
        log_file = setup_logging("test-proc")
        assert log_file.exists()

    def test_log_file_name_matches_process_name(self, tmp_path):
        log_file = setup_logging("myservice")
        assert log_file.name == "myservice.log"


class TestNamespaceHandling:
    def test_explicit_namespace_used(self, tmp_path):
        log_file = setup_logging("worker", namespace="xr-render-demo")
        # Directory name contains the namespace.
        assert "xr-render-demo" in str(log_file.parent)

    def test_namespace_env_var_as_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XR_AI_LOG_NAMESPACE", "from-env")
        # First call sets the timestamp env var.
        monkeypatch.delenv("XR_AI_LOG_TIMESTAMP", raising=False)
        log_file = setup_logging("worker")
        assert "from-env" in str(log_file.parent)

    def test_name_used_as_namespace_when_both_absent(self, tmp_path):
        log_file = setup_logging("fallback-name")
        assert "fallback-name" in str(log_file.parent)

    def test_namespace_env_var_stamped_for_subprocesses(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XR_AI_LOG_NAMESPACE", raising=False)
        setup_logging("orch", namespace="my-demo")
        assert os.environ.get("XR_AI_LOG_NAMESPACE") == "my-demo"


class TestTimestampStamping:
    def test_timestamp_env_var_set_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XR_AI_LOG_TIMESTAMP", raising=False)
        setup_logging("proc")
        assert "XR_AI_LOG_TIMESTAMP" in os.environ

    def test_existing_timestamp_reused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XR_AI_LOG_TIMESTAMP", "2026-01-01_00-00-00")
        log_file = setup_logging("proc")
        assert "2026-01-01_00-00-00" in str(log_file)
        # Env var must not change.
        assert os.environ["XR_AI_LOG_TIMESTAMP"] == "2026-01-01_00-00-00"


class TestInterceptHandler:
    def test_stdlib_log_record_reaches_loguru(self, tmp_path, caplog):
        """Records emitted via stdlib logging must be intercepted without error."""
        setup_logging("test-intercept")
        std_logger = logging.getLogger("test.intercept.module")
        # Should not raise even though loguru is now the backend.
        std_logger.warning("stdlib warning via intercept handler")


class TestVerboseFlag:
    def test_non_verbose_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XR_AI_VERBOSE", raising=False)
        # Should not raise; just verifies it runs without verbose=True issues.
        setup_logging("proc")

    def test_verbose_flag_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XR_AI_VERBOSE", "1")
        setup_logging("verbose-proc")  # must not raise
