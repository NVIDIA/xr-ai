# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._cloudxr_env.load_cloudxr_env."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from xr_ai_launcher._cloudxr_env import XR_RUNTIME_VAR, load_cloudxr_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove XR_RUNTIME_JSON from the environment for each test."""
    monkeypatch.delenv(XR_RUNTIME_VAR, raising=False)
    monkeypatch.delenv("CLOUDXR_EXTRA", raising=False)
    monkeypatch.delenv("QUOTED_DOUBLE", raising=False)
    monkeypatch.delenv("QUOTED_SINGLE", raising=False)
    monkeypatch.delenv("NO_EXPORT", raising=False)


def _write_env(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cloudxr.env"
    p.write_text(textwrap.dedent(content))
    return p


class TestBasicParsing:
    def test_export_prefix_stripped(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\
            export {XR_RUNTIME_VAR}=/etc/xr/runtime.json
        """)
        load_cloudxr_env(env_file)
        assert os.environ[XR_RUNTIME_VAR] == "/etc/xr/runtime.json"

    def test_no_export_prefix(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\
            NO_EXPORT=plain_value
        """)
        load_cloudxr_env(env_file)
        assert os.environ["NO_EXPORT"] == "plain_value"

    def test_double_quoted_value_stripped(self, tmp_path):
        env_file = _write_env(tmp_path, """\
            export QUOTED_DOUBLE="/some/path/with spaces"
        """)
        load_cloudxr_env(env_file)
        assert os.environ["QUOTED_DOUBLE"] == "/some/path/with spaces"

    def test_single_quoted_value_stripped(self, tmp_path):
        env_file = _write_env(tmp_path, """\
            export QUOTED_SINGLE='another value'
        """)
        load_cloudxr_env(env_file)
        assert os.environ["QUOTED_SINGLE"] == "another value"

    def test_multiple_vars_loaded(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\
            export {XR_RUNTIME_VAR}=/runtime.json
            export CLOUDXR_EXTRA=extra_val
        """)
        load_cloudxr_env(env_file)
        assert os.environ[XR_RUNTIME_VAR] == "/runtime.json"
        assert os.environ["CLOUDXR_EXTRA"] == "extra_val"


class TestCommentAndBlankHandling:
    def test_blank_lines_ignored(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\

            export {XR_RUNTIME_VAR}=/runtime.json

        """)
        load_cloudxr_env(env_file)
        assert os.environ[XR_RUNTIME_VAR] == "/runtime.json"

    def test_comment_lines_ignored(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\
            # This is a comment
            export {XR_RUNTIME_VAR}=/runtime.json
            # Another comment
        """)
        load_cloudxr_env(env_file)
        assert os.environ[XR_RUNTIME_VAR] == "/runtime.json"

    def test_invalid_lines_skipped(self, tmp_path):
        env_file = _write_env(tmp_path, f"""\
            this is not valid
            export {XR_RUNTIME_VAR}=/runtime.json
            ===also invalid
        """)
        load_cloudxr_env(env_file)
        # Only the valid line should be loaded.
        assert os.environ[XR_RUNTIME_VAR] == "/runtime.json"


class TestEdgeCases:
    def test_empty_file_ok(self, tmp_path):
        env_file = tmp_path / "cloudxr.env"
        env_file.write_text("")
        load_cloudxr_env(env_file)  # must not raise

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_cloudxr_env(tmp_path / "nonexistent.env")

    def test_value_with_equals_sign(self, tmp_path):
        """Values that contain '=' must not be truncated."""
        env_file = _write_env(tmp_path, """\
            export CLOUDXR_EXTRA=key=value
        """)
        load_cloudxr_env(env_file)
        assert os.environ["CLOUDXR_EXTRA"] == "key=value"
