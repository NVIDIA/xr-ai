# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._cloudxr_env.load_cloudxr_env."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from xr_ai_launcher._cloudxr_env import (
    NATIVE_DEVICE_PROFILES,
    XR_RUNTIME_VAR,
    is_native_profile,
    load_cloudxr_env,
    read_device_profile,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove XR_RUNTIME_JSON from the environment for each test."""
    monkeypatch.delenv(XR_RUNTIME_VAR, raising=False)
    monkeypatch.delenv("CLOUDXR_EXTRA", raising=False)
    monkeypatch.delenv("QUOTED_DOUBLE", raising=False)
    monkeypatch.delenv("QUOTED_SINGLE", raising=False)
    monkeypatch.delenv("NO_EXPORT", raising=False)
    monkeypatch.delenv("NV_DEVICE_PROFILE", raising=False)


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


class TestIsNativeProfile:
    @pytest.mark.parametrize("profile", sorted(NATIVE_DEVICE_PROFILES))
    def test_native_profiles(self, profile):
        assert is_native_profile(profile) is True

    def test_webrtc_profile_is_not_native(self):
        assert is_native_profile("auto-webrtc") is False

    def test_empty_string_is_not_native(self):
        assert is_native_profile("") is False

    def test_whitespace_is_stripped(self):
        assert is_native_profile("  auto-native  ") is True

    def test_case_insensitive(self):
        assert is_native_profile("Apple-Vision-Pro") is True


class TestReadDeviceProfile:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "cloudxr_runtime.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_env_set_wins_over_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NV_DEVICE_PROFILE", "auto-native")
        yaml_path = self._write_yaml(tmp_path, """\
            cloudxr_env:
              NV_DEVICE_PROFILE: auto-webrtc
        """)
        assert read_device_profile(yaml_path) == "auto-native"

    def test_env_unset_falls_back_to_yaml(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path, """\
            cloudxr_env:
              NV_DEVICE_PROFILE: "apple-vision-pro"
        """)
        assert read_device_profile(yaml_path) == "apple-vision-pro"

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_device_profile(tmp_path / "nonexistent.yaml") == ""

    def test_no_match_in_yaml_returns_empty(self, tmp_path):
        yaml_path = self._write_yaml(tmp_path, """\
            cloudxr_env:
              SOMETHING_ELSE: value
        """)
        assert read_device_profile(yaml_path) == ""

    def test_falsy_path_returns_empty(self):
        assert read_device_profile(None) == ""
