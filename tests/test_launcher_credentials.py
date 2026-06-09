# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._credentials (read/write/load/ensure helpers)."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

# Import the private helpers directly to exercise them in isolation.
import xr_ai_launcher._credentials as _creds


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN",    raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)


@pytest.fixture()
def fake_creds_dir(tmp_path, monkeypatch):
    """Point _CREDS_FILE and _HF_TOKEN_FILE to tmp_path so tests never
    touch ~/.config or ~/.cache."""
    creds_file   = tmp_path / "credentials.json"
    hf_token_file = tmp_path / "hf_token"
    monkeypatch.setattr(_creds, "_CREDS_FILE",    creds_file)
    monkeypatch.setattr(_creds, "_HF_TOKEN_FILE", hf_token_file)
    return creds_file, hf_token_file


class TestReadWrite:
    def test_read_missing_file_returns_empty(self, fake_creds_dir):
        assert _creds._read() == {}

    def test_read_valid_json(self, fake_creds_dir):
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "tok123"}))
        assert _creds._read() == {"HF_TOKEN": "tok123"}

    def test_read_corrupt_json_returns_empty(self, fake_creds_dir):
        creds_file, _ = fake_creds_dir
        creds_file.write_text("not json{{{")
        assert _creds._read() == {}

    def test_write_creates_file_with_restricted_permissions(self, fake_creds_dir):
        _creds._write({"HF_TOKEN": "abc"})
        creds_file, _ = fake_creds_dir
        assert creds_file.exists()
        assert json.loads(creds_file.read_text()) == {"HF_TOKEN": "abc"}
        assert oct(creds_file.stat().st_mode & 0o777) == oct(0o600)

    def test_read_filters_non_string_values(self, fake_creds_dir):
        """_read skips non-string values written by other tools."""
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "tok", "BAD": 123}))
        result = _creds._read()
        assert "BAD" not in result
        assert result["HF_TOKEN"] == "tok"


class TestHFTokenFile:
    def test_read_hf_token_file_missing_returns_empty(self, fake_creds_dir):
        assert _creds._read_hf_token_file() == ""

    def test_read_hf_token_file_strips_whitespace(self, fake_creds_dir):
        _, hf_file = fake_creds_dir
        hf_file.write_text("  mytoken\n")
        assert _creds._read_hf_token_file() == "mytoken"

    def test_write_hf_token_file_sets_permissions(self, fake_creds_dir):
        _, hf_file = fake_creds_dir
        _creds._write_hf_token_file("secret")
        assert hf_file.read_text().strip() == "secret"
        assert oct(hf_file.stat().st_mode & 0o777) == oct(0o600)


class TestLoadCredentials:
    def test_load_injects_saved_values(self, fake_creds_dir, monkeypatch):
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "saved_tok"}))
        _creds.load_credentials()
        assert os.environ.get("HF_TOKEN") == "saved_tok"

    def test_load_does_not_override_existing_env(self, fake_creds_dir, monkeypatch):
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "saved_tok"}))
        monkeypatch.setenv("HF_TOKEN", "env_tok")
        _creds.load_credentials()
        assert os.environ["HF_TOKEN"] == "env_tok"

    def test_load_picks_up_hf_token_file(self, fake_creds_dir, monkeypatch):
        _, hf_file = fake_creds_dir
        hf_file.write_text("file_token\n")
        _creds.load_credentials()
        assert os.environ.get("HF_TOKEN") == "file_token"

    def test_load_env_takes_precedence_over_hf_file(self, fake_creds_dir, monkeypatch):
        _, hf_file = fake_creds_dir
        hf_file.write_text("file_token\n")
        monkeypatch.setenv("HF_TOKEN", "env_token")
        _creds.load_credentials()
        assert os.environ["HF_TOKEN"] == "env_token"


class TestEnsureCredentials:
    def test_env_already_set_is_not_prompted(self, fake_creds_dir, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "already_set")
        # If getpass were called this would hang; the test would timeout.
        _creds.ensure_credentials("HF_TOKEN")
        assert os.environ["HF_TOKEN"] == "already_set"

    def test_saved_credential_loaded_without_prompt(self, fake_creds_dir, monkeypatch):
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "cached_tok"}))
        _creds.ensure_credentials("HF_TOKEN")
        assert os.environ.get("HF_TOKEN") == "cached_tok"

    def test_hf_token_file_used_without_prompt(self, fake_creds_dir, monkeypatch):
        _, hf_file = fake_creds_dir
        hf_file.write_text("file_tok\n")
        _creds.ensure_credentials("HF_TOKEN")
        assert os.environ.get("HF_TOKEN") == "file_tok"

    def test_skipped_token_not_saved(self, fake_creds_dir, monkeypatch):
        """Empty input (press Enter) leaves the token unset and nothing is written."""
        with patch("getpass.getpass", return_value=""):
            _creds.ensure_credentials("HF_TOKEN")
        assert not os.environ.get("HF_TOKEN")
        creds_file, _ = fake_creds_dir
        assert not creds_file.exists()

    def test_entered_token_saved_and_set(self, fake_creds_dir, monkeypatch):
        with patch("getpass.getpass", return_value="new_tok"):
            _creds.ensure_credentials("HF_TOKEN")
        creds_file, hf_file = fake_creds_dir
        assert os.environ.get("HF_TOKEN") == "new_tok"
        saved = json.loads(creds_file.read_text())
        assert saved["HF_TOKEN"] == "new_tok"
        assert hf_file.read_text().strip() == "new_tok"


class TestWarnIfMissing:
    """warn_if_missing surfaces a missing token but NEVER prompts."""

    def test_missing_token_does_not_prompt(self, fake_creds_dir, monkeypatch):
        # If getpass were called this would raise; warn_if_missing must not call it.
        with patch("getpass.getpass", side_effect=AssertionError("must not prompt")):
            _creds.warn_if_missing("HF_TOKEN")
        assert not os.environ.get("HF_TOKEN")  # still unset, nothing saved

    def test_missing_token_not_saved(self, fake_creds_dir, monkeypatch):
        _creds.warn_if_missing("HF_TOKEN")
        creds_file, _ = fake_creds_dir
        assert not creds_file.exists()

    def test_present_token_is_loaded_from_saved(self, fake_creds_dir, monkeypatch):
        creds_file, _ = fake_creds_dir
        creds_file.write_text(json.dumps({"HF_TOKEN": "saved_tok"}))
        with patch("getpass.getpass", side_effect=AssertionError("must not prompt")):
            _creds.warn_if_missing("HF_TOKEN")
        assert os.environ.get("HF_TOKEN") == "saved_tok"

    def test_env_token_short_circuits(self, fake_creds_dir, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_tok")
        with patch("getpass.getpass", side_effect=AssertionError("must not prompt")):
            _creds.warn_if_missing("HF_TOKEN")
        assert os.environ.get("HF_TOKEN") == "env_tok"
