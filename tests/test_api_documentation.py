# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static public API documentation contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = _ROOT / "docs" / "source" / "_api_contract.py"


def _load_contract() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_api_contract_test", _CONTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "example_api"
    package.mkdir()
    (package / "__init__.py").write_text(source, encoding="utf-8")
    return package


def test_documented_public_api_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(
        tmp_path,
        '''
class Client:
    """A documented client."""

    def run(self) -> None:
        """Run the client."""


DEFAULT_CLIENT = Client()
"""The default client instance."""

__all__ = ["Client", "DEFAULT_CLIENT"]
''',
    )
    monkeypatch.setattr(contract, "API_PACKAGE_DIRS", (package,))

    assert contract.validate_public_api() == []


def test_undocumented_public_members_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(
        tmp_path,
        '''
class Client:
    """A documented client."""

    def run(self) -> None:
        pass


DEFAULT_CLIENT = Client()
__all__ = ["Client", "DEFAULT_CLIENT", "Missing"]
''',
    )
    monkeypatch.setattr(contract, "API_PACKAGE_DIRS", (package,))

    assert contract.validate_public_api() == [
        "example_api: public method Client.run has no docstring",
        "example_api: exported DEFAULT_CLIENT has no docstring",
        "example_api: exported Missing does not resolve",
    ]
