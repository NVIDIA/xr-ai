# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static public API documentation contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = _ROOT / "docs" / "source" / "_api_contract.py"
_REFERENCE_CHECK = _ROOT / "docs" / "source" / "_api_reference_check.py"


def _load_contract() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_api_contract_test", _CONTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def _load_reference_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_api_reference_check_test", _REFERENCE_CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "example_api"
    package.mkdir()
    (package / "__init__.py").write_text(source, encoding="utf-8")
    return package


def _enroll(
    monkeypatch: pytest.MonkeyPatch,
    contract: ModuleType,
    package: Path,
    public_modules: tuple[str, ...] = (),
    exclusions: tuple[str, ...] = (),
) -> None:
    monkeypatch.setattr(contract, "API_PACKAGE_DIRS", (package,))
    monkeypatch.setattr(contract, "PUBLIC_API_MODULES", public_modules)
    monkeypatch.setattr(contract, "PUBLIC_API_EXCLUSIONS", exclusions)


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
    _enroll(monkeypatch, contract, package)

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
    _enroll(monkeypatch, contract, package)

    assert contract.validate_public_api() == [
        "example_api: public method Client.run has no docstring",
        "example_api: exported DEFAULT_CLIENT has no docstring",
        "example_api: exported Missing does not resolve",
    ]


def test_unbound_submodule_definition_does_not_satisfy_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(tmp_path, '__all__ = ["Client"]\n')
    (package / "implementation.py").write_text(
        '''
class Client:
    """A class that the facade never imports."""
''',
        encoding="utf-8",
    )
    _enroll(
        monkeypatch,
        contract,
        package,
        exclusions=("example_api.implementation",),
    )

    assert contract.validate_public_api() == ["example_api: exported Client does not resolve"]


def test_reexport_resolves_exact_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(
        tmp_path,
        '''
from .public import Client

__all__ = ["Client"]
''',
    )
    (package / "public.py").write_text(
        '''
class Client:
    """The class bound by the facade."""
''',
        encoding="utf-8",
    )
    (package / "unrelated.py").write_text(
        """
class Client:
    pass
""",
        encoding="utf-8",
    )
    _enroll(
        monkeypatch,
        contract,
        package,
        exclusions=("example_api.public", "example_api.unrelated"),
    )

    assert contract.validate_public_api() == []


def test_import_source_must_bind_exported_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(
        tmp_path,
        '''
from .source import Client

__all__ = ["Client"]
''',
    )
    (package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "unrelated.py").write_text(
        '''
class Client:
    """An unrelated documented class."""
''',
        encoding="utf-8",
    )
    _enroll(
        monkeypatch,
        contract,
        package,
        exclusions=("example_api.source", "example_api.unrelated"),
    )

    assert contract.validate_public_api() == ["example_api: exported Client does not resolve"]


def test_undocumented_dataclass_and_pydantic_fields_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(
        tmp_path,
        '''
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class Item:
    """A documented dataclass."""

    documented: str
    """A documented field."""

    missing: int


class Request(BaseModel):
    """A documented Pydantic model."""

    missing: str


__all__ = ["Item", "Request"]
''',
    )
    _enroll(monkeypatch, contract, package)

    assert contract.validate_public_api() == [
        "example_api: public field Item.missing has no docstring",
        "example_api: public field Request.missing has no docstring",
    ]


def test_public_module_checks_its_local_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(tmp_path, "__all__ = []\n")
    (package / "public.py").write_text(
        '''
class Client:
    pass


__all__ = ["Client"]
''',
        encoding="utf-8",
    )
    (package / "other.py").write_text(
        '''
class Client:
    """An unrelated documented class with the same name."""
''',
        encoding="utf-8",
    )
    _enroll(
        monkeypatch,
        contract,
        package,
        ("example_api.public",),
        ("example_api.other",),
    )

    assert contract.validate_public_api() == [
        "example_api.public: exported Client has no docstring"
    ]


def test_public_module_resolves_documented_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(tmp_path, "__all__ = []\n")
    (package / "public.py").write_text(
        '''
from .implementation import Client

__all__ = ["Client"]
''',
        encoding="utf-8",
    )
    (package / "implementation.py").write_text(
        '''
class Client:
    """A documented client."""
''',
        encoding="utf-8",
    )
    _enroll(
        monkeypatch,
        contract,
        package,
        ("example_api.public",),
        ("example_api.implementation",),
    )

    assert contract.validate_public_api() == []


def test_public_subpackage_is_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(tmp_path, "__all__ = []\n")
    subpackage = package / "extras"
    subpackage.mkdir()
    subpackage.joinpath("__init__.py").write_text(
        '''
class Client:
    """A documented client in a public subpackage."""


__all__ = ["Client"]
''',
        encoding="utf-8",
    )
    _enroll(monkeypatch, contract, package, ("example_api.extras",))

    assert contract.validate_public_api() == []


def test_importable_module_must_be_enrolled_or_explicitly_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load_contract()
    package = _package(tmp_path, "__all__ = []\n")
    (package / "extra.py").write_text("VALUE = 1\n", encoding="utf-8")
    _enroll(monkeypatch, contract, package)

    assert contract.validate_public_api() == [
        "example_api.extra: importable module is neither enrolled nor excluded"
    ]

    monkeypatch.setattr(contract, "PUBLIC_API_EXCLUSIONS", ("example_api.extra",))
    assert contract.validate_public_api() == []


def test_generated_reference_rejects_private_implementation_details(
    tmp_path: Path,
) -> None:
    reference_check = _load_reference_check()
    models = tmp_path / "reference" / "python" / "xr_ai_models" / "index.html"
    models.parent.mkdir(parents=True)
    models.write_text("xr_ai_models._protocols.LLMService", encoding="utf-8")
    voice = tmp_path / "reference" / "python" / "xr_ai_voice" / "index.html"
    voice.parent.mkdir(parents=True)
    voice.write_text(
        "Pipecat XRMediaHubInputTransport XRMediaHubOutputTransport",
        encoding="utf-8",
    )

    assert reference_check.validate_generated_api(tmp_path) == [
        "generated API exposes private module path: xr_ai_models._protocols.LLMService",
        "generated API exposes private implementation detail: Pipecat",
        "generated API exposes private implementation detail: XRMediaHubInputTransport",
        "generated API exposes private implementation detail: XRMediaHubOutputTransport",
    ]


def test_generated_reference_accepts_public_surface(tmp_path: Path) -> None:
    reference_check = _load_reference_check()
    reference = tmp_path / "reference" / "python" / "xr_ai_voice" / "index.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("VoiceAgent HubVoiceTransport", encoding="utf-8")

    assert reference_check.validate_generated_api(tmp_path) == []


def test_utility_packages_are_enrolled() -> None:
    contract = _load_contract()

    enrolled = {path.name for path in contract.API_PACKAGE_DIRS}

    assert {
        "xr_ai_launcher",
        "xr_ai_logging",
        "xr_ai_vad",
        "xr_ai_vllm",
        "xr_ai_voicegate",
    } <= enrolled
