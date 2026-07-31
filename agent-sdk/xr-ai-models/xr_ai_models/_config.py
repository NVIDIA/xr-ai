# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``models.yaml`` schema, typed specs, and the loader.

One ``models.yaml`` lives next to each sample (``yaml/models.yaml``) and maps
logical names (``llm``, ``agent_llm``, ``vlm``, ``stt``, ``tts``, …) to a spec
that either inlines all knobs or references a built-in preset
(``kind: preset:<name>``).  Presets fill in everything except ``base_url``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from . import presets as _presets
from ._utils import merge_dicts


Category = Literal["llm", "vlm", "stt", "tts"]
ModelKind = Literal["openai_compat"]
Ownership = Literal["managed", "reused", "external"]

KIND_OPENAI_COMPAT: ModelKind = "openai_compat"


@dataclass(frozen=True)
class DeploymentSpec:
    """Process ownership for the service behind a model role."""

    ownership: Ownership = "external"
    service: str | None = None


@dataclass(frozen=True)
class LLMSpec:
    kind: ModelKind = KIND_OPENAI_COMPAT
    base_url: str = ""
    model_name: str = ""
    api_key_env: str | None = None
    reasoning_field: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    default_extras: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    health_check: bool = True
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)


@dataclass(frozen=True)
class VLMSpec:
    kind: ModelKind = KIND_OPENAI_COMPAT
    base_url: str = ""
    model_name: str = ""
    api_key_env: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    default_extras: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    health_check: bool = True
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)


@dataclass(frozen=True)
class STTSpec:
    kind: ModelKind = KIND_OPENAI_COMPAT
    base_url: str = ""
    api_key_env: str | None = None
    timeout: float = 30.0
    health_check: bool = True
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)


@dataclass(frozen=True)
class TTSSpec:
    kind: ModelKind = KIND_OPENAI_COMPAT
    base_url: str = ""
    api_key_env: str | None = None
    timeout: float = 30.0
    health_check: bool = True
    deployment: DeploymentSpec = field(default_factory=DeploymentSpec)


Spec = LLMSpec | VLMSpec | STTSpec | TTSSpec
T = TypeVar("T", LLMSpec, VLMSpec, STTSpec, TTSSpec)


@dataclass(frozen=True)
class ModelsConfig:
    """Logical-name → spec map for one sample/process.

    Use :func:`load_models_config` to build from YAML.  ``entries`` is keyed
    by logical name; the typed accessors (:py:meth:`llm`, :py:meth:`vlm`,
    :py:meth:`stt`, :py:meth:`tts`) cast and validate at lookup time.
    """
    entries: dict[str, Spec]

    def llm(self, name: str) -> LLMSpec:
        return _typed(self.entries, name, LLMSpec)

    def vlm(self, name: str) -> VLMSpec:
        return _typed(self.entries, name, VLMSpec)

    def stt(self, name: str) -> STTSpec:
        return _typed(self.entries, name, STTSpec)

    def tts(self, name: str) -> TTSSpec:
        return _typed(self.entries, name, TTSSpec)

    @property
    def required_credentials(self) -> tuple[str, ...]:
        return tuple(sorted({
            spec.api_key_env
            for spec in self.entries.values()
            if spec.api_key_env
        }))


def _typed(entries: dict[str, Spec], name: str, cls: type[T]) -> T:
    try:
        spec = entries[name]
    except KeyError as exc:
        raise KeyError(f"no spec named {name!r} in models config") from exc
    if not isinstance(spec, cls):
        raise TypeError(
            f"spec {name!r} is {type(spec).__name__}, expected {cls.__name__}"
        )
    return spec


def load_models_config(path: Path | str) -> ModelsConfig:
    """Load a ``models.yaml`` and resolve any ``kind: preset:<name>`` refs."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return load_models_config_from_dict(raw, source=str(path))


def load_models_config_from_dict(
    raw: dict[str, Any], *, source: str = "<dict>"
) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from an already-parsed mapping.

    Same semantics as :func:`load_models_config` minus the file I/O — callers
    that already hold the mapping (e.g. an embedded YAML block in a larger
    config) can skip the disk round-trip. ``source`` is only used to label
    validation errors.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top-level must be a mapping")
    models = raw.get("models", raw)
    if not isinstance(models, dict):
        raise ValueError(f"{source}: 'models' must be a mapping")
    entries: dict[str, Spec] = {}
    for name, body in models.items():
        if not isinstance(body, dict):
            raise ValueError(f"{source}: entry {name!r} must be a mapping")
        entries[name] = _build_spec(name, _flatten_profile_entry(name, body))
    return ModelsConfig(entries=entries)


def _flatten_profile_entry(name: str, body: dict[str, Any]) -> dict[str, Any]:
    if not any(key in body for key in ("adapter", "endpoint", "deployment")):
        return dict(body)

    adapter = body.get("adapter") or {}
    endpoint = body.get("endpoint") or {}
    deployment = body.get("deployment") or {}
    for label, value in (("adapter", adapter), ("endpoint", endpoint), ("deployment", deployment)):
        if not isinstance(value, dict):
            raise ValueError(f"{name!r}: {label} must be a mapping")
    if "api_key_env" in body:
        raise ValueError(
            f"{name!r}: structured profiles must declare api_key_env in endpoint"
        )

    flattened = {
        key: value
        for key, value in body.items()
        if key not in {"adapter", "endpoint", "deployment"}
    }
    flattened.update(adapter)
    flattened.update(endpoint)
    if "preset" in adapter:
        flattened["kind"] = f"preset:{adapter['preset']}"
    if "readiness" in endpoint:
        readiness = endpoint["readiness"]
        if readiness not in {"health", "none"}:
            raise ValueError(f"unsupported readiness policy: {readiness!r}")
        flattened["health_check"] = readiness == "health"
    flattened["deployment"] = deployment
    return flattened


def _build_spec(name: str, body: dict[str, Any]) -> Spec:
    resolved, preset_category = _resolve_preset(body)
    explicit_category = resolved.get("category")
    if preset_category is not None and explicit_category is not None \
            and explicit_category != preset_category:
        raise ValueError(
            f"{name!r}: category mismatch — preset gave {preset_category!r}"
            f" but entry overrides to {explicit_category!r}"
        )
    category = preset_category or explicit_category
    if category not in {"llm", "vlm", "stt", "tts"}:
        raise ValueError(
            f"{name!r}: missing or unknown category (got {category!r});"
            " presets set it implicitly — set ``category:`` if not using one"
        )
    return _construct(category, resolved)


def _resolve_preset(body: dict[str, Any]) -> tuple[dict[str, Any], Category | None]:
    kind = body.get("kind", KIND_OPENAI_COMPAT)
    if not isinstance(kind, str) or not kind.startswith("preset:"):
        return dict(body), None
    preset_name = kind.split(":", 1)[1]
    preset = _presets.get_preset(preset_name)
    forbidden = {"base_url", "api_key_env", "health_check", "deployment"} & preset.keys()
    if forbidden:
        raise ValueError(
            f"preset {preset_name!r} contains endpoint or deployment fields: "
            f"{sorted(forbidden)}"
        )
    merged = merge_dicts(preset, body, skip_keys=("kind",))
    merged["kind"] = preset.get("kind", KIND_OPENAI_COMPAT)
    return merged, preset["category"]


def _construct(category: Category, body: dict[str, Any]) -> Spec:
    common: dict[str, Any] = {
        "kind":     body.get("kind", KIND_OPENAI_COMPAT),
        "base_url": _require_str(body, "base_url"),
        # Remote endpoints (e.g. hosted NIM) have no local /health route; set
        # ``health_check: false`` so the worker readiness gate doesn't block.
        "health_check": bool(body.get("health_check", True)),
        "deployment": _deployment(body.get("deployment") or {}),
    }
    if "api_key_env" in body:
        common["api_key_env"] = _require_str(body, "api_key_env")
    if category in ("llm", "vlm"):
        return _construct_chat(category, body, common)
    if category == "stt":
        return STTSpec(**common, timeout=float(body.get("timeout", 30.0)))
    if category == "tts":
        return TTSSpec(**common, timeout=float(body.get("timeout", 30.0)))
    raise AssertionError(category)


def _deployment(body: dict[str, Any]) -> DeploymentSpec:
    ownership = body.get("ownership", "external")
    if ownership not in {"managed", "reused", "external"}:
        raise ValueError(f"unsupported deployment ownership: {ownership!r}")
    service = body.get("service")
    if service is not None and (not isinstance(service, str) or not service):
        raise ValueError("deployment service must be a non-empty string")
    if ownership != "external" and not service:
        raise ValueError(f"{ownership} deployments require a service name")
    return DeploymentSpec(ownership=ownership, service=service)


def _construct_chat(category: Category, body: dict[str, Any], common: dict[str, Any]) -> Spec:
    chat_common = {
        **common,
        "model_name":     _require_str(body, "model_name"),
        "capabilities":   dict(body.get("capabilities") or {}),
        "default_extras": dict(body.get("default_extras") or {}),
        "timeout":        float(body.get("timeout", 60.0)),
    }
    if category == "llm":
        return LLMSpec(reasoning_field=body.get("reasoning_field"), **chat_common)
    return VLMSpec(**chat_common)


def _require_str(body: dict[str, Any], key: str) -> str:
    val = body.get(key)
    if not isinstance(val, str) or not val:
        raise ValueError(f"missing required string field {key!r}")
    return val
